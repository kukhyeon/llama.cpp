// GGML_HTP_CHAIN_SPLITS: consecutive splits on a backend that provides
// "ggml_backend_graph_compute_nowait" are submitted without waiting, and the
// last split of the run waits for all of them. A fake NPU executes graphs in
// order on its own thread after a delay, like the Hexagon DSP queue, so a
// scheduler that reads an output early or skips a needed wait computes wrong
// values. Route candidates isolate NPU nodes into separate splits, as in the
// TriPrefill attention-output -> FFN-input -> FFN-norm chain.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool check(bool condition, const char * scenario, const char * detail) {
    if (!condition) {
        std::fprintf(stderr, "%s: %s\n", scenario, detail);
    }
    return condition;
}

struct npu_job {
    ggml_cgraph graph;
    std::string first_node;
    bool nowait = false;
    bool fail = false;
    int64_t submit_us = 0;
    int64_t start_us = 0;
    int64_t end_us = 0;
};

struct fake_npu {
    ggml_backend_t inner = nullptr;
    ggml_backend_buffer_type_t inner_buft = nullptr;
    ggml_backend_buffer_type buft = {};
    ggml_backend_device device = {};
    ggml_backend_reg reg = {};
    bool provide_nowait = true;
    std::string fail_node; // a graph starting with this node reports failure

    std::mutex mutex;
    std::condition_variable cv;
    std::deque<npu_job *> queue;
    std::vector<npu_job *> done;
    bool busy = false;
    bool stopping = false;
    bool failed = false;
    std::thread worker;

    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            cv.wait(lock, [&] { return stopping || !queue.empty(); });
            if (queue.empty()) {
                return;
            }
            npu_job * job = queue.front();
            queue.pop_front();
            busy = true;
            lock.unlock();
            job->start_us = now_us();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const bool ok = ggml_backend_graph_compute(inner, &job->graph) == GGML_STATUS_SUCCESS && !job->fail;
            job->end_us = now_us();
            lock.lock();
            failed = failed || !ok;
            done.push_back(job);
            busy = false;
            cv.notify_all();
        }
    }

    void submit(ggml_cgraph * graph, bool nowait) {
        auto * job = new npu_job;
        job->graph = *graph; // the node array outlives the call; the header may not
        job->first_node = graph->n_nodes > 0 ? ggml_get_name(graph->nodes[0]) : "";
        job->nowait = nowait;
        job->fail = !fail_node.empty() && job->first_node == fail_node;
        job->submit_us = now_us();
        std::lock_guard<std::mutex> lock(mutex);
        queue.push_back(job);
        cv.notify_all();
    }

    // Returns the failure state of everything submitted so far and clears it,
    // like the Hexagon session status collected by its next waiting call.
    bool wait_idle() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return queue.empty() && !busy; });
        const bool ok = !failed;
        failed = false;
        return ok;
    }

    ~fake_npu() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            cv.notify_all();
        }
        if (worker.joinable()) {
            worker.join();
        }
        for (npu_job * job : done) {
            delete job;
        }
        ggml_backend_free(inner);
    }
};

static fake_npu * npu_of(ggml_backend_t backend) { return static_cast<fake_npu *>(backend->context); }
static fake_npu * npu_of(ggml_backend_dev_t device) { return static_cast<fake_npu *>(device->context); }
static fake_npu * npu_of(ggml_backend_buffer_type_t buft) { return static_cast<fake_npu *>(buft->context); }
static fake_npu * npu_of(ggml_backend_reg_t reg) { return static_cast<fake_npu *>(reg->context); }

static const char * npu_buft_name(ggml_backend_buffer_type_t) { return "HTP-ChainTest"; }
static ggml_backend_buffer_t npu_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(npu_of(buft)->inner_buft, size);
    if (buffer != nullptr) {
        buffer->buft = buft;
    }
    return buffer;
}
static size_t npu_buft_alignment(ggml_backend_buffer_type_t buft) { return ggml_backend_buft_get_alignment(npu_of(buft)->inner_buft); }
static size_t npu_buft_max_size(ggml_backend_buffer_type_t buft) { return ggml_backend_buft_get_max_size(npu_of(buft)->inner_buft); }
static size_t npu_buft_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    return ggml_backend_buft_get_alloc_size(npu_of(buft)->inner_buft, t);
}
// Host memory, like rpcmem: the CPU reads NPU outputs in place.
static bool npu_buft_is_host(ggml_backend_buffer_type_t) { return true; }

static const ggml_backend_buffer_type_i npu_buft_iface = {
    /* .get_name       = */ npu_buft_name,
    /* .alloc_buffer   = */ npu_buft_alloc,
    /* .get_alignment  = */ npu_buft_alignment,
    /* .get_max_size   = */ npu_buft_max_size,
    /* .get_alloc_size = */ npu_buft_alloc_size,
    /* .is_host        = */ npu_buft_is_host,
};

static const char * npu_name(ggml_backend_t) { return "HTP-ChainTest"; }
static void npu_free(ggml_backend_t backend) {
    delete npu_of(backend);
    delete backend;
}
static void npu_synchronize(ggml_backend_t backend) { (void) npu_of(backend)->wait_idle(); }
static enum ggml_status npu_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    fake_npu * npu = npu_of(backend);
    npu->submit(graph, false);
    return npu->wait_idle() ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}
static enum ggml_status npu_graph_compute_nowait(ggml_backend_t backend, ggml_cgraph * graph) {
    npu_of(backend)->submit(graph, true);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i npu_iface = {
    /* .get_name                = */ npu_name,
    /* .free                    = */ npu_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ npu_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ npu_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static const char * npu_dev_name(ggml_backend_dev_t) { return "HTP-ChainTest"; }
static void npu_dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = 1u << 30; *total = 1u << 30; }
static enum ggml_backend_dev_type npu_dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
static void npu_dev_props(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    props->name = npu_dev_name(device);
    props->description = props->name;
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->memory_free = 1u << 30;
    props->memory_total = 1u << 30;
    props->device_id = nullptr;
    props->caps = {};
}
static ggml_backend_t npu_dev_init(ggml_backend_dev_t, const char *) { return nullptr; }
static ggml_backend_buffer_type_t npu_dev_buft(ggml_backend_dev_t device) { return &npu_of(device)->buft; }
static ggml_backend_buffer_type_t npu_dev_host_buft(ggml_backend_dev_t) { return nullptr; }
static bool npu_dev_supports_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    return ggml_backend_supports_op(npu_of(device)->inner, op);
}
// Only its own memory: CPU tensors reach the NPU through scheduler copies.
static bool npu_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return buft == &npu_of(device)->buft;
}
static bool npu_dev_offload_op(ggml_backend_dev_t, const ggml_tensor *) { return false; }

static const ggml_backend_device_i npu_dev_iface = {
    /* .get_name             = */ npu_dev_name,
    /* .get_description      = */ npu_dev_name,
    /* .get_memory           = */ npu_dev_memory,
    /* .get_type             = */ npu_dev_type,
    /* .get_props            = */ npu_dev_props,
    /* .init_backend         = */ npu_dev_init,
    /* .get_buffer_type      = */ npu_dev_buft,
    /* .get_host_buffer_type = */ npu_dev_host_buft,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ npu_dev_supports_op,
    /* .supports_buft        = */ npu_dev_supports_buft,
    /* .offload_op           = */ npu_dev_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * npu_reg_name(ggml_backend_reg_t) { return "HTP-ChainTest"; }
static size_t npu_reg_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t npu_reg_device(ggml_backend_reg_t reg, size_t) { return &npu_of(reg)->device; }
static void * npu_reg_proc(ggml_backend_reg_t reg, const char * name) {
    if (npu_of(reg)->provide_nowait && std::strcmp(name, "ggml_backend_graph_compute_nowait") == 0) {
        return reinterpret_cast<void *>(npu_graph_compute_nowait);
    }
    return nullptr;
}

static const ggml_backend_reg_i npu_reg_iface = {
    /* .get_name         = */ npu_reg_name,
    /* .get_device_count = */ npu_reg_count,
    /* .get_device       = */ npu_reg_device,
    /* .get_proc_address = */ npu_reg_proc,
};

static ggml_guid_t npu_guid() {
    static ggml_guid guid = {
        0x6a, 0x10, 0x3c, 0x52, 0x9e, 0x41, 0x4b, 0x07,
        0xa2, 0x5d, 0x81, 0xe4, 0x2f, 0x93, 0x0b, 0x6c,
    };
    return &guid;
}

static ggml_backend_t make_fake_npu(bool provide_nowait) {
    auto * npu = new fake_npu;
    npu->inner = ggml_backend_cpu_init();
    if (npu->inner == nullptr) {
        delete npu;
        return nullptr;
    }
    npu->provide_nowait = provide_nowait;
    npu->inner_buft = ggml_backend_get_default_buffer_type(npu->inner);
    npu->reg = { GGML_BACKEND_API_VERSION, npu_reg_iface, npu };
    npu->device = { npu_dev_iface, &npu->reg, npu };
    npu->buft = { npu_buft_iface, &npu->device, npu };
    npu->worker = std::thread(&fake_npu::run, npu);
    return new ggml_backend { npu_guid(), npu_iface, &npu->device, npu };
}

static void set_chain_env(const char * value) {
#if defined(_WIN32)
    (void) _putenv_s("GGML_HTP_CHAIN_SPLITS", value ? value : "");
#else
    if (value) {
        (void) setenv("GGML_HTP_CHAIN_SPLITS", value, 1);
    } else {
        (void) unsetenv("GGML_HTP_CHAIN_SPLITS");
    }
#endif
}

struct expected_job {
    const char * first_node;
    bool nowait;
};

// x (CPU) -> [y = x CPU] -> a = 2x (NPU) -> b = 3a or a + y (NPU) -> c = b/2 (NPU) -> out = c + x (CPU)
// a and b are route candidates with alternates directly after them, so each
// is its own split and an inactive alternate split lies between a, b and c.
struct fixture {
    static constexpr int64_t n = 4096;

    ggml_context * ctx = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t npu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<float> values;
    float expected_scale = 0.0f;
    ggml_tensor * group_nodes[2] = {};

    ~fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(npu);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }

    bool build(const char * chain_env, bool provide_nowait, bool b_reads_cpu, uint64_t plan, bool npu_output = false,
            bool checkpoint = false, bool group = false) {
        const ggml_init_params params = {
            /* .mem_size   = */ 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        cpu = ggml_backend_cpu_init();
        npu = make_fake_npu(provide_nowait);
        if (!ctx || !cpu || !npu) {
            return false;
        }
        graph = ggml_new_graph_custom(ctx, 64, false);
        x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_name(x, "chain-x");
        ggml_set_input(x);

        ggml_tensor * y = nullptr;
        if (b_reads_cpu) {
            y = ggml_scale(ctx, x, 1.0f);
            ggml_set_name(y, "chain-y");
            ggml_build_forward_expand(graph, y);
        }
        ggml_tensor * a = ggml_scale(ctx, x, 2.0f);
        ggml_set_name(a, "chain-a");
        ggml_build_forward_expand(graph, a);
        ggml_tensor * a_alt = ggml_scale(ctx, x, 2.0f);
        ggml_set_name(a_alt, "chain-a-alt");
        ggml_build_forward_expand(graph, a_alt);
        // Route variants are single nodes with the canonical node's sources.
        ggml_tensor * b = b_reads_cpu ? ggml_add(ctx, a, y) : ggml_scale(ctx, a, 3.0f);
        ggml_tensor * b_alt = b_reads_cpu ? ggml_add(ctx, a, y) : ggml_scale(ctx, a, 3.0f);
        ggml_set_name(b, "chain-b");
        ggml_build_forward_expand(graph, b);
        ggml_set_name(b_alt, "chain-b-alt");
        ggml_build_forward_expand(graph, b_alt);
        ggml_tensor * c = ggml_scale(ctx, b, 0.5f);
        ggml_set_name(c, "chain-c");
        ggml_tensor * last = c;
        ggml_tensor * d = nullptr;
        ggml_tensor * d_alt = nullptr;
        if (checkpoint) {
            // c is a layer boundary and the next layer starts on the NPU.
            ggml_build_forward_expand(graph, c);
            d = ggml_scale(ctx, c, 1.0f);
            ggml_set_name(d, "chain-d");
            ggml_build_forward_expand(graph, d);
            d_alt = ggml_scale(ctx, c, 1.0f);
            ggml_set_name(d_alt, "chain-d-alt");
            ggml_build_forward_expand(graph, d_alt);
            last = d;
        }
        ggml_tensor * npu_branch = nullptr;
        ggml_tensor * cpu_branch = nullptr;
        if (group) {
            // A parallel FFN group whose first branch is on the NPU reads c.
            // Each branch starts with a copy (CONT) like an FFN matmul: an
            // in-place first op would overwrite c while the sibling reads it.
            ggml_tensor * npu_read = ggml_cont(ctx, c);
            ggml_set_name(npu_read, "ffn_gate.npu-0");
            npu_branch = ggml_scale(ctx, npu_read, 2.0f);
            ggml_set_name(npu_branch, "ffn_down.npu-0");
            ggml_tensor * cpu_read = ggml_cont(ctx, c);
            ggml_set_name(cpu_read, "ffn_gate.cpu-0");
            cpu_branch = ggml_scale(ctx, cpu_read, 3.0f);
            ggml_set_name(cpu_branch, "ffn_down.cpu-0");
            group_nodes[0] = npu_read;
            group_nodes[1] = cpu_read;
            out = ggml_add(ctx, npu_branch, cpu_branch);
            ggml_set_name(out, "ffn_parallel_reduce-0");
        } else {
            out = npu_output ? last : ggml_add(ctx, last, x);
            if (!npu_output) {
                ggml_set_name(out, "chain-out");
            }
        }
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        expected_scale = group ? 15.0f : (b_reads_cpu ? 0.5f * 3.0f : 3.0f) + (npu_output ? 0.0f : 1.0f);

        set_chain_env(chain_env);
        ggml_backend_t backends[] = { npu, cpu };
        sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, true);
        set_chain_env(nullptr);
        if (sched == nullptr) {
            return false;
        }
        if (checkpoint && !ggml_backend_sched_prepare_layer_checkpoints(sched, graph, &c, 1, nullptr,
                    checkpoint_callback, this)) {
            return false;
        }
        if (checkpoint && (!ggml_backend_sched_register_route_candidate(sched, d, d, 0, 1) ||
                    !ggml_backend_sched_register_route_candidate(sched, d, d_alt, 1, 1))) {
            return false;
        }
        if (!ggml_backend_sched_register_route_candidate(sched, a, a, 0, 0) ||
                !ggml_backend_sched_register_route_candidate(sched, a, a_alt, 1, 0) ||
                !ggml_backend_sched_register_route_candidate(sched, b, b, 0, 0) ||
                !ggml_backend_sched_register_route_candidate(sched, b, b_alt, 1, 0)) {
            return false;
        }
        ggml_backend_sched_layer_checkpoint_set_plan_id(sched, plan);
        if (y) {
            ggml_backend_sched_set_tensor_backend(sched, y, cpu);
        }
        if (group) {
            ggml_backend_sched_set_tensor_backend(sched, group_nodes[1], cpu);
            ggml_backend_sched_set_tensor_backend(sched, cpu_branch, cpu);
        }
        for (ggml_tensor * t : { a, a_alt, b, b_alt, c, d, d_alt, group_nodes[0], npu_branch }) {
            if (t) {
                ggml_backend_sched_set_tensor_backend(sched, t, npu);
            }
        }
        if (!npu_output) {
            ggml_backend_sched_set_tensor_backend(sched, out, cpu);
        }
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return false;
        }
        values.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            values[(size_t) i] = 0.5f + (float) (i % 29) * 0.125f;
        }
        ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
        return true;
    }

    // NPU graphs complete when the layer checkpoint callback ran.
    std::vector<size_t> done_at_checkpoint;

    static uint64_t checkpoint_callback(int checkpoint_index, const ggml_tensor *, int, uint64_t active_plan_id,
            uint64_t, void * user_data) {
        auto * self = static_cast<fixture *>(user_data);
        if (checkpoint_index < 0) {
            return active_plan_id; // graph start, before any split
        }
        fake_npu * f = npu_of(self->npu);
        std::lock_guard<std::mutex> lock(f->mutex);
        self->done_at_checkpoint.push_back(f->done.size());
        return active_plan_id;
    }

    bool output_ok(const char * scenario) {
        std::vector<float> result((size_t) n);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        for (int64_t i = 0; i < n; ++i) {
            const float expected = expected_scale * values[(size_t) i];
            if (std::fabs(result[(size_t) i] - expected) > 1e-4f) {
                std::fprintf(stderr, "%s: out[%lld] = %f, expected %f\n", scenario,
                    (long long) i, result[(size_t) i], expected);
                return false;
            }
        }
        return true;
    }

    // NPU submissions in order, with their wait mode. Chained submissions must
    // reach the NPU before the previous graph has finished.
    bool jobs_match(const std::vector<expected_job> & expected, const char * scenario) {
        fake_npu * f = npu_of(npu);
        std::lock_guard<std::mutex> lock(f->mutex);
        if (f->done.size() != expected.size()) {
            std::fprintf(stderr, "%s: %zu NPU graphs, expected %zu\n", scenario, f->done.size(), expected.size());
            return false;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            const npu_job * job = f->done[i];
            if (job->first_node != expected[i].first_node || job->nowait != expected[i].nowait) {
                std::fprintf(stderr, "%s: NPU graph %zu is %s (%s), expected %s (%s)\n", scenario, i,
                    job->first_node.c_str(), job->nowait ? "nowait" : "wait",
                    expected[i].first_node, expected[i].nowait ? "nowait" : "wait");
                return false;
            }
            if (i > 0) {
                const bool overlapped = job->submit_us < f->done[i - 1]->end_us;
                if (overlapped != f->done[i - 1]->nowait) {
                    std::fprintf(stderr, "%s: NPU graph %zu submitted %s the previous one finished\n",
                        scenario, i, overlapped ? "before" : "after");
                    return false;
                }
            }
        }
        return true;
    }
};

// Canonical plan: a and b each run back to back with their successor; c waits
// because the CPU reads its output next. The inactive alternate splits between
// them must not break the chain.
static bool run_chain() {
    const char * scenario = "chain a -> b -> c";
    fixture f;
    return check(f.build("1", true, false, 0), scenario, "setup failed") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") &&
        f.output_ok(scenario) &&
        f.jobs_match({ { "chain-a", true }, { "chain-b", true }, { "chain-c", false } }, scenario);
}

// Default (unset), explicit off, and a backend without the proc: every split waits.
static bool run_disabled(const char * env, bool provide_nowait, const char * scenario) {
    fixture f;
    return check(f.build(env, provide_nowait, false, 0), scenario, "setup failed") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") &&
        f.output_ok(scenario) &&
        f.jobs_match({ { "chain-a", false }, { "chain-b", false }, { "chain-c", false } }, scenario);
}

// b copies a CPU tensor into NPU memory before it runs, so a must finish first:
// the copy target may reuse memory that a is still reading.
static bool run_input_copy_breaks_chain() {
    const char * scenario = "input copy before the next split";
    fixture f;
    return check(f.build("1", true, true, 0), scenario, "setup failed") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") &&
        f.output_ok(scenario) &&
        f.jobs_match({ { "chain-a", false }, { "chain-b", true }, { "chain-c", false } }, scenario);
}

// Alternate plan: each route group ends by committing the alternate output to
// the canonical tensor on the host, so neither alternate may stay in flight.
static bool run_alternate_commit_breaks_chain() {
    const char * scenario = "alternate route commit";
    fixture f;
    return check(f.build("1", true, false, 1), scenario, "setup failed") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") &&
        f.output_ok(scenario) &&
        f.jobs_match({ { "chain-a-alt", false }, { "chain-b-alt", false }, { "chain-c", false } }, scenario);
}

// The graph is computed twice: the second run must not see stale chain state.
static bool run_repeated() {
    const char * scenario = "repeated graph";
    fixture f;
    if (!check(f.build("on", true, false, 0), scenario, "setup failed")) {
        return false;
    }
    for (int i = 0; i < 2; ++i) {
        if (!check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") ||
                !f.output_ok(scenario)) {
            return false;
        }
    }
    return f.jobs_match({ { "chain-a", true }, { "chain-b", true }, { "chain-c", false },
                          { "chain-a", true }, { "chain-b", true }, { "chain-c", false } }, scenario);
}

// A split followed by a layer checkpoint waits even when the next layer starts
// on the same NPU: the checkpoint may select another plan for what follows,
// and the policy callback observes a completed layer.
static bool run_checkpoint_breaks_chain() {
    const char * scenario = "layer checkpoint";
    fixture f;
    return check(f.build("1", true, false, 0, false, true), scenario, "setup failed") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") &&
        f.output_ok(scenario) &&
        f.jobs_match({ { "chain-a", true }, { "chain-b", true }, { "chain-c", false }, { "chain-d", false } }, scenario) &&
        check(f.done_at_checkpoint.size() == 1 && f.done_at_checkpoint[0] == 3, scenario,
            "the checkpoint callback ran before the layer's NPU work had finished");
}

// The next split starts a parallel group on the same NPU. The group's CPU
// branch reads c in place while its NPU branch runs, so c must be complete.
static bool run_group_breaks_chain() {
    const char * scenario = "parallel group after the split";
    fixture f;
    return check(f.build("1", true, false, 0, false, false, true), scenario, "setup failed") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") &&
        f.output_ok(scenario) &&
        f.jobs_match({ { "chain-a", true }, { "chain-b", true }, { "chain-c", false }, { "ffn_gate.npu-0", false } },
            scenario);
}

// The NPU computes the graph output: the last split must wait, because the
// caller reads outputs through host pointers once compute returns.
static bool run_npu_output() {
    const char * scenario = "NPU split at graph end";
    fixture f;
    if (!check(f.build("1", true, false, 0, true), scenario, "setup failed") ||
            !check(ggml_backend_sched_graph_compute_async(f.sched, f.graph) == GGML_STATUS_SUCCESS,
                scenario, "compute failed")) {
        return false;
    }
    const bool ok = f.output_ok(scenario) &&
        f.jobs_match({ { "chain-a", true }, { "chain-b", true }, { "chain-c", false } }, scenario);
    ggml_backend_sched_synchronize(f.sched);
    return ok;
}

// A failure in a graph that was submitted without waiting is reported by the
// waiting call at the end of the chain, and the scheduler returns it.
static bool run_failure_reported() {
    const char * scenario = "failure in a chained split";
    fixture f;
    if (!check(f.build("1", true, false, 0), scenario, "setup failed")) {
        return false;
    }
    npu_of(f.npu)->fail_node = "chain-a";
    return check(ggml_backend_sched_graph_compute(f.sched, f.graph) != GGML_STATUS_SUCCESS,
        scenario, "the failure of a chained split was lost");
}

} // namespace

int main() {
    ggml_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level != GGML_LOG_LEVEL_DEBUG) {
            std::fputs(text, stderr);
        }
    }, nullptr);
    bool ok = true;
    ok = run_chain() && ok;
    ok = run_disabled(nullptr, true, "chain unset") && ok;
    ok = run_disabled("0", true, "chain off") && ok;
    ok = run_disabled("1", false, "backend without nowait") && ok;
    ok = run_input_copy_breaks_chain() && ok;
    ok = run_alternate_commit_breaks_chain() && ok;
    ok = run_repeated() && ok;
    ok = run_npu_output() && ok;
    ok = run_checkpoint_breaks_chain() && ok;
    ok = run_group_breaks_chain() && ok;
    ok = run_failure_reported() && ok;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
