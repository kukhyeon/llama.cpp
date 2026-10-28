// GGML_FFN_REPACK_PREFETCH scheduling. Before a split makes the scheduler wait
// on one backend (the NPU attention output), and only splits on that backend
// run before the next parallel FFN group, a helper thread calls
// "ggml_backend_shared_repack_prefetch" for the group's other branches. The
// scheduler must join the helper before it uses those backends again. Fake
// CPU/GPU/NPU backends compute on the CPU and log every call, so the test can
// check the overlap, the order and that no backend is ever entered twice.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-shared-weights.h"
#include "ggml.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

enum class role { cpu, gpu, npu };

struct call {
    std::string backend;
    std::string kind; // "compute" or "prefetch"
    std::string first_node;
    std::thread::id thread;
    int64_t start_us;
    int64_t end_us;
};

struct call_log {
    std::mutex mutex;
    std::vector<call> calls;
    bool reentered = false;

    void add(call c) {
        std::lock_guard<std::mutex> lock(mutex);
        calls.push_back(std::move(c));
    }
};

struct fake {
    ggml_backend_t inner = nullptr;
    ggml_backend_buffer_type_t inner_buft = nullptr;
    ggml_backend_buffer_type buft = {};
    ggml_backend_device device = {};
    ggml_backend_reg reg = {};
    role kind = role::cpu;
    std::string name;
    call_log * log = nullptr;
    int compute_ms = 0;
    int prefetch_ms = 0;
    bool provide_prefetch = false;
    bool zero_copy = false; // compute buffer registered as a zero-copy role
    std::string fail_node;
    std::atomic<int> in_use { 0 };
};

static fake * fake_of(ggml_backend_t backend) { return static_cast<fake *>(backend->context); }
static fake * fake_of(ggml_backend_dev_t device) { return static_cast<fake *>(device->context); }
static fake * fake_of(ggml_backend_buffer_type_t buft) { return static_cast<fake *>(buft->context); }
static fake * fake_of(ggml_backend_reg_t reg) { return static_cast<fake *>(reg->context); }

static std::string first_node(const ggml_cgraph * graph) {
    return graph->n_nodes > 0 ? ggml_get_name(graph->nodes[0]) : "";
}

struct use_guard {
    fake * f;
    explicit use_guard(fake * f) : f(f) {
        if (f->in_use.fetch_add(1) != 0) {
            std::lock_guard<std::mutex> lock(f->log->mutex);
            f->log->reentered = true;
        }
    }
    ~use_guard() { f->in_use.fetch_sub(1); }
};

static const char * buft_name(ggml_backend_buffer_type_t buft) { return fake_of(buft)->name.c_str(); }
static ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(fake_of(buft)->inner_buft, size);
    if (buffer != nullptr) {
        buffer->buft = buft;
    }
    return buffer;
}
static size_t buft_alignment(ggml_backend_buffer_type_t buft) { return ggml_backend_buft_get_alignment(fake_of(buft)->inner_buft); }
static size_t buft_max_size(ggml_backend_buffer_type_t buft) { return ggml_backend_buft_get_max_size(fake_of(buft)->inner_buft); }
static size_t buft_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    return ggml_backend_buft_get_alloc_size(fake_of(buft)->inner_buft, t);
}
static bool buft_is_host(ggml_backend_buffer_type_t buft) { return fake_of(buft)->kind == role::cpu || fake_of(buft)->zero_copy; }

static const ggml_backend_buffer_type_i buft_iface = {
    /* .get_name       = */ buft_name,
    /* .alloc_buffer   = */ buft_alloc,
    /* .get_alignment  = */ buft_alignment,
    /* .get_max_size   = */ buft_max_size,
    /* .get_alloc_size = */ buft_alloc_size,
    /* .is_host        = */ buft_is_host,
};

static const char * backend_name(ggml_backend_t backend) { return fake_of(backend)->name.c_str(); }
static void backend_free(ggml_backend_t backend) {
    fake * f = fake_of(backend);
    if (f->zero_copy) {
        ggml_zero_copy_unregister_buft(&f->buft);
    }
    ggml_backend_free(f->inner);
    delete f;
    delete backend;
}
static enum ggml_status backend_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    fake * f = fake_of(backend);
    use_guard guard(f);
    call c { f->name, "compute", first_node(graph), std::this_thread::get_id(), now_us(), 0 };
    if (f->compute_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(f->compute_ms));
    }
    enum ggml_status status = ggml_backend_graph_compute(f->inner, graph);
    if (!f->fail_node.empty() && c.first_node == f->fail_node) {
        status = GGML_STATUS_FAILED;
    }
    c.end_us = now_us();
    f->log->add(c);
    return status;
}

static const ggml_backend_i backend_iface = {
    /* .get_name                = */ backend_name,
    /* .free                    = */ backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ backend_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static int backend_prefetch(ggml_backend_t backend, struct ggml_cgraph * graph) {
    fake * f = fake_of(backend);
    use_guard guard(f);
    call c { f->name, "prefetch", first_node(graph), std::this_thread::get_id(), now_us(), 0 };
    std::this_thread::sleep_for(std::chrono::milliseconds(f->prefetch_ms));
    c.end_us = now_us();
    f->log->add(c);
    return 1;
}

static const char * dev_name(ggml_backend_dev_t device) { return fake_of(device)->name.c_str(); }
static void dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = 1u << 30; *total = 1u << 30; }
static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t device) {
    switch (fake_of(device)->kind) {
        case role::cpu: return GGML_BACKEND_DEVICE_TYPE_CPU;
        case role::gpu: return GGML_BACKEND_DEVICE_TYPE_GPU;
        case role::npu: return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}
static void dev_props(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    props->name = dev_name(device);
    props->description = props->name;
    props->memory_free = 1u << 30;
    props->memory_total = 1u << 30;
    props->type = dev_type(device);
    props->device_id = nullptr;
    props->caps = {};
}
static ggml_backend_t dev_init(ggml_backend_dev_t, const char *) { return nullptr; }
static ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t device) { return &fake_of(device)->buft; }
static ggml_backend_buffer_type_t dev_host_buft(ggml_backend_dev_t) { return nullptr; }
static bool dev_supports_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    return ggml_backend_supports_op(fake_of(device)->inner, op);
}
// Only its own memory, so every cross-backend input is a scheduler copy, or,
// with zero-copy buffers, every zero-copy buffer in place.
static bool dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return buft == &fake_of(device)->buft || (fake_of(device)->zero_copy && ggml_zero_copy_buft_role(buft) >= 0);
}
static bool dev_offload_op(ggml_backend_dev_t, const ggml_tensor *) { return false; }

static const ggml_backend_device_i dev_iface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_name,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init,
    /* .get_buffer_type      = */ dev_buft,
    /* .get_host_buffer_type = */ dev_host_buft,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ dev_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * reg_name(ggml_backend_reg_t reg) { return fake_of(reg)->name.c_str(); }
static size_t reg_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t reg_device(ggml_backend_reg_t reg, size_t) { return &fake_of(reg)->device; }
static void * reg_proc(ggml_backend_reg_t reg, const char * name) {
    if (fake_of(reg)->provide_prefetch && std::strcmp(name, "ggml_backend_shared_repack_prefetch") == 0) {
        ggml_backend_shared_repack_prefetch_t fn = backend_prefetch;
        return reinterpret_cast<void *>(fn);
    }
    return nullptr;
}

static const ggml_backend_reg_i reg_iface = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_count,
    /* .get_device       = */ reg_device,
    /* .get_proc_address = */ reg_proc,
};

static ggml_guid_t fake_guid() {
    static ggml_guid guid = {
        0x41, 0x7e, 0x93, 0x0f, 0xc2, 0x5b, 0x48, 0x6d,
        0x9a, 0x31, 0x0d, 0x77, 0xe8, 0x24, 0xb5, 0x1c,
    };
    return &guid;
}

static ggml_backend_t make_fake(role kind, const char * name, call_log * log, int compute_ms, int prefetch_ms,
        bool provide_prefetch, bool zero_copy = false) {
    auto * f = new fake;
    f->inner = ggml_backend_cpu_init();
    if (f->inner == nullptr) {
        delete f;
        return nullptr;
    }
    f->kind = kind;
    f->name = name;
    f->log = log;
    f->compute_ms = compute_ms;
    f->prefetch_ms = prefetch_ms;
    f->provide_prefetch = provide_prefetch;
    f->zero_copy = zero_copy;
    f->inner_buft = ggml_backend_get_default_buffer_type(f->inner);
    f->reg = { GGML_BACKEND_API_VERSION, reg_iface, f };
    f->device = { dev_iface, &f->reg, f };
    f->buft = { buft_iface, &f->device, f };
    if (zero_copy) {
        ggml_zero_copy_register_buft(&f->buft, kind == role::npu ? GGML_ZERO_COPY_NPU :
            kind == role::gpu ? GGML_ZERO_COPY_GPU : GGML_ZERO_COPY_CPU);
    }
    return new ggml_backend { fake_guid(), backend_iface, &f->device, f };
}

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    (void) _putenv_s(name, value ? value : "");
#else
    if (value) {
        (void) setenv(name, value, 1);
    } else {
        (void) unsetenv(name);
    }
#endif
}

struct options {
    const char * prefetch_env = "1";
    const char * persistent_env = nullptr;
    int layers = 1;
    bool cpu_between = false;    // a CPU node between the attention output and the FFN group
    bool trigger_on_cpu = false; // the attention output itself runs on the CPU
    bool no_group = false;       // no FFN group after the attention output
    bool slow_prefetch = false;  // prefetches outlast the attention output
    bool npu_branch_first = false; // the group starts on the attention output's backend
    // An NPU split "ffn_inp" between the attention output and the group (both
    // route candidates, so they are separate splits): mid = attn + y.
    enum class mid_kind { none, plain, cpu_input, gpu_input } mid = mid_kind::none;
    uint64_t plan = 0;           // 1 selects the alternates, which are committed
    bool checkpoint_between = false; // a layer checkpoint on that NPU split
    bool zero_copy = false;      // zero-copy compute buffers (read in place)
};

static uint64_t keep_plan(int, const ggml_tensor *, int, uint64_t active_plan_id, uint64_t, void *) {
    return active_plan_id;
}

// Per layer: attn_out (NPU, slow) -> FFN group with GPU, CPU and NPU branches
// -> reduce on the CPU. Branches start with CONT so no branch overwrites the
// tensor its siblings read in place of a copy.
struct fixture {
    static constexpr int64_t n = 2048;

    call_log log;
    ggml_context * ctx = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_backend_t npu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<float> values;
    float expected_scale = 1.0f;

    ~fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(npu);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }

    bool build(const options & o) {
        const ggml_init_params params = {
            /* .mem_size   = */ 256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(256, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        cpu = make_fake(role::cpu, "PrefetchCPU", &log, 0, o.slow_prefetch ? 60 : 15, true, o.zero_copy);
        gpu = make_fake(role::gpu, "PrefetchGPU", &log, 0, o.slow_prefetch ? 30 : 5, true, o.zero_copy);
        npu = make_fake(role::npu, "HTP-PrefetchNPU", &log, 40, 0, false, o.zero_copy);
        if (!ctx || !cpu || !gpu || !npu) {
            return false;
        }
        graph = ggml_new_graph_custom(ctx, 256, false);
        x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_name(x, "prefetch-x");
        ggml_set_input(x);

        std::vector<std::pair<ggml_tensor *, ggml_backend_t>> placement;
        auto put = [&](ggml_tensor * t, const std::string & name, ggml_backend_t backend) {
            ggml_set_name(t, name.c_str());
            placement.push_back({ t, backend });
            return t;
        };
        ggml_tensor * cur = x;
        ggml_tensor * y = nullptr;
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> routes; // canonical, alternate
        ggml_tensor * checkpoint = nullptr;
        using mid_kind = options::mid_kind;
        if (o.mid == mid_kind::cpu_input || o.mid == mid_kind::gpu_input) {
            // Computed before the attention output, read by the NPU split after it.
            y = put(ggml_scale(ctx, x, 1.0f), "prefetch-y", o.mid == mid_kind::cpu_input ? cpu : gpu);
            ggml_build_forward_expand(graph, y);
        }
        for (int layer = 0; layer < o.layers; ++layer) {
            const std::string l = std::to_string(layer);
            ggml_tensor * attn = put(ggml_scale(ctx, cur, 2.0f), "attn_out-" + l, o.trigger_on_cpu ? cpu : npu);
            expected_scale *= 2.0f;
            if (o.mid != mid_kind::none) {
                // Route candidates keep attn_out and mid in separate NPU splits.
                ggml_build_forward_expand(graph, attn);
                ggml_tensor * attn_alt = put(ggml_scale(ctx, cur, 2.0f), "attn_out_alt-" + l, npu);
                ggml_build_forward_expand(graph, attn_alt);
                routes.push_back({ attn, attn_alt });
                // mid = attn + y = 2x + x (one layer), or attn.
                auto make_mid = [&](const std::string & name) {
                    return put(y ? ggml_add(ctx, attn, y) : ggml_scale(ctx, attn, 1.0f), name, npu);
                };
                ggml_tensor * mid = make_mid("ffn_inp-" + l);
                ggml_build_forward_expand(graph, mid);
                ggml_tensor * mid_alt = make_mid("ffn_inp_alt-" + l);
                ggml_build_forward_expand(graph, mid_alt);
                routes.push_back({ mid, mid_alt });
                attn = mid;
                if (y) {
                    expected_scale = 3.0f;
                }
            }
            if (o.checkpoint_between) {
                attn = put(ggml_scale(ctx, attn, 1.0f), "ffn_norm-" + l, npu);
                checkpoint = attn;
            }
            if (o.cpu_between) {
                attn = put(ggml_scale(ctx, attn, 1.0f), "ffn_inp-" + l, cpu);
            }
            if (o.no_group) {
                cur = put(ggml_scale(ctx, attn, 1.0f), "l_out-" + l, cpu);
                continue;
            }
            ggml_tensor * p = nullptr;
            if (o.npu_branch_first) {
                p = put(ggml_cont(ctx, attn), "ffn_gate.npu-" + l, npu);
                p = put(ggml_scale(ctx, p, 3.0f), "ffn_down.npu-" + l, npu);
                ggml_build_forward_expand(graph, p);
            }
            ggml_tensor * g = put(ggml_cont(ctx, attn), "ffn_gate.gpu-" + l, gpu);
            g = put(ggml_scale(ctx, g, 1.0f), "ffn_down.gpu-" + l, gpu);
            ggml_build_forward_expand(graph, g);
            ggml_tensor * c = put(ggml_cont(ctx, attn), "ffn_gate.cpu-" + l, cpu);
            c = put(ggml_scale(ctx, c, 2.0f), "ffn_down.cpu-" + l, cpu);
            ggml_build_forward_expand(graph, c);
            if (!o.npu_branch_first) {
                p = put(ggml_cont(ctx, attn), "ffn_gate.npu-" + l, npu);
                p = put(ggml_scale(ctx, p, 3.0f), "ffn_down.npu-" + l, npu);
            }
            ggml_tensor * r = put(ggml_add(ctx, g, c), "ffn_parallel_reduce-" + l, cpu);
            cur = put(ggml_add(ctx, r, p), "ffn_parallel_out-" + l, cpu);
            expected_scale *= 6.0f;
        }
        out = cur;
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);

        set_env("GGML_FFN_REPACK_PREFETCH", o.prefetch_env);
        set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", o.persistent_env);
        ggml_backend_t backends[] = { gpu, npu, cpu };
        sched = ggml_backend_sched_new(backends, nullptr, 3, 256, false, true);
        set_env("GGML_FFN_REPACK_PREFETCH", nullptr);
        set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", nullptr);
        if (sched == nullptr) {
            return false;
        }
        if (checkpoint != nullptr && !ggml_backend_sched_prepare_layer_checkpoints(sched, graph, &checkpoint, 1,
                    nullptr, keep_plan, nullptr)) {
            return false;
        }
        for (const auto & r : routes) {
            if (!ggml_backend_sched_register_route_candidate(sched, r.first, r.first, 0, 0) ||
                    !ggml_backend_sched_register_route_candidate(sched, r.first, r.second, 1, 0)) {
                return false;
            }
        }
        if (!routes.empty()) {
            ggml_backend_sched_layer_checkpoint_set_plan_id(sched, o.plan);
        }
        ggml_backend_sched_set_tensor_backend(sched, x, cpu);
        for (const auto & p : placement) {
            ggml_backend_sched_set_tensor_backend(sched, p.first, p.second);
        }
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return false;
        }
        values.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            values[(size_t) i] = 0.25f + (float) (i % 23) * 0.0625f;
        }
        ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
        return true;
    }

    bool output_ok(const char * scenario) {
        std::vector<float> result((size_t) n);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        for (int64_t i = 0; i < n; ++i) {
            const float expected = expected_scale * values[(size_t) i];
            if (std::fabs(result[(size_t) i] - expected) > 1e-3f * std::fabs(expected)) {
                std::fprintf(stderr, "%s: out[%lld] = %f, expected %f\n", scenario,
                    (long long) i, result[(size_t) i], expected);
                return false;
            }
        }
        return true;
    }

    std::vector<call> calls(const char * kind, const char * backend = nullptr) {
        std::lock_guard<std::mutex> lock(log.mutex);
        std::vector<call> result;
        for (const auto & c : log.calls) {
            if (c.kind == kind && (backend == nullptr || c.backend == backend)) {
                result.push_back(c);
            }
        }
        return result;
    }

    const call * find(const char * kind, const std::string & node) {
        std::lock_guard<std::mutex> lock(log.mutex);
        for (const auto & c : log.calls) {
            if (c.kind == kind && c.first_node == node) {
                return &c;
            }
        }
        return nullptr;
    }
};

// Both prefetches run on one helper thread while the NPU computes the
// attention output (GPU first: it only enqueues), and finish before their
// branches start. No backend is ever used by two threads at once.
static bool run_prefetch_overlaps_attention(const char * persistent, const char * scenario) {
    fixture f;
    options o;
    o.persistent_env = persistent;
    o.layers = 2;
    if (!check(f.build(o), scenario, "setup failed") ||
            !check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") ||
            !f.output_ok(scenario)) {
        return false;
    }
    const auto prefetches = f.calls("prefetch");
    if (!check(prefetches.size() == 4, scenario, "expected one GPU and one CPU prefetch per layer") ||
            !check(!f.log.reentered, scenario, "a backend was entered by two threads at once")) {
        return false;
    }
    const std::thread::id main_thread = std::this_thread::get_id();
    for (int layer = 0; layer < 2; ++layer) {
        const std::string l = std::to_string(layer);
        const call * attn = f.find("compute", "attn_out-" + l);
        const call * gpu_prefetch = f.find("prefetch", "ffn_gate.gpu-" + l);
        const call * cpu_prefetch = f.find("prefetch", "ffn_gate.cpu-" + l);
        const call * gpu_branch = f.find("compute", "ffn_gate.gpu-" + l);
        const call * cpu_branch = f.find("compute", "ffn_gate.cpu-" + l);
        if (!check(attn && gpu_prefetch && cpu_prefetch && gpu_branch && cpu_branch, scenario, "missing calls")) {
            return false;
        }
        if (!check(gpu_prefetch->thread != main_thread && gpu_prefetch->thread == cpu_prefetch->thread,
                    scenario, "prefetches must run on one helper thread") ||
                !check(gpu_prefetch->end_us <= cpu_prefetch->start_us, scenario, "the GPU prefetch must run first") ||
                !check(cpu_prefetch->end_us <= attn->end_us, scenario,
                    "the prefetches did not overlap the attention output") ||
                !check(gpu_prefetch->end_us <= gpu_branch->start_us && cpu_prefetch->end_us <= cpu_branch->start_us,
                    scenario, "a branch started before its prefetch finished")) {
            return false;
        }
    }
    return true;
}

// The prefetches outlast the attention output: the scheduler waits for them
// before the first split that could use a prefetching backend.
static bool run_join_waits(const options & o, const char * scenario, const char * first_after) {
    fixture f;
    if (!check(f.build(o), scenario, "setup failed") ||
            !check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") ||
            !f.output_ok(scenario)) {
        return false;
    }
    const call * cpu_prefetch = f.find("prefetch", "ffn_gate.cpu-0");
    const call * attn = f.find("compute", o.plan == 1 ? "attn_out_alt-0" : "attn_out-0");
    const call * after = f.find("compute", first_after);
    if (!check(cpu_prefetch && attn && after, scenario, "missing calls")) {
        return false;
    }
    for (const auto & c : f.calls("compute")) {
        if (c.first_node.rfind("ffn_gate.", 0) == 0 && c.start_us < cpu_prefetch->end_us) {
            std::fprintf(stderr, "%s: branch %s started before the prefetch finished\n", scenario, c.first_node.c_str());
            return false;
        }
    }
    return check(cpu_prefetch->end_us > attn->end_us, scenario, "the prefetch did not outlast the attention; scenario is void") &&
        check(after->start_us >= cpu_prefetch->end_us, scenario, "a split ran before the prefetch was joined") &&
        check(!f.log.reentered, scenario, "a backend was entered by two threads at once");
}

// Control for the join scenarios: an NPU split that neither copies from nor
// waits on a prefetching backend runs while the prefetch is still going.
static bool run_join_waits_not(options o, const char * scenario) {
    o.slow_prefetch = true;
    fixture f;
    if (!check(f.build(o), scenario, "setup failed") ||
            !check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") ||
            !f.output_ok(scenario)) {
        return false;
    }
    const call * cpu_prefetch = f.find("prefetch", "ffn_gate.cpu-0");
    const call * mid = f.find("compute", "ffn_inp-0");
    return check(cpu_prefetch && mid, scenario, "missing calls") &&
        check(mid->start_us < cpu_prefetch->end_us, scenario, "the NPU split waited for the prefetch") &&
        check(!f.log.reentered, scenario, "a backend was entered by two threads at once");
}

static bool expect_no_prefetch(const options & o, const char * scenario) {
    fixture f;
    return check(f.build(o), scenario, "setup failed") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario, "compute failed") &&
        f.output_ok(scenario) &&
        check(f.calls("prefetch").empty(), scenario, "a prefetch ran") &&
        check(!f.log.reentered, scenario, "a backend was entered by two threads at once");
}

// The graph fails after the prefetch started: the scheduler still joins it,
// and the next run works.
static bool run_failure_joins() {
    const char * scenario = "failure after the prefetch started";
    fixture f;
    options o;
    o.slow_prefetch = true;
    if (!check(f.build(o), scenario, "setup failed")) {
        return false;
    }
    fake_of(f.npu)->fail_node = "attn_out-0";
    if (!check(ggml_backend_sched_graph_compute(f.sched, f.graph) != GGML_STATUS_SUCCESS, scenario,
                "the failure was not reported")) {
        return false;
    }
    const int64_t returned_us = now_us();
    const auto failed_run = f.calls("prefetch");
    if (!check(failed_run.size() == 2, scenario, "expected the prefetch to have started") ||
            !check(failed_run[0].end_us <= returned_us && failed_run[1].end_us <= returned_us, scenario,
                "the prefetch was still running when compute returned")) {
        return false;
    }
    fake_of(f.npu)->fail_node.clear();
    ggml_backend_tensor_set(f.x, f.values.data(), 0, f.values.size() * sizeof(float));
    return check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS, scenario,
            "the next compute failed") &&
        f.output_ok(scenario) &&
        check(f.calls("prefetch").size() == 4, scenario, "expected a prefetch in each run") &&
        check(!f.log.reentered, scenario, "a backend was entered by two threads at once");
}

} // namespace

int main() {
    // Read once; only fixtures that register zero-copy buffers use it.
#if defined(_WIN32)
    (void) _putenv_s("GGML_ACTIVATION_ZERO_COPY", "7");
#else
    (void) setenv("GGML_ACTIVATION_ZERO_COPY", "7", 1);
#endif
    ggml_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level != GGML_LOG_LEVEL_DEBUG) {
            std::fputs(text, stderr);
        }
    }, nullptr);
    bool ok = true;
    ok = run_prefetch_overlaps_attention(nullptr, "prefetch overlaps attention") && ok;
    ok = run_prefetch_overlaps_attention("1", "prefetch with persistent device workers") && ok;
    {
        options o;
        o.prefetch_env = nullptr;
        ok = expect_no_prefetch(o, "prefetch unset") && ok;
        o.prefetch_env = "0";
        ok = expect_no_prefetch(o, "prefetch off") && ok;
    }
    {
        options o;
        o.cpu_between = true;
        ok = expect_no_prefetch(o, "CPU split before the group") && ok;
    }
    {
        options o;
        o.trigger_on_cpu = true;
        ok = expect_no_prefetch(o, "attention output on a prefetching backend") && ok;
    }
    {
        options o;
        o.no_group = true;
        ok = expect_no_prefetch(o, "no FFN group") && ok;
    }
    {
        options o;
        o.slow_prefetch = true;
        ok = run_join_waits(o, "slow prefetch", "ffn_gate.gpu-0") && ok;
        o.npu_branch_first = true;
        ok = run_join_waits(o, "slow prefetch, group starts on the NPU", "ffn_gate.npu-0") && ok;
        o.npu_branch_first = false;
        o.mid = options::mid_kind::cpu_input;
        ok = run_join_waits(o, "slow prefetch, NPU split copies a CPU tensor", "ffn_inp-0") && ok;
        o.mid = options::mid_kind::plain;
        o.plan = 1;
        ok = run_join_waits(o, "slow prefetch, alternate route committed", "ffn_inp_alt-0") && ok;
        o.plan = 0;
        o.mid = options::mid_kind::gpu_input;
        o.zero_copy = true;
        ok = run_join_waits(o, "slow prefetch, NPU split waits on the GPU", "ffn_inp-0") && ok;
    }
    {
        options o;
        o.mid = options::mid_kind::plain;
        ok = run_join_waits_not(o, "plain NPU split before the group") && ok;
        o.checkpoint_between = true;
        ok = expect_no_prefetch(o, "layer checkpoint before the group") && ok;
    }
    ok = run_failure_joins() && ok;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
