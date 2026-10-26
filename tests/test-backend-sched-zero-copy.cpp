// Scheduler ordering for GGML_ACTIVATION_ZERO_COPY. Fake CPU/GPU/NPU backends
// share CPU memory; the GPU computes asynchronously after a delay, like an
// OpenCL queue, so a missing synchronization reads or overwrites data early.

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
#include <string>
#include <thread>
#include <vector>

namespace {

static bool check(bool condition, const char * scenario, const char * detail) {
    if (!condition) {
        std::fprintf(stderr, "%s: %s\n", scenario, detail);
    }
    return condition;
}

enum class fake_role { cpu, gpu, npu };

struct fake_backend_context {
    ggml_backend_t inner = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_buffer_type_t inner_buft = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    fake_role role = fake_role::cpu;
    enum ggml_backend_dev_type device_type = GGML_BACKEND_DEVICE_TYPE_CPU;
    const char * name = nullptr;
    bool zero_copy = false;
    std::thread pending;
    std::atomic<int> synchronizes { 0 };
};

static fake_backend_context * backend_ctx(ggml_backend_t backend) {
    return static_cast<fake_backend_context *>(backend->context);
}

static fake_backend_context * device_ctx(ggml_backend_dev_t device) {
    return static_cast<fake_backend_context *>(device->context);
}

static fake_backend_context * buft_ctx(ggml_backend_buffer_type_t buft) {
    return static_cast<fake_backend_context *>(buft->context);
}

static const char * fake_buft_name(ggml_backend_buffer_type_t buft) {
    return buft_ctx(buft)->name;
}

static ggml_backend_buffer_t fake_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft_ctx(buft)->inner_buft, size);
    if (buffer != nullptr) {
        buffer->buft = buft;
    }
    return buffer;
}

static size_t fake_buft_alignment(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_alignment(buft_ctx(buft)->inner_buft);
}

static size_t fake_buft_max_size(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_max_size(buft_ctx(buft)->inner_buft);
}

static size_t fake_buft_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(buft_ctx(buft)->inner_buft, tensor);
}

static bool fake_buft_is_host(ggml_backend_buffer_type_t buft) {
    // rpcmem compute buffers are host memory; device copy buffers are not.
    return buft_ctx(buft)->role == fake_role::cpu || buft_ctx(buft)->zero_copy;
}

static const ggml_backend_buffer_type_i fake_buft_iface = {
    /* .get_name       = */ fake_buft_name,
    /* .alloc_buffer   = */ fake_buft_alloc,
    /* .get_alignment  = */ fake_buft_alignment,
    /* .get_max_size   = */ fake_buft_max_size,
    /* .get_alloc_size = */ fake_buft_alloc_size,
    /* .is_host        = */ fake_buft_is_host,
};

static const char * fake_backend_name(ggml_backend_t backend) {
    return backend_ctx(backend)->name;
}

static void fake_backend_synchronize(ggml_backend_t backend) {
    fake_backend_context * ctx = backend_ctx(backend);
    ctx->synchronizes.fetch_add(1, std::memory_order_relaxed);
    if (ctx->pending.joinable()) {
        ctx->pending.join();
    }
    ggml_backend_synchronize(ctx->inner);
}

static void fake_backend_free(ggml_backend_t backend) {
    fake_backend_context * ctx = backend_ctx(backend);
    fake_backend_synchronize(backend);
    ggml_backend_free(ctx->inner);
    if (ctx->zero_copy) {
        ggml_zero_copy_unregister_buft(ctx->buft);
    }
    delete ctx->buft;
    delete ctx->device;
    delete ctx;
    delete backend;
}

static enum ggml_status fake_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    fake_backend_context * ctx = backend_ctx(backend);
    if (ctx->role != fake_role::gpu) {
        return ggml_backend_graph_compute(ctx->inner, graph);
    }
    // Like an OpenCL queue: return at once, read the inputs and write the
    // outputs later. The node array outlives the call; the graph header may not.
    if (ctx->pending.joinable()) {
        ctx->pending.join();
    }
    ggml_backend_t inner = ctx->inner;
    ctx->pending = std::thread([inner, copy = *graph]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        (void) ggml_backend_graph_compute(inner, &copy);
    });
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i fake_backend_iface = {
    /* .get_name                = */ fake_backend_name,
    /* .free                    = */ fake_backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ fake_backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ fake_backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static const char * fake_device_name(ggml_backend_dev_t device) {
    return device_ctx(device)->name;
}

static void fake_device_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    ggml_backend_dev_memory(ggml_backend_get_device(device_ctx(device)->inner), free, total);
}

static enum ggml_backend_dev_type fake_device_type(ggml_backend_dev_t device) {
    return device_ctx(device)->device_type;
}

static void fake_device_props(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    fake_backend_context * ctx = device_ctx(device);
    ggml_backend_dev_get_props(ggml_backend_get_device(ctx->inner), props);
    props->name = ctx->name;
    props->description = ctx->name;
    props->type = ctx->device_type;
    props->caps = {};
}

static ggml_backend_t fake_device_init(ggml_backend_dev_t, const char *) {
    return nullptr;
}

static ggml_backend_buffer_type_t fake_device_buffer_type(ggml_backend_dev_t device) {
    return device_ctx(device)->buft;
}

static ggml_backend_buffer_type_t fake_device_host_buffer_type(ggml_backend_dev_t) {
    return nullptr;
}

static bool fake_device_supports_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    return ggml_backend_dev_supports_op(ggml_backend_get_device(device_ctx(device)->inner), op);
}

static bool fake_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    // As with the Hexagon/OpenCL backends: every processor uses every
    // registered zero-copy buffer in place.
    return buft == device_ctx(device)->buft || ggml_zero_copy_buft_role(buft) >= 0;
}

static bool fake_device_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static const ggml_backend_device_i fake_device_iface = {
    /* .get_name             = */ fake_device_name,
    /* .get_description      = */ fake_device_name,
    /* .get_memory           = */ fake_device_memory,
    /* .get_type             = */ fake_device_type,
    /* .get_props            = */ fake_device_props,
    /* .init_backend         = */ fake_device_init,
    /* .get_buffer_type      = */ fake_device_buffer_type,
    /* .get_host_buffer_type = */ fake_device_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ fake_device_supports_op,
    /* .supports_buft        = */ fake_device_supports_buft,
    /* .offload_op           = */ fake_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_guid_t fake_backend_guid() {
    static ggml_guid guid = {
        0x2b, 0x91, 0x6e, 0x0d, 0x75, 0x4a, 0x4c, 0x1f,
        0x8e, 0x33, 0xa0, 0x5c, 0x19, 0xd2, 0x67, 0x41,
    };
    return &guid;
}

// The NPU name starts with "HTP": the scheduler treats it, like the CPU, as
// complete when graph_compute returns, and the GPU as asynchronous.
static ggml_backend_t make_fake_backend(fake_role role, const char * name,
        enum ggml_backend_dev_type device_type, bool zero_copy) {
    auto * ctx = new fake_backend_context;
    ctx->inner = ggml_backend_cpu_init();
    if (ctx->inner == nullptr) {
        delete ctx;
        return nullptr;
    }
    ctx->role = role;
    ctx->device_type = device_type;
    ctx->name = name;
    ctx->zero_copy = zero_copy;
    ctx->inner_buft = ggml_backend_get_default_buffer_type(ctx->inner);
    ctx->device = new ggml_backend_device {
        /* .iface   = */ fake_device_iface,
        /* .reg     = */ nullptr,
        /* .context = */ ctx,
    };
    ctx->buft = new ggml_backend_buffer_type {
        /* .iface   = */ fake_buft_iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
    if (zero_copy) {
        ggml_zero_copy_register_buft(ctx->buft,
            role == fake_role::npu ? GGML_ZERO_COPY_NPU :
            role == fake_role::gpu ? GGML_ZERO_COPY_GPU : GGML_ZERO_COPY_CPU);
    }
    return new ggml_backend {
        /* .guid    = */ fake_backend_guid(),
        /* .iface   = */ fake_backend_iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
}

struct fixture {
    static constexpr int64_t n = 1 << 14;

    ggml_context * ctx = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_backend_t npu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    std::vector<float> values;

    explicit fixture(bool zero_copy) {
        const ggml_init_params params = {
            /* .mem_size   = */ 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        cpu = make_fake_backend(fake_role::cpu, "ZeroCopyCPU", GGML_BACKEND_DEVICE_TYPE_CPU, zero_copy);
        gpu = make_fake_backend(fake_role::gpu, "ZeroCopyGPU", GGML_BACKEND_DEVICE_TYPE_GPU, zero_copy);
        npu = make_fake_backend(fake_role::npu, "HTP-ZeroCopyNPU", GGML_BACKEND_DEVICE_TYPE_ACCEL, zero_copy);
        graph = ggml_new_graph_custom(ctx, 64, false);
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_name(input, "zero-copy-input");
        ggml_set_input(input);
    }

    ~fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(npu);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }

    bool ready() const {
        return ctx && cpu && gpu && npu;
    }

    ggml_tensor * op(ggml_tensor * t, ggml_backend_t backend, const char * name) {
        ggml_set_name(t, name);
        node_backends.push_back({t, backend});
        return t;
    }

    bool alloc(ggml_tensor * output) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        ggml_backend_t backends[] = { gpu, npu, cpu };
        ggml_backend_buffer_type_t bufts[] = {
            ggml_backend_get_default_buffer_type(gpu),
            ggml_backend_get_default_buffer_type(npu),
            ggml_backend_get_default_buffer_type(cpu),
        };
        sched = ggml_backend_sched_new(backends, bufts, 3, 64, false, false);
        if (sched == nullptr) {
            return false;
        }
        ggml_backend_sched_set_tensor_backend(sched, input, cpu);
        for (const auto & nb : node_backends) {
            ggml_backend_sched_set_tensor_backend(sched, nb.first, nb.second);
        }
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return false;
        }
        for (const auto & nb : node_backends) {
            // In-place reuse must not move an output into another backend's buffer.
            if (nb.first->buffer == nullptr ||
                    nb.first->buffer->buft != ggml_backend_get_default_buffer_type(nb.second)) {
                std::fprintf(stderr, "%s is not in the buffer of %s\n", ggml_get_name(nb.first),
                    ggml_backend_name(nb.second));
                return false;
            }
        }
        values.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            values[(size_t) i] = 1.0f + (float) (i % 37) * 0.25f;
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        return true;
    }

    bool matches(ggml_tensor * t, float scale, const char * scenario) {
        std::vector<float> result((size_t) n);
        ggml_backend_tensor_get(t, result.data(), 0, result.size() * sizeof(float));
        for (int64_t i = 0; i < n; ++i) {
            if (std::fabs(result[(size_t) i] - scale * values[(size_t) i]) > 1e-4f) {
                std::fprintf(stderr, "%s: %s[%lld] = %f, expected %f\n", scenario, ggml_get_name(t),
                    (long long) i, result[(size_t) i], scale * values[(size_t) i]);
                return false;
            }
        }
        return true;
    }

    int synchronizes(ggml_backend_t backend) const {
        return backend_ctx(backend)->synchronizes.load(std::memory_order_relaxed);
    }

    std::vector<std::pair<ggml_tensor *, ggml_backend_t>> node_backends;
};

// Copy path: without registered zero-copy buffers the CPU reads a copy of the
// GPU output, and the scheduler synchronizes the GPU for that copy.
static bool run_copy_baseline() {
    const char * scenario = "copy baseline";
    fixture f(false);
    if (!check(f.ready(), scenario, "fixture setup failed")) {
        return false;
    }
    ggml_tensor * g = f.op(ggml_scale(f.ctx, f.input, 2.0f), f.gpu, "zc-gpu");
    ggml_tensor * out = f.op(ggml_add(f.ctx, g, f.input), f.cpu, "zc-out");
    return check(f.alloc(out), scenario, "allocation failed") &&
        check(out->src[0] != g, scenario, "CPU read the GPU output without a copy") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed") &&
        f.matches(out, 3.0f, scenario);
}

// RAW: the CPU reads the asynchronous GPU output in place, so the GPU must
// finish before the CPU split runs. The GPU reads no CPU memory here, so no
// write-after-read wait can stand in for the read-after-write one.
static bool run_read_after_write() {
    const char * scenario = "zero-copy read after asynchronous write";
    fixture f(true);
    if (!check(f.ready(), scenario, "fixture setup failed")) {
        return false;
    }
    ggml_tensor * n = f.op(ggml_scale(f.ctx, f.input, 1.0f), f.npu, "zc-npu");
    ggml_tensor * g = f.op(ggml_scale(f.ctx, n, 2.0f), f.gpu, "zc-gpu");
    ggml_tensor * out = f.op(ggml_add(f.ctx, g, n), f.cpu, "zc-out");
    return check(f.alloc(out), scenario, "allocation failed") &&
        check(g->src[0] == n && out->src[0] == g && out->src[1] == n,
            scenario, "outputs were not read in place") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed") &&
        f.matches(out, 3.0f, scenario);
}

// WAR: the GPU reads a CPU output in place, then a later CPU split reuses that
// memory. The CPU split must wait for the GPU read.
static bool run_write_after_read() {
    const char * scenario = "zero-copy write after asynchronous read";
    fixture f(true);
    if (!check(f.ready(), scenario, "fixture setup failed")) {
        return false;
    }
    ggml_set_output(f.input); // keep c from overwriting the input in place
    ggml_tensor * a = f.op(ggml_scale(f.ctx, f.input, 1.0f), f.cpu, "zc-cpu-a");
    ggml_tensor * b = f.op(ggml_scale(f.ctx, a, 2.0f), f.gpu, "zc-gpu-b");
    ggml_tensor * c = f.op(ggml_scale(f.ctx, f.input, 5.0f), f.cpu, "zc-cpu-c");
    ggml_tensor * d = f.op(ggml_scale(f.ctx, c, 1.0f), f.npu, "zc-npu-d");
    ggml_tensor * out = f.op(ggml_add(f.ctx, b, d), f.cpu, "zc-out");
    return check(f.alloc(out), scenario, "allocation failed") &&
        check(b->src[0] == a && d->src[0] == c, scenario, "GPU/NPU did not read CPU outputs in place") &&
        check(c->data == a->data, scenario, "the CPU did not reuse the GPU input's memory; scenario is void") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed") &&
        f.matches(out, 7.0f, scenario);
}

// Graph end: callers read outputs through host pointers right after
// compute_async, so outstanding GPU work must be complete when it returns.
static bool run_graph_end() {
    const char * scenario = "zero-copy output after compute_async";
    fixture f(true);
    if (!check(f.ready(), scenario, "fixture setup failed")) {
        return false;
    }
    ggml_tensor * g = f.op(ggml_scale(f.ctx, f.input, 2.0f), f.gpu, "zc-gpu-out");
    const bool ok = check(f.alloc(g), scenario, "allocation failed") &&
        check(ggml_backend_sched_graph_compute_async(f.sched, f.graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed") &&
        f.matches(g, 2.0f, scenario);
    ggml_backend_sched_synchronize(f.sched);
    return ok;
}

// Parallel group: the CPU branch reads a GPU-owned tensor in place while the
// GPU branch runs. The tensor stays allocated until the group ends, so the GPU
// branch output cannot take its memory.
static bool run_parallel_group_lifetime() {
    const char * scenario = "zero-copy parallel branch input lifetime";
    fixture f(true);
    if (!check(f.ready(), scenario, "fixture setup failed")) {
        return false;
    }
    ggml_tensor * t = f.op(ggml_scale(f.ctx, f.input, 1.0f), f.gpu, "zc-gpu-shared");
    ggml_tensor * cpu_branch = f.op(ggml_scale(f.ctx, t, 3.0f), f.cpu, "ffn_down.cpu-0");
    ggml_tensor * gpu_branch = f.op(ggml_scale(f.ctx, f.input, 2.0f), f.gpu, "ffn_down.gpu-0");
    ggml_tensor * npu_branch = f.op(ggml_scale(f.ctx, f.input, 4.0f), f.npu, "ffn_down.npu-0");
    ggml_tensor * pair = f.op(ggml_add(f.ctx, cpu_branch, gpu_branch), f.cpu, "ffn_parallel_reduce-0");
    ggml_tensor * out = f.op(ggml_add(f.ctx, pair, npu_branch), f.cpu, "ffn_parallel_out-0");
    return check(f.alloc(out), scenario, "allocation failed") &&
        check(cpu_branch->src[0] == t, scenario, "CPU branch did not read the GPU tensor in place") &&
        check(gpu_branch->data != t->data, scenario, "GPU branch output reused a tensor its sibling still reads") &&
        check(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed") &&
        f.matches(out, 9.0f, scenario) &&
        check(f.synchronizes(f.gpu) > 0, scenario, "GPU was never synchronized");
}

} // namespace

int main() {
    // Read once by ggml; set before any zero-copy query.
#if defined(_WIN32)
    (void) _putenv_s("GGML_ACTIVATION_ZERO_COPY", "7");
#else
    (void) setenv("GGML_ACTIVATION_ZERO_COPY", "7", 1);
#endif
    bool ok = true;
    ok = run_copy_baseline() && ok;
    ok = run_read_after_write() && ok;
    ok = run_write_after_read() && ok;
    ok = run_graph_end() && ok;
    ok = run_parallel_group_lifetime() && ok;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
