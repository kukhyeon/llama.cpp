#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

static bool check(bool condition, const char * scenario, const char * detail) {
    if (!condition) {
        std::fprintf(stderr, "%s: %s\n", scenario, detail);
        return false;
    }
    return true;
}

enum class instrumented_backend_role {
    cpu,
    npu,
    gpu,
};

enum class pending_graph_kind {
    none,
    attn_out_gpu_branch,
    attn_out_join,
};

struct deferred_sync_control {
    std::atomic<bool> gpu_branch_submitted { false };
    std::atomic<bool> cpu_branch_complete { false };
    std::atomic<int> cpu_wait_timeouts { 0 };
    std::atomic<int> gpu_branch_async_calls { 0 };
    std::atomic<int> gpu_join_async_calls { 0 };
    std::atomic<int> gpu_async_flush_calls { 0 };
    std::atomic<int> gpu_join_submitted_while_branch_pending { 0 };
    std::atomic<int> gpu_branch_sync_before_cpu_complete { 0 };
    std::atomic<int> gpu_branch_sync_after_cpu_complete { 0 };
    std::atomic<int> gpu_join_pending_syncs { 0 };
    std::atomic<int> gpu_pending_syncs { 0 };
};

struct instrumented_backend_context {
    ggml_backend_t inner = nullptr;
    ggml_backend_t wrapper = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_buffer_type_t inner_buft = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    instrumented_backend_role role = instrumented_backend_role::cpu;
    enum ggml_backend_dev_type device_type = GGML_BACKEND_DEVICE_TYPE_CPU;
    const char * name = nullptr;
    deferred_sync_control * control = nullptr;
    std::mutex pending_mutex;
    ggml_cgraph * pending_graph = nullptr;
    pending_graph_kind pending_kind = pending_graph_kind::none;
};

static instrumented_backend_context * instrumented_backend_ctx(ggml_backend_t backend) {
    return static_cast<instrumented_backend_context *>(backend->context);
}

static instrumented_backend_context * instrumented_backend_dev_ctx(ggml_backend_dev_t device) {
    return static_cast<instrumented_backend_context *>(device->context);
}

static instrumented_backend_context * instrumented_backend_buft_ctx(
        ggml_backend_buffer_type_t buft) {
    return static_cast<instrumented_backend_context *>(buft->context);
}

static const char * instrumented_buft_name(ggml_backend_buffer_type_t buft) {
    return instrumented_backend_buft_ctx(buft)->name;
}

static ggml_backend_buffer_t instrumented_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    instrumented_backend_context * ctx = instrumented_backend_buft_ctx(buft);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ctx->inner_buft, size);
    if (buffer != nullptr) {
        // Preserve the CPU buffer implementation while giving every fake
        // device a distinct buffer-type identity for scheduler assignment.
        buffer->buft = buft;
    }
    return buffer;
}

static size_t instrumented_buft_alignment(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_alignment(instrumented_backend_buft_ctx(buft)->inner_buft);
}

static size_t instrumented_buft_max_size(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_max_size(instrumented_backend_buft_ctx(buft)->inner_buft);
}

static size_t instrumented_buft_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(
            instrumented_backend_buft_ctx(buft)->inner_buft, tensor);
}

static bool instrumented_buft_is_host(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(instrumented_backend_buft_ctx(buft)->inner_buft);
}

static const ggml_backend_buffer_type_i instrumented_buft_iface = {
    /* .get_name       = */ instrumented_buft_name,
    /* .alloc_buffer   = */ instrumented_buft_alloc_buffer,
    /* .get_alignment  = */ instrumented_buft_alignment,
    /* .get_max_size   = */ instrumented_buft_max_size,
    /* .get_alloc_size = */ instrumented_buft_alloc_size,
    /* .is_host        = */ instrumented_buft_is_host,
};

static bool graph_has_node_prefix(const ggml_cgraph * graph, const char * prefix) {
    const size_t prefix_size = std::strlen(prefix);
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node != nullptr && std::strncmp(node->name, prefix, prefix_size) == 0) {
            return true;
        }
    }
    return false;
}

static const char * instrumented_backend_name(ggml_backend_t backend) {
    return instrumented_backend_ctx(backend)->name;
}

static void instrumented_backend_free(ggml_backend_t backend) {
    instrumented_backend_context * ctx = instrumented_backend_ctx(backend);
    ggml_backend_free(ctx->inner);
    delete ctx->buft;
    delete ctx->device;
    delete ctx;
    delete backend;
}

static void instrumented_backend_synchronize(ggml_backend_t backend) {
    instrumented_backend_context * ctx = instrumented_backend_ctx(backend);
    if (ctx->role != instrumented_backend_role::gpu) {
        ggml_backend_synchronize(ctx->inner);
        return;
    }

    ggml_cgraph * pending_graph = nullptr;
    pending_graph_kind kind = pending_graph_kind::none;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        pending_graph = ctx->pending_graph;
        kind = ctx->pending_kind;
        ctx->pending_graph = nullptr;
        ctx->pending_kind = pending_graph_kind::none;
    }
    if (pending_graph == nullptr) {
        return;
    }

    ctx->control->gpu_pending_syncs.fetch_add(1, std::memory_order_relaxed);
    if (kind == pending_graph_kind::attn_out_gpu_branch) {
        if (ctx->control->cpu_branch_complete.load(std::memory_order_acquire)) {
            ctx->control->gpu_branch_sync_after_cpu_complete.fetch_add(1, std::memory_order_relaxed);
        } else {
            ctx->control->gpu_branch_sync_before_cpu_complete.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (kind == pending_graph_kind::attn_out_join) {
        ctx->control->gpu_join_pending_syncs.fetch_add(1, std::memory_order_relaxed);
    }

    (void) ggml_backend_graph_compute(ctx->inner, pending_graph);
}

static enum ggml_status instrumented_backend_graph_compute(
        ggml_backend_t backend, ggml_cgraph * graph) {
    instrumented_backend_context * ctx = instrumented_backend_ctx(backend);
    if (ctx->role == instrumented_backend_role::cpu) {
        if (graph_has_node_prefix(graph, "attn_out_shard.cpu.")) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!ctx->control->gpu_branch_submitted.load(std::memory_order_acquire) &&
                    std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            if (!ctx->control->gpu_branch_submitted.load(std::memory_order_acquire)) {
                ctx->control->cpu_wait_timeouts.fetch_add(1, std::memory_order_relaxed);
            }

            // Keep the caller-owned CPU branch open long enough that an
            // immediate GPU synchronize is deterministically observable.
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const enum ggml_status status = ggml_backend_graph_compute_async(ctx->inner, graph);
            ctx->control->cpu_branch_complete.store(true, std::memory_order_release);
            return status;
        }
        return ggml_backend_graph_compute_async(ctx->inner, graph);
    }

    if (ctx->role == instrumented_backend_role::npu) {
        return ggml_backend_graph_compute_async(ctx->inner, graph);
    }

    pending_graph_kind kind = pending_graph_kind::none;
    if (graph_has_node_prefix(graph, "attn_out_shard.gpu.")) {
        kind = pending_graph_kind::attn_out_gpu_branch;
        ctx->control->gpu_branch_async_calls.fetch_add(1, std::memory_order_relaxed);
        ctx->control->gpu_branch_submitted.store(true, std::memory_order_release);
    } else if (graph_has_node_prefix(graph, "attn_out_parallel_concat-")) {
        kind = pending_graph_kind::attn_out_join;
        ctx->control->gpu_join_async_calls.fetch_add(1, std::memory_order_relaxed);
    } else {
        return ggml_backend_graph_compute_async(ctx->inner, graph);
    }

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    if (ctx->pending_graph != nullptr) {
        if (kind == pending_graph_kind::attn_out_join &&
                ctx->pending_kind == pending_graph_kind::attn_out_gpu_branch) {
            ctx->control->gpu_join_submitted_while_branch_pending.fetch_add(
                    1, std::memory_order_relaxed);
        }
        return GGML_STATUS_FAILED;
    }
    ctx->pending_graph = graph;
    ctx->pending_kind = kind;
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i instrumented_backend_iface = {
    /* .get_name                = */ instrumented_backend_name,
    /* .free                    = */ instrumented_backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ instrumented_backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ instrumented_backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static const char * instrumented_device_name(ggml_backend_dev_t device) {
    return instrumented_backend_dev_ctx(device)->name;
}

static const char * instrumented_device_description(ggml_backend_dev_t device) {
    return instrumented_backend_dev_ctx(device)->name;
}

static void instrumented_device_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    const auto * ctx = instrumented_backend_dev_ctx(device);
    ggml_backend_dev_memory(ggml_backend_get_device(ctx->inner), free, total);
}

static enum ggml_backend_dev_type instrumented_device_type(ggml_backend_dev_t device) {
    return instrumented_backend_dev_ctx(device)->device_type;
}

static void instrumented_device_props(
        ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    const auto * ctx = instrumented_backend_dev_ctx(device);
    ggml_backend_dev_get_props(ggml_backend_get_device(ctx->inner), props);
    props->name = ctx->name;
    props->description = ctx->name;
    props->type = ctx->device_type;
    props->caps.async = ctx->role == instrumented_backend_role::gpu;
    props->caps.events = false;
}

static ggml_backend_t instrumented_device_init(ggml_backend_dev_t, const char *) {
    return nullptr;
}

static ggml_backend_buffer_type_t instrumented_device_buffer_type(ggml_backend_dev_t device) {
    return instrumented_backend_dev_ctx(device)->buft;
}

static ggml_backend_buffer_type_t instrumented_device_host_buffer_type(ggml_backend_dev_t device) {
    return instrumented_backend_dev_ctx(device)->buft;
}

static bool instrumented_device_supports_op(
        ggml_backend_dev_t device, const ggml_tensor * op) {
    const auto * ctx = instrumented_backend_dev_ctx(device);
    return ggml_backend_dev_supports_op(ggml_backend_get_device(ctx->inner), op);
}

static bool instrumented_device_supports_buft(
        ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return buft == instrumented_backend_dev_ctx(device)->buft;
}

static bool instrumented_device_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static void instrumented_backend_async_flush(ggml_backend_t backend) {
    instrumented_backend_context * ctx = instrumented_backend_ctx(backend);
    ctx->control->gpu_async_flush_calls.fetch_add(1, std::memory_order_relaxed);
}

static const char * instrumented_registry_name(ggml_backend_reg_t) {
    return "InstrumentedOpenCLTest";
}

static size_t instrumented_registry_device_count(ggml_backend_reg_t) {
    return 0;
}

static ggml_backend_dev_t instrumented_registry_device(ggml_backend_reg_t, size_t) {
    return nullptr;
}

static void * instrumented_registry_proc_address(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_async_flush") == 0) {
        return reinterpret_cast<void *>(instrumented_backend_async_flush);
    }
    return nullptr;
}

static const ggml_backend_reg_i instrumented_registry_iface = {
    /* .get_name         = */ instrumented_registry_name,
    /* .get_device_count = */ instrumented_registry_device_count,
    /* .get_device       = */ instrumented_registry_device,
    /* .get_proc_address = */ instrumented_registry_proc_address,
};

static ggml_backend_reg instrumented_opencl_registry = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ instrumented_registry_iface,
    /* .context     = */ nullptr,
};

static const ggml_backend_device_i instrumented_device_iface = {
    /* .get_name             = */ instrumented_device_name,
    /* .get_description      = */ instrumented_device_description,
    /* .get_memory           = */ instrumented_device_memory,
    /* .get_type             = */ instrumented_device_type,
    /* .get_props            = */ instrumented_device_props,
    /* .init_backend         = */ instrumented_device_init,
    /* .get_buffer_type      = */ instrumented_device_buffer_type,
    /* .get_host_buffer_type = */ instrumented_device_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ instrumented_device_supports_op,
    /* .supports_buft        = */ instrumented_device_supports_buft,
    /* .offload_op           = */ instrumented_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_guid_t instrumented_backend_guid() {
    static ggml_guid guid = {
        0x5a, 0x84, 0x98, 0x22, 0x11, 0xc7, 0x49, 0xa0,
        0xb3, 0x1e, 0x64, 0x2f, 0x18, 0x73, 0x9c, 0xd2,
    };
    return &guid;
}

static ggml_backend_t make_instrumented_backend(
        instrumented_backend_role role,
        const char * name,
        enum ggml_backend_dev_type device_type,
        deferred_sync_control * control) {
    auto * ctx = new instrumented_backend_context;
    ctx->inner = ggml_backend_cpu_init();
    if (ctx->inner == nullptr) {
        delete ctx;
        return nullptr;
    }

    ctx->role = role;
    ctx->device_type = device_type;
    ctx->name = name;
    ctx->control = control;
    ctx->inner_buft = ggml_backend_get_default_buffer_type(ctx->inner);
    ctx->device = new ggml_backend_device {
        /* .iface   = */ instrumented_device_iface,
        /* .reg     = */ role == instrumented_backend_role::gpu
            ? &instrumented_opencl_registry
            : nullptr,
        /* .context = */ ctx,
    };
    ctx->buft = new ggml_backend_buffer_type {
        /* .iface   = */ instrumented_buft_iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
    ctx->wrapper = new ggml_backend {
        /* .guid    = */ instrumented_backend_guid(),
        /* .iface   = */ instrumented_backend_iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
    return ctx->wrapper;
}

class temporary_trace_file {
public:
    explicit temporary_trace_file(const char * tag) {
        std::error_code ec;
        const std::filesystem::path directory = std::filesystem::temp_directory_path(ec);
        if (ec) {
            throw std::runtime_error("failed to locate the temporary directory");
        }

        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            const std::filesystem::path candidate = directory /
                ("llama-test-backend-sched-parallel-" + std::to_string(stamp) + "-" +
                 tag + "-" + std::to_string(attempt) + ".csv");
            if (std::filesystem::exists(candidate, ec)) {
                if (ec) {
                    throw std::runtime_error("failed to inspect a temporary trace path");
                }
                continue;
            }

            std::ofstream file(candidate, std::ios::binary | std::ios::trunc);
            if (file) {
                path_ = candidate.string();
                return;
            }
        }

        throw std::runtime_error("failed to create a temporary scheduler trace");
    }

    ~temporary_trace_file() {
        std::error_code ec;
        (void) std::filesystem::remove(path_, ec);
    }

    temporary_trace_file(const temporary_trace_file &) = delete;
    temporary_trace_file & operator=(const temporary_trace_file &) = delete;

    const std::string & path() const {
        return path_;
    }

private:
    std::string path_;
};

class scheduler_trace_capture {
public:
    scheduler_trace_capture(const std::string & path, int query_id) {
        ggml_backend_sched_trace_set_enabled(false);
        ggml_backend_sched_trace_set_path(path.c_str());
        ggml_backend_sched_trace_set_enabled(true);
        ggml_backend_sched_trace_reset();
        ggml_backend_sched_trace_set_query_id(query_id);
        ggml_backend_sched_trace_set_ubatch(0, 0, 1);
        active_ = true;
    }

    ~scheduler_trace_capture() {
        if (active_) {
            ggml_backend_sched_trace_set_enabled(false);
        }
    }

    scheduler_trace_capture(const scheduler_trace_capture &) = delete;
    scheduler_trace_capture & operator=(const scheduler_trace_capture &) = delete;

    void finish() {
        if (!active_) {
            return;
        }
        ggml_backend_sched_trace_flush();
        ggml_backend_sched_trace_set_enabled(false);
        active_ = false;
    }

private:
    bool active_ = false;
};

struct qkv_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    ~qkv_fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    bool build(bool include_k) {
        constexpr size_t graph_size = 32;
        const ggml_init_params params = {
            /* .mem_size   = */ 64 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        backend = ggml_backend_cpu_init();
        ggml_backend_t backends[] = { backend };
        sched = backend != nullptr
            ? ggml_backend_sched_new(backends, nullptr, 1, 128, false, true)
            : nullptr;
        if (ctx == nullptr || sched == nullptr) {
            return false;
        }

        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_input(input);
        ggml_set_name(input, "attn-qkv-input");

        ggml_tensor * q = ggml_scale(ctx, input, 2.0f);
        ggml_tensor * v = ggml_scale(ctx, input, 3.0f);
        ggml_set_name(q, "attn_qkv.q.proj-0");
        ggml_set_name(v, "attn_qkv.v.proj-0");

        graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, v);

        ggml_tensor * joined = ggml_add(ctx, q, v);
        if (include_k) {
            ggml_tensor * k = ggml_scale(ctx, input, 4.0f);
            ggml_set_name(k, "attn_qkv.k.proj-0");
            ggml_build_forward_expand(graph, k);
            joined = ggml_add(ctx, joined, k);
            ggml_backend_sched_set_tensor_backend(sched, k, backend);
        }

        ggml_set_name(joined, include_k ? "attn-qkv-join-0" : "attn-qv-join-0");
        ggml_set_output(joined);
        ggml_build_forward_expand(graph, joined);
        output = joined;

        ggml_backend_sched_set_tensor_backend(sched, input, backend);
        ggml_backend_sched_set_tensor_backend(sched, q, backend);
        ggml_backend_sched_set_tensor_backend(sched, v, backend);
        ggml_backend_sched_set_tensor_backend(sched, output, backend);
        return ggml_backend_sched_alloc_graph(sched, graph);
    }

    bool compute_and_check(float expected_scale, const char * scenario) {
        const float values[4] = { -2.0f, -0.5f, 1.25f, 3.0f };
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario,
                    "graph compute failed")) {
            return false;
        }

        float result[4] = {};
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 4; ++i) {
            const float expected = values[i] * expected_scale;
            if (!check(std::fabs(result[i] - expected) < 1e-5f, scenario, "output mismatch")) {
                std::fprintf(
                        stderr,
                        "  index %d: got %.6f, expected %.6f\n",
                        i,
                        result[i],
                        expected);
                return false;
            }
        }
        return true;
    }
};

struct qkv_shard_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    ~qkv_shard_fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    bool build(int lane_count) {
        constexpr size_t graph_size = 64;
        const ggml_init_params params = {
            /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        backend = ggml_backend_cpu_init();
        ggml_backend_t backends[] = { backend };
        sched = backend != nullptr
            ? ggml_backend_sched_new(backends, nullptr, 1, 128, false, true)
            : nullptr;
        if (ctx == nullptr || sched == nullptr || lane_count <= 0) {
            return false;
        }

        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_input(input);
        ggml_set_name(input, "attn-qkv-shard-input");

        graph = ggml_new_graph_custom(ctx, graph_size, false);
        std::vector<ggml_tensor *> branch_nodes;
        static const char * projections[] = { "q", "k", "v" };
        float scale = 1.0f;
        for (int lane = 0; lane < lane_count; ++lane) {
            for (const char * projection : projections) {
                ggml_tensor * node = ggml_scale(ctx, input, scale++);
                const std::string name =
                    "attn_qkv_shard.lane" + std::to_string(lane) + "." +
                    projection + ".proj-0";
                ggml_set_name(node, name.c_str());
                ggml_build_forward_expand(graph, node);
                ggml_backend_sched_set_tensor_backend(sched, node, backend);
                branch_nodes.push_back(node);
            }
        }

        ggml_tensor * joined = branch_nodes[0];
        for (size_t i = 1; i < branch_nodes.size(); ++i) {
            joined = ggml_add(ctx, joined, branch_nodes[i]);
            const std::string name = "attn-qkv-shard-join-" + std::to_string(i);
            ggml_set_name(joined, name.c_str());
            ggml_backend_sched_set_tensor_backend(sched, joined, backend);
        }
        ggml_set_output(joined);
        ggml_build_forward_expand(graph, joined);
        output = joined;

        ggml_backend_sched_set_tensor_backend(sched, input, backend);
        return ggml_backend_sched_alloc_graph(sched, graph);
    }

    bool compute_and_check(float expected_scale, const char * scenario) {
        const float values[4] = { -2.0f, -0.5f, 1.25f, 3.0f };
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario,
                    "graph compute failed")) {
            return false;
        }

        float result[4] = {};
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 4; ++i) {
            const float expected = values[i] * expected_scale;
            if (!check(std::fabs(result[i] - expected) < 1e-5f, scenario, "output mismatch")) {
                std::fprintf(
                        stderr,
                        "  index %d: got %.6f, expected %.6f\n",
                        i,
                        result[i],
                        expected);
                return false;
            }
        }
        return true;
    }
};

struct attn_out_shard_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    int lane_count = 0;

    ~attn_out_shard_fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    bool build(int lane_count) {
        constexpr size_t graph_size = 64;
        const ggml_init_params params = {
            /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        backend = ggml_backend_cpu_init();
        ggml_backend_t backends[] = { backend };
        sched = backend != nullptr
            ? ggml_backend_sched_new(backends, nullptr, 1, 128, false, true)
            : nullptr;
        if (ctx == nullptr || sched == nullptr || lane_count <= 0) {
            return false;
        }

        constexpr int64_t lane_width = 2;
        constexpr int64_t parent_width = 6;
        constexpr int64_t rows = 2;
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, parent_width, rows);
        ggml_set_input(input);
        ggml_set_name(input, "attn-out-shard-source");
        this->lane_count = lane_count;

        graph = ggml_new_graph_custom(ctx, graph_size, false);
        std::vector<ggml_tensor *> views;
        views.reserve((size_t) lane_count);
        for (int lane = 0; lane < lane_count; ++lane) {
            const std::string lane_name = "lane" + std::to_string(lane);
            ggml_tensor * view = ggml_view_2d(
                    ctx, input, lane_width, rows, input->nb[1],
                    (size_t) lane * lane_width * input->nb[0]);
            ggml_set_name(view, ("attn_out_shard." + lane_name + ".view-0").c_str());
            views.push_back(view);
            ggml_build_forward_expand(graph, view);
        }

        std::vector<ggml_tensor *> partials;
        partials.reserve((size_t) lane_count);
        for (int lane = 0; lane < lane_count; ++lane) {
            const std::string lane_name = "lane" + std::to_string(lane);
            ggml_tensor * compact = ggml_cont(ctx, views[(size_t) lane]);
            ggml_set_name(compact, ("attn_out_shard." + lane_name + ".input-0").c_str());
            ggml_tensor * projection = ggml_scale(ctx, compact, (float) lane + 2.0f);
            ggml_set_name(projection, ("attn_out_shard." + lane_name + ".proj-0").c_str());

            ggml_backend_sched_set_tensor_backend(sched, compact, backend);
            ggml_backend_sched_set_tensor_backend(sched, projection, backend);
            ggml_build_forward_expand(graph, projection);
            partials.push_back(projection);
        }

        ggml_tensor * joined = partials.back();
        for (size_t i = partials.size() - 1; i > 0; --i) {
            joined = ggml_add(ctx, partials[i - 1], joined);
            const std::string name = "attn-out-shard-sum-" + std::to_string(i - 1);
            ggml_set_name(joined, name.c_str());
            ggml_backend_sched_set_tensor_backend(sched, joined, backend);
        }
        ggml_set_output(joined);
        ggml_build_forward_expand(graph, joined);
        output = joined;

        ggml_backend_sched_set_tensor_backend(sched, input, backend);
        return ggml_backend_sched_alloc_graph(sched, graph);
    }

    bool compute_and_check(const char * scenario) {
        constexpr int lane_width = 2;
        constexpr int parent_width = 6;
        constexpr int rows = 2;
        const float values[parent_width * rows] = {
            -2.0f, -0.5f,  1.25f,  3.0f, -4.0f,  0.75f,
             0.5f,  2.0f, -1.50f, -3.0f,  4.5f, -0.25f,
        };
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario,
                    "graph compute failed")) {
            return false;
        }

        float result[lane_width * rows] = {};
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < lane_width; ++col) {
                float expected = 0.0f;
                for (int lane = 0; lane < lane_count; ++lane) {
                    expected += values[row * parent_width + lane * lane_width + col] *
                        ((float) lane + 2.0f);
                }
                const int index = row * lane_width + col;
                if (check(std::fabs(result[index] - expected) < 1e-5f, scenario, "output mismatch")) {
                    continue;
                }
                std::fprintf(
                        stderr,
                        "  index %d: got %.6f, expected %.6f\n",
                        index,
                        result[index],
                        expected);
                return false;
            }
        }
        return true;
    }
};

struct attn_out_output_axis_shard_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    ~attn_out_output_axis_shard_fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    bool build() {
        constexpr size_t graph_size = 64;
        const ggml_init_params params = {
            /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        backend = ggml_backend_cpu_init();
        ggml_backend_t backends[] = { backend };
        sched = backend != nullptr
            ? ggml_backend_sched_new(backends, nullptr, 1, 128, false, true)
            : nullptr;
        if (ctx == nullptr || sched == nullptr) {
            return false;
        }

        constexpr int64_t lane_width = 2;
        constexpr int64_t rows = 2;
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lane_width, rows);
        ggml_set_input(input);
        ggml_set_name(input, "attn-out-output-axis-source");

        graph = ggml_new_graph_custom(ctx, graph_size, false);
        static const char * lane_names[] = { "cpu", "npu", "gpu" };
        std::vector<ggml_tensor *> projections;
        projections.reserve(3);
        for (int lane = 0; lane < 3; ++lane) {
            ggml_tensor * projection = ggml_scale(ctx, input, (float) lane + 2.0f);
            const std::string name =
                "attn_out_shard." + std::string(lane_names[lane]) + ".proj-0";
            ggml_set_name(projection, name.c_str());
            ggml_backend_sched_set_tensor_backend(sched, projection, backend);
            ggml_build_forward_expand(graph, projection);
            projections.push_back(projection);
        }

        // Match the output-axis W_O graph: each branch produces a disjoint
        // output range and the canonical result is assembled without a sum.
        ggml_tensor * joined = projections.back();
        for (size_t i = projections.size() - 1; i > 0; --i) {
            joined = ggml_concat(ctx, projections[i - 1], joined, 0);
            const std::string name =
                "attn_out_parallel_concat-" + std::to_string(i - 1);
            ggml_set_name(joined, name.c_str());
            // The real policy may choose OpenCL for both the last lane and
            // CONCAT. Reusing this backend pins the equivalent split boundary.
            ggml_backend_sched_set_tensor_backend(sched, joined, backend);
        }
        ggml_set_output(joined);
        ggml_build_forward_expand(graph, joined);
        output = joined;

        ggml_backend_sched_set_tensor_backend(sched, input, backend);
        return ggml_backend_sched_alloc_graph(sched, graph);
    }

    bool compute_and_check(const char * scenario) {
        constexpr int lane_count = 3;
        constexpr int lane_width = 2;
        constexpr int rows = 2;
        const float values[lane_width * rows] = {
            -2.0f, 0.5f,
             1.25f, 3.0f,
        };
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario,
                    "graph compute failed")) {
            return false;
        }

        float result[lane_count * lane_width * rows] = {};
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int row = 0; row < rows; ++row) {
            for (int lane = 0; lane < lane_count; ++lane) {
                for (int col = 0; col < lane_width; ++col) {
                    const int index =
                        row * lane_count * lane_width + lane * lane_width + col;
                    const float expected = values[row * lane_width + col] *
                        ((float) lane + 2.0f);
                    if (check(
                                std::fabs(result[index] - expected) < 1e-5f,
                                scenario,
                                "output mismatch")) {
                        continue;
                    }
                    std::fprintf(
                            stderr,
                            "  index %d: got %.6f, expected %.6f\n",
                            index,
                            result[index],
                            expected);
                    return false;
                }
            }
        }
        return true;
    }
};

struct attn_out_deferred_sync_fixture {
    deferred_sync_control control;
    ggml_context * ctx = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t npu_backend = nullptr;
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    ~attn_out_deferred_sync_fixture() {
        ggml_backend_sched_free(sched);
        if (gpu_backend != nullptr) {
            ggml_backend_free(gpu_backend);
        }
        if (npu_backend != nullptr) {
            ggml_backend_free(npu_backend);
        }
        if (cpu_backend != nullptr) {
            ggml_backend_free(cpu_backend);
        }
        ggml_free(ctx);
    }

    bool build(bool join_on_gpu = true) {
        constexpr size_t graph_size = 96;
        const ggml_init_params params = {
            /* .mem_size   = */ 192 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        cpu_backend = make_instrumented_backend(
                instrumented_backend_role::cpu,
                "TestCPU",
                GGML_BACKEND_DEVICE_TYPE_CPU,
                &control);
        npu_backend = make_instrumented_backend(
                instrumented_backend_role::npu,
                "HTP0-Test",
                GGML_BACKEND_DEVICE_TYPE_ACCEL,
                &control);
        gpu_backend = make_instrumented_backend(
                instrumented_backend_role::gpu,
                "OpenCLTest",
                GGML_BACKEND_DEVICE_TYPE_GPU,
                &control);
        if (ctx == nullptr || cpu_backend == nullptr ||
                npu_backend == nullptr || gpu_backend == nullptr) {
            return false;
        }

        // The scheduler requires the CPU fallback to be the final backend.
        ggml_backend_t backends[] = { gpu_backend, npu_backend, cpu_backend };
        ggml_backend_buffer_type_t bufts[] = {
            ggml_backend_get_default_buffer_type(gpu_backend),
            ggml_backend_get_default_buffer_type(npu_backend),
            ggml_backend_get_default_buffer_type(cpu_backend),
        };
        sched = ggml_backend_sched_new(backends, bufts, 3, 192, false, true);
        if (sched == nullptr) {
            return false;
        }

        constexpr int64_t lane_width = 2;
        constexpr int64_t rows = 2;
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lane_width, rows);
        ggml_set_input(input);
        ggml_set_name(input, "attn-out-deferred-source");

        graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_backend_t lane_backends[] = { cpu_backend, npu_backend, gpu_backend };
        static const char * lane_names[] = { "cpu", "npu", "gpu" };
        std::vector<ggml_tensor *> projections;
        projections.reserve(3);
        for (int lane = 0; lane < 3; ++lane) {
            ggml_tensor * projection = ggml_scale(ctx, input, (float) lane + 2.0f);
            const std::string name =
                "attn_out_shard." + std::string(lane_names[lane]) + ".proj-0";
            ggml_set_name(projection, name.c_str());
            ggml_backend_sched_set_tensor_backend(sched, projection, lane_backends[lane]);
            ggml_build_forward_expand(graph, projection);
            projections.push_back(projection);
        }

        ggml_backend_t join_backend = join_on_gpu ? gpu_backend : npu_backend;
        ggml_tensor * joined = projections.back();
        for (size_t i = projections.size() - 1; i > 0; --i) {
            joined = ggml_concat(ctx, projections[i - 1], joined, 0);
            // Both CONCAT nodes belong to the same transformer layer, as in
            // the production output-axis W_O builder.
            ggml_set_name(joined, "attn_out_parallel_concat-deferred-0");
            ggml_backend_sched_set_tensor_backend(sched, joined, join_backend);
        }

        // A consumer on a different backend forces the deferred CONCAT itself
        // to complete before its result is copied out of OpenCL.
        output = ggml_scale(ctx, joined, 0.5f);
        ggml_set_name(output, "attn-out-deferred-consumer-0");
        ggml_set_output(output);
        ggml_backend_sched_set_tensor_backend(sched, output, cpu_backend);
        ggml_build_forward_expand(graph, output);

        ggml_backend_sched_set_tensor_backend(sched, input, cpu_backend);
        return ggml_backend_sched_alloc_graph(sched, graph);
    }

    bool compute_and_check(const char * scenario) {
        constexpr int lane_count = 3;
        constexpr int lane_width = 2;
        constexpr int rows = 2;
        const float values[lane_width * rows] = {
            -2.0f, 0.5f,
             1.25f, 3.0f,
        };
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario,
                    "graph compute failed")) {
            return false;
        }

        float result[lane_count * lane_width * rows] = {};
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int row = 0; row < rows; ++row) {
            for (int lane = 0; lane < lane_count; ++lane) {
                for (int col = 0; col < lane_width; ++col) {
                    const int index =
                        row * lane_count * lane_width + lane * lane_width + col;
                    const float expected = values[row * lane_width + col] *
                        ((float) lane + 2.0f) * 0.5f;
                    if (!check(
                                std::fabs(result[index] - expected) < 1e-5f,
                                scenario,
                                "deferred join output mismatch")) {
                        std::fprintf(
                                stderr,
                                "  index %d: got %.6f, expected %.6f\n",
                                index,
                                result[index],
                                expected);
                        return false;
                    }
                }
            }
        }
        return true;
    }
};

struct trace_row {
    int split_id = -1;
    int group_id = -1;
    int node_count = -1;
    bool is_parallel_group = false;
    std::string first_node;
    std::string last_node;
    std::string timing_mode;
    std::string parallel_kind;
    std::string parallel_branch;
};

static bool parse_csv_line(const std::string & line, std::vector<std::string> * fields) {
    fields->clear();
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields->push_back(std::move(field));
            field.clear();
        } else if (ch != '\r') {
            field.push_back(ch);
        }
    }
    if (quoted) {
        return false;
    }
    fields->push_back(std::move(field));
    return true;
}

static int column_index(const std::vector<std::string> & columns, const char * name) {
    const auto it = std::find(columns.begin(), columns.end(), name);
    return it == columns.end() ? -1 : (int) std::distance(columns.begin(), it);
}

static bool parse_int(const std::string & text, int * value) {
    char * end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *value = (int) parsed;
    return true;
}

static bool read_trace(
        const std::string & path,
        const char * scenario,
        std::vector<trace_row> * rows) {
    std::ifstream file(path, std::ios::binary);
    if (!check((bool) file, scenario, "failed to open scheduler trace")) {
        return false;
    }

    std::string line;
    std::vector<std::string> columns;
    if (!check(
                std::getline(file, line) && parse_csv_line(line, &columns),
                scenario,
                "failed to read scheduler trace header")) {
        return false;
    }

    const int split_id_col = column_index(columns, "split_id");
    const int group_id_col = column_index(columns, "group_id");
    const int node_count_col = column_index(columns, "node_count");
    const int first_node_col = column_index(columns, "first_node");
    const int last_node_col = column_index(columns, "last_node");
    const int timing_mode_col = column_index(columns, "timing_mode");
    const int is_parallel_group_col = column_index(columns, "is_parallel_group");
    const int parallel_kind_col = column_index(columns, "parallel_group_kind");
    const int parallel_branch_col = column_index(columns, "parallel_branch");
    const int cpu_idle_before_us_col = column_index(columns, "cpu_idle_before_us");
    const int cpu_prewake_requested_col = column_index(columns, "cpu_prewake_requested");
    const int required_columns[] = {
        split_id_col,
        group_id_col,
        node_count_col,
        first_node_col,
        last_node_col,
        timing_mode_col,
        is_parallel_group_col,
        parallel_kind_col,
        parallel_branch_col,
        cpu_idle_before_us_col,
        cpu_prewake_requested_col,
    };
    for (int column : required_columns) {
        if (!check(column >= 0, scenario, "scheduler trace is missing a required column")) {
            return false;
        }
    }

    rows->clear();
    std::vector<std::string> fields;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (!check(parse_csv_line(line, &fields), scenario, "malformed scheduler trace row") ||
                !check(fields.size() == columns.size(), scenario, "scheduler trace column count mismatch")) {
            return false;
        }

        trace_row row;
        if (!check(parse_int(fields[split_id_col], &row.split_id), scenario, "invalid split id") ||
                !check(parse_int(fields[group_id_col], &row.group_id), scenario, "invalid group id") ||
                !check(parse_int(fields[node_count_col], &row.node_count), scenario, "invalid node count")) {
            return false;
        }
        row.is_parallel_group = fields[is_parallel_group_col] == "1";
        row.first_node = fields[first_node_col];
        row.last_node = fields[last_node_col];
        row.timing_mode = fields[timing_mode_col];
        row.parallel_kind = fields[parallel_kind_col];
        row.parallel_branch = fields[parallel_branch_col];
        rows->push_back(std::move(row));
    }

    return check(!rows->empty(), scenario, "scheduler trace contained no rows");
}

static bool check_attn_out_group_timing_mode(
        const std::string & path,
        const char * scenario,
        const char * expected_mode) {
    std::vector<trace_row> rows;
    if (!read_trace(path, scenario, &rows)) {
        return false;
    }

    int grouped_rows = 0;
    for (const trace_row & row : rows) {
        if (!row.is_parallel_group || row.parallel_kind != "attn_out_shard") {
            continue;
        }
        grouped_rows++;
        if (!check(row.timing_mode == expected_mode, scenario,
                    "unexpected attn_out group timing mode")) {
            std::fprintf(
                    stderr,
                    "  branch %s: got %s, expected %s\n",
                    row.parallel_branch.c_str(),
                    row.timing_mode.c_str(),
                    expected_mode);
            return false;
        }
    }
    return check(grouped_rows == 3, scenario,
        "expected exactly three traced attn_out group rows");
}

static bool run_exact_qvk_group_case() {
    const char * scenario = "exact-qvk-group";
    temporary_trace_file trace_file(scenario);
    qkv_fixture fixture;
    if (!check(fixture.build(true), scenario, "fixture setup failed") ||
            !check(
                ggml_backend_sched_get_n_splits(fixture.sched) == 4,
                scenario,
                "Q/V/K branches were not isolated from each other and the join")) {
        return false;
    }

    {
        scheduler_trace_capture capture(trace_file.path(), 101);
        if (!fixture.compute_and_check(9.0f, scenario)) {
            return false;
        }
        capture.finish();
    }

    std::vector<trace_row> rows;
    if (!read_trace(trace_file.path(), scenario, &rows)) {
        return false;
    }

    std::vector<trace_row> grouped;
    for (const trace_row & row : rows) {
        if (row.is_parallel_group) {
            grouped.push_back(row);
        }
    }
    if (!check(grouped.size() == 3, scenario, "expected exactly three grouped trace rows")) {
        return false;
    }

    const char * expected_branches[] = { "q", "v", "k" };
    const char * expected_nodes[] = {
        "attn_qkv.q.proj-0",
        "attn_qkv.v.proj-0",
        "attn_qkv.k.proj-0",
    };
    const int group_id = grouped[0].group_id;
    for (int i = 0; i < 3; ++i) {
        const trace_row & row = grouped[i];
        if (!check(row.parallel_kind == "attn_qkv", scenario, "parallel group kind mismatch") ||
                !check(row.parallel_branch == expected_branches[i], scenario, "parallel branch order mismatch") ||
                !check(row.timing_mode == "serial_fallback", scenario, "single-backend group was not serialized") ||
                !check(row.group_id == group_id && group_id >= 0, scenario, "parallel group id mismatch") ||
                !check(row.node_count == 1, scenario, "parallel branch split was not pure") ||
                !check(row.first_node == expected_nodes[i], scenario, "parallel split first node mismatch") ||
                !check(row.last_node == expected_nodes[i], scenario, "parallel split last node mismatch")) {
            return false;
        }
    }
    return true;
}

static bool run_incomplete_qv_case() {
    const char * scenario = "incomplete-qv-non-group";
    temporary_trace_file trace_file(scenario);
    qkv_fixture fixture;
    if (!check(fixture.build(false), scenario, "fixture setup failed") ||
            !check(
                ggml_backend_sched_get_n_splits(fixture.sched) == 3,
                scenario,
                "Q/V branches were not isolated from each other and the join")) {
        return false;
    }

    {
        scheduler_trace_capture capture(trace_file.path(), 102);
        if (!fixture.compute_and_check(5.0f, scenario)) {
            return false;
        }
        capture.finish();
    }

    std::vector<trace_row> rows;
    if (!read_trace(trace_file.path(), scenario, &rows)) {
        return false;
    }

    for (const trace_row & row : rows) {
        if (!check(!row.is_parallel_group, scenario, "incomplete Q/V set was collected as a group")) {
            return false;
        }
    }

    const char * branch_nodes[] = { "attn_qkv.q.proj-0", "attn_qkv.v.proj-0" };
    int branch_split_ids[2] = { -1, -1 };
    for (int i = 0; i < 2; ++i) {
        const auto it = std::find_if(rows.begin(), rows.end(), [&](const trace_row & row) {
            return row.first_node == branch_nodes[i] && row.last_node == branch_nodes[i];
        });
        if (!check(it != rows.end(), scenario, "isolated Q/V branch row was not traced") ||
                !check(it->node_count == 1, scenario, "incomplete branch split was not pure")) {
            return false;
        }
        branch_split_ids[i] = it->split_id;
    }
    return check(
        branch_split_ids[0] != branch_split_ids[1],
        scenario,
        "incomplete Q/V branches shared a split");
}

static bool run_exact_qkv_shard_lane_group_case() {
    const char * scenario = "exact-qkv-shard-lane-group";
    temporary_trace_file trace_file(scenario);
    qkv_shard_fixture fixture;
    if (!check(fixture.build(3), scenario, "fixture setup failed") ||
            !check(
                ggml_backend_sched_get_n_splits(fixture.sched) == 4,
                scenario,
                "three composite lanes were not isolated from each other and the join")) {
        return false;
    }

    {
        scheduler_trace_capture capture(trace_file.path(), 103);
        if (!fixture.compute_and_check(45.0f, scenario)) {
            return false;
        }
        capture.finish();
    }

    std::vector<trace_row> rows;
    if (!read_trace(trace_file.path(), scenario, &rows)) {
        return false;
    }

    std::vector<trace_row> grouped;
    for (const trace_row & row : rows) {
        if (row.is_parallel_group) {
            grouped.push_back(row);
        }
    }
    if (!check(grouped.size() == 3, scenario, "expected exactly three grouped lane rows")) {
        return false;
    }

    const int group_id = grouped[0].group_id;
    for (int lane = 0; lane < 3; ++lane) {
        const trace_row & row = grouped[lane];
        const std::string expected_lane = "lane" + std::to_string(lane);
        const std::string expected_first =
            "attn_qkv_shard." + expected_lane + ".q.proj-0";
        const std::string expected_last =
            "attn_qkv_shard." + expected_lane + ".v.proj-0";
        if (!check(row.parallel_kind == "attn_qkv_shard", scenario, "parallel group kind mismatch") ||
                !check(row.parallel_branch == expected_lane, scenario, "parallel lane order mismatch") ||
                !check(row.timing_mode == "serial_fallback", scenario, "single-backend lanes were not serialized") ||
                !check(row.group_id == group_id && group_id >= 0, scenario, "parallel group id mismatch") ||
                !check(row.node_count == 3, scenario, "Q/K/V nodes did not remain in one lane split") ||
                !check(row.first_node == expected_first, scenario, "lane split first node mismatch") ||
                !check(row.last_node == expected_last, scenario, "lane split last node mismatch")) {
            return false;
        }
    }
    return true;
}

static bool run_incomplete_qkv_shard_lane_case() {
    const char * scenario = "incomplete-qkv-shard-lane-non-group";
    temporary_trace_file trace_file(scenario);
    qkv_shard_fixture fixture;
    if (!check(fixture.build(2), scenario, "fixture setup failed") ||
            !check(
                ggml_backend_sched_get_n_splits(fixture.sched) == 3,
                scenario,
                "two composite lanes were not isolated from each other and the join")) {
        return false;
    }

    {
        scheduler_trace_capture capture(trace_file.path(), 104);
        if (!fixture.compute_and_check(21.0f, scenario)) {
            return false;
        }
        capture.finish();
    }

    std::vector<trace_row> rows;
    if (!read_trace(trace_file.path(), scenario, &rows)) {
        return false;
    }
    for (const trace_row & row : rows) {
        if (!check(!row.is_parallel_group, scenario, "incomplete lane set was collected as a group")) {
            return false;
        }
    }

    int lane_split_ids[2] = { -1, -1 };
    for (int lane = 0; lane < 2; ++lane) {
        const std::string expected_lane = "lane" + std::to_string(lane);
        const std::string expected_first =
            "attn_qkv_shard." + expected_lane + ".q.proj-0";
        const std::string expected_last =
            "attn_qkv_shard." + expected_lane + ".v.proj-0";
        const auto it = std::find_if(rows.begin(), rows.end(), [&](const trace_row & row) {
            return row.first_node == expected_first && row.last_node == expected_last;
        });
        if (!check(it != rows.end(), scenario, "composite lane row was not traced") ||
                !check(it->node_count == 3, scenario, "incomplete lane split was not pure")) {
            return false;
        }
        lane_split_ids[lane] = it->split_id;
    }
    return check(
        lane_split_ids[0] != lane_split_ids[1],
        scenario,
        "incomplete lanes shared a split");
}

static bool run_exact_attn_out_shard_lane_group_case() {
    const char * scenario = "exact-attn-out-shard-lane-group";
    temporary_trace_file trace_file(scenario);
    attn_out_shard_fixture fixture;
    if (!check(fixture.build(3), scenario, "fixture setup failed")) {
        return false;
    }
    const int n_splits = ggml_backend_sched_get_n_splits(fixture.sched);
    if (!check(n_splits == 5, scenario,
                "three W_O lanes were not isolated from each other and the ADD join")) {
        std::fprintf(stderr, "  got %d splits, expected 5\n", n_splits);
        return false;
    }

    {
        scheduler_trace_capture capture(trace_file.path(), 105);
        if (!fixture.compute_and_check(scenario)) {
            return false;
        }
        capture.finish();
    }

    std::vector<trace_row> rows;
    if (!read_trace(trace_file.path(), scenario, &rows)) {
        return false;
    }

    std::vector<trace_row> grouped;
    for (const trace_row & row : rows) {
        if (row.is_parallel_group) {
            grouped.push_back(row);
        }
    }
    if (!check(grouped.size() == 3, scenario, "expected exactly three grouped W_O lane rows")) {
        return false;
    }

    const int group_id = grouped[0].group_id;
    for (int lane = 0; lane < 3; ++lane) {
        const trace_row & row = grouped[lane];
        const std::string expected_lane = "lane" + std::to_string(lane);
        const std::string expected_first =
            "attn_out_shard." + expected_lane + ".input-0";
        const std::string expected_last =
            "attn_out_shard." + expected_lane + ".proj-0";
        const int expected_node_count = 2;
        if (row.node_count != expected_node_count ||
                row.first_node != expected_first || row.last_node != expected_last) {
            std::fprintf(stderr, "  lane %d trace: nodes=%d first=%s last=%s\n",
                    lane, row.node_count, row.first_node.c_str(), row.last_node.c_str());
        }
        if (!check(row.parallel_kind == "attn_out_shard", scenario, "parallel group kind mismatch") ||
                !check(row.parallel_branch == expected_lane, scenario, "parallel lane order mismatch") ||
                !check(row.timing_mode == "serial_fallback", scenario, "single-backend lanes were not serialized") ||
                !check(row.group_id == group_id && group_id >= 0, scenario, "parallel group id mismatch") ||
                !check(row.node_count == expected_node_count, scenario,
                    "W_O executable input/projection did not remain in one lane split") ||
                !check(row.first_node == expected_first, scenario, "lane split first node mismatch") ||
                !check(row.last_node == expected_last, scenario, "lane split last node mismatch")) {
            return false;
        }
    }
    return true;
}

static bool run_incomplete_attn_out_shard_lane_case() {
    const char * scenario = "incomplete-attn-out-shard-lane-non-group";
    temporary_trace_file trace_file(scenario);
    attn_out_shard_fixture fixture;
    if (!check(fixture.build(2), scenario, "fixture setup failed")) {
        return false;
    }
    const int n_splits = ggml_backend_sched_get_n_splits(fixture.sched);
    if (!check(n_splits == 4, scenario,
                "two W_O lanes were not isolated from each other and the ADD join")) {
        std::fprintf(stderr, "  got %d splits, expected 4\n", n_splits);
        return false;
    }

    {
        scheduler_trace_capture capture(trace_file.path(), 106);
        if (!fixture.compute_and_check(scenario)) {
            return false;
        }
        capture.finish();
    }

    std::vector<trace_row> rows;
    if (!read_trace(trace_file.path(), scenario, &rows)) {
        return false;
    }
    for (const trace_row & row : rows) {
        if (!check(!row.is_parallel_group, scenario, "incomplete W_O lane set was collected as a group")) {
            return false;
        }
    }

    int lane_split_ids[2] = { -1, -1 };
    for (int lane = 0; lane < 2; ++lane) {
        const std::string expected_lane = "lane" + std::to_string(lane);
        const std::string expected_first =
            "attn_out_shard." + expected_lane + ".input-0";
        const std::string expected_last =
            "attn_out_shard." + expected_lane + ".proj-0";
        const int expected_node_count = 2;
        const auto it = std::find_if(rows.begin(), rows.end(), [&](const trace_row & row) {
            return row.first_node == expected_first && row.last_node == expected_last;
        });
        if (!check(it != rows.end(), scenario, "isolated W_O lane row was not traced") ||
                !check(it->node_count == expected_node_count, scenario,
                    "incomplete W_O executable lane split was not pure")) {
            return false;
        }
        lane_split_ids[lane] = it->split_id;
    }
    return check(
        lane_split_ids[0] != lane_split_ids[1],
        scenario,
        "incomplete W_O lanes shared a split");
}

static bool run_exact_attn_out_output_axis_shard_lane_group_case() {
    const char * scenario = "exact-attn-out-output-axis-shard-lane-group";
    temporary_trace_file trace_file(scenario);
    attn_out_output_axis_shard_fixture fixture;
    if (!check(fixture.build(), scenario, "fixture setup failed")) {
        return false;
    }
    const int n_splits = ggml_backend_sched_get_n_splits(fixture.sched);
    if (!check(n_splits == 4, scenario,
                "three projection-only W_O lanes were not isolated from the CONCAT join")) {
        std::fprintf(stderr, "  got %d splits, expected 4\n", n_splits);
        return false;
    }

    {
        scheduler_trace_capture capture(trace_file.path(), 107);
        if (!fixture.compute_and_check(scenario)) {
            return false;
        }
        capture.finish();
    }

    std::vector<trace_row> rows;
    if (!read_trace(trace_file.path(), scenario, &rows)) {
        return false;
    }

    std::vector<trace_row> grouped;
    for (const trace_row & row : rows) {
        if (row.is_parallel_group) {
            grouped.push_back(row);
        }
    }
    if (!check(grouped.size() == 3, scenario,
                "expected exactly three grouped output-axis W_O lane rows")) {
        return false;
    }

    static const char * lane_names[] = { "cpu", "npu", "gpu" };
    const int group_id = grouped[0].group_id;
    for (int lane = 0; lane < 3; ++lane) {
        const trace_row & row = grouped[lane];
        const std::string expected_node =
            "attn_out_shard." + std::string(lane_names[lane]) + ".proj-0";
        if (!check(row.parallel_kind == "attn_out_shard", scenario,
                    "parallel group kind mismatch") ||
                !check(row.parallel_branch == lane_names[lane], scenario,
                    "parallel lane order mismatch") ||
                !check(row.timing_mode == "serial_fallback", scenario,
                    "single-backend lanes were not serialized") ||
                !check(row.group_id == group_id && group_id >= 0, scenario,
                    "parallel group id mismatch") ||
                !check(row.node_count == 1, scenario,
                    "output-axis W_O lane was not projection-only") ||
                !check(row.first_node == expected_node, scenario,
                    "lane split first node mismatch") ||
                !check(row.last_node == expected_node, scenario,
                    "lane split last node mismatch")) {
            return false;
        }
    }

    const auto join = std::find_if(rows.begin(), rows.end(), [](const trace_row & row) {
        return row.first_node == "attn_out_parallel_concat-1" &&
            row.last_node == "attn_out_parallel_concat-0";
    });
    if (!check(join != rows.end(), scenario, "isolated CONCAT join split was not traced") ||
            !check(!join->is_parallel_group, scenario, "CONCAT join was included in the lane group") ||
            !check(join->node_count == 2, scenario, "CONCAT tree did not remain in one join split") ||
            !check(join->split_id != grouped.back().split_id, scenario,
                "last gpu lane fused with the CONCAT join")) {
        return false;
    }
    return true;
}

static bool run_attn_out_output_axis_deferred_sync_case() {
    const char * scenario = "attn-out-output-axis-deferred-sync";
    temporary_trace_file trace_file(scenario);
    attn_out_deferred_sync_fixture fixture;
    if (!check(fixture.build(), scenario, "fixture setup failed")) {
        return false;
    }

    const int n_splits = ggml_backend_sched_get_n_splits(fixture.sched);
    if (!check(n_splits == 5, scenario,
                "expected CPU/NPU/GPU lanes, OpenCL CONCAT, and CPU consumer splits")) {
        std::fprintf(stderr, "  got %d splits, expected 5\n", n_splits);
        return false;
    }
    {
        scheduler_trace_capture capture(trace_file.path(), 108);
        if (!fixture.compute_and_check(scenario)) {
            return false;
        }
        capture.finish();
    }
    if (!check_attn_out_group_timing_mode(
                trace_file.path(), scenario, "persistent_deferred_sync")) {
        return false;
    }

    const deferred_sync_control & control = fixture.control;
    const int branch_async = control.gpu_branch_async_calls.load(std::memory_order_relaxed);
    const int join_async = control.gpu_join_async_calls.load(std::memory_order_relaxed);
    const int async_flush = control.gpu_async_flush_calls.load(std::memory_order_relaxed);
    const int early_branch_sync =
        control.gpu_branch_sync_before_cpu_complete.load(std::memory_order_relaxed);
    const int joined_branch_sync =
        control.gpu_branch_sync_after_cpu_complete.load(std::memory_order_relaxed);
    const int join_pending_sync =
        control.gpu_join_pending_syncs.load(std::memory_order_relaxed);
    const int pending_syncs = control.gpu_pending_syncs.load(std::memory_order_relaxed);
    const int join_while_pending =
        control.gpu_join_submitted_while_branch_pending.load(std::memory_order_relaxed);
    const int cpu_wait_timeouts = control.cpu_wait_timeouts.load(std::memory_order_relaxed);

    if (branch_async != 1 || join_async != 1 || async_flush != 1 || early_branch_sync != 0 ||
            joined_branch_sync != 1 || join_pending_sync != 1 || pending_syncs != 2 ||
            join_while_pending != 0 || cpu_wait_timeouts != 0) {
        std::fprintf(
                stderr,
                "  async(branch/join/flush)=%d/%d/%d sync(early/at-join/join/pending)=%d/%d/%d/%d "
                "join-while-pending=%d cpu-wait-timeouts=%d\n",
                branch_async,
                join_async,
                async_flush,
                early_branch_sync,
                joined_branch_sync,
                join_pending_sync,
                pending_syncs,
                join_while_pending,
                cpu_wait_timeouts);
    }

    return check(branch_async == 1, scenario, "GPU branch was not submitted asynchronously") &&
        check(join_async == 1, scenario, "OpenCL CONCAT was not submitted asynchronously") &&
        check(async_flush == 1, scenario,
            "GPU branch did not flush its deferred OpenCL command queue") &&
        check(early_branch_sync == 0, scenario,
            "GPU branch performed an immediate synchronize inside the parallel worker") &&
        check(joined_branch_sync == 1, scenario,
            "GPU branch was not synchronized at the CONCAT boundary") &&
        check(join_pending_sync == 1, scenario,
            "OpenCL CONCAT was not synchronized before its CPU consumer") &&
        check(pending_syncs == 2, scenario,
            "unexpected number of pending OpenCL graph synchronizations") &&
        check(join_while_pending == 0, scenario,
            "OpenCL CONCAT was submitted before the pending GPU branch completed") &&
        check(cpu_wait_timeouts == 0, scenario,
            "CPU branch did not observe GPU async submission");
}

static bool run_attn_out_deferred_sync_backend_guard_case() {
    const char * scenario = "attn-out-deferred-sync-backend-guard";
    temporary_trace_file trace_file(scenario);
    attn_out_deferred_sync_fixture fixture;
    if (!check(fixture.build(false), scenario, "fixture setup failed")) {
        return false;
    }

    const int n_splits = ggml_backend_sched_get_n_splits(fixture.sched);
    if (!check(n_splits == 5, scenario,
                "expected CPU/NPU/GPU lanes, NPU CONCAT, and CPU consumer splits")) {
        std::fprintf(stderr, "  got %d splits, expected 5\n", n_splits);
        return false;
    }
    {
        scheduler_trace_capture capture(trace_file.path(), 109);
        if (!fixture.compute_and_check(scenario)) {
            return false;
        }
        capture.finish();
    }
    if (!check_attn_out_group_timing_mode(
                trace_file.path(), scenario, "persistent_synced_wall")) {
        return false;
    }

    const deferred_sync_control & control = fixture.control;
    const int branch_async = control.gpu_branch_async_calls.load(std::memory_order_relaxed);
    const int join_async = control.gpu_join_async_calls.load(std::memory_order_relaxed);
    const int async_flush = control.gpu_async_flush_calls.load(std::memory_order_relaxed);
    const int early_branch_sync =
        control.gpu_branch_sync_before_cpu_complete.load(std::memory_order_relaxed);
    const int joined_branch_sync =
        control.gpu_branch_sync_after_cpu_complete.load(std::memory_order_relaxed);
    const int join_pending_sync =
        control.gpu_join_pending_syncs.load(std::memory_order_relaxed);
    const int pending_syncs = control.gpu_pending_syncs.load(std::memory_order_relaxed);
    const int cpu_wait_timeouts = control.cpu_wait_timeouts.load(std::memory_order_relaxed);

    if (branch_async != 1 || join_async != 0 || async_flush != 0 ||
            early_branch_sync != 1 || joined_branch_sync != 0 ||
            join_pending_sync != 0 || pending_syncs != 1 || cpu_wait_timeouts != 0) {
        std::fprintf(
                stderr,
                "  async(branch/join/flush)=%d/%d/%d "
                "sync(early/at-join/join/pending)=%d/%d/%d/%d cpu-wait-timeouts=%d\n",
                branch_async,
                join_async,
                async_flush,
                early_branch_sync,
                joined_branch_sync,
                join_pending_sync,
                pending_syncs,
                cpu_wait_timeouts);
    }

    return check(branch_async == 1, scenario, "GPU branch was not submitted asynchronously") &&
        check(join_async == 0, scenario, "NPU CONCAT unexpectedly ran on the fake OpenCL backend") &&
        check(async_flush == 0, scenario,
            "deferred OpenCL flush escaped the same-backend CONCAT guard") &&
        check(early_branch_sync == 1, scenario,
            "mismatched CONCAT backend did not preserve legacy immediate synchronize") &&
        check(joined_branch_sync == 0, scenario,
            "mismatched CONCAT backend deferred the GPU branch to the join") &&
        check(join_pending_sync == 0, scenario,
            "fake OpenCL backend unexpectedly received the NPU CONCAT") &&
        check(pending_syncs == 1, scenario,
            "unexpected number of pending OpenCL graph synchronizations") &&
        check(cpu_wait_timeouts == 0, scenario,
            "CPU branch did not observe GPU async submission");
}

static void force_trace_all_phases() {
#if defined(_WIN32)
    (void) _putenv_s("SCHED_TRACE_PHASE", "both");
#else
    (void) setenv("SCHED_TRACE_PHASE", "both", 1);
#endif
}

static void force_persistent_device_workers() {
#if defined(_WIN32)
    (void) _putenv_s("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    (void) _putenv_s("GGML_FFN_GPU_WORKER_CPU", "-1");
    (void) _putenv_s("GGML_FFN_NPU_WORKER_CPU", "-1");
    (void) _putenv_s("GGML_ATTN_OUT_DEFER_OPENCL_SYNC", "1");
#else
    (void) setenv("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1", 1);
    (void) setenv("GGML_FFN_GPU_WORKER_CPU", "-1", 1);
    (void) setenv("GGML_FFN_NPU_WORKER_CPU", "-1", 1);
    (void) setenv("GGML_ATTN_OUT_DEFER_OPENCL_SYNC", "1", 1);
#endif
}

} // namespace

int main() {
    force_trace_all_phases();
    force_persistent_device_workers();
    ggml_backend_sched_profile_set_phase(GGML_BACKEND_SCHED_PROFILE_PREFILL);

    try {
        const bool ok = run_exact_qvk_group_case() && run_incomplete_qv_case() &&
            run_exact_qkv_shard_lane_group_case() && run_incomplete_qkv_shard_lane_case() &&
            run_exact_attn_out_shard_lane_group_case() && run_incomplete_attn_out_shard_lane_case() &&
            run_exact_attn_out_output_axis_shard_lane_group_case() &&
            run_attn_out_output_axis_deferred_sync_case() &&
            run_attn_out_deferred_sync_backend_guard_case();
        if (ok) {
            std::puts("scheduler parallel-group tests passed");
            return 0;
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "scheduler parallel-group test failed: %s\n", error.what());
    }
    return 1;
}
