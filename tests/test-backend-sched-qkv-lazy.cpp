#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
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

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    (void) _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        (void) setenv(name, value, 1);
    } else {
        (void) unsetenv(name);
    }
#endif
}

enum class fake_role {
    cpu,
    npu,
    gpu,
};

struct fake_control {
    std::atomic<int> flush_calls { 0 };
    std::atomic<int> async_uploads { 0 };
    std::atomic<int> event_records { 0 };
    std::atomic<int> event_synchronizes { 0 };
    std::atomic<int> pending_queue_synchronizes { 0 };
    std::atomic<int> branch_synchronizes_before_consumer { 0 };
    std::atomic<int> consumers_submitted_while_branch_pending { 0 };
};

struct fake_command {
    uint64_t sequence = 0;
    std::function<void()> run;
};

struct fake_backend_context {
    ggml_backend_t inner = nullptr;
    ggml_backend_t wrapper = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_buffer_type_t inner_buft = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    fake_role role = fake_role::cpu;
    enum ggml_backend_dev_type device_type = GGML_BACKEND_DEVICE_TYPE_CPU;
    const char * name = nullptr;
    fake_control * control = nullptr;
    std::mutex queue_mutex;
    std::deque<fake_command> queue;
    uint64_t next_sequence = 1;
    bool device_branch_pending = false;
    bool consumer_queued_behind_branch = false;
};

struct fake_event_context {
    fake_backend_context * backend = nullptr;
    uint64_t sequence = 0;
    bool recorded = false;
    std::mutex mutex;
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

static bool graph_has_prefix(const ggml_cgraph * graph, const char * prefix) {
    const size_t length = std::strlen(prefix);
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node != nullptr && std::strncmp(node->name, prefix, length) == 0) {
            return true;
        }
    }
    return false;
}

static uint64_t enqueue(fake_backend_context * ctx, std::function<void()> command) {
    std::lock_guard<std::mutex> lock(ctx->queue_mutex);
    const uint64_t sequence = ctx->next_sequence++;
    ctx->queue.push_back({ sequence, std::move(command) });
    return sequence;
}

static void run_through(fake_backend_context * ctx, uint64_t sequence) {
    for (;;) {
        std::function<void()> command;
        {
            std::lock_guard<std::mutex> lock(ctx->queue_mutex);
            if (ctx->queue.empty() || ctx->queue.front().sequence > sequence) {
                break;
            }
            command = std::move(ctx->queue.front().run);
            ctx->queue.pop_front();
        }
        command();
    }
}

static void run_all(fake_backend_context * ctx) {
    run_through(ctx, UINT64_MAX);
}

static const char * fake_buft_name(ggml_backend_buffer_type_t buft) {
    return buft_ctx(buft)->name;
}

static ggml_backend_buffer_t fake_buft_alloc(
        ggml_backend_buffer_type_t buft, size_t size) {
    fake_backend_context * ctx = buft_ctx(buft);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ctx->inner_buft, size);
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

static size_t fake_buft_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(buft_ctx(buft)->inner_buft, tensor);
}

static bool fake_buft_is_host(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft_ctx(buft)->inner_buft);
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
    if (ctx->role != fake_role::gpu) {
        ggml_backend_synchronize(ctx->inner);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(ctx->queue_mutex);
        if (!ctx->queue.empty()) {
            ctx->control->pending_queue_synchronizes.fetch_add(
                    1, std::memory_order_relaxed);
        }
        if (ctx->device_branch_pending && !ctx->consumer_queued_behind_branch) {
            ctx->control->branch_synchronizes_before_consumer.fetch_add(
                    1, std::memory_order_relaxed);
        }
    }
    run_all(ctx);
}

static enum ggml_status fake_backend_graph_compute(
        ggml_backend_t backend, ggml_cgraph * graph) {
    fake_backend_context * ctx = backend_ctx(backend);
    if (ctx->role != fake_role::gpu) {
        return ggml_backend_graph_compute(ctx->inner, graph);
    }

    const bool branch = graph_has_prefix(graph, "attn_qkv.") ||
        graph_has_prefix(graph, "attn_qkv_shard.") ||
        graph_has_prefix(graph, "ffn_down.");
    const bool consumer = graph_has_prefix(graph, "qkv-lazy-consumer-") ||
        graph_has_prefix(graph, "ffn_parallel_add4-");
    {
        std::lock_guard<std::mutex> lock(ctx->queue_mutex);
        if (consumer && ctx->device_branch_pending) {
            ctx->control->consumers_submitted_while_branch_pending.fetch_add(
                    1, std::memory_order_relaxed);
            ctx->consumer_queued_behind_branch = true;
        }
        if (branch) {
            ctx->device_branch_pending = true;
            ctx->consumer_queued_behind_branch = false;
        }
    }
    enqueue(ctx, [ctx, graph, branch, consumer]() {
        (void) ggml_backend_graph_compute(ctx->inner, graph);
        if (branch || consumer) {
            std::lock_guard<std::mutex> lock(ctx->queue_mutex);
            if (branch) {
                ctx->device_branch_pending = false;
            }
            if (consumer) {
                ctx->consumer_queued_behind_branch = false;
            }
        }
    });
    return GGML_STATUS_SUCCESS;
}

static void fake_backend_set_tensor_async(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    fake_backend_context * ctx = backend_ctx(backend);
    if (ctx->role != fake_role::gpu) {
        ggml_backend_synchronize(ctx->inner);
        ggml_backend_tensor_set(tensor, data, offset, size);
        return;
    }
    ctx->control->async_uploads.fetch_add(1, std::memory_order_relaxed);
    enqueue(ctx, [tensor, data, offset, size]() {
        // Deliberately consume the pointer later. Reusing a staging slot before
        // its event completes therefore produces a deterministic bad result.
        ggml_backend_tensor_set(tensor, data, offset, size);
    });
}

static bool fake_backend_copy_tensor_async(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src,
        ggml_tensor * dst) {
    if (backend_src != backend_dst || backend_ctx(backend_dst)->role != fake_role::gpu) {
        return false;
    }
    fake_backend_context * ctx = backend_ctx(backend_dst);
    enqueue(ctx, [src, dst]() { ggml_backend_tensor_copy(src, dst); });
    return true;
}

static void fake_backend_event_record(
        ggml_backend_t backend, ggml_backend_event_t event) {
    fake_backend_context * ctx = backend_ctx(backend);
    auto * event_ctx = static_cast<fake_event_context *>(event->context);
    const uint64_t sequence = enqueue(ctx, []() {});
    {
        std::lock_guard<std::mutex> lock(event_ctx->mutex);
        event_ctx->backend = ctx;
        event_ctx->sequence = sequence;
        event_ctx->recorded = true;
    }
    ctx->control->event_records.fetch_add(1, std::memory_order_relaxed);
}

static void fake_backend_event_wait(
        ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    auto * event_ctx = static_cast<fake_event_context *>(event->context);
    fake_backend_context * event_backend = nullptr;
    uint64_t sequence = 0;
    {
        std::lock_guard<std::mutex> lock(event_ctx->mutex);
        if (!event_ctx->recorded) {
            return;
        }
        event_backend = event_ctx->backend;
        sequence = event_ctx->sequence;
    }
    run_through(event_backend, sequence);
}

static const ggml_backend_i fake_backend_iface = {
    /* .get_name                = */ fake_backend_name,
    /* .free                    = */ nullptr,
    /* .set_tensor_async        = */ fake_backend_set_tensor_async,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ fake_backend_copy_tensor_async,
    /* .synchronize             = */ fake_backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ fake_backend_graph_compute,
    /* .event_record            = */ fake_backend_event_record,
    /* .event_wait              = */ fake_backend_event_wait,
    /* .graph_optimize          = */ nullptr,
};

static const char * fake_device_name(ggml_backend_dev_t device) {
    return device_ctx(device)->name;
}

static const char * fake_device_description(ggml_backend_dev_t device) {
    return device_ctx(device)->name;
}

static void fake_device_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    ggml_backend_dev_memory(ggml_backend_get_device(device_ctx(device)->inner), free, total);
}

static enum ggml_backend_dev_type fake_device_type(ggml_backend_dev_t device) {
    return device_ctx(device)->device_type;
}

static void fake_device_props(
        ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    fake_backend_context * ctx = device_ctx(device);
    ggml_backend_dev_get_props(ggml_backend_get_device(ctx->inner), props);
    props->name = ctx->name;
    props->description = ctx->name;
    props->type = ctx->device_type;
    // The QKV path must use its targeted proc rather than changing global
    // scheduler capabilities or host-buffer selection.
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

static bool fake_device_supports_op(
        ggml_backend_dev_t device, const ggml_tensor * op) {
    fake_backend_context * ctx = device_ctx(device);
    return ggml_backend_dev_supports_op(ggml_backend_get_device(ctx->inner), op);
}

static bool fake_device_supports_buft(
        ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return buft == device_ctx(device)->buft;
}

static bool fake_device_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static ggml_backend_event_t fake_device_event_new(ggml_backend_dev_t device) {
    if (device_ctx(device)->role != fake_role::gpu) {
        return nullptr;
    }
    return new ggml_backend_event {
        /* .device  = */ device,
        /* .context = */ new fake_event_context,
    };
}

static void fake_device_event_free(
        ggml_backend_dev_t, ggml_backend_event_t event) {
    delete static_cast<fake_event_context *>(event->context);
    delete event;
}

static void fake_device_event_synchronize(
        ggml_backend_dev_t, ggml_backend_event_t event) {
    auto * event_ctx = static_cast<fake_event_context *>(event->context);
    fake_backend_context * backend = nullptr;
    uint64_t sequence = 0;
    {
        std::lock_guard<std::mutex> lock(event_ctx->mutex);
        if (!event_ctx->recorded) {
            return;
        }
        backend = event_ctx->backend;
        sequence = event_ctx->sequence;
    }
    backend->control->event_synchronizes.fetch_add(1, std::memory_order_relaxed);
    run_through(backend, sequence);
}

static const ggml_backend_device_i fake_device_iface = {
    /* .get_name             = */ fake_device_name,
    /* .get_description      = */ fake_device_description,
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
    /* .event_new            = */ fake_device_event_new,
    /* .event_free           = */ fake_device_event_free,
    /* .event_synchronize    = */ fake_device_event_synchronize,
};

static void fake_async_flush(ggml_backend_t backend) {
    backend_ctx(backend)->control->flush_calls.fetch_add(1, std::memory_order_relaxed);
}

static ggml_backend_buffer_type_t fake_async_staging_buffer_type(
        ggml_backend_t backend) {
    fake_backend_context * ctx = backend_ctx(backend);
    const char * enabled = std::getenv("GGML_OPENCL_ASYNC_EVENTS");
    return ctx->role == fake_role::gpu && enabled != nullptr && std::atoi(enabled) != 0
        ? ctx->inner_buft
        : nullptr;
}

static const char * fake_registry_name(ggml_backend_reg_t) {
    return "QKVLazyFakeOpenCL";
}

static size_t fake_registry_device_count(ggml_backend_reg_t) {
    return 0;
}

static ggml_backend_dev_t fake_registry_device(ggml_backend_reg_t, size_t) {
    return nullptr;
}

static void * fake_registry_proc(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_async_flush") == 0) {
        return reinterpret_cast<void *>(fake_async_flush);
    }
    if (std::strcmp(name, "ggml_backend_async_staging_buffer_type") == 0) {
        return reinterpret_cast<void *>(fake_async_staging_buffer_type);
    }
    return nullptr;
}

static const ggml_backend_reg_i fake_registry_iface = {
    /* .get_name         = */ fake_registry_name,
    /* .get_device_count = */ fake_registry_device_count,
    /* .get_device       = */ fake_registry_device,
    /* .get_proc_address = */ fake_registry_proc,
};

static ggml_backend_reg fake_registry = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ fake_registry_iface,
    /* .context     = */ nullptr,
};

static ggml_guid_t fake_backend_guid() {
    static ggml_guid guid = {
        0x61, 0x51, 0xf3, 0xe8, 0x42, 0x89, 0x47, 0x1d,
        0xa0, 0xef, 0x2a, 0x5e, 0xe2, 0x79, 0x11, 0x05,
    };
    return &guid;
}

static ggml_backend_t make_fake_backend(
        fake_role role,
        const char * name,
        enum ggml_backend_dev_type device_type,
        fake_control * control) {
    auto * ctx = new fake_backend_context;
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
        /* .iface   = */ fake_device_iface,
        /* .reg     = */ role == fake_role::gpu ? &fake_registry : nullptr,
        /* .context = */ ctx,
    };
    ctx->buft = new ggml_backend_buffer_type {
        /* .iface   = */ fake_buft_iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
    ggml_backend_i iface = fake_backend_iface;
    iface.free = [](ggml_backend_t backend) {
        fake_backend_context * ctx = backend_ctx(backend);
        fake_backend_synchronize(backend);
        ggml_backend_free(ctx->inner);
        delete ctx->buft;
        delete ctx->device;
        delete ctx;
        delete backend;
    };
    ctx->wrapper = new ggml_backend {
        /* .guid    = */ fake_backend_guid(),
        /* .iface   = */ iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
    return ctx->wrapper;
}

struct qkv_lazy_fixture {
    fake_control control;
    ggml_context * ctx = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t npu = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    float expected_scale = 0.0f;

    ~qkv_lazy_fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(gpu);
        ggml_backend_free(npu);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }

    bool build(
            const std::vector<std::string> & projection_order,
            const std::vector<fake_role> & projection_roles,
            int layers,
            fake_role consumer_role,
            bool register_canonical_route = false,
            bool register_alternate_route = false) {
        constexpr size_t graph_size = 128;
        const ggml_init_params params = {
            /* .mem_size   = */ 256 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        cpu = make_fake_backend(
                fake_role::cpu, "QKVLazyCPU", GGML_BACKEND_DEVICE_TYPE_CPU, &control);
        npu = make_fake_backend(
                fake_role::npu, "QKVLazyHTP", GGML_BACKEND_DEVICE_TYPE_ACCEL, &control);
        gpu = make_fake_backend(
                fake_role::gpu, "QKVLazyOpenCL", GGML_BACKEND_DEVICE_TYPE_GPU, &control);
        if (ctx == nullptr || cpu == nullptr || npu == nullptr || gpu == nullptr ||
                projection_order.size() != 3 || projection_roles.size() != 3 ||
                (register_canonical_route && register_alternate_route)) {
            return false;
        }
        ggml_backend_t backends[] = { gpu, npu, cpu };
        ggml_backend_buffer_type_t bufts[] = {
            ggml_backend_get_default_buffer_type(gpu),
            ggml_backend_get_default_buffer_type(npu),
            ggml_backend_get_default_buffer_type(cpu),
        };
        sched = ggml_backend_sched_new(backends, bufts, 3, 256, false, true);
        if (sched == nullptr) {
            return false;
        }

        auto role_backend = [&](fake_role role) {
            return role == fake_role::gpu ? gpu : role == fake_role::npu ? npu : cpu;
        };
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_set_input(input);
        ggml_set_name(input, "qkv-lazy-input");
        ggml_backend_sched_set_tensor_backend(sched, input, cpu);
        graph = ggml_new_graph_custom(ctx, graph_size, false);

        std::vector<ggml_tensor *> layer_outputs;
        float next_scale = 1.0f;
        for (int layer = 0; layer < layers; ++layer) {
            std::vector<ggml_tensor *> branches;
            std::vector<ggml_tensor *> alternate_branches;
            auto append_branches = [&](
                    std::vector<ggml_tensor *> * destination,
                    bool selected_outputs) {
                for (size_t i = 0; i < projection_order.size(); ++i) {
                    ggml_tensor * branch = ggml_scale(ctx, input, next_scale);
                    ggml_set_name(branch, (
                            "attn_qkv." + projection_order[i] + ".proj-" +
                            std::to_string(layer)).c_str());
                    ggml_backend_sched_set_tensor_backend(
                            sched, branch, role_backend(projection_roles[i]));
                    ggml_build_forward_expand(graph, branch);
                    destination->push_back(branch);
                    if (selected_outputs) {
                        expected_scale += next_scale;
                    }
                    next_scale += 1.0f;
                }
            };
            append_branches(&branches, !register_alternate_route);
            if (register_alternate_route) {
                append_branches(&alternate_branches, true);
            }
            ggml_tensor * joined_pair = ggml_add(ctx, branches[0], branches[1]);
            ggml_set_name(joined_pair, (
                    "qkv-lazy-consumer-" + std::to_string(layer) + "-pair").c_str());
            ggml_backend_sched_set_tensor_backend(
                    sched, joined_pair, role_backend(consumer_role));
            ggml_tensor * joined = ggml_add(ctx, joined_pair, branches[2]);
            ggml_set_name(joined, ("qkv-lazy-consumer-" + std::to_string(layer)).c_str());
            ggml_backend_sched_set_tensor_backend(sched, joined, role_backend(consumer_role));
            ggml_build_forward_expand(graph, joined);
            if ((register_canonical_route || register_alternate_route) &&
                    !ggml_backend_sched_register_route_subgraph_bundle_range(
                            sched,
                            branches.front(),
                            branches.back(),
                            branches.data(),
                            register_alternate_route
                                ? alternate_branches.front()
                                : branches.front(),
                            register_alternate_route
                                ? alternate_branches.back()
                                : branches.back(),
                            register_alternate_route
                                ? alternate_branches.data()
                                : branches.data(),
                            (int) branches.size(),
                            register_alternate_route ? 1 : 0,
                            layer)) {
                return false;
            }
            layer_outputs.push_back(joined);
        }

        output = layer_outputs[0];
        for (size_t i = 1; i < layer_outputs.size(); ++i) {
            output = ggml_add(ctx, output, layer_outputs[i]);
            ggml_backend_sched_set_tensor_backend(sched, output, role_backend(consumer_role));
        }
        ggml_set_name(output, "qkv-lazy-output");
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        if (register_alternate_route) {
            ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 1);
        }
        return ggml_backend_sched_alloc_graph(sched, graph);
    }

    bool compute_and_check(const char * scenario) {
        float values[16];
        for (int i = 0; i < 16; ++i) {
            values[i] = (float) i * 0.25f - 1.5f;
        }
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario,
                    "graph compute failed")) {
            return false;
        }
        float result[16] = {};
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 16; ++i) {
            const float expected = values[i] * expected_scale;
            if (!check(
                        std::fabs(result[i] - expected) < 1e-5f,
                        scenario,
                        "output mismatch")) {
                std::fprintf(stderr, "  index %d got %.7g expected %.7g\n",
                        i, result[i], expected);
                return false;
            }
        }
        return true;
    }
};

struct ffn_add4_lazy_fixture {
    fake_control control;
    ggml_context * ctx = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t npu = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    float expected_scale = 0.0f;

    ~ffn_add4_lazy_fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(gpu);
        ggml_backend_free(npu);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }

    bool build(int layers) {
        constexpr size_t graph_size = 128;
        const ggml_init_params params = {
            /* .mem_size   = */ 256 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        cpu = make_fake_backend(
                fake_role::cpu, "ADD4LazyCPU", GGML_BACKEND_DEVICE_TYPE_CPU, &control);
        npu = make_fake_backend(
                fake_role::npu, "ADD4LazyHTP", GGML_BACKEND_DEVICE_TYPE_ACCEL, &control);
        gpu = make_fake_backend(
                fake_role::gpu, "ADD4LazyOpenCL", GGML_BACKEND_DEVICE_TYPE_GPU, &control);
        if (ctx == nullptr || cpu == nullptr || npu == nullptr || gpu == nullptr ||
                layers <= 0) {
            return false;
        }

        ggml_backend_t backends[] = { gpu, npu, cpu };
        ggml_backend_buffer_type_t bufts[] = {
            ggml_backend_get_default_buffer_type(gpu),
            ggml_backend_get_default_buffer_type(npu),
            ggml_backend_get_default_buffer_type(cpu),
        };
        sched = ggml_backend_sched_new(backends, bufts, 3, 256, false, true);
        if (sched == nullptr) {
            return false;
        }

        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_set_input(input);
        ggml_set_name(input, "ffn-add4-lazy-input");
        ggml_backend_sched_set_tensor_backend(sched, input, gpu);
        graph = ggml_new_graph_custom(ctx, graph_size, false);

        ggml_backend_t branch_backends[] = { cpu, gpu, npu };
        static const char * branch_names[] = { "cpu", "gpu", "npu" };
        std::vector<ggml_tensor *> layer_outputs;
        float next_scale = 1.0f;
        for (int layer = 0; layer < layers; ++layer) {
            ggml_tensor * branches[3] = {};
            float layer_scale = 1.0f; // residual source
            for (int lane = 0; lane < 3; ++lane) {
                branches[lane] = ggml_scale(ctx, input, next_scale);
                ggml_set_name(
                        branches[lane],
                        (std::string("ffn_down.") + branch_names[lane] + "-" +
                         std::to_string(layer)).c_str());
                ggml_backend_sched_set_tensor_backend(
                        sched, branches[lane], branch_backends[lane]);
                ggml_build_forward_expand(graph, branches[lane]);
                layer_scale += next_scale;
                next_scale += 1.0f;
            }
            ggml_tensor * fused = ggml_add4(
                    ctx, branches[0], branches[1], branches[2], input);
            ggml_set_name(
                    fused, ("ffn_parallel_add4-" + std::to_string(layer)).c_str());
            ggml_backend_sched_set_tensor_backend(sched, fused, gpu);
            ggml_build_forward_expand(graph, fused);
            layer_outputs.push_back(fused);
            expected_scale += layer_scale;
        }

        output = layer_outputs[0];
        for (size_t layer = 1; layer < layer_outputs.size(); ++layer) {
            output = ggml_add(ctx, output, layer_outputs[layer]);
            ggml_backend_sched_set_tensor_backend(sched, output, gpu);
        }
        ggml_set_name(output, "ffn-add4-lazy-output");
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        return ggml_backend_sched_alloc_graph(sched, graph);
    }

    bool compute_and_check(const char * scenario) {
        float values[16];
        for (int i = 0; i < 16; ++i) {
            values[i] = (float) i * 0.375f - 2.0f;
        }
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario,
                    "graph compute failed")) {
            return false;
        }
        float result[16] = {};
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 16; ++i) {
            const float expected = values[i] * expected_scale;
            if (!check(
                        std::fabs(result[i] - expected) < 1e-5f,
                        scenario,
                        "output mismatch")) {
                std::fprintf(stderr, "  index %d got %.7g expected %.7g\n",
                        i, result[i], expected);
                return false;
            }
        }
        return true;
    }
};

static bool run_lazy_case(
        const char * scenario,
        const std::vector<std::string> & order,
        const std::vector<fake_role> & roles) {
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", "1");
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    qkv_lazy_fixture fixture;
    if (!check(fixture.build(order, roles, 2, fake_role::gpu), scenario, "fixture build failed") ||
            !fixture.compute_and_check(scenario)) {
        return false;
    }
    return check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 4,
                scenario,
                "expected two remote staged uploads per layer") &&
        check(
                fixture.control.event_records.load(std::memory_order_relaxed) == 4,
                scenario,
                "each staging upload must have a completion event") &&
        check(
                fixture.control.event_synchronizes.load(std::memory_order_relaxed) >= 2,
                scenario,
                "staging slots were reused without waiting for their events") &&
        check(
                fixture.control.consumers_submitted_while_branch_pending.load(
                        std::memory_order_relaxed) >= 2,
                scenario,
                "consumer was not queued behind the still-pending GPU branch") &&
        check(
                fixture.control.flush_calls.load(std::memory_order_relaxed) >= 2,
                scenario,
                "deferred GPU branch was not flushed");
}

static bool run_default_off_case() {
    const char * scenario = "QKV lazy default-off equivalence";
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", nullptr);
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    qkv_lazy_fixture fixture;
    return check(
                fixture.build(
                        { "k", "v", "q" },
                        { fake_role::cpu, fake_role::gpu, fake_role::npu },
                        1,
                        fake_role::gpu),
                scenario,
                "fixture build failed") &&
        fixture.compute_and_check(scenario) &&
        check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 0,
                scenario,
                "default-off path unexpectedly used async staging") &&
        check(
                fixture.control.consumers_submitted_while_branch_pending.load(
                        std::memory_order_relaxed) == 0,
                scenario,
                "default-off path skipped the legacy branch synchronization");
}

static bool run_incompatible_consumer_fallback_case() {
    const char * scenario = "QKV lazy incompatible-consumer fallback";
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", "1");
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    qkv_lazy_fixture fixture;
    return check(
                fixture.build(
                        { "q", "k", "v" },
                        { fake_role::npu, fake_role::gpu, fake_role::cpu },
                        1,
                        fake_role::cpu),
                scenario,
                "fixture build failed") &&
        fixture.compute_and_check(scenario) &&
        check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 0,
                scenario,
                "incompatible consumer unexpectedly used lazy staging");
}

static bool always_continue_eval_callback(
        ggml_tensor *, bool, void *) {
    return true;
}

static bool run_callback_fallback_case() {
    const char * scenario = "QKV lazy callback fallback";
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", "1");
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    qkv_lazy_fixture fixture;
    if (!check(
                fixture.build(
                        { "k", "v", "q" },
                        { fake_role::cpu, fake_role::gpu, fake_role::npu },
                        1,
                        fake_role::gpu),
                scenario,
                "fixture build failed")) {
        return false;
    }
    ggml_backend_sched_set_eval_callback(
            fixture.sched, always_continue_eval_callback, nullptr);
    return fixture.compute_and_check(scenario) &&
        check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 0,
                scenario,
                "callback path unexpectedly used lazy staging");
}

static bool run_canonical_route_case() {
    const char * scenario = "QKV lazy canonical-route dedup";
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", "1");
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    qkv_lazy_fixture fixture;
    return check(
                fixture.build(
                        { "k", "v", "q" },
                        { fake_role::cpu, fake_role::gpu, fake_role::npu },
                        1,
                        fake_role::gpu,
                        true),
                scenario,
                "fixture build failed") &&
        fixture.compute_and_check(scenario) &&
        check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 2,
                scenario,
                "canonical no-commit route was incorrectly forced to legacy sync") &&
        check(
                fixture.control.consumers_submitted_while_branch_pending.load(
                        std::memory_order_relaxed) == 1,
                scenario,
                "canonical route did not preserve the lazy same-queue consumer");
}

static bool run_alternate_route_fallback_case() {
    const char * scenario = "QKV lazy alternate-route commit fallback";
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", "1");
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    qkv_lazy_fixture fixture;
    return check(
                fixture.build(
                        { "k", "v", "q" },
                        { fake_role::cpu, fake_role::gpu, fake_role::npu },
                        1,
                        fake_role::gpu,
                        false,
                        true),
                scenario,
                "fixture build failed") &&
        fixture.compute_and_check(scenario) &&
        check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 0,
                scenario,
                "alternate route escaped the blocking canonical-commit fallback") &&
        check(
                fixture.control.consumers_submitted_while_branch_pending.load(
                        std::memory_order_relaxed) == 0,
                scenario,
                "alternate route submitted its consumer before canonical commit");
}

static bool run_ffn_add4_default_off_case() {
    const char * scenario = "FFN ADD4 lazy default-off equivalence";
    set_env("LLAMA_FFN_FUSED_ADD4", nullptr);
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", nullptr);
    ffn_add4_lazy_fixture fixture;
    return check(fixture.build(1), scenario, "fixture build failed") &&
        fixture.compute_and_check(scenario) &&
        check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 0,
                scenario,
                "default-off ADD4 path unexpectedly used async staging") &&
        check(
                fixture.control.consumers_submitted_while_branch_pending.load(
                        std::memory_order_relaxed) == 0,
                scenario,
                "default-off ADD4 path skipped the legacy GPU branch wait");
}

static bool run_ffn_add4_async_events_fallback_case() {
    const char * scenario = "FFN ADD4 lazy async-events fallback";
    set_env("LLAMA_FFN_FUSED_ADD4", "1");
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", nullptr);
    set_env("GGML_OPENCL_ASYNC_EVENTS", "0");
    ffn_add4_lazy_fixture fixture;
    const bool ok = check(fixture.build(1), scenario, "fixture build failed") &&
        fixture.compute_and_check(scenario) &&
        check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 0,
                scenario,
                "ADD4 used staging without async-event backend support") &&
        check(
                fixture.control.consumers_submitted_while_branch_pending.load(
                        std::memory_order_relaxed) == 0,
                scenario,
                "ADD4 consumer bypassed the capability fallback wait");
    set_env("GGML_OPENCL_ASYNC_EVENTS", "1");
    return ok;
}

static bool run_ffn_add4_lazy_case() {
    const char * scenario = "FFN ADD4 lazy OpenCL two-layer reuse";
    set_env("LLAMA_FFN_FUSED_ADD4", "1");
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", nullptr);
    set_env("GGML_OPENCL_ASYNC_EVENTS", "1");
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    ffn_add4_lazy_fixture fixture;
    if (!check(fixture.build(2), scenario, "fixture build failed") ||
            !fixture.compute_and_check(scenario)) {
        return false;
    }
    return check(
                fixture.control.async_uploads.load(std::memory_order_relaxed) == 4,
                scenario,
                "expected CPU/NPU staged uploads for both ADD4 consumers") &&
        check(
                fixture.control.event_records.load(std::memory_order_relaxed) == 4,
                scenario,
                "each ADD4 staging upload must have a completion event") &&
        check(
                fixture.control.event_synchronizes.load(std::memory_order_relaxed) >= 2,
                scenario,
                "ADD4 staging slots were reused before their events") &&
        check(
                fixture.control.consumers_submitted_while_branch_pending.load(
                        std::memory_order_relaxed) >= 2,
                scenario,
                "ADD4 consumer was not queued behind the pending GPU branch") &&
        check(
                fixture.control.branch_synchronizes_before_consumer.load(
                        std::memory_order_relaxed) == 0,
                scenario,
                "ADD4 lazy path synchronized the GPU branch before queuing its consumer") &&
        check(
                fixture.control.flush_calls.load(std::memory_order_relaxed) >= 2,
                scenario,
                "deferred ADD4 GPU branches were not flushed");
}

} // namespace

int main() {
    set_env("GGML_OPENCL_ASYNC_EVENTS", "1");
    const bool ok =
        run_default_off_case() &&
        run_lazy_case(
                "QKV lazy arbitrary order K/V/Q",
                { "k", "v", "q" },
                { fake_role::cpu, fake_role::gpu, fake_role::npu }) &&
        run_lazy_case(
                "QKV lazy arbitrary placement V/Q/K",
                { "v", "q", "k" },
                { fake_role::npu, fake_role::cpu, fake_role::gpu }) &&
        run_canonical_route_case() &&
        run_alternate_route_fallback_case() &&
        run_callback_fallback_case() &&
        run_incompatible_consumer_fallback_case() &&
        run_ffn_add4_default_off_case() &&
        run_ffn_add4_async_events_fallback_case() &&
        run_ffn_add4_lazy_case();
    set_env("GGML_QKV_LAZY_OPENCL_SYNC", nullptr);
    set_env("LLAMA_FFN_FUSED_ADD4", nullptr);
    set_env("GGML_OPENCL_ASYNC_EVENTS", nullptr);
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", nullptr);
    return ok ? 0 : 1;
}
