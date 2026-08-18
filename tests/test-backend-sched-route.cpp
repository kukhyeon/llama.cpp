#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct callback_step {
    int checkpoint_index;
    uint64_t accepted_plan;
    bool publish_plan_one;
};

struct callback_observation {
    int checkpoint_index;
    uint64_t active_plan;
    uint64_t requested_plan;
};

struct callback_script {
    ggml_backend_sched_t sched = nullptr;
    std::vector<callback_step> steps;
    std::vector<callback_observation> observations;
    std::vector<int> producer_observations;
    int producer_publish_checkpoint = -1;
    uint64_t producer_plan = 0;
    size_t next_step = 0;
    bool failed = false;
};

static void route_request_producer(
        int checkpoint_index,
        const ggml_tensor *,
        int,
        uint64_t,
        void * user_data) {
    auto * script = static_cast<callback_script *>(user_data);
    script->producer_observations.push_back(checkpoint_index);
    if (checkpoint_index == script->producer_publish_checkpoint) {
        ggml_backend_sched_layer_checkpoint_set_plan_id(
                script->sched, script->producer_plan);
    }
}

static uint64_t route_checkpoint_callback(
        int checkpoint_index,
        const ggml_tensor *,
        int,
        uint64_t active_plan_id,
        uint64_t requested_plan_id,
        void * user_data) {
    auto * script = static_cast<callback_script *>(user_data);
    script->observations.push_back({ checkpoint_index, active_plan_id, requested_plan_id });

    if (script->next_step >= script->steps.size()) {
        script->failed = true;
        return active_plan_id;
    }

    const callback_step & step = script->steps[script->next_step++];
    if (step.checkpoint_index != checkpoint_index) {
        script->failed = true;
        return active_plan_id;
    }
    if (step.publish_plan_one) {
        // The scheduler has already taken requested_plan_id for this
        // checkpoint. This request must remain pending until the next one.
        ggml_backend_sched_layer_checkpoint_set_plan_id(script->sched, 1);
    }
    return step.accepted_plan;
}

struct route_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_t backends[2] = { nullptr, nullptr };
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * y = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<ggml_tensor *> canonicals;
    std::vector<ggml_tensor *> alternates;
    std::vector<ggml_tensor *> boundaries;

    ~route_fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    }

    bool build(
            int n_layers,
            callback_script * script,
            uint64_t initial_request,
            bool configure_routes = true,
            bool clear_after_registration = false,
            bool use_request_producer = false) {
        const size_t graph_size = 64;
        const ggml_init_params params = {
            /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            return false;
        }

        x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_input(x);
        ggml_set_input(y);
        ggml_set_name(x, "route-x");
        ggml_set_name(y, "route-y");

        graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_tensor * layer_input = nullptr;
        for (int layer = 0; layer < n_layers; ++layer) {
            ggml_tensor * canonical = nullptr;
            ggml_tensor * alternate = nullptr;
            if (layer == 0) {
                canonical = ggml_add(ctx, x, y);
                alternate = ggml_add(ctx, x, y);
            } else if (layer % 2 == 1) {
                canonical = ggml_mul(ctx, layer_input, y);
                alternate = ggml_mul(ctx, layer_input, y);
            } else {
                canonical = ggml_add(ctx, layer_input, y);
                alternate = ggml_add(ctx, layer_input, y);
            }

            const std::string canonical_name = "route-canonical-" + std::to_string(layer);
            const std::string alternate_name = "route-alternate-" + std::to_string(layer);
            ggml_set_name(canonical, canonical_name.c_str());
            ggml_set_name(alternate, alternate_name.c_str());
            canonicals.push_back(canonical);
            alternates.push_back(alternate);

            // Incremental graph expansion preserves the required order:
            // canonical, then its otherwise-unreferenced alternate clone.
            ggml_build_forward_expand(graph, canonical);
            ggml_build_forward_expand(graph, alternate);

            if (layer + 1 < n_layers) {
                ggml_tensor * boundary = ggml_scale(ctx, canonical, 1.0f);
                const std::string boundary_name = "route-boundary-" + std::to_string(layer);
                ggml_set_name(boundary, boundary_name.c_str());
                boundaries.push_back(boundary);
                ggml_build_forward_expand(graph, boundary);
                layer_input = boundary;
            } else {
                layer_input = canonical;
            }
        }

        out = ggml_scale(ctx, layer_input, 1.0f);
        ggml_set_name(out, "route-output");
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);

        backends[0] = ggml_backend_cpu_init();
        backends[1] = ggml_backend_cpu_init();
        if (backends[0] == nullptr || backends[1] == nullptr) {
            return false;
        }
        sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false, true);
        if (sched == nullptr) {
            return false;
        }
        script->sched = sched;

        // All policy/candidate state is lazy. Exercise the null-safe APIs and
        // prove that a request before preparation is intentionally a no-op.
        ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 77);
        ggml_backend_sched_layer_checkpoint_stats empty_checkpoint_stats = {};
        ggml_backend_sched_get_layer_checkpoint_stats(sched, &empty_checkpoint_stats);
        ggml_backend_sched_route_candidate_stats empty_route_stats = {};
        ggml_backend_sched_get_route_candidate_stats(sched, &empty_route_stats);
        if (ggml_backend_sched_layer_checkpoint_get_active_plan_id(sched) != 0 ||
                empty_checkpoint_stats.graph_runs != 0 ||
                empty_route_stats.graph_runs != 0) {
            return false;
        }
        ggml_backend_sched_clear_layer_checkpoints(sched);
        ggml_backend_sched_clear_route_candidates(sched);

        if (configure_routes) {
            if (!ggml_backend_sched_prepare_layer_checkpoints(
                        sched,
                        graph,
                        boundaries.data(),
                        (int) boundaries.size(),
                        use_request_producer ? route_request_producer : nullptr,
                        route_checkpoint_callback,
                        script)) {
                return false;
            }
            for (int layer = 0; layer < n_layers; ++layer) {
                if (!ggml_backend_sched_register_route_candidate(
                            sched, canonicals[layer], canonicals[layer], 0, layer) ||
                        !ggml_backend_sched_register_route_candidate(
                            sched, canonicals[layer], alternates[layer], 1, layer)) {
                    return false;
                }
            }
            if (clear_after_registration) {
                ggml_backend_sched_clear_route_candidates(sched);
                ggml_backend_sched_clear_layer_checkpoints(sched);
            }
        }

        for (int layer = 0; layer < n_layers; ++layer) {
            ggml_backend_sched_set_tensor_backend(sched, canonicals[layer], backends[0]);
            ggml_backend_sched_set_tensor_backend(sched, alternates[layer], backends[1]);
        }
        for (ggml_tensor * boundary : boundaries) {
            ggml_backend_sched_set_tensor_backend(sched, boundary, backends[0]);
        }
        ggml_backend_sched_set_tensor_backend(sched, out, backends[0]);
        if (configure_routes && !clear_after_registration) {
            ggml_backend_sched_layer_checkpoint_set_plan_id(sched, initial_request);
        }

        return ggml_backend_sched_alloc_graph(sched, graph);
    }
};

static bool check(bool condition, const char * scenario, const char * detail) {
    if (!condition) {
        std::fprintf(stderr, "%s: %s\n", scenario, detail);
        return false;
    }
    return true;
}

static void set_attn_out_direct_scatter_env(const char * value) {
#if defined(_WIN32)
    (void) _putenv_s("GGML_ATTN_OUT_DIRECT_SCATTER", value != nullptr ? value : "");
#else
    if (value != nullptr) {
        (void) setenv("GGML_ATTN_OUT_DIRECT_SCATTER", value, 1);
    } else {
        (void) unsetenv("GGML_ATTN_OUT_DIRECT_SCATTER");
    }
#endif
}

struct queued_commit_control {
    int async_copy_calls = 0;
    int flush_calls = 0;
    int synchronize_calls = 0;
    int fail_async_copy_call = 0;
    int direct_scatter_calls = 0;
    ggml_tensor * direct_scatter_dst = nullptr;
};

struct queued_commit_backend_context {
    ggml_backend_t inner = nullptr;
    ggml_backend_t wrapper = nullptr;
    ggml_backend_dev_t device = nullptr;
    queued_commit_control * control = nullptr;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> pending_copies;
};

static queued_commit_backend_context * queued_commit_backend_ctx(ggml_backend_t backend) {
    return static_cast<queued_commit_backend_context *>(backend->context);
}

static queued_commit_backend_context * queued_commit_device_ctx(ggml_backend_dev_t device) {
    return static_cast<queued_commit_backend_context *>(device->context);
}

static const char * queued_commit_backend_name(ggml_backend_t) {
    return "QueuedCommitTest";
}

static void queued_commit_backend_free(ggml_backend_t backend) {
    queued_commit_backend_context * ctx = queued_commit_backend_ctx(backend);
    ggml_backend_free(ctx->inner);
    delete ctx->device;
    delete ctx;
    delete backend;
}

static bool queued_commit_backend_copy_async(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src,
        ggml_tensor * dst) {
    if (backend_src != backend_dst || src == nullptr || dst == nullptr ||
            src->data == nullptr || dst->data == nullptr ||
            ggml_nbytes(src) != ggml_nbytes(dst)) {
        return false;
    }

    queued_commit_backend_context * ctx = queued_commit_backend_ctx(backend_dst);
    queued_commit_control * control = ctx->control;
    ++control->async_copy_calls;
    if (control->fail_async_copy_call == control->async_copy_calls) {
        return false;
    }

    ctx->pending_copies.emplace_back(src, dst);
    return true;
}

static void queued_commit_backend_drain(queued_commit_backend_context * ctx) {
    for (const auto & copy : ctx->pending_copies) {
        std::memcpy(copy.second->data, copy.first->data, ggml_nbytes(copy.first));
    }
    ctx->pending_copies.clear();
}

static void queued_commit_backend_synchronize(ggml_backend_t backend) {
    queued_commit_backend_context * ctx = queued_commit_backend_ctx(backend);
    ++ctx->control->synchronize_calls;
    queued_commit_backend_drain(ctx);
    ggml_backend_synchronize(ctx->inner);
}

static enum ggml_status queued_commit_backend_graph_compute(
        ggml_backend_t backend, ggml_cgraph * graph) {
    return ggml_backend_graph_compute_async(
            queued_commit_backend_ctx(backend)->inner, graph);
}

static const ggml_backend_i queued_commit_backend_iface = {
    /* .get_name                = */ queued_commit_backend_name,
    /* .free                    = */ queued_commit_backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ queued_commit_backend_copy_async,
    /* .synchronize             = */ queued_commit_backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ queued_commit_backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static const char * queued_commit_device_name(ggml_backend_dev_t) {
    return "QueuedCommitTest";
}

static const char * queued_commit_device_description(ggml_backend_dev_t) {
    return "CPU-backed queued route commit test device";
}

static void queued_commit_device_memory(
        ggml_backend_dev_t device, size_t * free_memory, size_t * total_memory) {
    const queued_commit_backend_context * ctx = queued_commit_device_ctx(device);
    ggml_backend_dev_memory(
            ggml_backend_get_device(ctx->inner), free_memory, total_memory);
}

static enum ggml_backend_dev_type queued_commit_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static void queued_commit_device_props(
        ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    const queued_commit_backend_context * ctx = queued_commit_device_ctx(device);
    ggml_backend_dev_get_props(ggml_backend_get_device(ctx->inner), props);
    props->name = "QueuedCommitTest";
    props->description = "CPU-backed queued route commit test device";
    props->caps.async = true;
    props->caps.events = false;
}

static ggml_backend_t queued_commit_device_init(ggml_backend_dev_t, const char *) {
    return nullptr;
}

static ggml_backend_buffer_type_t queued_commit_device_buffer_type(
        ggml_backend_dev_t device) {
    return ggml_backend_get_default_buffer_type(queued_commit_device_ctx(device)->inner);
}

static ggml_backend_buffer_type_t queued_commit_device_host_buffer_type(
        ggml_backend_dev_t device) {
    return ggml_backend_get_default_buffer_type(queued_commit_device_ctx(device)->inner);
}

static bool queued_commit_device_supports_op(
        ggml_backend_dev_t device, const ggml_tensor * op) {
    const queued_commit_backend_context * ctx = queued_commit_device_ctx(device);
    return ggml_backend_dev_supports_op(ggml_backend_get_device(ctx->inner), op);
}

static bool queued_commit_device_supports_buft(
        ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    const queued_commit_backend_context * ctx = queued_commit_device_ctx(device);
    return ggml_backend_dev_supports_buft(ggml_backend_get_device(ctx->inner), buft);
}

static bool queued_commit_device_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static void queued_commit_backend_flush(ggml_backend_t backend) {
    queued_commit_backend_context * ctx = queued_commit_backend_ctx(backend);
    ++ctx->control->flush_calls;
    queued_commit_backend_drain(ctx);
}

static bool queued_commit_backend_attn_out_direct_scatter(
        ggml_backend_t backend,
        ggml_tensor * dst,
        const ggml_backend_attn_out_scatter_slice * slices,
        size_t n_slices) {
    queued_commit_backend_context * ctx = queued_commit_backend_ctx(backend);
    ++ctx->control->direct_scatter_calls;
    ctx->control->direct_scatter_dst = dst;
    if (dst == nullptr || slices == nullptr || n_slices < 2 || n_slices > 3 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }

    int64_t covered = 0;
    for (size_t slice_id = 0; slice_id < n_slices; ++slice_id) {
        const ggml_tensor * src = slices[slice_id].src;
        if (src == nullptr || src->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(src) || src->ne[1] != dst->ne[1] ||
                slices[slice_id].dst_offset != covered ||
                src->ne[0] > dst->ne[0] - covered) {
            return false;
        }
        std::vector<float> values((size_t) ggml_nelements(src));
        ggml_backend_tensor_get(
                src, values.data(), 0, values.size() * sizeof(values[0]));
        for (int64_t row = 0; row < src->ne[1]; ++row) {
            ggml_backend_tensor_set(
                    dst,
                    values.data() + row * src->ne[0],
                    (size_t) (row * dst->ne[0] + covered) * sizeof(float),
                    (size_t) src->ne[0] * sizeof(float));
        }
        covered += src->ne[0];
    }
    return covered == dst->ne[0];
}

static const char * queued_commit_registry_name(ggml_backend_reg_t) {
    return "QueuedCommitTest";
}

static size_t queued_commit_registry_device_count(ggml_backend_reg_t) {
    return 0;
}

static ggml_backend_dev_t queued_commit_registry_device(ggml_backend_reg_t, size_t) {
    return nullptr;
}

static void * queued_commit_registry_proc_address(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_async_flush") == 0) {
        return reinterpret_cast<void *>(queued_commit_backend_flush);
    }
    if (std::strcmp(name, "ggml_backend_attn_out_direct_scatter") == 0) {
        return reinterpret_cast<void *>(
                queued_commit_backend_attn_out_direct_scatter);
    }
    return nullptr;
}

static const ggml_backend_reg_i queued_commit_registry_iface = {
    /* .get_name         = */ queued_commit_registry_name,
    /* .get_device_count = */ queued_commit_registry_device_count,
    /* .get_device       = */ queued_commit_registry_device,
    /* .get_proc_address = */ queued_commit_registry_proc_address,
};

static ggml_backend_reg queued_commit_registry = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ queued_commit_registry_iface,
    /* .context     = */ nullptr,
};

static const ggml_backend_device_i queued_commit_device_iface = {
    /* .get_name             = */ queued_commit_device_name,
    /* .get_description      = */ queued_commit_device_description,
    /* .get_memory           = */ queued_commit_device_memory,
    /* .get_type             = */ queued_commit_device_type,
    /* .get_props            = */ queued_commit_device_props,
    /* .init_backend         = */ queued_commit_device_init,
    /* .get_buffer_type      = */ queued_commit_device_buffer_type,
    /* .get_host_buffer_type = */ queued_commit_device_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ queued_commit_device_supports_op,
    /* .supports_buft        = */ queued_commit_device_supports_buft,
    /* .offload_op           = */ queued_commit_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_guid_t queued_commit_backend_guid() {
    static ggml_guid guid = {
        0xd9, 0x30, 0xae, 0x0c, 0x21, 0x9a, 0x4c, 0x31,
        0xb7, 0x84, 0xe8, 0x3c, 0xe6, 0x9d, 0x7a, 0x55,
    };
    return &guid;
}

static ggml_backend_t make_queued_commit_backend(queued_commit_control * control) {
    auto * ctx = new queued_commit_backend_context;
    ctx->inner = ggml_backend_cpu_init();
    ctx->control = control;
    if (ctx->inner == nullptr) {
        delete ctx;
        return nullptr;
    }

    ctx->device = new ggml_backend_device {
        /* .iface   = */ queued_commit_device_iface,
        /* .reg     = */ &queued_commit_registry,
        /* .context = */ ctx,
    };
    ctx->wrapper = new ggml_backend {
        /* .guid    = */ queued_commit_backend_guid(),
        /* .iface   = */ queued_commit_backend_iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
    return ctx->wrapper;
}

class route_trace_capture {
public:
    explicit route_trace_capture(const char * tag) {
        std::error_code ec;
        const std::filesystem::path directory = std::filesystem::temp_directory_path(ec);
        if (ec) {
            return;
        }
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            const std::filesystem::path candidate = directory /
                ("llama-test-backend-sched-route-" + std::to_string(stamp) + "-" +
                 tag + "-" + std::to_string(attempt) + ".csv");
            if (std::filesystem::exists(candidate, ec)) {
                if (ec) {
                    return;
                }
                continue;
            }
            std::ofstream file(candidate, std::ios::binary | std::ios::trunc);
            if (file) {
                path_ = candidate.string();
                break;
            }
        }
    }

    ~route_trace_capture() {
        finish();
        std::error_code ec;
        if (!path_.empty()) {
            (void) std::filesystem::remove(path_, ec);
        }
    }

    bool start(int query_id) {
        if (path_.empty()) {
            return false;
        }
        ggml_backend_sched_trace_set_enabled(false);
        ggml_backend_sched_trace_set_path(path_.c_str());
        ggml_backend_sched_trace_set_enabled(true);
        ggml_backend_sched_trace_reset();
        ggml_backend_sched_trace_set_query_id(query_id);
        ggml_backend_sched_trace_set_ubatch(0, 0, 1);
        active_ = true;
        return true;
    }

    void finish() {
        if (!active_) {
            return;
        }
        ggml_backend_sched_trace_flush();
        ggml_backend_sched_trace_set_enabled(false);
        active_ = false;
    }

    const std::string & path() const {
        return path_;
    }

private:
    std::string path_;
    bool active_ = false;
};

struct route_trace_row {
    int group_id = -1;
    int node_count = -1;
    bool is_parallel_group = false;
    std::string first_node;
    std::string last_node;
    std::string parallel_kind;
    std::string parallel_branch;
};

static bool parse_route_trace_csv_line(
        const std::string & line,
        std::vector<std::string> * fields) {
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

static int route_trace_column(
        const std::vector<std::string> & columns,
        const char * name) {
    const auto it = std::find(columns.begin(), columns.end(), name);
    return it == columns.end() ? -1 : (int) std::distance(columns.begin(), it);
}

static bool parse_route_trace_int(const std::string & text, int * value) {
    char * end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *value = (int) parsed;
    return true;
}

static bool read_route_trace(
        const std::string & path,
        const char * scenario,
        std::vector<route_trace_row> * rows) {
    std::ifstream file(path, std::ios::binary);
    if (!check((bool) file, scenario, "failed to open scheduler trace")) {
        return false;
    }

    std::string line;
    std::vector<std::string> columns;
    if (!check(
                std::getline(file, line) &&
                    parse_route_trace_csv_line(line, &columns),
                scenario, "failed to read scheduler trace header")) {
        return false;
    }
    const int group_id_col = route_trace_column(columns, "group_id");
    const int node_count_col = route_trace_column(columns, "node_count");
    const int first_node_col = route_trace_column(columns, "first_node");
    const int last_node_col = route_trace_column(columns, "last_node");
    const int is_parallel_group_col = route_trace_column(columns, "is_parallel_group");
    const int parallel_kind_col = route_trace_column(columns, "parallel_group_kind");
    const int parallel_branch_col = route_trace_column(columns, "parallel_branch");
    const int required_columns[] = {
        group_id_col,
        node_count_col,
        first_node_col,
        last_node_col,
        is_parallel_group_col,
        parallel_kind_col,
        parallel_branch_col,
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
        if (!check(parse_route_trace_csv_line(line, &fields),
                    scenario, "malformed scheduler trace row") ||
                !check(fields.size() == columns.size(),
                    scenario, "scheduler trace column count mismatch")) {
            return false;
        }
        route_trace_row row;
        if (!check(parse_route_trace_int(fields[group_id_col], &row.group_id),
                    scenario, "invalid scheduler trace group id") ||
                !check(parse_route_trace_int(fields[node_count_col], &row.node_count),
                    scenario, "invalid scheduler trace node count")) {
            return false;
        }
        row.is_parallel_group = fields[is_parallel_group_col] == "1";
        row.first_node = fields[first_node_col];
        row.last_node = fields[last_node_col];
        row.parallel_kind = fields[parallel_kind_col];
        row.parallel_branch = fields[parallel_branch_col];
        rows->push_back(std::move(row));
    }
    return true;
}

static bool check_output(route_fixture & fixture, int n_layers, const char * scenario) {
    const float x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float y[4] = { 2.0f, 3.0f, 4.0f, 5.0f };
    ggml_backend_tensor_set(fixture.x, x, 0, sizeof(x));
    ggml_backend_tensor_set(fixture.y, y, 0, sizeof(y));
    if (!check(
                ggml_backend_sched_graph_compute(fixture.sched, fixture.graph) ==
                    GGML_STATUS_SUCCESS,
                scenario,
                "graph compute failed")) {
        return false;
    }

    float output[4] = {};
    ggml_backend_tensor_get(fixture.out, output, 0, sizeof(output));
    for (int i = 0; i < 4; ++i) {
        float expected = x[i] + y[i];
        for (int layer = 1; layer < n_layers; ++layer) {
            expected = layer % 2 == 1 ? expected * y[i] : expected + y[i];
        }
        if (!check(std::fabs(output[i] - expected) < 1e-5f, scenario, "output mismatch")) {
            std::fprintf(stderr, "  index %d: got %.6f, expected %.6f\n", i, output[i], expected);
            return false;
        }
    }
    return true;
}

static bool run_two_layer_case(
        const char * scenario,
        uint64_t initial_request,
        uint64_t graph_start_plan,
        uint64_t next_layer_plan,
        uint64_t expected_canonical,
        uint64_t expected_alternate,
        uint64_t expected_commits,
        uint64_t expected_misses) {
    callback_script script;
    script.steps = {
        { -1, graph_start_plan, false },
        {  0, next_layer_plan, false },
    };
    route_fixture fixture;
    if (!check(fixture.build(2, &script, initial_request), scenario, "fixture setup failed") ||
            !check_output(fixture, 2, scenario)) {
        return false;
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(fixture.sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(fixture.sched, &checkpoint_stats);

    return
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(script.observations.size() == 2, scenario, "unexpected callback count") &&
        check(route_stats.graph_runs == 1, scenario, "unexpected route graph count") &&
        check(route_stats.groups_executed == 2, scenario, "unexpected route group count") &&
        check(route_stats.canonical_selected == expected_canonical, scenario, "canonical selection mismatch") &&
        check(route_stats.alternate_selected == expected_alternate, scenario, "alternate selection mismatch") &&
        check(route_stats.inactive_splits_skipped == 2, scenario, "inactive split count mismatch") &&
        check(route_stats.canonical_commits == expected_commits, scenario, "canonical commit count mismatch") &&
        check(route_stats.plan_misses == expected_misses, scenario, "plan miss count mismatch") &&
        check(route_stats.plan_latches == 2, scenario, "plan latch count mismatch") &&
        check(checkpoint_stats.graph_runs == 1, scenario, "unexpected checkpoint graph count") &&
        check(checkpoint_stats.checkpoint_hits == 1, scenario, "unexpected checkpoint hit count") &&
        check(checkpoint_stats.callback_calls == 2, scenario, "unexpected checkpoint callback count") &&
        check(
            ggml_backend_sched_layer_checkpoint_get_active_plan_id(fixture.sched) == next_layer_plan,
            scenario,
            "final active plan mismatch");
}

static bool run_deferred_request_case() {
    const char * scenario = "deferred-request";
    callback_script script;
    script.steps = {
        { -1, 0, false },
        {  0, 0, true  }, // publish 1 after snapshot, reject for this boundary
        {  1, 1, false }, // the same request must still be pending here
    };
    route_fixture fixture;
    if (!check(fixture.build(3, &script, 0), scenario, "fixture setup failed") ||
            !check_output(fixture, 3, scenario)) {
        return false;
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(fixture.sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(fixture.sched, &checkpoint_stats);

    return
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(script.observations.size() == 3, scenario, "unexpected callback count") &&
        check(script.observations[0].checkpoint_index == -1, scenario, "missing graph-start callback") &&
        check(script.observations[0].requested_plan == 0, scenario, "wrong graph-start request") &&
        check(script.observations[1].checkpoint_index == 0, scenario, "missing first checkpoint") &&
        check(script.observations[1].active_plan == 0, scenario, "first checkpoint active plan changed early") &&
        check(script.observations[1].requested_plan == 0, scenario, "first checkpoint snapshot was not stable") &&
        check(script.observations[2].checkpoint_index == 1, scenario, "missing second checkpoint") &&
        check(script.observations[2].active_plan == 0, scenario, "rejected request became active early") &&
        check(script.observations[2].requested_plan == 1, scenario, "deferred request was not retained") &&
        check(route_stats.groups_executed == 3, scenario, "unexpected route group count") &&
        check(route_stats.canonical_selected == 2, scenario, "canonical selection mismatch") &&
        check(route_stats.alternate_selected == 1, scenario, "alternate selection mismatch") &&
        check(route_stats.inactive_splits_skipped == 3, scenario, "inactive split count mismatch") &&
        check(route_stats.canonical_commits == 1, scenario, "canonical commit count mismatch") &&
        check(route_stats.plan_misses == 0, scenario, "unexpected plan miss") &&
        check(route_stats.plan_latches == 3, scenario, "plan latch count mismatch") &&
        check(checkpoint_stats.checkpoint_hits == 2, scenario, "unexpected checkpoint hit count") &&
        check(checkpoint_stats.callback_calls == 3, scenario, "unexpected checkpoint callback count") &&
        check(
            ggml_backend_sched_layer_checkpoint_get_active_plan_id(fixture.sched) == 1,
            scenario,
            "deferred plan was not accepted");
}

static bool run_boundary_producer_case() {
    const char * scenario = "boundary-producer";
    callback_script script;
    script.producer_publish_checkpoint = 0;
    script.producer_plan = 1;
    script.steps = {
        { -1, 0, false },
        {  0, 1, false },
    };
    route_fixture fixture;
    if (!check(
                fixture.build(2, &script, 0, true, false, true),
                scenario,
                "fixture setup failed") ||
            !check_output(fixture, 2, scenario)) {
        return false;
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(fixture.sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(fixture.sched, &checkpoint_stats);

    return
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(script.producer_observations == std::vector<int>({ 0 }), scenario, "producer ran at graph start or wrong checkpoint") &&
        check(script.observations.size() == 2, scenario, "unexpected callback count") &&
        check(script.observations[0].checkpoint_index == -1, scenario, "missing graph-start callback") &&
        check(script.observations[0].requested_plan == 0, scenario, "producer affected graph start") &&
        check(script.observations[1].checkpoint_index == 0, scenario, "missing layer checkpoint") &&
        check(script.observations[1].requested_plan == 1, scenario, "producer publication missed same checkpoint snapshot") &&
        check(route_stats.canonical_selected == 1, scenario, "first layer route mismatch") &&
        check(route_stats.alternate_selected == 1, scenario, "producer did not route next layer") &&
        check(checkpoint_stats.producer_calls == 1, scenario, "producer stats mismatch") &&
        check(checkpoint_stats.producer_total_us >= checkpoint_stats.producer_max_us, scenario, "invalid producer timing stats") &&
        check(
            ggml_backend_sched_layer_checkpoint_get_active_plan_id(fixture.sched) == 1,
            scenario,
            "producer plan was not accepted");
}

static bool run_boundary_producer_cancels_stale_request_case() {
    const char * scenario = "boundary-producer-cancels-stale-request";
    callback_script script;
    // Plan 1 is pending at graph start but rejected. The boundary producer
    // observes that plan 0 is still the correct active plan and republishes
    // it, cancelling the stale request before the checkpoint snapshot.
    script.producer_publish_checkpoint = 0;
    script.producer_plan = 0;
    script.steps = {
        { -1, 0, false },
        {  0, 0, false },
    };
    route_fixture fixture;
    if (!check(
                fixture.build(2, &script, 1, true, false, true),
                scenario,
                "fixture setup failed") ||
            !check_output(fixture, 2, scenario)) {
        return false;
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(fixture.sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(fixture.sched, &checkpoint_stats);

    return
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(script.producer_observations == std::vector<int>({ 0 }), scenario, "producer ran at wrong checkpoint") &&
        check(script.observations.size() == 2, scenario, "unexpected callback count") &&
        check(script.observations[0].requested_plan == 1, scenario, "stale request was not present at graph start") &&
        check(script.observations[1].active_plan == 0, scenario, "stale request became active early") &&
        check(script.observations[1].requested_plan == 0, scenario, "producer did not cancel stale request") &&
        check(route_stats.canonical_selected == 2, scenario, "cancelled request changed the route") &&
        check(route_stats.alternate_selected == 0, scenario, "stale alternate route was selected") &&
        check(checkpoint_stats.producer_calls == 1, scenario, "producer stats mismatch") &&
        check(
            ggml_backend_sched_layer_checkpoint_get_active_plan_id(fixture.sched) == 0,
            scenario,
            "stale plan remained active");
}

static bool run_subgraph_layer_switch_case() {
    const char * scenario = "subgraph-layer-switch";
    const size_t graph_size = 128;
    const ggml_init_params params = {
        /* .mem_size   = */ 256 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backends[2] = {
        ggml_backend_cpu_init(),
        ggml_backend_cpu_init(),
    };
    ggml_backend_sched_t sched = backends[0] != nullptr && backends[1] != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 2, 256, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_name(x, "route-subgraph-x");
    ggml_set_name(y, "route-subgraph-y");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    std::vector<ggml_tensor *> canonical_firsts;
    std::vector<ggml_tensor *> canonical_outputs;
    std::vector<ggml_tensor *> alternate_firsts;
    std::vector<ggml_tensor *> alternate_outputs;
    ggml_tensor * layer_input = x;
    ggml_tensor * boundary = nullptr;

    for (int layer = 0; layer < 2; ++layer) {
        ggml_tensor * c1 = ggml_add(ctx, layer_input, y);
        ggml_tensor * c2 = ggml_mul(ctx, c1, y);
        ggml_tensor * c3 = ggml_add(ctx, c2, y);
        ggml_tensor * a1 = ggml_mul(ctx, layer_input, y);
        ggml_tensor * a2 = ggml_add(ctx, a1, y);
        ggml_tensor * a3 = ggml_mul(ctx, a2, y);
        // Reuse the scheduler's FFN branch-purity boundary to force two
        // splits per variant even though this host-only test has two CPU
        // instances with the same buffer type (which pass 3 coalesces).
        ggml_set_name(c1, ("ffn_up.route-subgraph-canonical-" + std::to_string(layer)).c_str());
        ggml_set_name(c2, ("ffn_down.route-subgraph-canonical-" + std::to_string(layer)).c_str());
        ggml_set_name(c3, ("route-subgraph-c3-" + std::to_string(layer)).c_str());
        ggml_set_name(a1, ("ffn_up.route-subgraph-alternate-" + std::to_string(layer)).c_str());
        ggml_set_name(a2, ("ffn_down.route-subgraph-alternate-" + std::to_string(layer)).c_str());
        ggml_set_name(a3, ("route-subgraph-a3-" + std::to_string(layer)).c_str());
        canonical_firsts.push_back(c1);
        canonical_outputs.push_back(c3);
        alternate_firsts.push_back(a1);
        alternate_outputs.push_back(a3);

        ggml_build_forward_expand(graph, c3);
        ggml_build_forward_expand(graph, a3);

        ggml_backend_sched_set_tensor_backend(sched, c1, backends[0]);
        ggml_backend_sched_set_tensor_backend(sched, c2, backends[0]);
        ggml_backend_sched_set_tensor_backend(sched, c3, backends[1]);
        ggml_backend_sched_set_tensor_backend(sched, a1, backends[1]);
        ggml_backend_sched_set_tensor_backend(sched, a2, backends[1]);
        ggml_backend_sched_set_tensor_backend(sched, a3, backends[0]);

        if (layer == 0) {
            boundary = ggml_scale(ctx, c3, 1.0f);
            ggml_set_name(boundary, "route-subgraph-boundary-0");
            ggml_build_forward_expand(graph, boundary);
            ggml_backend_sched_set_tensor_backend(sched, boundary, backends[0]);
            layer_input = boundary;
        }
    }

    ggml_tensor * out = ggml_scale(ctx, canonical_outputs.back(), 1.0f);
    ggml_set_name(out, "route-subgraph-output");
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);
    ggml_backend_sched_set_tensor_backend(sched, out, backends[0]);

    callback_script script;
    script.sched = sched;
    script.steps = {
        { -1, 0, false },
        {  0, 1, false },
    };
    ggml_tensor * boundaries[] = { boundary };
    bool setup_ok = ggml_backend_sched_prepare_layer_checkpoints(
            sched, graph, boundaries, 1, nullptr,
            route_checkpoint_callback, &script);
    for (int layer = 0; setup_ok && layer < 2; ++layer) {
        setup_ok = ggml_backend_sched_register_route_subgraph(
                sched,
                canonical_firsts[layer], canonical_outputs[layer],
                canonical_firsts[layer], canonical_outputs[layer],
                0, layer) &&
            ggml_backend_sched_register_route_subgraph(
                sched,
                canonical_firsts[layer], canonical_outputs[layer],
                alternate_firsts[layer], alternate_outputs[layer],
                1, layer);
    }
    setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 0);
    if (!check(setup_ok, scenario, "subgraph preparation failed")) {
        cleanup();
        return false;
    }

    const float x_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float y_data[4] = { 2.0f, 3.0f, 4.0f, 5.0f };
    ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
    ggml_backend_tensor_set(y, y_data, 0, sizeof(y_data));
    bool ok = check(
            ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed");

    const float expected[4] = { 36.0f, 171.0f, 528.0f, 1275.0f };
    float actual[4] = {};
    if (ok) {
        ggml_backend_tensor_get(out, actual, 0, sizeof(actual));
        for (int i = 0; i < 4; ++i) {
            ok = check(std::fabs(actual[i] - expected[i]) < 1e-5f,
                    scenario, "subgraph output mismatch") && ok;
        }
    }

    ggml_backend_sched_route_candidate_stats route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(sched, &route_stats);
    ggml_backend_sched_layer_checkpoint_stats checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(sched, &checkpoint_stats);
    if (route_stats.prepared_candidate_splits != 8) {
        std::fprintf(stderr,
                "%s: got %d candidate splits (%d total), expected 8\n",
                scenario,
                route_stats.prepared_candidate_splits,
                ggml_backend_sched_get_n_splits(sched));
    }
    ok =
        check(!script.failed && script.next_step == script.steps.size(), scenario, "callback script mismatch") &&
        check(route_stats.registered_mappings == 4, scenario, "mapping count mismatch") &&
        check(route_stats.prepared_groups == 2, scenario, "prepared group count mismatch") &&
        check(route_stats.prepared_candidate_splits == 8, scenario, "candidate split coalescing mismatch") &&
        check(route_stats.groups_executed == 2, scenario, "executed group count mismatch") &&
        check(route_stats.canonical_selected == 1, scenario, "canonical selection mismatch") &&
        check(route_stats.alternate_selected == 1, scenario, "alternate selection mismatch") &&
        check(route_stats.inactive_splits_skipped == 4, scenario, "inactive subgraph split mismatch") &&
        check(route_stats.canonical_commits == 1, scenario, "terminal commit count mismatch") &&
        check(route_stats.plan_misses == 0, scenario, "unexpected plan miss") &&
        check(route_stats.plan_latches == 2, scenario, "plan latch count mismatch") &&
        check(checkpoint_stats.checkpoint_hits == 1, scenario, "checkpoint hit count mismatch") &&
        check(checkpoint_stats.callback_calls == 2, scenario, "checkpoint callback count mismatch") &&
        check(ggml_backend_sched_layer_checkpoint_get_active_plan_id(sched) == 1,
                scenario, "final active plan mismatch") && ok;

    cleanup();
    return ok;
}

static bool run_alternate_terminal_lifetime_case() {
    const char * scenario = "alternate-terminal-lifetime";
    constexpr int n_alternates = 24;
    constexpr int64_t n_elements = 256 * 1024; // 1 MiB per F32 terminal
    const size_t graph_size = 64;
    const ggml_init_params params = {
        /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t sched = backend != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 1, 128, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(input);
    ggml_set_name(input, "route-lifetime-input");
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);

    ggml_tensor * canonical = ggml_scale(ctx, input, 1.0f);
    ggml_set_name(canonical, "route-lifetime-canonical");
    ggml_build_forward_expand(graph, canonical);
    ggml_backend_sched_set_tensor_backend(sched, canonical, backend);

    std::vector<ggml_tensor *> alternates;
    alternates.reserve(n_alternates);
    for (int i = 0; i < n_alternates; ++i) {
        ggml_tensor * alternate = ggml_scale(ctx, input, (float) (i + 2));
        const std::string name = "route-lifetime-alternate-" + std::to_string(i);
        ggml_set_name(alternate, name.c_str());
        ggml_build_forward_expand(graph, alternate);
        ggml_backend_sched_set_tensor_backend(sched, alternate, backend);
        alternates.push_back(alternate);
    }

    ggml_tensor * output = ggml_scale(ctx, canonical, 1.0f);
    ggml_set_name(output, "route-lifetime-output");
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_set_tensor_backend(sched, output, backend);

    bool setup_ok = ggml_backend_sched_register_route_subgraph(
            sched, canonical, canonical, canonical, canonical, 0, 0);
    for (int i = 0; setup_ok && i < n_alternates; ++i) {
        setup_ok = ggml_backend_sched_register_route_subgraph(
                sched, canonical, canonical,
                alternates[i], alternates[i], (uint64_t) i + 1, 0);
    }

    size_t sizes[1] = {};
    if (setup_ok) {
        ggml_backend_sched_reserve_size(sched, graph, sizes);
    }

    // Input + canonical/output + one reusable alternate scratch terminal fit
    // comfortably below 8 MiB. Without scheduler lifetime dependencies, the
    // 24 alternate roots alone require 24 MiB and this assertion fails.
    const bool ok =
        check(setup_ok, scenario, "route registration failed") &&
        check(sizes[0] > 0, scenario, "reserve size was not reported") &&
        check(sizes[0] < 8 * 1024 * 1024,
                scenario, "alternate terminals remained live to graph end");

    cleanup();
    return ok;
}

static bool run_multi_consumer_commit_case() {
    const char * scenario = "multi-consumer-canonical-commit";
    const size_t graph_size = 32;
    const ggml_init_params params = {
        /* .mem_size   = */ 64 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backends[2] = {
        ggml_backend_cpu_init(),
        ggml_backend_cpu_init(),
    };
    ggml_backend_sched_t sched = backends[0] != nullptr && backends[1] != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 2, 64, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_name(x, "route-fanout-x");
    ggml_set_name(y, "route-fanout-y");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_tensor * canonical_first = ggml_add(ctx, x, y);
    ggml_tensor * canonical = ggml_scale(ctx, canonical_first, 1.0f);
    ggml_tensor * alternate_first = ggml_mul(ctx, x, y);
    ggml_tensor * alternate = ggml_scale(ctx, alternate_first, 1.0f);
    ggml_set_name(canonical_first, "route-fanout-canonical-first");
    ggml_set_name(canonical, "route-fanout-canonical");
    ggml_set_name(alternate_first, "route-fanout-alternate-first");
    ggml_set_name(alternate, "route-fanout-alternate");
    ggml_build_forward_expand(graph, canonical);
    ggml_build_forward_expand(graph, alternate);

    // Model ffn_inp's fan-out: the committed canonical value feeds multiple
    // later consumers, which may themselves begin separate route subgraphs.
    ggml_tensor * consumer_a = ggml_scale(ctx, canonical, 2.0f);
    ggml_tensor * consumer_b = ggml_add(ctx, canonical, y);
    ggml_tensor * output = ggml_add(ctx, consumer_a, consumer_b);
    ggml_set_name(consumer_a, "route-fanout-consumer-a");
    ggml_set_name(consumer_b, "route-fanout-consumer-b");
    ggml_set_name(output, "route-fanout-output");
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ggml_backend_sched_set_tensor_backend(sched, canonical_first, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, canonical, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, alternate_first, backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, alternate, backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, consumer_a, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, consumer_b, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, output, backends[0]);

    bool setup_ok =
        ggml_backend_sched_register_route_subgraph(
                sched, canonical_first, canonical,
                canonical_first, canonical, 0, 0) &&
        ggml_backend_sched_register_route_subgraph(
                sched, canonical_first, canonical,
                alternate_first, alternate, 1, 0);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 1);
    setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
    if (!check(setup_ok, scenario, "route preparation failed")) {
        cleanup();
        return false;
    }

    const float x_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float y_data[4] = { 2.0f, 3.0f, 4.0f, 5.0f };
    ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
    ggml_backend_tensor_set(y, y_data, 0, sizeof(y_data));
    bool ok = check(
            ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed");

    float actual[4] = {};
    if (ok) {
        ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
        for (int i = 0; i < 4; ++i) {
            // The selected alternate terminal is x*y. Both consumers must
            // observe the committed value:
            // 2*(x*y) + ((x*y)+y) = 3*x*y+y.
            const float expected = 3.0f * x_data[i] * y_data[i] + y_data[i];
            ok = check(std::fabs(actual[i] - expected) < 1e-5f,
                    scenario, "fan-out consumer observed stale canonical data") && ok;
        }
    }

    ggml_backend_sched_route_candidate_stats stats = {};
    ggml_backend_sched_get_route_candidate_stats(sched, &stats);
    ok =
        check(stats.groups_executed == 1, scenario, "unexpected route group count") &&
        check(stats.alternate_selected == 1, scenario, "alternate was not selected") &&
        check(stats.canonical_commits == 1, scenario, "alternate was not committed") && ok;

    cleanup();
    return ok;
}

static bool run_multi_output_bundle_case() {
    const char * scenario = "multi-output-atomic-bundle";
    constexpr int n_outputs = 3;
    const size_t graph_size = 64;
    const ggml_init_params params = {
        /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backends[2] = {
        ggml_backend_cpu_init(),
        ggml_backend_cpu_init(),
    };
    ggml_backend_sched_t sched = backends[0] != nullptr && backends[1] != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 2, 128, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_name(x, "route-bundle-x");
    ggml_set_name(y, "route-bundle-y");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_tensor * canonical_outputs[n_outputs] = {
        ggml_add(ctx, x, y),
        ggml_mul(ctx, x, y),
        ggml_sub(ctx, x, y),
    };
    ggml_tensor * alternate_outputs[n_outputs] = {
        ggml_mul(ctx, x, y),
        ggml_add(ctx, x, y),
        ggml_sub(ctx, y, x),
    };
    const char * canonical_names[n_outputs] = {
        "route-bundle-canonical-q",
        "route-bundle-canonical-k",
        "route-bundle-canonical-v",
    };
    const char * alternate_names[n_outputs] = {
        "route-bundle-alternate-q",
        "route-bundle-alternate-k",
        "route-bundle-alternate-v",
    };
    for (int output_index = 0; output_index < n_outputs; ++output_index) {
        ggml_set_name(canonical_outputs[output_index], canonical_names[output_index]);
        ggml_backend_sched_set_tensor_backend(
                sched, canonical_outputs[output_index], backends[0]);
        ggml_build_forward_expand(graph, canonical_outputs[output_index]);
    }
    for (int output_index = 0; output_index < n_outputs; ++output_index) {
        ggml_set_name(alternate_outputs[output_index], alternate_names[output_index]);
        ggml_backend_sched_set_tensor_backend(
                sched, alternate_outputs[output_index], backends[1]);
        ggml_build_forward_expand(graph, alternate_outputs[output_index]);
    }

    // Each downstream node observes a different canonical terminal. If the
    // alternate route commits only the final V value, or releases Q/K scratch
    // storage before V completes, at least one output check below fails.
    ggml_tensor * outputs[n_outputs] = {};
    for (int output_index = 0; output_index < n_outputs; ++output_index) {
        outputs[output_index] = ggml_scale(ctx, canonical_outputs[output_index], 1.0f);
        const std::string name = "route-bundle-output-" + std::to_string(output_index);
        ggml_set_name(outputs[output_index], name.c_str());
        ggml_set_output(outputs[output_index]);
        ggml_backend_sched_set_tensor_backend(sched, outputs[output_index], backends[0]);
        ggml_build_forward_expand(graph, outputs[output_index]);
    }

    // Step 7 consumes all three canonical slots in one common node after the
    // selected route commits. This catches partial/non-atomic commits that a
    // set of independent one-input consumers could otherwise miss.
    ggml_tensor * fused_output = ggml_add4(
            ctx,
            canonical_outputs[0], canonical_outputs[1],
            canonical_outputs[2], x);
    ggml_set_name(fused_output, "route-bundle-common-add4");
    ggml_set_output(fused_output);
    ggml_backend_sched_set_tensor_backend(sched, fused_output, backends[0]);
    ggml_build_forward_expand(graph, fused_output);

    // Reject a malformed bundle without changing the scheduler registration
    // state. All valid plan mappings below share one canonical bundle.
    ggml_tensor * wrong_layout = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * mismatched_outputs[n_outputs] = {
        alternate_outputs[0], alternate_outputs[1], wrong_layout,
    };
    bool setup_ok = !ggml_backend_sched_register_route_subgraph_bundle(
            sched,
            canonical_outputs[0], canonical_outputs,
            alternate_outputs[0], mismatched_outputs,
            n_outputs, 99, 0);
    setup_ok = setup_ok &&
        ggml_backend_sched_register_route_subgraph_bundle(
                sched,
                canonical_outputs[0], canonical_outputs,
                canonical_outputs[0], canonical_outputs,
                n_outputs, 0, 0) &&
        ggml_backend_sched_register_route_subgraph_bundle(
                sched,
                canonical_outputs[0], canonical_outputs,
                alternate_outputs[0], alternate_outputs,
                n_outputs, 1, 0) &&
        // Multiple policy profiles may inherit exactly the same topology.
        ggml_backend_sched_register_route_subgraph_bundle(
                sched,
                canonical_outputs[0], canonical_outputs,
                alternate_outputs[0], alternate_outputs,
                n_outputs, 2, 0);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 0);
    setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
    if (!check(setup_ok, scenario, "bundle route preparation failed")) {
        cleanup();
        return false;
    }

    const float x_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float y_data[4] = { 5.0f, 6.0f, 7.0f, 8.0f };
    auto compute_and_check = [&](uint64_t plan_id, bool alternate) {
        // User-owned graph inputs may be refreshed between scheduler runs;
        // exercise every plan from the same explicit input state.
        ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
        ggml_backend_tensor_set(y, y_data, 0, sizeof(y_data));
        ggml_backend_sched_layer_checkpoint_set_plan_id(sched, plan_id);
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario, "graph compute failed")) {
            return false;
        }
        for (int output_index = 0; output_index < n_outputs; ++output_index) {
            float actual[4] = {};
            ggml_backend_tensor_get(outputs[output_index], actual, 0, sizeof(actual));
            for (int i = 0; i < 4; ++i) {
                float expected = 0.0f;
                if (!alternate) {
                    expected = output_index == 0 ? x_data[i] + y_data[i] :
                        output_index == 1 ? x_data[i] * y_data[i] :
                        x_data[i] - y_data[i];
                } else {
                    expected = output_index == 0 ? x_data[i] * y_data[i] :
                        output_index == 1 ? x_data[i] + y_data[i] :
                        y_data[i] - x_data[i];
                }
                if (!check(std::fabs(actual[i] - expected) < 1e-5f,
                            scenario, "bundle output mismatch")) {
                    std::fprintf(stderr,
                            "  plan %llu output %d index %d: got %.6f, expected %.6f\n",
                            (unsigned long long) plan_id, output_index, i,
                            actual[i], expected);
                    return false;
                }
            }
        }

        float fused_actual[4] = {};
        ggml_backend_tensor_get(fused_output, fused_actual, 0, sizeof(fused_actual));
        for (int i = 0; i < 4; ++i) {
            const float q = alternate ? x_data[i] * y_data[i] : x_data[i] + y_data[i];
            const float k = alternate ? x_data[i] + y_data[i] : x_data[i] * y_data[i];
            const float v = alternate ? y_data[i] - x_data[i] : x_data[i] - y_data[i];
            const float expected = (q + (k + v)) + x_data[i];
            if (!check(std::fabs(fused_actual[i] - expected) < 1e-5f,
                        scenario, "common ADD4 observed a partial route commit")) {
                return false;
            }
        }
        return true;
    };

    bool ok =
        compute_and_check(0, false) &&
        compute_and_check(1, true) &&
        compute_and_check(2, true) &&
        // An unknown profile falls back to the complete canonical bundle.
        compute_and_check(77, false);

    ggml_backend_sched_route_candidate_stats stats = {};
    ggml_backend_sched_get_route_candidate_stats(sched, &stats);
    ok =
        check(stats.registered_mappings == 3, scenario, "mapping count mismatch") &&
        check(stats.prepared_groups == 1, scenario, "prepared group count mismatch") &&
        check(stats.groups_executed == 4, scenario, "executed group count mismatch") &&
        check(stats.canonical_selected == 2, scenario, "canonical selection mismatch") &&
        check(stats.alternate_selected == 2, scenario, "alternate selection mismatch") &&
        check(stats.canonical_commits == 2, scenario, "bundle commit count mismatch") &&
        check(stats.plan_misses == 1, scenario, "unknown plan fallback mismatch") && ok;

    cleanup();
    return ok;
}

static bool run_explicit_range_end_bundle_case() {
    const char * scenario = "explicit-range-end-qkv-bundle";
    constexpr int n_outputs = 3;
    const size_t graph_size = 64;
    const ggml_init_params params = {
        /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backends[2] = {
        ggml_backend_cpu_init(),
        ggml_backend_cpu_init(),
    };
    ggml_backend_sched_t sched = backends[0] != nullptr && backends[1] != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 2, 128, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_name(x, "range-end-qkv-x");
    ggml_set_name(y, "range-end-qkv-y");
    ggml_backend_sched_set_tensor_backend(sched, x, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, y, backends[0]);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    struct qkv_variant {
        ggml_tensor * first = nullptr;
        ggml_tensor * range_end = nullptr;
        ggml_tensor * outputs[n_outputs] = {};
    };
    auto build_variant = [&](const char * tag, ggml_backend_t backend, bool alternate) {
        qkv_variant variant;

        // Deliberately build Q, then V, and finish with a two-node K path.
        // The semantic output bundle remains {Q, K, V}, so outputs[2] is not
        // the graph range endpoint. Projection-local QKV assembly has this
        // shape when (for example) V is a single GPU result while K finishes
        // later in a CPU/NPU join.
        ggml_tensor * q = alternate ? ggml_mul(ctx, x, y) : ggml_add(ctx, x, y);
        ggml_tensor * v = alternate ? ggml_add(ctx, x, y) : ggml_sub(ctx, x, y);
        ggml_tensor * k_partial = alternate ? ggml_sub(ctx, y, x) : ggml_sub(ctx, x, y);
        ggml_tensor * k = ggml_scale(ctx, k_partial, alternate ? 0.5f : 2.0f);

        ggml_set_name(q, ("range-end-qkv-" + std::string(tag) + "-q").c_str());
        ggml_set_name(v, ("range-end-qkv-" + std::string(tag) + "-v").c_str());
        ggml_set_name(k_partial,
                ("range-end-qkv-" + std::string(tag) + "-k-partial").c_str());
        ggml_set_name(k, ("range-end-qkv-" + std::string(tag) + "-k-join").c_str());

        for (ggml_tensor * node : { q, v, k_partial, k }) {
            ggml_backend_sched_set_tensor_backend(sched, node, backend);
            ggml_build_forward_expand(graph, node);
        }

        variant.first = q;
        variant.range_end = k;
        variant.outputs[0] = q;
        variant.outputs[1] = k;
        variant.outputs[2] = v;
        return variant;
    };

    const qkv_variant canonical = build_variant("canonical", backends[0], false);
    const qkv_variant alternate = build_variant("alternate", backends[1], true);

    ggml_tensor * downstream[n_outputs] = {};
    for (int output_index = 0; output_index < n_outputs; ++output_index) {
        downstream[output_index] =
            ggml_scale(ctx, canonical.outputs[output_index], 1.0f);
        const std::string name =
            "range-end-qkv-downstream-" + std::to_string(output_index);
        ggml_set_name(downstream[output_index], name.c_str());
        ggml_set_output(downstream[output_index]);
        ggml_backend_sched_set_tensor_backend(sched, downstream[output_index], backends[0]);
        ggml_build_forward_expand(graph, downstream[output_index]);
    }

    bool setup_ok =
        ggml_backend_sched_register_route_subgraph_bundle_range(
                sched,
                canonical.first, canonical.range_end, canonical.outputs,
                canonical.first, canonical.range_end, canonical.outputs,
                n_outputs, 0, 0) &&
        ggml_backend_sched_register_route_subgraph_bundle_range(
                sched,
                canonical.first, canonical.range_end, canonical.outputs,
                alternate.first, alternate.range_end, alternate.outputs,
                n_outputs, 1, 0);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 1);
    setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
    if (!check(setup_ok, scenario, "explicit range-end route preparation failed")) {
        cleanup();
        return false;
    }

    const float x_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float y_data[4] = { 5.0f, 6.0f, 7.0f, 8.0f };
    ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
    ggml_backend_tensor_set(y, y_data, 0, sizeof(y_data));
    bool ok = check(
            ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed");
    for (int output_index = 0; ok && output_index < n_outputs; ++output_index) {
        float actual[4] = {};
        ggml_backend_tensor_get(downstream[output_index], actual, 0, sizeof(actual));
        for (int i = 0; i < 4; ++i) {
            const float expected = output_index == 0 ? x_data[i] * y_data[i] :
                output_index == 1 ? 0.5f * (y_data[i] - x_data[i]) :
                x_data[i] + y_data[i];
            if (!check(std::fabs(actual[i] - expected) < 1e-5f,
                        scenario, "explicit range-end bundle output mismatch")) {
                std::fprintf(stderr,
                        "  output %d index %d: got %.6f, expected %.6f\n",
                        output_index, i, actual[i], expected);
                ok = false;
                break;
            }
        }
    }

    ggml_backend_sched_route_candidate_stats stats = {};
    ggml_backend_sched_get_route_candidate_stats(sched, &stats);
    ok =
        check(stats.registered_mappings == 2, scenario, "mapping count mismatch") &&
        check(stats.groups_executed == 1, scenario, "route group count mismatch") &&
        check(stats.alternate_selected == 1, scenario, "alternate route was not selected") &&
        check(stats.canonical_commits == 1, scenario, "QKV bundle was not committed") && ok;

    cleanup();
    return ok;
}

static bool run_queued_multi_output_commit_case() {
    constexpr int n_outputs = 3;

    auto run_one = [](const char * scenario, int fail_async_copy_call) {
        const size_t graph_size = 64;
        const ggml_init_params params = {
            /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };

        queued_commit_control control;
        control.fail_async_copy_call = fail_async_copy_call;
        ggml_context * ctx = ggml_init(params);
        ggml_backend_t backend = make_queued_commit_backend(&control);
        ggml_backend_t backends[] = { backend };
        ggml_backend_sched_t sched = ctx != nullptr && backend != nullptr
            ? ggml_backend_sched_new(backends, nullptr, 1, 128, false, true)
            : nullptr;
        auto cleanup = [&]() {
            ggml_backend_sched_free(sched);
            ggml_backend_free(backend);
            ggml_free(ctx);
        };
        if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
            cleanup();
            return false;
        }

        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_input(x);
        ggml_set_input(y);
        ggml_set_name(x, "queued-route-x");
        ggml_set_name(y, "queued-route-y");

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_tensor * canonical_outputs[n_outputs] = {
            ggml_add(ctx, x, y),
            ggml_mul(ctx, x, y),
            ggml_sub(ctx, x, y),
        };
        ggml_tensor * alternate_outputs[n_outputs] = {
            ggml_mul(ctx, x, y),
            ggml_add(ctx, x, y),
            ggml_sub(ctx, y, x),
        };
        for (int output_index = 0; output_index < n_outputs; ++output_index) {
            const std::string canonical_name =
                "queued-route-canonical-" + std::to_string(output_index);
            const std::string alternate_name =
                "queued-route-alternate-" + std::to_string(output_index);
            ggml_set_name(canonical_outputs[output_index], canonical_name.c_str());
            ggml_set_name(alternate_outputs[output_index], alternate_name.c_str());
            ggml_backend_sched_set_tensor_backend(
                    sched, canonical_outputs[output_index], backend);
            ggml_backend_sched_set_tensor_backend(
                    sched, alternate_outputs[output_index], backend);
            ggml_build_forward_expand(graph, canonical_outputs[output_index]);
        }
        for (ggml_tensor * alternate_output : alternate_outputs) {
            ggml_build_forward_expand(graph, alternate_output);
        }

        ggml_tensor * outputs[n_outputs] = {};
        for (int output_index = 0; output_index < n_outputs; ++output_index) {
            outputs[output_index] =
                ggml_scale(ctx, canonical_outputs[output_index], 1.0f);
            const std::string output_name =
                "queued-route-output-" + std::to_string(output_index);
            ggml_set_name(outputs[output_index], output_name.c_str());
            ggml_set_output(outputs[output_index]);
            ggml_backend_sched_set_tensor_backend(sched, outputs[output_index], backend);
            ggml_build_forward_expand(graph, outputs[output_index]);
        }

        bool setup_ok =
            ggml_backend_sched_register_route_subgraph_bundle(
                    sched,
                    canonical_outputs[0], canonical_outputs,
                    canonical_outputs[0], canonical_outputs,
                    n_outputs, 0, 0) &&
            ggml_backend_sched_register_route_subgraph_bundle(
                    sched,
                    canonical_outputs[0], canonical_outputs,
                    alternate_outputs[0], alternate_outputs,
                    n_outputs, 1, 0);
        ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 1);
        setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
        if (!check(setup_ok, scenario, "queued route preparation failed")) {
            cleanup();
            return false;
        }

        const float x_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        const float y_data[4] = { 5.0f, 6.0f, 7.0f, 8.0f };
        ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
        ggml_backend_tensor_set(y, y_data, 0, sizeof(y_data));
        bool ok = check(
                ggml_backend_sched_graph_compute_async(sched, graph) ==
                    GGML_STATUS_SUCCESS,
                scenario, "async graph compute failed");

        const int expected_async_copies = fail_async_copy_call == 0
            ? n_outputs : fail_async_copy_call;
        const int expected_flushes = fail_async_copy_call == 0 ? 1 : 0;
        // A later callback rejection drains the partially queued bundle once;
        // the blocking fallback recognizes that the same producer backend has
        // already been synchronized and does not wait on it again.
        const int expected_internal_syncs = fail_async_copy_call == 0 ? 0 : 1;
        if (control.async_copy_calls != expected_async_copies ||
                control.flush_calls != expected_flushes ||
                control.synchronize_calls != expected_internal_syncs) {
            std::fprintf(stderr,
                    "  async-copy/flush/synchronize=%d/%d/%d, expected %d/%d/%d\n",
                    control.async_copy_calls,
                    control.flush_calls,
                    control.synchronize_calls,
                    expected_async_copies,
                    expected_flushes,
                    expected_internal_syncs);
        }
        ok =
            check(control.async_copy_calls == expected_async_copies,
                scenario, "unexpected async-copy callback count") &&
            check(control.flush_calls == expected_flushes,
                scenario, "unexpected queued-commit flush count") &&
            check(control.synchronize_calls == expected_internal_syncs,
                scenario, "queued commit used an unexpected blocking wait") && ok;

        for (int output_index = 0; ok && output_index < n_outputs; ++output_index) {
            float actual[4] = {};
            ggml_backend_tensor_get(outputs[output_index], actual, 0, sizeof(actual));
            for (int i = 0; i < 4; ++i) {
                const float expected = output_index == 0
                    ? x_data[i] * y_data[i]
                    : output_index == 1
                        ? x_data[i] + y_data[i]
                        : y_data[i] - x_data[i];
                ok = check(std::fabs(actual[i] - expected) < 1e-5f,
                        scenario, "downstream observed a mixed route bundle") && ok;
            }
        }

        ggml_backend_sched_route_candidate_stats stats = {};
        ggml_backend_sched_get_route_candidate_stats(sched, &stats);
        ok =
            check(stats.alternate_selected == 1,
                scenario, "alternate bundle was not selected") &&
            check(stats.canonical_commits == 1,
                scenario, "alternate bundle was not committed") && ok;

        // Account separately for the caller-requested final synchronization;
        // it must not be confused with a host wait inside the fast path.
        ggml_backend_sched_synchronize(sched);
        ok = check(control.synchronize_calls == expected_internal_syncs + 1,
                scenario, "final scheduler synchronization count mismatch") && ok;

        cleanup();
        return ok;
    };

    return run_one("queued-multi-output-commit", 0) &&
        run_one("queued-multi-output-partial-failure", 2);
}

static bool run_qkv_composite_lane_bundle_case() {
    const char * scenario = "qkv-composite-lane-route-bundle";
    constexpr int n_lanes = 3;
    constexpr int n_outputs = 3;
    constexpr int64_t output_width = 12;
    const size_t graph_size = 128;
    const ggml_init_params params = {
        /* .mem_size   = */ 192 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backends[n_lanes] = {
        ggml_backend_cpu_init(),
        ggml_backend_cpu_init(),
        ggml_backend_cpu_init(),
    };
    ggml_backend_sched_t sched =
        backends[0] != nullptr && backends[1] != nullptr && backends[2] != nullptr
        ? ggml_backend_sched_new(backends, nullptr, n_lanes, 256, false, true)
        : nullptr;
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        for (ggml_backend_t backend : backends) {
            ggml_backend_free(backend);
        }
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output_width, 1);
    ggml_set_input(input);
    ggml_set_name(input, "qkv-route-composite-input");
    ggml_backend_sched_set_tensor_backend(sched, input, backends[0]);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);

    struct composite_variant {
        ggml_tensor * first = nullptr;
        ggml_tensor * lane_nodes[n_lanes][n_outputs] = {};
        ggml_tensor * outputs[n_outputs] = {};
    };
    struct selector_weight {
        ggml_tensor * tensor = nullptr;
        int64_t start = 0;
        int64_t size = 0;
        float factor = 0.0f;
    };
    std::vector<selector_weight> selector_weights;
    static const char * projection_names[n_outputs] = { "q", "k", "v" };
    auto build_variant = [&](const char * tag, int factor_base, const int sizes[n_lanes]) {
        composite_variant variant;
        int64_t start = 0;
        std::vector<int> active_lanes;

        // Match the production graph's backend-major ordering: one lane owns
        // Q, K, and V consecutively before graph construction moves to the
        // next processor lane.
        for (int lane = 0; lane < n_lanes; ++lane) {
            if (sizes[lane] == 0) {
                continue;
            }
            active_lanes.push_back(lane);
            for (int projection = 0; projection < n_outputs; ++projection) {
                const float factor =
                    (float) (factor_base + projection * n_lanes + lane + 1);
                ggml_tensor * weight =
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output_width, sizes[lane]);
                ggml_set_input(weight);
                ggml_set_name(weight, ("qkv-route-weight-" + std::string(tag) + "-lane" +
                        std::to_string(lane) + "-" + projection_names[projection]).c_str());
                ggml_tensor * node = ggml_mul_mat(ctx, weight, input);
                const std::string name =
                    "attn_qkv_shard." + std::string(tag) + "-lane" +
                    std::to_string(lane) + "." + projection_names[projection] +
                    ".proj-0";
                ggml_set_name(node, name.c_str());
                ggml_backend_sched_set_tensor_backend(sched, node, backends[lane]);
                ggml_build_forward_expand(graph, node);
                selector_weights.push_back({ weight, start, sizes[lane], factor });
                variant.lane_nodes[lane][projection] = node;
                if (variant.first == nullptr) {
                    variant.first = node;
                }
            }
            start += sizes[lane];
        }
        GGML_ASSERT(!active_lanes.empty() && active_lanes.size() <= 3);
        GGML_ASSERT(start == output_width);

        // Assemble Q, K, and V independently after every active lane's
        // projection operations. V is expanded last and therefore explicitly
        // delimits the route range.
        for (int projection = 0; projection < n_outputs; ++projection) {
            const std::string prefix =
                "attn_qkv_join." + std::string(projection_names[projection]) +
                "." + tag;
            ggml_tensor * output = nullptr;
            if (active_lanes.size() == 1) {
                // Production QKV single-lane routing uses the three REGION
                // matmuls directly as its terminal bundle. No identity CONCAT
                // or fake assemble-backend node belongs on this path.
                output = variant.lane_nodes[active_lanes[0]][projection];
            } else if (active_lanes.size() == 2) {
                output = ggml_concat(
                        ctx,
                        variant.lane_nodes[active_lanes[0]][projection],
                        variant.lane_nodes[active_lanes[1]][projection],
                        0);
            } else {
                ggml_tensor * inner = ggml_concat(
                        ctx,
                        variant.lane_nodes[active_lanes[1]][projection],
                        variant.lane_nodes[active_lanes[2]][projection],
                        0);
                ggml_set_name(inner, (prefix + ".concat1-0").c_str());
                ggml_backend_sched_set_tensor_backend(sched, inner, backends[0]);
                output = ggml_concat(
                        ctx,
                        variant.lane_nodes[active_lanes[0]][projection],
                        inner,
                        0);
            }
            if (active_lanes.size() == 1) {
                // Match production: the REGION tensor remains on its lane
                // backend, but its routed terminal name no longer advertises
                // an incomplete parallel branch.
                ggml_set_name(output, (prefix + ".output2d-0").c_str());
            } else {
                ggml_set_name(output, (prefix + ".concat0-0").c_str());
                ggml_backend_sched_set_tensor_backend(sched, output, backends[0]);
            }
            ggml_build_forward_expand(graph, output);
            variant.outputs[projection] = output;
        }
        return variant;
    };

    const int canonical_sizes[n_lanes] = { 4, 4, 4 };
    const int alternate_sizes[n_lanes] = { 0, 0, 12 };
    const composite_variant canonical = build_variant("canonical", 0, canonical_sizes);
    const composite_variant alternate = build_variant("alternate", 9, alternate_sizes);

    ggml_tensor * downstream[n_outputs] = {};
    for (int projection = 0; projection < n_outputs; ++projection) {
        downstream[projection] = ggml_scale(ctx, canonical.outputs[projection], 1.0f);
        const std::string name =
            "qkv-route-composite-downstream-" + std::to_string(projection);
        ggml_set_name(downstream[projection], name.c_str());
        ggml_set_output(downstream[projection]);
        ggml_backend_sched_set_tensor_backend(sched, downstream[projection], backends[0]);
        ggml_build_forward_expand(graph, downstream[projection]);
    }

    bool setup_ok =
        ggml_backend_sched_register_route_subgraph_bundle(
                sched,
                canonical.first, canonical.outputs,
                canonical.first, canonical.outputs,
                n_outputs, 0, 0) &&
        ggml_backend_sched_register_route_subgraph_bundle(
                sched,
                canonical.first, canonical.outputs,
                alternate.first, alternate.outputs,
                n_outputs, 1, 0);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 1);
    setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
    if (!check(setup_ok, scenario, "QKV composite route preparation failed")) {
        cleanup();
        return false;
    }

    route_trace_capture trace(scenario);
    if (!check(trace.start(201), scenario, "failed to start scheduler trace")) {
        cleanup();
        return false;
    }
    const float input_data[output_width] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
    };
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    for (const selector_weight & selector : selector_weights) {
        std::vector<float> values(
                (size_t) output_width * (size_t) selector.size, 0.0f);
        for (int64_t row = 0; row < selector.size; ++row) {
            values[(size_t) row * (size_t) output_width +
                   (size_t) selector.start + (size_t) row] = selector.factor;
        }
        ggml_backend_tensor_set(
                selector.tensor, values.data(), 0, values.size() * sizeof(values[0]));
    }
    bool ok = check(
            ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed");
    trace.finish();

    // All three alternate projection terminals must be committed before these
    // canonical downstream nodes execute.
    for (int projection = 0; ok && projection < n_outputs; ++projection) {
        float actual[output_width] = {};
        ggml_backend_tensor_get(
                downstream[projection], actual, 0, sizeof(actual));
        int index = 0;
        for (int lane = 0; lane < n_lanes; ++lane) {
            if (alternate_sizes[lane] == 0) {
                continue;
            }
            const float factor = (float) (9 + projection * n_lanes + lane + 1);
            for (int element = 0; element < alternate_sizes[lane]; ++element, ++index) {
                const float expected = input_data[index] * factor;
                if (!check(std::fabs(actual[index] - expected) < 1e-5f,
                            scenario, "committed Q/K/V output mismatch")) {
                    std::fprintf(stderr,
                            "  projection %s lane %d index %d: got %.6f, expected %.6f\n",
                            projection_names[projection], lane, index,
                            actual[index], expected);
                    ok = false;
                    break;
                }
            }
        }
    }

    ggml_backend_sched_route_candidate_stats stats = {};
    ggml_backend_sched_get_route_candidate_stats(sched, &stats);
    ok =
        check(stats.registered_mappings == 2, scenario, "mapping count mismatch") &&
        check(stats.prepared_groups == 1, scenario, "prepared group count mismatch") &&
        check(stats.prepared_candidate_splits == 5,
            scenario, "three-way/single-lane QKV candidate split count mismatch") &&
        check(stats.groups_executed == 1, scenario, "route group count mismatch") &&
        check(stats.canonical_selected == 0, scenario, "canonical route unexpectedly selected") &&
        check(stats.alternate_selected == 1, scenario, "alternate route was not selected") &&
        check(stats.inactive_splits_skipped == 4,
            scenario, "non-selected canonical range was not fully skipped") &&
        check(stats.canonical_commits == 1, scenario, "QKV bundle commit count mismatch") && ok;

    std::vector<route_trace_row> trace_rows;
    ok = read_route_trace(trace.path(), scenario, &trace_rows) && ok;
    std::vector<route_trace_row> selected_rows;
    bool canonical_range_was_traced = false;
    for (const route_trace_row & row : trace_rows) {
        canonical_range_was_traced = canonical_range_was_traced ||
            row.first_node.find("attn_qkv_shard.canonical-") != std::string::npos ||
            row.last_node.find("attn_qkv_shard.canonical-") != std::string::npos;
        if (row.first_node.find("attn_qkv_join.q.alternate.output2d") != std::string::npos ||
                row.last_node.find("attn_qkv_join.v.alternate.output2d") != std::string::npos) {
            selected_rows.push_back(row);
        }
    }
    ok =
        check(!canonical_range_was_traced,
            scenario, "inactive canonical lane range appeared in execution trace") &&
        check(selected_rows.size() == 1,
            scenario, "selected single-lane QKV route did not remain one scheduler split") && ok;
    if (selected_rows.size() == 1) {
        const route_trace_row & row = selected_rows.front();
        ok =
            check(!row.is_parallel_group,
                scenario, "single QKV lane entered the parallel worker path") &&
            check(row.node_count == n_outputs,
                scenario, "single-lane Q/K/V did not remain one backend-major split") &&
            check(row.first_node == "attn_qkv_join.q.alternate.output2d-0",
                scenario, "single-lane QKV split did not begin with Q") &&
            check(row.last_node == "attn_qkv_join.v.alternate.output2d-0",
                scenario, "single-lane QKV split did not end with V") &&
            check(row.first_node.find("concat") == std::string::npos &&
                    row.last_node.find("concat") == std::string::npos,
                scenario, "single-lane QKV route manufactured a CONCAT") && ok;
    }

    cleanup();
    return ok;
}

static bool run_attn_out_fork_join_subgraph_case() {
    const char * scenario = "attn-out-fork-join-subgraph";
    const size_t graph_size = 64;
    const ggml_init_params params = {
        /* .mem_size   = */ 128 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    queued_commit_control direct_scatter_control;
    ggml_backend_t backends[2] = {
        make_queued_commit_backend(&direct_scatter_control),
        ggml_backend_cpu_init(),
    };
    set_attn_out_direct_scatter_env("on");
    ggml_backend_sched_t sched = backends[0] != nullptr && backends[1] != nullptr
        ? ggml_backend_sched_new(backends, nullptr, 2, 128, false, true)
        : nullptr;
    set_attn_out_direct_scatter_env(nullptr);
    auto cleanup = [&]() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);
        ggml_backend_free(backends[1]);
        ggml_free(ctx);
    };
    if (!check(ctx != nullptr && sched != nullptr, scenario, "backend setup failed")) {
        cleanup();
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 1);
    ggml_set_input(input);
    ggml_set_name(input, "attn-out-route-input");
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);

    struct fork_join_variant {
        ggml_tensor * first = nullptr;
        ggml_tensor * output = nullptr;
    };
    struct selector_weight {
        ggml_tensor * tensor = nullptr;
        int64_t start = 0;
        int64_t size = 0;
    };
    std::vector<selector_weight> selector_weights;
    auto build_variant = [&](const int sizes[3], const char * tag,
                             ggml_backend_t lane0_backend,
                             ggml_backend_t lane1_backend) {
        static const char * lane_ids[3] = { "cpu", "gpu", "npu" };
        ggml_tensor * lanes[3] = {};
        std::vector<ggml_tensor *> active_lanes;
        int64_t start = 0;
        for (int lane = 0; lane < 3; ++lane) {
            if (sizes[lane] == 0) {
                continue;
            }
            // Mirror output-axis W_O: every projection consumes the same full
            // activation and independently produces a disjoint-width result.
            // Selector matrices make the two different partitions assemble
            // to the same canonical vector without introducing VIEW nodes,
            // which real output-axis W_O also does not contain.
            ggml_tensor * weight =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, sizes[lane]);
            ggml_set_input(weight);
            ggml_set_name(weight, (std::string("attn-out-weight-") + tag + "-" +
                    lane_ids[lane]).c_str());
            lanes[lane] = ggml_mul_mat(ctx, weight, input);
            selector_weights.push_back({ weight, start, sizes[lane] });
            const std::string name = std::string("attn_out_shard.") +
                lane_ids[lane] + "-" + tag + ".proj-0";
            ggml_set_name(lanes[lane], name.c_str());
            ggml_backend_sched_set_tensor_backend(
                    sched, lanes[lane], lane == 1 ? lane1_backend : lane0_backend);
            active_lanes.push_back(lanes[lane]);
            start += sizes[lane];
        }

        GGML_ASSERT(!active_lanes.empty() && active_lanes.size() <= 3);
        ggml_tensor * joined = active_lanes.front();
        if (active_lanes.size() >= 2) {
            joined = ggml_concat(ctx, active_lanes[0], active_lanes[1], 0);
            ggml_set_name(joined, (std::string("attn-out-inner-") + tag).c_str());
            ggml_backend_sched_set_tensor_backend(sched, joined, lane0_backend);
            if (active_lanes.size() == 3) {
                joined = ggml_concat(ctx, joined, active_lanes[2], 0);
            }
        }
        // Mirror the production single-lane path: the complete projection is
        // renamed to the canonical terminal and remains on its lane backend,
        // rather than retaining an attn_out_shard branch name or acquiring a
        // synthetic CONCAT on reduce_backend.
        ggml_set_name(joined, (std::string(active_lanes.size() == 1
                ? "attn-out-terminal-" : "attn-out-joined-") + tag).c_str());
        if (active_lanes.size() > 1) {
            ggml_backend_sched_set_tensor_backend(sched, joined, lane0_backend);
        }
        ggml_build_forward_expand(graph, joined);
        return fork_join_variant { active_lanes.front(), joined };
    };

    const int canonical_sizes[3] = { 2, 2, 2 };
    const int alternate_sizes[3] = { 0, 2, 4 };
    const int single_sizes[3] = { 0, 0, 6 };
    const fork_join_variant canonical = build_variant(
            canonical_sizes, "r0000000000000000", backends[0], backends[1]);
    const fork_join_variant alternate = build_variant(
            alternate_sizes, "r0000000000000001", backends[0], backends[1]);
    const fork_join_variant single = build_variant(
            single_sizes, "r0000000000000002", backends[1], backends[0]);
    const bool single_terminal_kept_lane_backend =
        ggml_backend_sched_get_tensor_backend(sched, single.output) == backends[1];

    ggml_tensor * output = ggml_scale(ctx, canonical.output, 1.0f);
    ggml_set_name(output, "attn-out-route-output");
    ggml_set_output(output);
    ggml_backend_sched_set_tensor_backend(sched, output, backends[0]);
    ggml_build_forward_expand(graph, output);

    bool setup_ok =
        ggml_backend_sched_register_route_subgraph(
                sched, canonical.first, canonical.output,
                canonical.first, canonical.output, 0, 0) &&
        ggml_backend_sched_register_route_subgraph(
                sched, canonical.first, canonical.output,
                alternate.first, alternate.output, 1, 0) &&
        ggml_backend_sched_register_route_subgraph(
                sched, canonical.first, canonical.output,
                single.first, single.output, 2, 0);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 1);
    setup_ok = setup_ok && ggml_backend_sched_alloc_graph(sched, graph);
    if (!check(setup_ok, scenario, "fork/join route preparation failed")) {
        cleanup();
        return false;
    }

    const float expected[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    ggml_backend_tensor_set(input, expected, 0, sizeof(expected));
    for (const auto & selector : selector_weights) {
        std::vector<float> values((size_t) 6 * (size_t) selector.size, 0.0f);
        for (int64_t row = 0; row < selector.size; ++row) {
            values[(size_t) row * 6 + (size_t) selector.start + (size_t) row] = 1.0f;
        }
        ggml_backend_tensor_set(
                selector.tensor, values.data(), 0, values.size() * sizeof(values[0]));
    }
    route_trace_capture trace(scenario);
    if (!check(trace.start(202), scenario, "failed to start scheduler trace")) {
        cleanup();
        return false;
    }
    bool ok = check(
            ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            scenario, "graph compute failed");
    trace.finish();
    float actual[6] = {};
    if (ok) {
        ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
        for (int i = 0; i < 6; ++i) {
            ok = check(std::fabs(actual[i] - expected[i]) < 1e-5f,
                    scenario, "alternate fork/join output mismatch") && ok;
        }
    }

    ggml_backend_sched_route_candidate_stats stats = {};
    ggml_backend_sched_get_route_candidate_stats(sched, &stats);
    ok =
        check(stats.groups_executed == 1, scenario, "unexpected route group count") &&
        check(stats.alternate_selected == 1, scenario, "alternate fork/join was not selected") &&
        check(stats.canonical_commits == 1, scenario, "alternate output was not committed") &&
        check(stats.plan_misses == 0, scenario, "unexpected plan miss") &&
        check(direct_scatter_control.direct_scatter_calls == 1,
            scenario, "selected W_O route did not use direct scatter") &&
        check(direct_scatter_control.direct_scatter_dst == alternate.output,
            scenario, "direct scatter wrote canonical storage before variant publication") && ok;

    std::vector<route_trace_row> trace_rows;
    ok = read_route_trace(trace.path(), scenario, &trace_rows) && ok;
    std::vector<route_trace_row> grouped_rows;
    for (const route_trace_row & row : trace_rows) {
        if (row.is_parallel_group && row.parallel_kind == "attn_out_shard") {
            grouped_rows.push_back(row);
        }
    }
    ok = check(grouped_rows.size() == 2, scenario,
            "selected two-way W_O route did not execute one two-lane group") && ok;

    // Reuse the allocated graph with the one-lane route selected. A complete
    // single projection must still commit into the canonical route output, but
    // it must not be collected as a parallel group or execute a join node.
    ggml_backend_sched_reset_route_candidate_stats(sched);
    ggml_backend_sched_layer_checkpoint_set_plan_id(sched, 2);
    route_trace_capture single_trace("attn-out-single-lane-subgraph");
    if (!check(single_trace.start(203), scenario, "failed to start single-lane scheduler trace")) {
        cleanup();
        return false;
    }
    ok = check(
            ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            scenario, "single-lane graph compute failed") && ok;
    single_trace.finish();
    if (ok) {
        ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
        for (int i = 0; i < 6; ++i) {
            ok = check(std::fabs(actual[i] - expected[i]) < 1e-5f,
                    scenario, "single-lane route output mismatch") && ok;
        }
    }
    ggml_backend_sched_get_route_candidate_stats(sched, &stats);
    ok =
        check(stats.groups_executed == 1, scenario, "unexpected single-lane route group count") &&
        check(stats.alternate_selected == 1, scenario, "single-lane route was not selected") &&
        check(stats.canonical_commits == 1, scenario, "single-lane output was not committed") &&
        check(single_terminal_kept_lane_backend,
            scenario, "single-lane terminal moved off its execution backend") && ok;

    trace_rows.clear();
    ok = read_route_trace(single_trace.path(), scenario, &trace_rows) && ok;
    const bool single_has_group = std::any_of(
            trace_rows.begin(), trace_rows.end(), [](const route_trace_row & row) {
                return row.is_parallel_group && row.parallel_kind == "attn_out_shard";
            });
    ok = check(!single_has_group, scenario,
            "single-lane W_O was incorrectly collected as a parallel group") && ok;

    cleanup();
    return ok;
}

static bool run_default_off_case() {
    const char * scenario = "default-off";
    callback_script baseline_script;
    route_fixture baseline;
    if (!check(
                baseline.build(2, &baseline_script, 0, false, false),
                scenario,
                "baseline setup failed") ||
            !check_output(baseline, 2, scenario)) {
        return false;
    }
    const int baseline_splits = ggml_backend_sched_get_n_splits(baseline.sched);

    callback_script cleared_script;
    route_fixture cleared;
    if (!check(
                cleared.build(2, &cleared_script, 0, true, true),
                scenario,
                "registered-then-cleared setup failed") ||
            !check_output(cleared, 2, scenario)) {
        return false;
    }
    const int cleared_splits = ggml_backend_sched_get_n_splits(cleared.sched);

    ggml_backend_sched_route_candidate_stats baseline_route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(baseline.sched, &baseline_route_stats);
    ggml_backend_sched_route_candidate_stats cleared_route_stats = {};
    ggml_backend_sched_get_route_candidate_stats(cleared.sched, &cleared_route_stats);
    ggml_backend_sched_layer_checkpoint_stats baseline_checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(baseline.sched, &baseline_checkpoint_stats);
    ggml_backend_sched_layer_checkpoint_stats cleared_checkpoint_stats = {};
    ggml_backend_sched_get_layer_checkpoint_stats(cleared.sched, &cleared_checkpoint_stats);

    return
        check(baseline_splits == cleared_splits, scenario, "legacy split count changed after clear") &&
        check(baseline_route_stats.graph_runs == 0, scenario, "baseline entered route compute") &&
        check(cleared_route_stats.graph_runs == 0, scenario, "cleared graph entered route compute") &&
        check(cleared_route_stats.registered_mappings == 0, scenario, "route registrations survived clear") &&
        check(baseline_checkpoint_stats.graph_runs == 0, scenario, "baseline entered checkpoint compute") &&
        check(cleared_checkpoint_stats.graph_runs == 0, scenario, "checkpoint registration survived clear") &&
        check(baseline_checkpoint_stats.producer_calls == 0, scenario, "baseline invoked request producer") &&
        check(cleared_checkpoint_stats.producer_calls == 0, scenario, "cleared producer registration survived") &&
        check(
            ggml_backend_sched_get_n_layer_checkpoint_ranges(cleared.sched) == 0,
            scenario,
            "checkpoint ranges survived clear");
}

} // namespace

int main() {
#if defined(_WIN32)
    (void) _putenv_s("SCHED_TRACE_PHASE", "both");
#else
    (void) setenv("SCHED_TRACE_PHASE", "both", 1);
#endif
    bool ok = true;
    ok = run_default_off_case() && ok;
    ok = run_two_layer_case(
        "canonical-to-alternate",
        0, 0, 1,
        1, 1, 1, 0) && ok;
    ok = run_two_layer_case(
        "alternate-to-canonical",
        1, 1, 0,
        1, 1, 1, 0) && ok;
    ok = run_two_layer_case(
        "unknown-plan-fallback",
        99, 99, 99,
        2, 0, 0, 2) && ok;
    ok = run_deferred_request_case() && ok;
    ok = run_boundary_producer_case() && ok;
    ok = run_boundary_producer_cancels_stale_request_case() && ok;
    ok = run_subgraph_layer_switch_case() && ok;
    ok = run_multi_consumer_commit_case() && ok;
    ok = run_multi_output_bundle_case() && ok;
    ok = run_explicit_range_end_bundle_case() && ok;
    ok = run_queued_multi_output_commit_case() && ok;
    ok = run_qkv_composite_lane_bundle_case() && ok;
    ok = run_attn_out_fork_join_subgraph_case() && ok;
    ok = run_alternate_terminal_lifetime_case() && ok;

    if (ok) {
        std::puts("scheduler route-candidate tests passed");
        return 0;
    }
    return 1;
}
