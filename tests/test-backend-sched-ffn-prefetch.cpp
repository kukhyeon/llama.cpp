#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

static bool check(bool condition, const char * scenario, const char * detail) {
    if (!condition) {
        std::fprintf(stderr, "%s: %s\n", scenario, detail);
    }
    return condition;
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
    gpu,
    npu,
};

struct fake_control {
    std::atomic<bool> npu_graph_active { false };
    std::atomic<int> gpu_synchronizes { 0 };
    std::atomic<int> npu_synchronizes { 0 };
    std::atomic<int> gpu_sync_while_npu_active { 0 };

    void reset() {
        npu_graph_active.store(false, std::memory_order_relaxed);
        gpu_synchronizes.store(0, std::memory_order_relaxed);
        npu_synchronizes.store(0, std::memory_order_relaxed);
        gpu_sync_while_npu_active.store(0, std::memory_order_relaxed);
    }
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

static ggml_backend_buffer_t fake_buft_alloc(
        ggml_backend_buffer_type_t buft, size_t size) {
    fake_backend_context * ctx = buft_ctx(buft);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ctx->inner_buft, size);
    if (buffer != nullptr) {
        // Keep the well-tested CPU allocation implementation while giving every
        // fake processor a distinct scheduler-visible buffer identity.
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
    // Make remote buffers take the ordinary device-to-host get path even
    // though this deterministic test implements them with CPU memory.
    return buft_ctx(buft)->role == fake_role::cpu;
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

static void fake_backend_free(ggml_backend_t backend) {
    fake_backend_context * ctx = backend_ctx(backend);
    ggml_backend_synchronize(ctx->inner);
    ggml_backend_free(ctx->inner);
    delete ctx->buft;
    delete ctx->device;
    delete ctx;
    delete backend;
}

static void fake_backend_synchronize(ggml_backend_t backend) {
    fake_backend_context * ctx = backend_ctx(backend);
    if (ctx->role == fake_role::gpu) {
        ctx->control->gpu_synchronizes.fetch_add(1, std::memory_order_relaxed);
        if (ctx->control->npu_graph_active.load(std::memory_order_acquire)) {
            ctx->control->gpu_sync_while_npu_active.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (ctx->role == fake_role::npu) {
        ctx->control->npu_synchronizes.fetch_add(1, std::memory_order_relaxed);
    }
    ggml_backend_synchronize(ctx->inner);
}

static enum ggml_status fake_backend_graph_compute(
        ggml_backend_t backend, ggml_cgraph * graph) {
    fake_backend_context * ctx = backend_ctx(backend);
    if (ctx->role == fake_role::gpu) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } else if (ctx->role == fake_role::npu) {
        ctx->control->npu_graph_active.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(14));
    }
    const enum ggml_status status = ggml_backend_graph_compute(ctx->inner, graph);
    if (ctx->role == fake_role::npu) {
        ctx->control->npu_graph_active.store(false, std::memory_order_release);
    }
    return status;
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
    return ggml_backend_dev_supports_op(
            ggml_backend_get_device(device_ctx(device)->inner), op);
}

static bool fake_device_supports_buft(
        ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return buft == device_ctx(device)->buft;
}

static bool fake_device_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
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
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_guid_t fake_backend_guid() {
    static ggml_guid guid = {
        0x67, 0x04, 0x7a, 0xac, 0x4c, 0x5e, 0x43, 0xf3,
        0x92, 0xd8, 0x91, 0x26, 0x45, 0x13, 0x22, 0x09,
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
        /* .reg     = */ nullptr,
        /* .context = */ ctx,
    };
    ctx->buft = new ggml_backend_buffer_type {
        /* .iface   = */ fake_buft_iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
    ctx->wrapper = new ggml_backend {
        /* .guid    = */ fake_backend_guid(),
        /* .iface   = */ fake_backend_iface,
        /* .device  = */ ctx->device,
        /* .context = */ ctx,
    };
    return ctx->wrapper;
}

class temporary_trace_file {
public:
    temporary_trace_file() {
        std::error_code ec;
        const auto directory = std::filesystem::temp_directory_path(ec);
        if (ec) {
            return;
        }
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = (directory /
                ("llama-test-ffn-prefetch-" + std::to_string(stamp) + ".csv")).string();
        std::ofstream file(path_, std::ios::binary | std::ios::trunc);
        valid_ = (bool) file;
    }

    ~temporary_trace_file() {
        std::error_code ec;
        (void) std::filesystem::remove(path_, ec);
    }

    bool valid() const { return valid_; }
    const std::string & path() const { return path_; }

private:
    std::string path_;
    bool valid_ = false;
};

struct fixture {
    static constexpr int64_t element_count = 1 << 16;

    fake_control control;
    ggml_context * ctx = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_backend_t npu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * inputs[3] = {};
    ggml_tensor * output = nullptr;

    ~fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(npu);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }

    bool build(bool parallel_copies, bool callback) {
        constexpr size_t graph_size = 32;
        const ggml_init_params params = {
            /* .mem_size   = */ 64 * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(graph_size, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        cpu = make_fake_backend(
                fake_role::cpu, "FFNPrefetchCPU", GGML_BACKEND_DEVICE_TYPE_CPU, &control);
        gpu = make_fake_backend(
                fake_role::gpu, "FFNPrefetchGPU", GGML_BACKEND_DEVICE_TYPE_GPU, &control);
        npu = make_fake_backend(
                fake_role::npu, "FFNPrefetchNPU", GGML_BACKEND_DEVICE_TYPE_ACCEL, &control);
        if (ctx == nullptr || cpu == nullptr || gpu == nullptr || npu == nullptr) {
            return false;
        }

        ggml_backend_t backends[] = { gpu, npu, cpu };
        ggml_backend_buffer_type_t bufts[] = {
            ggml_backend_get_default_buffer_type(gpu),
            ggml_backend_get_default_buffer_type(npu),
            ggml_backend_get_default_buffer_type(cpu),
        };
        sched = ggml_backend_sched_new(
                backends, bufts, 3, graph_size, parallel_copies, true);
        if (sched == nullptr) {
            return false;
        }

        graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_backend_t branch_backends[] = { cpu, gpu, npu };
        static const char * branch_names[] = { "cpu", "gpu", "npu" };
        ggml_tensor * branches[3] = {};
        for (int lane = 0; lane < 3; ++lane) {
            inputs[lane] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
            ggml_set_input(inputs[lane]);
            ggml_set_name(inputs[lane],
                    (std::string("ffn-prefetch-input-") + branch_names[lane]).c_str());
            branches[lane] = ggml_dup(ctx, inputs[lane]);
            ggml_set_name(branches[lane],
                    (std::string("ffn_down.") + branch_names[lane] + "-0").c_str());
            ggml_backend_sched_set_tensor_backend(sched, inputs[lane], branch_backends[lane]);
            ggml_backend_sched_set_tensor_backend(sched, branches[lane], branch_backends[lane]);
            ggml_build_forward_expand(graph, branches[lane]);
        }

        ggml_tensor * pair = ggml_add(ctx, branches[0], branches[1]);
        ggml_set_name(pair, "ffn_parallel_reduce-0");
        ggml_backend_sched_set_tensor_backend(sched, pair, cpu);
        output = ggml_add(ctx, pair, branches[2]);
        ggml_set_name(output, "ffn_parallel_out-0");
        ggml_set_output(output);
        ggml_backend_sched_set_tensor_backend(sched, output, cpu);
        ggml_build_forward_expand(graph, output);

        if (callback) {
            ggml_backend_sched_set_eval_callback(
                    sched,
                    [](ggml_tensor *, bool, void *) { return true; },
                    nullptr);
        }
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return false;
        }
        control.reset();
        return true;
    }

    bool compute_and_check(int query, const char * scenario) {
        std::vector<float> values[3];
        for (int lane = 0; lane < 3; ++lane) {
            values[lane].resize((size_t) element_count);
            for (int64_t i = 0; i < element_count; ++i) {
                values[lane][(size_t) i] =
                    (float) query * 0.125f + (float) lane * 1.75f + (float) (i % 29) * 0.03125f;
            }
            ggml_backend_tensor_set(
                    inputs[lane], values[lane].data(), 0,
                    values[lane].size() * sizeof(values[lane][0]));
        }

        ggml_backend_sched_trace_set_query_id(query);
        if (!check(
                    ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    scenario,
                    "graph compute failed")) {
            return false;
        }

        std::vector<float> result((size_t) element_count);
        ggml_backend_tensor_get(
                output, result.data(), 0, result.size() * sizeof(result[0]));
        for (int64_t i = 0; i < element_count; ++i) {
            const float expected = values[0][(size_t) i] + values[1][(size_t) i] +
                values[2][(size_t) i];
            if (!check(
                        std::fabs(result[(size_t) i] - expected) < 1e-5f,
                        scenario,
                        "output mismatch")) {
                std::fprintf(stderr, "  index %lld got %.7g expected %.7g\n",
                        (long long) i, result[(size_t) i], expected);
                return false;
            }
        }
        return true;
    }
};

struct trace_row {
    int query_id = -1;
    int64_t copy_us = -1;
    bool parallel_group = false;
    std::string first_node;
    std::string timing_mode;
    std::string branch;
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
    fields->push_back(std::move(field));
    return !quoted;
}

static int column_index(const std::vector<std::string> & columns, const char * name) {
    for (size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == name) {
            return (int) i;
        }
    }
    return -1;
}

static bool read_trace(
        const std::string & path,
        std::vector<trace_row> * rows,
        const char * scenario) {
    std::ifstream file(path, std::ios::binary);
    std::string line;
    std::vector<std::string> columns;
    if (!check((bool) file, scenario, "failed to open trace") ||
            !check(std::getline(file, line) && parse_csv_line(line, &columns),
                scenario, "failed to read trace header")) {
        return false;
    }
    const int query_col = column_index(columns, "query_id");
    const int copy_col = column_index(columns, "copy_in_us");
    const int group_col = column_index(columns, "is_parallel_group");
    const int first_col = column_index(columns, "first_node");
    const int mode_col = column_index(columns, "timing_mode");
    const int branch_col = column_index(columns, "parallel_branch");
    if (!check(query_col >= 0 && copy_col >= 0 && group_col >= 0 &&
                    first_col >= 0 && mode_col >= 0 && branch_col >= 0,
                scenario, "trace schema is missing required columns")) {
        return false;
    }

    rows->clear();
    std::vector<std::string> fields;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (!check(parse_csv_line(line, &fields) && fields.size() == columns.size(),
                    scenario, "malformed trace row")) {
            return false;
        }
        trace_row row;
        row.query_id = std::atoi(fields[(size_t) query_col].c_str());
        row.copy_us = std::strtoll(fields[(size_t) copy_col].c_str(), nullptr, 10);
        row.parallel_group = fields[(size_t) group_col] == "1";
        row.first_node = fields[(size_t) first_col];
        row.timing_mode = fields[(size_t) mode_col];
        row.branch = fields[(size_t) branch_col];
        rows->push_back(std::move(row));
    }
    return check(!rows->empty(), scenario, "trace contains no rows");
}

static bool check_query_trace(
        const std::vector<trace_row> & rows,
        int query_id,
        bool expect_prefetch,
        const char * scenario) {
    int group_rows = 0;
    int remote_prefetch_rows = 0;
    int reduce_rows = 0;
    for (const trace_row & row : rows) {
        if (row.query_id != query_id) {
            continue;
        }
        if (row.parallel_group) {
            ++group_rows;
            const std::string expected = expect_prefetch
                ? "persistent_prefetch_reduce"
                : "persistent_synced_wall";
            if (!check(row.timing_mode == expected, scenario,
                        "unexpected FFN group timing mode")) {
                return false;
            }
            if (expect_prefetch && (row.branch == "gpu" || row.branch == "npu")) {
                ++remote_prefetch_rows;
                if (!check(row.copy_us > 0, scenario,
                            "remote branch prefetch was not attributed to copy_in_us")) {
                    return false;
                }
            }
        }
        if (row.first_node == "ffn_parallel_reduce-0") {
            ++reduce_rows;
            if (expect_prefetch) {
                if (!check(row.timing_mode == "sync_wall_prefetched_inputs" && row.copy_us == 0,
                            scenario, "reduce split did not skip prefetched inputs")) {
                    return false;
                }
            } else if (!check(row.timing_mode == "sync_wall" && row.copy_us > 0,
                        scenario, "legacy reduce copy was not traced")) {
                return false;
            }
        }
    }
    return check(group_rows == 3, scenario, "expected three FFN group trace rows") &&
        check(reduce_rows == 1, scenario, "expected one FFN reduce trace row") &&
        check(!expect_prefetch || remote_prefetch_rows == 2,
                scenario, "expected two traced remote prefetch copies");
}

static bool check_callback_fallback_trace(
        const std::vector<trace_row> & rows,
        int query_id,
        const char * scenario) {
    int query_rows = 0;
    int reduce_rows = 0;
    for (const trace_row & row : rows) {
        if (row.query_id != query_id) {
            continue;
        }
        ++query_rows;
        if (!check(!row.parallel_group, scenario,
                    "callback fallback unexpectedly used a parallel prefetch group") ||
                !check(row.timing_mode == "callback", scenario,
                    "callback fallback trace mode changed")) {
            return false;
        }
        if (row.first_node == "ffn_parallel_reduce-0") {
            ++reduce_rows;
            if (!check(row.copy_us > 0, scenario,
                        "callback reduce did not retain its legacy input copies")) {
                return false;
            }
        }
    }
    return check(query_rows > 0, scenario, "callback fallback trace contains no rows") &&
        check(reduce_rows == 1, scenario, "callback fallback reduce row is missing");
}

static bool run_default_off_case() {
    const char * scenario = "FFN prefetch default-off";
    set_env("GGML_FFN_PREFETCH_REDUCE_INPUTS", nullptr);
    fixture test;
    if (!check(test.build(false, false), scenario, "fixture build failed") ||
            !test.compute_and_check(600, scenario)) {
        return false;
    }
    const int gpu_syncs = test.control.gpu_synchronizes.load(std::memory_order_relaxed);
    const int npu_syncs = test.control.npu_synchronizes.load(std::memory_order_relaxed);
    if (gpu_syncs != 3 || npu_syncs != 3) {
        std::fprintf(stderr, "  observed sync counts: GPU %d, NPU %d\n", gpu_syncs, npu_syncs);
    }
    return check(gpu_syncs == 3,
                scenario, "default-off GPU path did not retain legacy reduce synchronization") &&
        check(npu_syncs == 3,
                scenario, "default-off NPU path did not retain legacy reduce synchronization");
}

static bool run_enabled_reuse_case() {
    const char * scenario = "FFN prefetch delayed two-query reuse";
    set_env("GGML_FFN_PREFETCH_REDUCE_INPUTS", "1");
    fixture test;
    if (!check(test.build(false, false), scenario, "fixture build failed") ||
            !test.compute_and_check(601, scenario) ||
            !test.compute_and_check(602, scenario)) {
        return false;
    }
    return check(test.control.gpu_synchronizes.load(std::memory_order_relaxed) == 4,
                scenario, "GPU source was redundantly synchronized by the reduce split") &&
        check(test.control.npu_synchronizes.load(std::memory_order_relaxed) == 4,
                scenario, "NPU source was redundantly synchronized by the reduce split") &&
        check(test.control.gpu_sync_while_npu_active.load(std::memory_order_relaxed) >= 2,
                scenario, "delayed sibling did not overlap the early GPU completion/prefetch point");
}

static bool run_multiple_copy_slots_fallback_case() {
    const char * scenario = "FFN prefetch multiple-copy fallback";
    set_env("GGML_FFN_PREFETCH_REDUCE_INPUTS", "1");
    fixture test;
    return check(test.build(true, false), scenario, "fixture build failed") &&
        test.compute_and_check(603, scenario);
}

static bool run_callback_fallback_case() {
    const char * scenario = "FFN prefetch callback fallback";
    set_env("GGML_FFN_PREFETCH_REDUCE_INPUTS", "1");
    fixture test;
    return check(test.build(false, true), scenario, "fixture build failed") &&
        test.compute_and_check(604, scenario);
}

} // namespace

int main() {
    const char * scenario = "FFN prefetch trace";
    temporary_trace_file trace;
    if (!check(trace.valid(), scenario, "failed to create temporary trace")) {
        return 1;
    }

    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", "1");
    ggml_backend_sched_trace_set_path(trace.path().c_str());
    ggml_backend_sched_trace_set_enabled(true);
    ggml_backend_sched_trace_reset();

    bool ok = run_default_off_case() &&
        run_enabled_reuse_case() &&
        run_multiple_copy_slots_fallback_case() &&
        run_callback_fallback_case();
    ggml_backend_sched_trace_flush();

    std::vector<trace_row> rows;
    ok = ok && read_trace(trace.path(), &rows, scenario) &&
        check_query_trace(rows, 600, false, "FFN prefetch default-off trace") &&
        check_query_trace(rows, 601, true, "FFN prefetch query 1 trace") &&
        check_query_trace(rows, 602, true, "FFN prefetch query 2 trace") &&
        check_query_trace(rows, 603, false, "FFN prefetch multiple-copy trace") &&
        check_callback_fallback_trace(
                rows, 604, "FFN prefetch callback fallback trace");

    ggml_backend_sched_trace_set_enabled(false);
    set_env("GGML_FFN_PREFETCH_REDUCE_INPUTS", nullptr);
    set_env("GGML_FFN_PERSISTENT_DEVICE_WORKERS", nullptr);
    return ok ? 0 : 1;
}
