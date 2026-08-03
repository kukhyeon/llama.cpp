#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
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

static void force_trace_all_phases() {
#if defined(_WIN32)
    (void) _putenv_s("SCHED_TRACE_PHASE", "both");
#else
    (void) setenv("SCHED_TRACE_PHASE", "both", 1);
#endif
}

} // namespace

int main() {
    force_trace_all_phases();
    ggml_backend_sched_profile_set_phase(GGML_BACKEND_SCHED_PROFILE_PREFILL);

    try {
        const bool ok = run_exact_qvk_group_case() && run_incomplete_qv_case() &&
            run_exact_qkv_shard_lane_group_case() && run_incomplete_qkv_shard_lane_case();
        if (ok) {
            std::puts("scheduler parallel-group tests passed");
            return 0;
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "scheduler parallel-group test failed: %s\n", error.what());
    }
    return 1;
}
