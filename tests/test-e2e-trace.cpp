#include "ggml-e2e-trace.h"
#include "../ggml/src/ggml-hexagon/e2e-clock.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define REQUIRE(condition) do { if (!(condition)) { \
    throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #condition); \
} } while (0)

namespace {
void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) { setenv(name, value, 1); } else { unsetenv(name); }
#endif
}

void close_to(double actual, double expected) {
    REQUIRE(std::abs(actual - expected) < 0.00001);
}

void begin(int query = 1, int64_t start = 100) {
    ggml_e2e_trace_reset();
    ggml_e2e_trace_set_enabled(true);
    ggml_e2e_trace_begin_query(query, start);
    REQUIRE(ggml_e2e_trace_active());
}

void record(ggml_e2e_category category, const char * backend, int64_t start, int64_t end,
        int64_t error = 0, ggml_e2e_clock_quality quality = GGML_E2E_CLOCK_HOST) {
    ggml_e2e_trace_record(ggml_e2e_trace_query_id(), category, backend, "synthetic", start, end, error, quality);
}

ggml_e2e_summary finish(int64_t end) {
    ggml_e2e_trace_end_query(end);
    REQUIRE(!ggml_e2e_trace_active());
    ggml_e2e_summary result{};
    REQUIRE(ggml_e2e_trace_get_summary(&result));
    double exposed = result.unattributed_us;
    for (double value : result.category_exposed_us) { exposed += value; }
    close_to(exposed, double(result.e2e_us));
    return result;
}

void test_overlap_and_exclusive_partition() {
    begin();
    record(GGML_E2E_COMPUTE, "CPU", 100, 350);
    record(GGML_E2E_COMPUTE, "GPU", 250, 500);
    record(GGML_E2E_COMPUTE, "HTP", 450, 600);
    record(GGML_E2E_COMMUNICATION, "HOST", 500, 800);
    record(GGML_E2E_TRANSFORM, "CPU", 750, 900);
    record(GGML_E2E_SAMPLING, "HOST", 850, 950);
    record(GGML_E2E_REQUEST, "HOST", 100, 1000);
    const auto s = finish(1000);
    REQUIRE(s.complete && !s.estimated && s.event_count == 7);
    close_to(s.model_active_us, 500); // Backend totals would double-count overlap.
    close_to(s.overhead_us, 400);
    close_to(s.overhead_pct, 40000.0 / 900);
    close_to(s.overhead_lower_us, 400);
    close_to(s.overhead_upper_us, 400);
    close_to(s.category_union_us[GGML_E2E_COMMUNICATION], 300);
    close_to(s.category_union_us[GGML_E2E_REQUEST], 900);
    close_to(s.category_exposed_us[GGML_E2E_COMPUTE], 500);
    close_to(s.category_exposed_us[GGML_E2E_TRANSFORM], 150);
    close_to(s.category_exposed_us[GGML_E2E_COMMUNICATION], 150);
    close_to(s.category_exposed_us[GGML_E2E_SAMPLING], 50);
    close_to(s.category_exposed_us[GGML_E2E_REQUEST], 50);
    close_to(s.unattributed_us, 0);
}

void test_clipping_and_invalid_intervals() {
    begin();
    record(GGML_E2E_COMPUTE, "CPU", 50, 120);
    record(GGML_E2E_COMPUTE, "GPU", 180, 250);
    record(GGML_E2E_TRANSFORM, "CPU", 120, 180);
    record(GGML_E2E_TRANSFORM, "CPU", 180, 170); // Invalid preparation is ignored.
    record(GGML_E2E_COMPUTE, "CPU", 140, 140); // Empty work is not a missing event.
    ggml_e2e_trace_record(2, GGML_E2E_COMPUTE, "CPU", "wrong_query", 100, 200, 0, GGML_E2E_CLOCK_HOST);
    const auto s = finish(200);
    REQUIRE(s.complete && s.event_count == 3);
    close_to(s.model_active_us, 40);
    close_to(s.overhead_us, 60);
    close_to(s.category_exposed_us[GGML_E2E_TRANSFORM], 60);

    begin();
    record(GGML_E2E_COMPUTE, "CPU", 110, 130);
    record(GGML_E2E_COMPUTE, "GPU", 150, 140);
    record(GGML_E2E_COMPUTE, "HTP", -1, 140);
    record(GGML_E2E_COMPUTE, "GPU", 140, 150, std::numeric_limits<int64_t>::max());
    const auto invalid = finish(200);
    REQUIRE(!invalid.complete && invalid.missing_compute_events == 3);
    REQUIRE(std::isnan(invalid.overhead_us) && std::isnan(invalid.overhead_pct));

    begin();
    const auto empty = finish(200);
    REQUIRE(!empty.complete && empty.event_count == 0);
    close_to(empty.unattributed_us, 100);
    REQUIRE(std::isnan(empty.overhead_us));
    begin();
    const auto inverted_window = finish(90);
    REQUIRE(!inverted_window.complete && inverted_window.e2e_us == 0);
    close_to(inverted_window.unattributed_us, 0);
}

void test_clock_uncertainty() {
    begin();
    record(GGML_E2E_COMPUTE, "CPU", 110, 130);
    record(GGML_E2E_COMPUTE, "GPU", 120, 180, 10, GGML_E2E_CLOCK_BOUNDED);
    const auto s = finish(200);
    REQUIRE(s.complete && s.estimated && s.clock_error_max_us == 10);
    close_to(s.model_active_us, 70);
    close_to(s.overhead_us, 30);
    close_to(s.overhead_lower_us, 20);
    close_to(s.overhead_upper_us, 40);

    // A midpoint outside the request still has an uncertainty envelope inside it.
    begin();
    record(GGML_E2E_COMPUTE, "CPU", 150, 160);
    record(GGML_E2E_COMPUTE, "GPU", 50, 90, 30, GGML_E2E_CLOCK_BOUNDED);
    const auto outside = finish(200);
    close_to(outside.overhead_us, 90);
    close_to(outside.overhead_lower_us, 70);
    close_to(outside.overhead_upper_us, 90);

    const int64_t limit = std::numeric_limits<int64_t>::max();
    begin(1, limit - 100);
    record(GGML_E2E_COMPUTE, "GPU", limit - 90, limit - 50, 30, GGML_E2E_CLOCK_BOUNDED);
    const auto extreme = finish(limit);
    close_to(extreme.overhead_us, 60);
    close_to(extreme.overhead_lower_us, 20);
    close_to(extreme.overhead_upper_us, 100);

    int64_t offset = 0, error = 0, mapped = 0;
    REQUIRE(ggml_hexagon_e2e_clock_bounds(100, 200, 1000, 1050, &offset, &error));
    REQUIRE(offset == -875 && error == 27);
    REQUIRE(ggml_hexagon_e2e_map_timestamp(1000, offset, &mapped) && mapped == 125);
    REQUIRE(ggml_hexagon_e2e_map_timestamp(1050, offset, &mapped) && mapped == 175);
    REQUIRE(ggml_hexagon_e2e_clock_bounds(1000, 1100, 100, 150, &offset, &error));
    REQUIRE(offset == 925 && error == 27);
    REQUIRE(!ggml_hexagon_e2e_clock_bounds(200, 100, 1000, 1050, &offset, &error));
    REQUIRE(!ggml_hexagon_e2e_clock_bounds(100, 200, 1050, 1000, &offset, &error));
    REQUIRE(!ggml_hexagon_e2e_clock_bounds(100, 120, 1000, 1050, &offset, &error));
    REQUIRE(!ggml_hexagon_e2e_map_timestamp(uint64_t(limit), 1, &mapped));
    REQUIRE(!ggml_hexagon_e2e_map_timestamp(0, -1, &mapped));
    REQUIRE(!ggml_hexagon_e2e_map_timestamp(uint64_t(limit) + 1, 0, &mapped));

    // Deferred HTP footer entries map exactly like the former response path.
    ggml_hexagon_e2e_pending_op op = {};
    op.host_lo_us = 100; op.host_hi_us = 200;
    op.batch_start_qtimer = 19200; op.batch_end_qtimer = 19200 + 960;
    op.batch_start_us = 0; op.batch_end_us = 50;
    op.op_start_qtimer = 19200 + 192; op.op_end_qtimer = 19200 + 768;
    op.op_start_us = 10; op.op_end_us = 40;
    op.payload_valid = true; op.status_ok = true;
    int64_t start = 0, end = 0;
    REQUIRE(ggml_hexagon_e2e_finalize(&op, &start, &end, &error) == nullptr);
    REQUIRE(start == 135 && end == 165 && error == 27);
    auto bad = op; bad.payload_valid = false;
    REQUIRE(std::strcmp(ggml_hexagon_e2e_finalize(&bad, &start, &end, &error), "htp_timeline_footer_missing_or_invalid") == 0);
    bad = op; bad.host_hi_us = 120;
    REQUIRE(std::strcmp(ggml_hexagon_e2e_finalize(&bad, &start, &end, &error), "htp_clock_mapping_unavailable") == 0);
    bad = op; bad.status_ok = false;
    REQUIRE(std::strcmp(ggml_hexagon_e2e_finalize(&bad, &start, &end, &error), "htp_operation_timeline_incomplete") == 0);
    bad = op; bad.op_end_us = 60;
    REQUIRE(std::strcmp(ggml_hexagon_e2e_finalize(&bad, &start, &end, &error), "htp_operation_timeline_incomplete") == 0);
    bad = op; bad.op_end_qtimer = op.batch_end_qtimer + 1;
    REQUIRE(std::strcmp(ggml_hexagon_e2e_finalize(&bad, &start, &end, &error), "htp_operation_timeline_incomplete") == 0);
}

struct collector_probe { int calls = 0; };
void probe_collector(void * user_data) {
    auto * probe = static_cast<collector_probe *>(user_data);
    ++probe->calls;
    ggml_e2e_trace_record(ggml_e2e_trace_query_id(), GGML_E2E_COMPUTE, "GPU", "deferred",
            140, 180, 3, GGML_E2E_CLOCK_BOUNDED);
}

void test_deferred_collectors() {
    collector_probe probe;
    ggml_e2e_trace_add_collector(probe_collector, &probe);
    ggml_e2e_trace_add_collector(probe_collector, &probe); // Duplicate registration is ignored.
    begin();
    record(GGML_E2E_COMPUTE, "CPU", 110, 150);
    ggml_e2e_trace_expect_compute(1, "GPU");
    ggml_e2e_summary s{};
    REQUIRE(!ggml_e2e_trace_get_summary(&s));
    REQUIRE(probe.calls == 0); // An open request window never runs collectors.
    const auto done = finish(200);
    REQUIRE(probe.calls == 1 && done.complete && done.missing_compute_events == 0);
    REQUIRE(done.event_count == 3); // Two events plus one counted registration.
    close_to(done.model_active_us, 70);

    ggml_e2e_trace_remove_collector(probe_collector, &probe);
    begin();
    record(GGML_E2E_COMPUTE, "CPU", 110, 150);
    ggml_e2e_trace_expect_compute(1, "GPU");
    const auto pending = finish(200);
    REQUIRE(probe.calls == 1 && !pending.complete && pending.missing_compute_events == 1);

    // Registrations are counted per query and per backend name.
    begin();
    record(GGML_E2E_COMPUTE, "CPU", 110, 150);
    for (int i = 0; i < 3; ++i) { ggml_e2e_trace_expect_compute(1, "HTP0"); }
    ggml_e2e_trace_expect_compute(1, "OpenCL");
    for (int i = 0; i < 3; ++i) {
        ggml_e2e_trace_record(1, GGML_E2E_COMPUTE, "HTP0", "op", 120 + i, 130 + i, 1, GGML_E2E_CLOCK_BOUNDED);
    }
    ggml_e2e_trace_record(1, GGML_E2E_COMPUTE, "OpenCL", "k", 120, 125, 1, GGML_E2E_CLOCK_BOUNDED);
    const auto counted = finish(200);
    REQUIRE(counted.complete && counted.missing_compute_events == 0 && counted.event_count == 9);
    ggml_e2e_trace_begin_query(2, 300);
    record(GGML_E2E_COMPUTE, "CPU", 310, 350);
    const auto next = finish(400);
    REQUIRE(next.complete && next.event_count == 1);
}

void test_lifecycle_and_pending_work() {
    ggml_e2e_trace_reset();
    ggml_e2e_trace_set_enabled(false);
    ggml_e2e_trace_begin_query(1, 100);
    ggml_e2e_trace_record(1, GGML_E2E_COMPUTE, "CPU", "off", 100, 200, 0, GGML_E2E_CLOCK_HOST);
    ggml_e2e_trace_expect_compute(1, "GPU");
    ggml_e2e_trace_missing(1, "GPU", "off");
    ggml_e2e_trace_thread_push(GGML_E2E_COMPUTE, "CPU", "off");
    ggml_e2e_trace_thread_pause(); ggml_e2e_trace_thread_resume(); ggml_e2e_trace_thread_pop();
    ggml_e2e_trace_end_query(200);
    ggml_e2e_summary s{};
    REQUIRE(!ggml_e2e_trace_enabled() && !ggml_e2e_trace_active());
    REQUIRE(!ggml_e2e_trace_get_summary(&s));

    begin();
    record(GGML_E2E_COMPUTE, "CPU", 110, 150);
    ggml_e2e_trace_expect_compute(1, "GPU");
    auto pending = finish(200);
    REQUIRE(!pending.complete && pending.missing_compute_events == 1);
    // Device events may be collected at the existing final synchronize boundary.
    record(GGML_E2E_COMPUTE, "GPU", 140, 180);
    REQUIRE(ggml_e2e_trace_get_summary(&s) && s.complete && s.missing_compute_events == 0);
    close_to(s.model_active_us, 70);
    ggml_e2e_trace_begin_query(2, 300);
    record(GGML_E2E_COMPUTE, "CPU", 320, 360);
    const auto next = finish(400);
    REQUIRE(next.query_id == 2 && next.event_count == 1 && next.complete);
    close_to(next.model_active_us, 40);
    ggml_e2e_trace_reset();
    REQUIRE(ggml_e2e_trace_query_id() == -1 && !ggml_e2e_trace_get_summary(&s));

    begin();
    record(GGML_E2E_COMPUTE, "CPU", 110, 150);
    ggml_e2e_trace_expect_compute(1, "GPU");
    ggml_e2e_trace_expect_compute(1, "GPU");
    record(GGML_E2E_COMPUTE, "GPU", 140, 180);
    ggml_e2e_trace_missing(1, "GPU", "no_device_timestamp");
    const auto missing = finish(200);
    REQUIRE(!missing.complete && missing.missing_compute_events == 1); // No double-counting.
}

// Only trace scopes and clocks run here: no graph, tensor kernel, or model exists.
void tick() {
    const int64_t until = ggml_time_us() + 100;
    while (ggml_time_us() < until) { std::this_thread::yield(); }
}

void test_thread_scopes() {
    begin(1, ggml_time_us());
    ggml_e2e_trace_thread_push(GGML_E2E_COMPUTE, "CPU", "outer"); tick();
    ggml_e2e_trace_thread_push(GGML_E2E_TRANSFORM, "CPU", "repack"); tick();
    ggml_e2e_trace_thread_pause(); ggml_e2e_trace_thread_pause(); tick();
    ggml_e2e_trace_thread_resume(); tick(); // Still paused until the matching resume.
    ggml_e2e_trace_thread_resume(); tick();
    ggml_e2e_trace_thread_pop(); tick();
    ggml_e2e_trace_thread_pop();
    const auto s = finish(ggml_time_us());
    REQUIRE(s.complete && s.missing_compute_events == 0);
    REQUIRE(s.category_union_us[GGML_E2E_COMPUTE] >= 200);
    REQUIRE(s.category_union_us[GGML_E2E_TRANSFORM] >= 200);
    REQUIRE(s.category_union_us[GGML_E2E_WAIT] >= 200);
    close_to(s.category_exposed_us[GGML_E2E_COMPUTE], s.category_union_us[GGML_E2E_COMPUTE]);
    close_to(s.category_exposed_us[GGML_E2E_TRANSFORM], s.category_union_us[GGML_E2E_TRANSFORM]);
    close_to(s.category_exposed_us[GGML_E2E_WAIT], s.category_union_us[GGML_E2E_WAIT]);

    begin(1, ggml_time_us());
    for (int i = 0; i < 35; ++i) { ggml_e2e_trace_thread_push(GGML_E2E_COMPUTE, "CPU", "depth"); }
    tick();
    for (int i = 0; i < 35; ++i) { ggml_e2e_trace_thread_pop(); }
    ggml_e2e_trace_thread_push(GGML_E2E_COMPUTE, "CPU", "after_depth"); tick();
    ggml_e2e_trace_thread_pop();
    const auto overflow = finish(ggml_time_us());
    REQUIRE(!overflow.complete && overflow.missing_compute_events == 3);
    REQUIRE(overflow.model_active_us >= 100);

    begin();
    std::thread first([] { record(GGML_E2E_COMPUTE, "CPU", 110, 170); });
    std::thread second([] { record(GGML_E2E_COMPUTE, "CPU", 140, 190); });
    first.join(); second.join();
    const auto workers = finish(200);
    REQUIRE(workers.complete && workers.event_count == 2);
    close_to(workers.model_active_us, 80); // Two retired worker buffers, one wall union.
}

std::vector<std::string> parse_csv(const std::string & row) {
    std::vector<std::string> fields(1);
    bool quoted = false;
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i] == '"') {
            if (quoted && i + 1 < row.size() && row[i + 1] == '"') { fields.back() += '"'; ++i; }
            else { quoted = !quoted; }
        } else if (row[i] == ',' && !quoted) { fields.emplace_back(); }
        else { fields.back() += row[i]; }
    }
    REQUIRE(!quoted);
    return fields;
}

std::vector<std::vector<std::string>> read_csv(const std::filesystem::path & path) {
    std::ifstream in(path);
    REQUIRE(in.good());
    std::vector<std::vector<std::string>> result;
    std::string line;
    while (std::getline(in, line)) { result.push_back(parse_csv(line)); }
    return result;
}

void test_csv() {
    const auto stem = std::filesystem::temp_directory_path() / ("llama-e2e-test-" + std::to_string(ggml_time_us()));
    const std::filesystem::path summary_path = stem.string() + "-summary.csv";
    const std::filesystem::path events_path = stem.string() + "-events.csv";
    struct cleanup {
        std::filesystem::path a, b;
        ~cleanup() { std::error_code ignored; std::filesystem::remove(a, ignored); std::filesystem::remove(b, ignored); }
    } files{summary_path, events_path};
    REQUIRE(!std::filesystem::exists(summary_path) && !std::filesystem::exists(events_path));
    ggml_e2e_trace_set_paths(summary_path.string().c_str(), events_path.string().c_str());
    begin();
    ggml_e2e_trace_record(1, GGML_E2E_COMPUTE, "GPU,\"test\"", "kernel,\"quoted\"",
            110, 150, 5, GGML_E2E_CLOCK_BOUNDED);
    const ggml_e2e_raw_clock raw = { GGML_E2E_RAW_CLOCK_OPENCL_NS, 100, 120, 5000, 5000, 15000, 55000 };
    ggml_e2e_trace_record_raw(1, GGML_E2E_COMPUTE, "OpenCL", "kernel_raw", 115, 155, 10, &raw);
    const ggml_e2e_raw_clock inconsistent = { GGML_E2E_RAW_CLOCK_QTIMER, 120, 100, 1, 2, 3, 4 };
    ggml_e2e_trace_record_raw(1, GGML_E2E_COMPUTE, "HTP0", "bad_raw", 115, 155, 10, &inconsistent);
    ggml_e2e_trace_record(1, GGML_E2E_TRANSFORM, "0123456789012345678901234567890123456789", "x",
            160, 170, 0, GGML_E2E_CLOCK_HOST);
    finish(200);
    ggml_e2e_trace_begin_query(2, 300);
    record(GGML_E2E_COMPUTE, "CPU", 310, 350);
    ggml_e2e_trace_expect_compute(2, "GPU");
    finish(400);
    ggml_e2e_trace_flush();
    const auto summaries = read_csv(summary_path);
    const auto events = read_csv(events_path);
    REQUIRE(summaries.size() == 3 && events.size() == 6);
    REQUIRE(summaries[1][2] == "estimated" && summaries[2][2] == "incomplete");
    REQUIRE(summaries[1].size() == summaries[0].size() && summaries[2].size() == summaries[0].size());
    for (int col : {5, 6, 7, 8}) { REQUIRE(summaries[2][col].empty()); }
    REQUIRE(summaries[2][10] == "2" && summaries[2][11] == "1"); // Registration counted, not written.
    REQUIRE(events[0].size() == 16 && events[0][9] == "raw_clock" && events[0][15] == "raw_dev_end");
    for (const auto & row : events) { REQUIRE(row.size() == 16); }
    REQUIRE(events[1][0] == "2" && events[1][3] == "GPU,\"test\"" && events[1][4] == "kernel,\"quoted\"");
    REQUIRE(events[1][8] == "bounded" && events[1][9].empty());
    REQUIRE(events[2][9] == "opencl_ns" && events[2][10] == "100" && events[2][11] == "120");
    REQUIRE(events[2][12] == "5000" && events[2][13] == "5000" && events[2][14] == "15000" && events[2][15] == "55000");
    REQUIRE(events[3][4] == "bad_raw" && events[3][8] == "bounded" && events[3][9].empty()); // Interval kept.
    REQUIRE(events[4][3] == "0123456789012345678901234567890" && events[4][8] == "host");
    REQUIRE(events[5][1] == "2" && events[5][3] == "CPU");
    ggml_e2e_summary s{};
    REQUIRE(ggml_e2e_trace_query_id() == -1 && !ggml_e2e_trace_get_summary(&s));

    // A file written with another schema is never appended to.
    const std::string old_events = "schema_version,query_id,category,backend,detail,start_us,end_us,clock_error_us,clock_quality\n";
    { std::ofstream out(events_path, std::ios::trunc); out << old_events; }
    begin();
    record(GGML_E2E_COMPUTE, "CPU", 110, 150);
    finish(200);
    ggml_e2e_trace_flush();
    {
        std::ifstream in(events_path);
        const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        REQUIRE(content == old_events);
    }
    REQUIRE(read_csv(summary_path).size() == 4); // The unchanged summary schema still appends.
    ggml_e2e_trace_set_paths("", "");
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--default-off") {
            set_env("GGML_E2E_LATENCY_TRACE", nullptr);
            REQUIRE(!ggml_e2e_trace_enabled());
            std::puts("E2E trace default off: PASS");
            return 0;
        }
        REQUIRE(argc == 1);
        set_env("GGML_E2E_LATENCY_TRACE", "1");
        REQUIRE(ggml_e2e_trace_enabled());
        ggml_e2e_trace_set_paths("", "");
        test_overlap_and_exclusive_partition();
        test_clipping_and_invalid_intervals();
        test_clock_uncertainty();
        test_lifecycle_and_pending_work();
        test_deferred_collectors();
        test_thread_scopes();
        test_csv();
        ggml_e2e_trace_reset();
        ggml_e2e_trace_set_enabled(false);
        std::puts("E2E trace synthetic host tests: PASS");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "E2E trace: %s\n", error.what());
        return 1;
    }
}
