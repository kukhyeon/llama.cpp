#include "ggml-e2e-trace.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct event {
    int query;
    ggml_e2e_category category;
    int64_t start, end, error;
    bool estimated, missing;
    uint8_t raw_domain;
    int64_t raw_host_lo, raw_host_hi;
    uint64_t raw_dev_lo, raw_dev_hi, raw_dev_start, raw_dev_end;
    char backend[32];
    char detail[GGML_MAX_NAME];
};
struct buffer {
    std::mutex mutex;
    std::vector<event> events;
    std::atomic<bool> retired { false };
};
struct window { int query; int64_t start, end; };
struct expected_count { int query; char backend[32]; int64_t count; };
struct collector { ggml_e2e_trace_collector_t fn; void * user_data; };
struct state {
    std::atomic<bool> enabled { false }, active { false };
    std::atomic<int> query { -1 };
    std::atomic<uint64_t> dropped { 0 };
    std::mutex mutex;
    int64_t start = 0;
    std::vector<window> windows;
    std::vector<std::shared_ptr<buffer>> buffers;
    std::mutex expected_mutex;
    std::vector<expected_count> expected;
    std::mutex collectors_mutex;
    std::vector<collector> collectors;
    std::string summary_path = "e2e_latency.csv";
    std::string events_path = "e2e_latency_events.csv";
    state() {
        const char * opt = std::getenv("GGML_E2E_LATENCY_TRACE");
        if (opt) {
            std::string value(opt);
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : char(c);
            });
            enabled = value == "1" || value == "on" || value == "true" || value == "yes";
        }
        if (const char * p = std::getenv("GGML_E2E_LATENCY_TRACE_PATH")) { summary_path = p; }
        if (const char * p = std::getenv("GGML_E2E_LATENCY_EVENTS_PATH")) { events_path = p; }
    }
};
state & global() { static state s; return s; }
struct local_buffer {
    std::shared_ptr<buffer> ptr;
    ~local_buffer() { if (ptr) { ptr->retired = true; } }
};
thread_local local_buffer local;

// Bounded copy without printf-style parsing; truncates like snprintf("%s").
void copy_text(char * dst, size_t size, const char * src) {
    size_t n = 0;
    if (src) {
        while (n + 1 < size && src[n] != '\0') { ++n; }
        memcpy(dst, src, n);
    }
    dst[n] = '\0';
}

void append(const event & e) {
    auto & s = global();
    try {
        if (!local.ptr) {
            auto b = std::make_shared<buffer>();
            b->events.reserve(2048);
            std::lock_guard<std::mutex> lock(s.mutex);
            s.buffers.push_back(b);
            local.ptr = std::move(b);
        }
        std::lock_guard<std::mutex> lock(local.ptr->mutex);
        local.ptr->events.push_back(e);
    } catch (...) {
        // Profiling allocation failure must not turn inference into an abort,
        // or silently produce a complete-looking measurement.
        s.dropped.fetch_add(1, std::memory_order_relaxed);
    }
}

using span = std::pair<int64_t, int64_t>;
double union_length(std::vector<span> spans) {
    if (spans.empty()) { return 0; }
    std::sort(spans.begin(), spans.end());
    int64_t start = spans[0].first, end = spans[0].second;
    double total = 0;
    for (size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first > end) {
            total += double(end - start);
            start = spans[i].first;
        }
        end = std::max(end, spans[i].second);
    }
    return total + double(end - start);
}

std::vector<event> collect(int query) {
    auto & s = global();
    std::vector<event> result;
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto & b : s.buffers) {
        std::lock_guard<std::mutex> block(b->mutex);
        for (const auto & e : b->events) {
            if (e.query == query) { result.push_back(e); }
        }
    }
    return result;
}

std::vector<expected_count> expected_snapshot() {
    auto & s = global();
    std::lock_guard<std::mutex> lock(s.expected_mutex);
    return s.expected;
}

// Deferred device timelines are recorded here, after the window has closed.
void run_collectors() {
    auto & s = global();
    std::lock_guard<std::mutex> lock(s.collectors_mutex);
    for (const auto & c : s.collectors) {
        try { c.fn(c.user_data); }
        catch (...) { s.dropped.fetch_add(1, std::memory_order_relaxed); }
    }
}

ggml_e2e_summary summarize(const window & w, const std::vector<event> & events,
        const std::vector<expected_count> & expected_counts) {
    ggml_e2e_summary out{};
    out.query_id = w.query;
    out.e2e_us = std::max<int64_t>(0, w.end - w.start);
    std::array<std::vector<span>, GGML_E2E_CATEGORY_COUNT> spans;
    std::vector<span> compute_lower, compute_upper;
    struct edge { int64_t time; int category; int delta; };
    std::vector<edge> edges;
    std::unordered_map<std::string, int64_t> expected, observed, missing;
    for (const auto & item : expected_counts) {
        if (item.query != w.query) { continue; }
        expected[item.backend] += item.count;
        out.event_count += (uint64_t) item.count;
    }
    for (const auto & e : events) {
        ++out.event_count;
        if (e.missing) { ++out.missing_compute_events; ++missing[e.backend]; continue; }
        if (e.category == GGML_E2E_COMPUTE) { ++observed[e.backend]; }
        out.estimated |= e.estimated;
        out.clock_error_max_us = std::max(out.clock_error_max_us, e.error);
        if (e.category == GGML_E2E_COMPUTE) {
            // Account for uncertainty even when the midpoint lies outside the
            // query window. Its possible interval can still overlap the query.
            const int64_t lo_start = std::max(w.start,
                    e.start <= INT64_MAX - e.error ? e.start + e.error : INT64_MAX);
            const int64_t lo_end = std::min(w.end, e.end - e.error);
            if (lo_end > lo_start) { compute_lower.emplace_back(lo_start, lo_end); }
            const int64_t hi_start = std::max(w.start, e.start > e.error ? e.start - e.error : int64_t(0));
            const int64_t hi_end = std::min(w.end, e.end <= INT64_MAX - e.error ? e.end + e.error : INT64_MAX);
            if (hi_end > hi_start) { compute_upper.emplace_back(hi_start, hi_end); }
        }
        const int64_t start = std::max(w.start, e.start);
        const int64_t end = std::min(w.end, e.end);
        if (end <= start) { continue; }
        spans[e.category].emplace_back(start, end);
        edges.push_back({start, e.category, 1});
        edges.push_back({end, e.category, -1});
    }
    for (const auto & item : expected) {
        out.missing_compute_events += std::max<int64_t>(0,
                item.second - observed[item.first] - missing[item.first]);
    }
    out.missing_compute_events += global().dropped.load(std::memory_order_relaxed);
    for (int c = 0; c < GGML_E2E_CATEGORY_COUNT; ++c) {
        out.category_union_us[c] = union_length(std::move(spans[c]));
    }
    // Compute wins over concurrent host preparation/copies/waits. Remaining
    // overlapping labels use the documented enum priority, so exposed columns
    // plus unattributed always partition the request wall time.
    std::sort(edges.begin(), edges.end(), [](const edge & a, const edge & b) { return a.time < b.time; });
    std::array<int, GGML_E2E_CATEGORY_COUNT> active{};
    int64_t previous = w.start;
    for (size_t i = 0; i < edges.size();) {
        const int64_t now = edges[i].time;
        int category = 0;
        while (category < GGML_E2E_CATEGORY_COUNT && active[category] == 0) { ++category; }
        if (category < GGML_E2E_CATEGORY_COUNT) {
            out.category_exposed_us[category] += double(now - previous);
        } else {
            out.unattributed_us += double(now - previous);
        }
        while (i < edges.size() && edges[i].time == now) {
            active[edges[i].category] += edges[i].delta;
            ++i;
        }
        previous = now;
    }
    out.unattributed_us += double(w.end - previous);
    out.model_active_us = out.category_union_us[GGML_E2E_COMPUTE];
    out.complete = out.e2e_us > 0 && out.model_active_us > 0 && out.missing_compute_events == 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    out.overhead_us = out.complete ? double(out.e2e_us) - out.model_active_us : nan;
    out.overhead_pct = out.complete ? 100.0 * out.overhead_us / double(out.e2e_us) : nan;
    out.overhead_lower_us = out.complete ? double(out.e2e_us) - union_length(std::move(compute_upper)) : nan;
    out.overhead_upper_us = out.complete ? double(out.e2e_us) - union_length(std::move(compute_lower)) : nan;
    return out;
}

const char * categories[] = {"model_compute", "transform", "communication", "sampling", "request_prepare", "graph_prepare", "dispatch", "synchronization"};
const char * raw_domains[] = {"", "opencl_ns", "qtimer_tick"};

std::string summary_header() {
    std::string h = "schema_version,query_id,status,e2e_ms,model_active_ms,overhead_ms,overhead_pct,overhead_lower_ms,overhead_upper_ms,clock_error_max_us,event_count,missing_compute_events";
    for (const char * c : categories) { h += std::string(",") + c + "_union_ms," + c + "_exposed_ms"; }
    return h + ",unattributed_ms\n";
}
const char * events_header =
    "schema_version,query_id,category,backend,detail,start_us,end_us,clock_error_us,clock_quality,"
    "raw_clock,raw_host_lo_us,raw_host_hi_us,raw_dev_lo,raw_dev_hi,raw_dev_start,raw_dev_end\n";

// Rows are appended. A file written with another schema is left untouched,
// because mixing column layouts would corrupt it for every reader.
FILE * open_csv(const std::string & path, const std::string & header, bool & write_header) {
    write_header = false;
    if (path.empty()) { return nullptr; }
    FILE * f = fopen(path.c_str(), "a+");
    if (!f) { return nullptr; }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) { write_header = true; return f; }
    std::string existing(header.size(), '\0');
    rewind(f);
    const size_t n = fread(&existing[0], 1, existing.size(), f);
    fseek(f, 0, SEEK_END);
    if (n != header.size() || existing != header) {
        GGML_LOG_WARN("e2e trace: %s has a different CSV header; not appending (use a fresh output path)\n", path.c_str());
        fclose(f);
        return nullptr;
    }
    return f;
}
void csv_string(FILE * f, const char * value) {
    fputc('"', f);
    for (const char * p = value; *p; ++p) {
        if (*p == '"') { fputc('"', f); }
        fputc(*p, f);
    }
    fputc('"', f);
}

void record_impl(int query, ggml_e2e_category category, const char * backend, const char * detail,
        int64_t start, int64_t end, int64_t error, ggml_e2e_clock_quality quality, const ggml_e2e_raw_clock * raw) {
    if (!ggml_e2e_trace_enabled() || query < 0 || query != ggml_e2e_trace_query_id()) { return; }
    if (category < 0 || category >= GGML_E2E_CATEGORY_COUNT) { return; }
    if (start < 0 || end < start || error < 0 || error > INT64_MAX / 4 ||
            (quality != GGML_E2E_CLOCK_HOST && quality != GGML_E2E_CLOCK_BOUNDED)) {
        if (category == GGML_E2E_COMPUTE) { ggml_e2e_trace_missing(query, backend, "invalid_interval"); }
        return;
    }
    if (start == end) { return; }
    event e{};
    e.query = query; e.category = category; e.start = start; e.end = end; e.error = error;
    e.estimated = quality == GGML_E2E_CLOCK_BOUNDED;
    if (raw && raw->domain > GGML_E2E_RAW_CLOCK_NONE && raw->domain <= GGML_E2E_RAW_CLOCK_QTIMER &&
            raw->host_lo_us >= 0 && raw->host_lo_us <= raw->host_hi_us &&
            raw->dev_lo <= raw->dev_hi && raw->dev_start <= raw->dev_end) {
        e.raw_domain = (uint8_t) raw->domain;
        e.raw_host_lo = raw->host_lo_us; e.raw_host_hi = raw->host_hi_us;
        e.raw_dev_lo = raw->dev_lo; e.raw_dev_hi = raw->dev_hi;
        e.raw_dev_start = raw->dev_start; e.raw_dev_end = raw->dev_end;
    }
    copy_text(e.backend, sizeof(e.backend), backend);
    copy_text(e.detail, sizeof(e.detail), detail);
    append(e);
}

struct frame { ggml_e2e_category category; const char * backend; const char * detail; int query; };
struct thread_scope {
    std::array<frame, 32> frames;
    size_t depth = 0;
    unsigned pauses = 0;
    size_t overflow = 0;
    int64_t start = 0, pause_start = 0;
};
thread_local thread_scope scope;
void close_segment(int64_t now) {
    if (scope.depth == 0 || scope.pauses != 0 || scope.start == 0) { return; }
    const auto & f = scope.frames[scope.depth - 1];
    ggml_e2e_trace_record(f.query, f.category, f.backend, f.detail, scope.start, now, 0, GGML_E2E_CLOCK_HOST);
    scope.start = 0;
}
} // namespace

bool ggml_e2e_trace_enabled(void) { return global().enabled.load(std::memory_order_relaxed); }
bool ggml_e2e_trace_active(void) { return ggml_e2e_trace_enabled() && global().active.load(std::memory_order_relaxed); }
int ggml_e2e_trace_query_id(void) { return global().query.load(std::memory_order_relaxed); }
void ggml_e2e_trace_set_enabled(bool enabled) {
    global().enabled = enabled;
    if (!enabled) { global().active = false; }
}
void ggml_e2e_trace_set_paths(const char * summary, const char * events) {
    auto & s = global();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (summary) { s.summary_path = summary; }
    if (events) { s.events_path = events; }
}
void ggml_e2e_trace_begin_query(int query, int64_t start) {
    if (!ggml_e2e_trace_enabled()) { return; }
    if (query < 0 || start < 0) { return; }
    auto & s = global();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.start = start;
    s.query = query;
    s.active = true;
}
void ggml_e2e_trace_end_query(int64_t end) {
    auto & s = global();
    if (!ggml_e2e_trace_active()) { return; }
    s.active = false;
    std::lock_guard<std::mutex> lock(s.mutex);
    try { s.windows.push_back({s.query.load(), s.start, std::max(s.start, end)}); }
    catch (...) { s.dropped.fetch_add(1); }
}
void ggml_e2e_trace_record(int query, ggml_e2e_category category, const char * backend, const char * detail,
        int64_t start, int64_t end, int64_t error, ggml_e2e_clock_quality quality) {
    record_impl(query, category, backend, detail, start, end, error, quality, nullptr);
}
void ggml_e2e_trace_record_raw(int query, ggml_e2e_category category, const char * backend, const char * detail,
        int64_t start, int64_t end, int64_t error, const ggml_e2e_raw_clock * raw) {
    record_impl(query, category, backend, detail, start, end, error, GGML_E2E_CLOCK_BOUNDED, raw);
}
void ggml_e2e_trace_missing(int query, const char * backend, const char * reason) {
    if (!ggml_e2e_trace_enabled() || query < 0 || query != ggml_e2e_trace_query_id()) { return; }
    event e{};
    e.query = query; e.category = GGML_E2E_COMPUTE; e.missing = true;
    copy_text(e.backend, sizeof(e.backend), backend);
    copy_text(e.detail, sizeof(e.detail), reason ? reason : "unavailable");
    append(e);
}
void ggml_e2e_trace_expect_compute(int query, const char * backend) {
    if (!ggml_e2e_trace_active() || query < 0 || query != ggml_e2e_trace_query_id()) { return; }
    char name[sizeof(expected_count::backend)];
    copy_text(name, sizeof(name), backend);
    auto & s = global();
    try {
        std::lock_guard<std::mutex> lock(s.expected_mutex);
        for (auto & item : s.expected) {
            if (item.query == query && strcmp(item.backend, name) == 0) { ++item.count; return; }
        }
        expected_count item{};
        item.query = query;
        memcpy(item.backend, name, sizeof(name));
        item.count = 1;
        s.expected.push_back(item);
    } catch (...) {
        s.dropped.fetch_add(1, std::memory_order_relaxed);
    }
}
void ggml_e2e_trace_add_collector(ggml_e2e_trace_collector_t fn, void * user_data) {
    if (!fn) { return; }
    auto & s = global();
    std::lock_guard<std::mutex> lock(s.collectors_mutex);
    for (const auto & c : s.collectors) {
        if (c.fn == fn && c.user_data == user_data) { return; }
    }
    try { s.collectors.push_back({fn, user_data}); }
    catch (...) { GGML_LOG_WARN("e2e trace: cannot register a device timeline collector\n"); }
}
void ggml_e2e_trace_remove_collector(ggml_e2e_trace_collector_t fn, void * user_data) {
    auto & s = global();
    std::lock_guard<std::mutex> lock(s.collectors_mutex);
    s.collectors.erase(std::remove_if(s.collectors.begin(), s.collectors.end(), [&](const collector & c) {
        return c.fn == fn && c.user_data == user_data;
    }), s.collectors.end());
}
bool ggml_e2e_trace_get_summary(ggml_e2e_summary * out) {
    if (!out || !ggml_e2e_trace_enabled()) { return false; }
    try {
        // Deferred device work belongs to a closed window only.
        if (!global().active.load()) { run_collectors(); }
        window w;
        {
            std::lock_guard<std::mutex> lock(global().mutex);
            if (global().windows.empty()) { return false; }
            w = global().windows.back();
        }
        *out = summarize(w, collect(w.query), expected_snapshot());
        return true;
    } catch (...) {
        return false;
    }
}
void ggml_e2e_trace_reset(void) {
    auto & s = global();
    s.active = false;
    s.query = -1;
    s.dropped = 0;
    scope = thread_scope{};
    {
        std::lock_guard<std::mutex> lock(s.expected_mutex);
        s.expected.clear();
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    s.windows.clear();
    for (const auto & b : s.buffers) {
        std::lock_guard<std::mutex> block(b->mutex);
        b->events.clear();
    }
}
void ggml_e2e_trace_flush(void) {
    if (!ggml_e2e_trace_enabled() || global().active.load()) { return; }
    FILE * sf = nullptr;
    FILE * ef = nullptr;
    try {
        run_collectors();
        std::vector<window> windows;
        std::string summary_path, events_path;
        {
            std::lock_guard<std::mutex> lock(global().mutex);
            windows = global().windows;
            summary_path = global().summary_path; events_path = global().events_path;
        }
        if (windows.empty()) { return; }
        const std::string sheader = summary_header();
        bool sh = false, eh = false;
        sf = open_csv(summary_path, sheader, sh);
        ef = open_csv(events_path, events_header, eh);
        if (!sf && !summary_path.empty()) { GGML_LOG_WARN("e2e trace: cannot write %s\n", summary_path.c_str()); }
        if (sh) { fputs(sheader.c_str(), sf); }
        if (eh) { fputs(events_header, ef); }
        const auto expected = expected_snapshot();
        for (const auto & w : windows) {
            const auto events = collect(w.query);
            const auto out = summarize(w, events, expected);
            if (sf) {
                fprintf(sf, "1,%d,%s,%.6f,%.6f,", out.query_id,
                        !out.complete ? "incomplete" : out.estimated ? "estimated" : "measured",
                        out.e2e_us / 1000.0, out.model_active_us / 1000.0);
                if (out.complete) {
                    fprintf(sf, "%.6f,%.6f,%.6f,%.6f,", out.overhead_us / 1000.0, out.overhead_pct,
                            out.overhead_lower_us / 1000.0, out.overhead_upper_us / 1000.0);
                } else { fprintf(sf, ",,,,"); }
                fprintf(sf, "%lld,%llu,%llu", (long long) out.clock_error_max_us,
                        (unsigned long long) out.event_count, (unsigned long long) out.missing_compute_events);
                for (int c = 0; c < GGML_E2E_CATEGORY_COUNT; ++c) {
                    fprintf(sf, ",%.6f,%.6f", out.category_union_us[c] / 1000.0, out.category_exposed_us[c] / 1000.0);
                }
                fprintf(sf, ",%.6f\n", out.unattributed_us / 1000.0);
            }
            if (ef) {
                for (const auto & e : events) {
                    fprintf(ef, "2,%d,%s,", e.query, e.missing ? "missing_compute" : categories[e.category]);
                    csv_string(ef, e.backend); fputc(',', ef); csv_string(ef, e.detail);
                    fprintf(ef, ",%lld,%lld,%lld,%s", (long long) e.start, (long long) e.end,
                            (long long) e.error, e.missing ? "unavailable" : e.estimated ? "bounded" : "host");
                    if (e.raw_domain) {
                        fprintf(ef, ",%s,%lld,%lld,%llu,%llu,%llu,%llu\n", raw_domains[e.raw_domain],
                                (long long) e.raw_host_lo, (long long) e.raw_host_hi,
                                (unsigned long long) e.raw_dev_lo, (unsigned long long) e.raw_dev_hi,
                                (unsigned long long) e.raw_dev_start, (unsigned long long) e.raw_dev_end);
                    } else {
                        fputs(",,,,,,,\n", ef);
                    }
                }
            }
        }
    } catch (...) {
        // CSV aggregation is outside the measured window. A trace allocation
        // failure must still leave the inference result and control flow intact.
        GGML_LOG_WARN("e2e trace: aggregation failed; discarding this trace\n");
    }
    if (sf) { fclose(sf); }
    if (ef) { fclose(ef); }
    ggml_e2e_trace_reset();
    // Thread-local references keep live worker buffers valid; retire only
    // buffers whose producer threads have already exited.
    std::lock_guard<std::mutex> lock(global().mutex);
    auto & buffers = global().buffers;
    buffers.erase(std::remove_if(buffers.begin(), buffers.end(), [](const std::shared_ptr<buffer> & b) {
        return b->retired.load();
    }), buffers.end());
}
void ggml_e2e_trace_thread_push(ggml_e2e_category category, const char * backend, const char * detail) {
    if (!ggml_e2e_trace_active()) { return; }
    const int64_t now = ggml_time_us();
    close_segment(now);
    if (scope.depth == scope.frames.size()) {
        ++scope.overflow;
        ggml_e2e_trace_missing(ggml_e2e_trace_query_id(), backend, "scope_depth_exceeded");
        return;
    }
    scope.frames[scope.depth++] = {category, backend, detail, ggml_e2e_trace_query_id()};
    scope.start = scope.pauses == 0 ? now : 0;
}
void ggml_e2e_trace_thread_pop(void) {
    if (scope.overflow) { --scope.overflow; return; }
    if (scope.depth == 0) { return; }
    const int64_t now = ggml_time_us();
    close_segment(now);
    --scope.depth;
    scope.start = scope.depth && scope.pauses == 0 ? now : 0;
}
void ggml_e2e_trace_thread_pause(void) {
    if (scope.depth == 0) { return; }
    if (scope.pauses++ == 0) {
        // close_segment observes the pre-pause state.
        --scope.pauses;
        const int64_t now = ggml_time_us(); close_segment(now);
        ++scope.pauses;
        scope.pause_start = now;
    }
}
void ggml_e2e_trace_thread_resume(void) {
    if (scope.pauses == 0 || --scope.pauses != 0) { return; }
    const int64_t now = ggml_time_us();
    if (scope.depth) {
        const auto & f = scope.frames[scope.depth - 1];
        ggml_e2e_trace_record(f.query, GGML_E2E_WAIT, f.backend, "worker_barrier",
                scope.pause_start, now, 0, GGML_E2E_CLOCK_HOST);
        scope.start = now;
    }
}
