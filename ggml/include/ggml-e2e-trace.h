#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opt-in operation timeline. All intervals use ggml_time_us()'s host clock.
// Device clocks must be mapped with an explicit error bound; elapsed backend
// API calls must never be reported as device compute intervals.
enum ggml_e2e_category {
    GGML_E2E_COMPUTE = 0,
    GGML_E2E_TRANSFORM,
    GGML_E2E_COMMUNICATION,
    GGML_E2E_SAMPLING,
    GGML_E2E_REQUEST,
    GGML_E2E_GRAPH,
    GGML_E2E_DISPATCH,
    GGML_E2E_WAIT,
    GGML_E2E_CATEGORY_COUNT,
};

enum ggml_e2e_clock_quality {
    GGML_E2E_CLOCK_HOST = 0,
    GGML_E2E_CLOCK_BOUNDED,
};

struct ggml_e2e_summary {
    int query_id;
    int64_t e2e_us;
    double model_active_us;
    double overhead_us;
    double overhead_pct;
    double overhead_lower_us;
    double overhead_upper_us;
    double category_union_us[GGML_E2E_CATEGORY_COUNT];
    double category_exposed_us[GGML_E2E_CATEGORY_COUNT];
    double unattributed_us;
    int64_t clock_error_max_us;
    uint64_t event_count;
    uint64_t missing_compute_events;
    bool complete;
    bool estimated;
};

// GGML_E2E_LATENCY_TRACE=1/on enables collection before backend construction.
// Default: off. Lifecycle calls do not synchronize any backend.
GGML_API bool ggml_e2e_trace_enabled(void);
GGML_API bool ggml_e2e_trace_active(void);
GGML_API void ggml_e2e_trace_set_enabled(bool enabled);
GGML_API void ggml_e2e_trace_set_paths(const char * summary_path, const char * events_path);
GGML_API int ggml_e2e_trace_query_id(void);
GGML_API void ggml_e2e_trace_begin_query(int query_id, int64_t start_us);
GGML_API void ggml_e2e_trace_end_query(int64_t end_us);
GGML_API void ggml_e2e_trace_flush(void);
GGML_API void ggml_e2e_trace_reset(void);
GGML_API bool ggml_e2e_trace_get_summary(struct ggml_e2e_summary * summary);
GGML_API void ggml_e2e_trace_record(
        int query_id, enum ggml_e2e_category category,
        const char * backend, const char * detail,
        int64_t start_us, int64_t end_us, int64_t clock_error_us,
        enum ggml_e2e_clock_quality quality);
GGML_API void ggml_e2e_trace_missing(int query_id, const char * backend, const char * reason);
// Register asynchronous model work before it is submitted. Uncollected work
// makes the summary incomplete instead of being silently counted as overhead.
// Registrations are counted per backend; they are not written as event rows.
GGML_API void ggml_e2e_trace_expect_compute(int query_id, const char * backend);

// Raw device-clock evidence kept with a bounded device event, so the
// host/device clock offset can be re-estimated offline across a whole query.
// Causality: dev_lo cannot precede host_lo_us and dev_hi cannot follow
// host_hi_us. Device values use the domain's native unit.
enum ggml_e2e_raw_clock_domain {
    GGML_E2E_RAW_CLOCK_NONE = 0,
    GGML_E2E_RAW_CLOCK_OPENCL_NS,   // OpenCL profiling counter, nanoseconds
    GGML_E2E_RAW_CLOCK_QTIMER,      // Hexagon QTIMER ticks
};

struct ggml_e2e_raw_clock {
    enum ggml_e2e_raw_clock_domain domain;
    int64_t  host_lo_us;
    int64_t  host_hi_us;
    uint64_t dev_lo;
    uint64_t dev_hi;
    uint64_t dev_start;
    uint64_t dev_end;
};

// Same as ggml_e2e_trace_record(..., GGML_E2E_CLOCK_BOUNDED) plus raw clock
// evidence. A NULL or inconsistent raw record is ignored, not the interval.
GGML_API void ggml_e2e_trace_record_raw(
        int query_id, enum ggml_e2e_category category,
        const char * backend, const char * detail,
        int64_t start_us, int64_t end_us, int64_t clock_error_us,
        const struct ggml_e2e_raw_clock * raw);

// Backends that defer device timestamp queries register a collector. The
// collectors run on the calling thread at ggml_e2e_trace_flush() and
// ggml_e2e_trace_get_summary(), after the request window has closed, so
// timestamp queries and clock mapping stay off the inference path.
typedef void (*ggml_e2e_trace_collector_t)(void * user_data);
GGML_API void ggml_e2e_trace_add_collector(ggml_e2e_trace_collector_t fn, void * user_data);
GGML_API void ggml_e2e_trace_remove_collector(ggml_e2e_trace_collector_t fn, void * user_data);

// CPU worker scopes support nesting (e.g. repack inside a model operation).
// Pausing around existing barriers excludes worker waiting without introducing
// new barriers. These calls use thread-local buffers and no global hot-path lock.
GGML_API void ggml_e2e_trace_thread_push(enum ggml_e2e_category category, const char * backend, const char * detail);
GGML_API void ggml_e2e_trace_thread_pop(void);
GGML_API void ggml_e2e_trace_thread_pause(void);
GGML_API void ggml_e2e_trace_thread_resume(void);

#ifdef __cplusplus
}

// Host scopes partition nested preparation/dispatch/copy work. They do not
// wait for asynchronous device work; device intervals are recorded separately.
class ggml_e2e_trace_scope {
    bool running;
public:
    ggml_e2e_trace_scope(ggml_e2e_category category, const char * backend, const char * detail)
        : running(ggml_e2e_trace_active()) {
        if (running) { ggml_e2e_trace_thread_push(category, backend, detail); }
    }
    ~ggml_e2e_trace_scope() { stop(); }
    void stop() {
        if (running) { ggml_e2e_trace_thread_pop(); running = false; }
    }
    ggml_e2e_trace_scope(const ggml_e2e_trace_scope &) = delete;
    ggml_e2e_trace_scope & operator=(const ggml_e2e_trace_scope &) = delete;
};
#endif
