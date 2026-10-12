#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

// A request must reach the DSP after host_start, and its response must reach the
// host after device_end. These causality bounds require no clock-sync RPC.
// Map each batch separately; do not assume a session-wide constant clock offset.
// This unit-slope estimate assumes both microsecond clocks advance at the same
// rate within a batch. The error covers transport/queue uncertainty and rounding,
// not independently measured clock-frequency skew (e.g. host monotonic slewing).
static inline bool ggml_hexagon_e2e_clock_bounds(
        int64_t host_start, int64_t host_end,
        uint64_t device_start, uint64_t device_end,
        int64_t * offset_us, int64_t * error_us) {
    if (host_start < 0 || host_end < host_start || device_end < device_start ||
            device_end > (uint64_t) INT64_MAX) {
        return false;
    }
    const int64_t lower = host_start - (int64_t) device_start;
    const int64_t upper = host_end - (int64_t) device_end;
    if (upper < lower) {
        return false;
    }
    // Unsigned subtraction represents the nonnegative span without signed
    // overflow, even when the allowed offset straddles zero.
    const uint64_t span = (uint64_t) upper - (uint64_t) lower;
    const uint64_t half = span / 2;
    if (half > (uint64_t) INT64_MAX - 3) {
        return false;
    }
    *offset_us = lower + (int64_t) half;
    // Include QTIMER delta conversion and integer midpoint rounding.
    *error_us = (int64_t) half + (int64_t) (span & 1U) + 2;
    return true;
}

static inline bool ggml_hexagon_e2e_map_timestamp(uint64_t device_us, int64_t offset_us, int64_t * host_us) {
    if (device_us > (uint64_t) INT64_MAX) {
        return false;
    }
    const int64_t device = (int64_t) device_us;
    if ((offset_us > 0 && device > INT64_MAX - offset_us) ||
            (offset_us < 0 && offset_us < -device)) {
        return false;
    }
    *host_us = device + offset_us;
    return true;
}

// One operation copied out of a response footer before the shared block is
// reused. Mapping runs later, after the request window has closed.
struct ggml_hexagon_e2e_pending_op {
    int64_t  host_lo_us;            // request enqueue, host clock
    int64_t  host_hi_us;            // response received, host clock
    uint64_t batch_start_qtimer;
    uint64_t batch_end_qtimer;
    uint64_t batch_start_us;        // relative to the batch start
    uint64_t batch_end_us;
    uint64_t op_start_qtimer;
    uint64_t op_end_qtimer;
    uint64_t op_start_us;           // relative to the batch start
    uint64_t op_end_us;
    bool     payload_valid;
    bool     status_ok;
};

// Maps a pending operation to the host clock. Returns NULL on success, or the
// reason the operation must be reported as missing instead.
static inline const char * ggml_hexagon_e2e_finalize(
        const struct ggml_hexagon_e2e_pending_op * op,
        int64_t * start_us, int64_t * end_us, int64_t * error_us) {
    int64_t offset_us = 0;
    int64_t error = 0;
    if (!op->payload_valid) {
        return "htp_timeline_footer_missing_or_invalid";
    }
    if (!ggml_hexagon_e2e_clock_bounds(op->host_lo_us, op->host_hi_us,
            op->batch_start_us, op->batch_end_us, &offset_us, &error)) {
        return "htp_clock_mapping_unavailable";
    }
    if (!op->status_ok ||
            op->op_start_qtimer < op->batch_start_qtimer ||
            op->op_end_qtimer < op->op_start_qtimer ||
            op->op_end_qtimer > op->batch_end_qtimer ||
            op->op_start_us < op->batch_start_us ||
            op->op_end_us < op->op_start_us || op->op_end_us > op->batch_end_us ||
            !ggml_hexagon_e2e_map_timestamp(op->op_start_us, offset_us, start_us) ||
            !ggml_hexagon_e2e_map_timestamp(op->op_end_us, offset_us, end_us)) {
        return "htp_operation_timeline_incomplete";
    }
    *error_us = error;
    return NULL;
}
