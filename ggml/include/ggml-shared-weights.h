#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable Q8_0 rows in HTP format: K signed bytes, then K/32 fp16 scales.
// This is a physical buffer layout, not a new logical ggml quantization type.
struct ggml_shared_weight_buffer {
    void * base;
    size_t size;           // usable bytes
    size_t allocation_size; // includes external-memory padding
    int fd;
    uint64_t allocation_id;
};

GGML_API bool ggml_shared_weights_enabled(void);
GGML_API void ggml_shared_weights_register_buft(ggml_backend_buffer_type_t buft);
GGML_API void ggml_shared_weights_unregister_buft(ggml_backend_buffer_type_t buft);
GGML_API bool ggml_shared_weights_is_buft(ggml_backend_buffer_type_t buft);
GGML_API void ggml_shared_weights_register_buffer(ggml_backend_buffer_t buffer,
        void * base, size_t size, size_t allocation_size, int fd);
GGML_API void ggml_shared_weights_unregister_buffer(ggml_backend_buffer_t buffer);
GGML_API const struct ggml_shared_weight_buffer * ggml_shared_weights_buffer(ggml_backend_buffer_t buffer);
GGML_API bool ggml_shared_weights_tensor(const struct ggml_tensor * tensor);
GGML_API bool ggml_shared_weights_supports_op(const struct ggml_tensor * op);

// Imported handles belong to the allocation, and are released before rpcmem.
// The caller serializes creation for a given backend context.
GGML_API void * ggml_shared_weights_attachment(ggml_backend_buffer_t buffer, const void * key);
GGML_API void ggml_shared_weights_attach(ggml_backend_buffer_t buffer, const void * key,
        void * handle, void (*release)(void *));

// Defaults: native scratch enabled with sharing; timing enabled with a CSV path.
GGML_API bool ggml_shared_weights_repack_enabled(void);
GGML_API bool ggml_shared_weights_timing_enabled(void);

// Private backend allocations outside scheduler tensor buffers. Logical request
// peaks are separate and need not occur together. CPU weights share work_data.
struct ggml_shared_weights_scratch_stats {
    const void * owner;
    size_t weight_request_bytes;
    size_t activation_request_bytes;
    size_t request_bytes;
    size_t allocation_bytes;
};
typedef bool (*ggml_backend_get_shared_weights_scratch_stats_t)(
        ggml_backend_t backend, struct ggml_shared_weights_scratch_stats * stats);
GGML_API bool ggml_shared_weights_get_scratch_stats(
        ggml_backend_t backend, struct ggml_shared_weights_scratch_stats * stats);

struct ggml_shared_weights_repack_event {
    int query_id; // -1 for warmup or callers without query metadata
    int phase;    // ggml_backend_sched_profile_phase, -1 if unknown
    const char * node_name;
    int64_t tokens;
    int64_t k_start, k_size, out_start, out_size;
    size_t packed_bytes;
    size_t scratch_request_bytes; // per-op weights + activations + required alignment
    double repack_us; // -1 if unavailable; never substitute host submit time
    const char * timing_kind;
};
GGML_API void ggml_shared_weights_set_query_id(int query_id);
GGML_API void ggml_shared_weights_set_phase(int phase);
GGML_API struct ggml_shared_weights_repack_event ggml_shared_weights_repack_begin(
        const struct ggml_tensor * op, const char * timing_kind,
        size_t packed_bytes, size_t scratch_request_bytes);
GGML_API void ggml_shared_weights_log_repack(const struct ggml_tensor * weight,
        const char * backend, const struct ggml_shared_weights_repack_event * event);

// Optional single CSV for functional diagnostics and repack timings.
GGML_API void ggml_shared_weights_log(const char * event, const struct ggml_tensor * tensor,
        const char * backend, int axis, int64_t start, int64_t size, const char * detail);

#ifdef __cplusplus
}
#endif
