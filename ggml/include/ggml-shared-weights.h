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

// Optional single CSV for functional diagnostics. No performance timers.
GGML_API void ggml_shared_weights_log(const char * event, const struct ggml_tensor * tensor,
        const char * backend, int axis, int64_t start, int64_t size, const char * detail);

#ifdef __cplusplus
}
#endif
