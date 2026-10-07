#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct      ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct             ggml_backend * ggml_backend_t;

// Tensor allocator
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;
    void * base;
    size_t alignment;
    size_t offset;
};

GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
GGML_API enum ggml_status    ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);

// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   ggml_set_output(): output tensors are never freed and never overwritten

typedef struct ggml_gallocr * ggml_gallocr_t;

GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
GGML_API void           ggml_gallocr_free(ggml_gallocr_t galloc);

// Optional allocation-plan diagnostics. Each chunk reports the live storage at
// the point which established its final required span. Different chunks may
// reach that span at different graph nodes; these are not simultaneous peaks.
// Views and in-place aliases do not contribute separate storage entries.
struct ggml_gallocr_tensor_trace {
    const struct ggml_tensor * tensor;
    size_t offset;
    size_t allocation_bytes; // backend allocation size, including alignment
};

struct ggml_gallocr_chunk_trace {
    uint64_t plan_id; // allocator-local sequence, incremented for each traced plan
    int buffer_id; // first buffer using this allocator (shared types are deduplicated)
    int chunk_id;
    int peak_node_index; // -1 when established while preparing initial inputs/leafs
    const struct ggml_tensor * peak_node;
    size_t required_bytes;
    size_t live_bytes;
    size_t gap_bytes; // required_bytes - live_bytes at the recorded point
    const struct ggml_gallocr_tensor_trace * tensors;
    size_t n_tensors;
};

// Called synchronously after planning, before backend buffers are allocated.
// The trace and tensor-entry array are valid only during the callback. The
// callback must not modify the graph or recursively use this allocator.
// Passing NULL disables tracing and its tracking allocations/scans.
typedef void (*ggml_gallocr_trace_callback)(const struct ggml_gallocr_chunk_trace * trace, void * user_data);
GGML_API void ggml_gallocr_set_trace_callback(
    ggml_gallocr_t galloc, ggml_gallocr_trace_callback callback, void * user_data);

// Increments before each real buffer allocation/replacement attempt, including
// failed attempts. Sizing-only plans do not increment it. Aliased buffer types
// return the generation of their shared owner.
GGML_API uint64_t ggml_gallocr_get_buffer_generation(ggml_gallocr_t galloc, int buffer_id);

// pre-allocate buffers from a measure graph - does not allocate or modify the graph
// call with a worst-case graph to avoid buffer reallocations
// not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
// returns false if the buffer allocation failed
// ggml_gallocr_resrve_n_size writes the buffer sizes per galloc buffer that would be allocated by ggml_gallocr_reserve_n to sizes
GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
GGML_API void ggml_gallocr_reserve_n_size(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids,
    size_t * sizes);
GGML_API bool ggml_gallocr_reserve_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);

// Utils
// Create a buffer and allocate all the tensors in a ggml_context
// ggml_backend_alloc_ctx_tensors_from_buft_size returns the size of the buffer that would be allocated by ggml_backend_alloc_ctx_tensors_from_buft
GGML_API size_t                       ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
