#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The tests exercise the real lifetime planner without running any kernels.
struct trace_backend {
    ggml_backend_buffer_type buft = {};
    size_t max_size;
    size_t input_padding = 0;
    bool fail_alloc = false;

    explicit trace_backend(size_t max_size = 4096) : max_size(max_size) {
        buft.context = this;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "trace-test"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 8; };
        buft.iface.get_max_size = [](ggml_backend_buffer_type_t type) {
            return static_cast<trace_backend *>(type->context)->max_size;
        };
        buft.iface.get_alloc_size = [](ggml_backend_buffer_type_t type, const ggml_tensor * tensor) {
            return ggml_nbytes(tensor) + ((tensor->flags & GGML_TENSOR_FLAG_INPUT)
                ? static_cast<trace_backend *>(type->context)->input_padding : 0);
        };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) -> ggml_backend_buffer_t {
            if (static_cast<trace_backend *>(type->context)->fail_alloc) {
                return nullptr;
            }
            ggml_backend_buffer_i iface = {};
            iface.get_base = [](ggml_backend_buffer_t buffer) { return buffer->context; };
            iface.free_buffer = [](ggml_backend_buffer_t buffer) { std::free(buffer->context); };
            return ggml_backend_buffer_init(type, iface, std::malloc(std::max(size, size_t(1))), size);
        };
    }
};

struct trace_graph {
    ggml_context_ptr ctx;
    ggml_cgraph * graph;

    trace_graph() : ctx(ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true})) {
        GGML_ASSERT(ctx);
        graph = ggml_new_graph(ctx.get());
    }

    ggml_tensor * input(int64_t elements, const char * name) {
        auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, elements);
        ggml_set_name(tensor, name);
        ggml_set_input(tensor);
        return tensor;
    }

    void finish(ggml_tensor * output) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
};

struct chunk_snapshot {
    ggml_gallocr_chunk_trace trace;
    std::vector<ggml_gallocr_tensor_trace> entries;
};

static void collect(const ggml_gallocr_chunk_trace * trace, void * user_data) {
    auto & snapshots = *static_cast<std::vector<chunk_snapshot> *>(user_data);
    chunk_snapshot snapshot = {*trace, {}};
    if (trace->n_tensors) {
        snapshot.entries.assign(trace->tensors, trace->tensors + trace->n_tensors);
    }
    snapshot.trace.tensors = nullptr; // callback-owned pointer must not escape
    size_t live = 0;
    for (size_t i = 0; i < snapshot.entries.size(); ++i) {
        const auto & entry = snapshot.entries[i];
        GGML_ASSERT(entry.tensor->view_src == nullptr);
        GGML_ASSERT(entry.offset + entry.allocation_bytes <= trace->required_bytes);
        GGML_ASSERT(entry.offset % 8 == 0 && entry.allocation_bytes % 8 == 0);
        live += entry.allocation_bytes;
        for (size_t j = 0; j < i; ++j) {
            const auto & other = snapshot.entries[j];
            GGML_ASSERT(entry.offset + entry.allocation_bytes <= other.offset ||
                        other.offset + other.allocation_bytes <= entry.offset);
        }
    }
    GGML_ASSERT(live == trace->live_bytes);
    GGML_ASSERT(trace->required_bytes == trace->live_bytes + trace->gap_bytes);
    snapshots.push_back(std::move(snapshot));
}

static bool contains(const chunk_snapshot & snapshot, const ggml_tensor * tensor) {
    return std::any_of(snapshot.entries.begin(), snapshot.entries.end(),
        [tensor](const ggml_gallocr_tensor_trace & entry) { return entry.tensor == tensor; });
}

static void test_gap_and_external_storage() {
    trace_backend backend;
    trace_graph g;
    auto * a = g.input(4, "a");
    auto * b = g.input(4, "b");
    auto * c = ggml_concat(g.ctx.get(), a, b, 0);
    auto * d = ggml_concat(g.ctx.get(), c, c, 0);
    float external[16] = {};
    auto * weight = ggml_new_tensor_1d(g.ctx.get(), GGML_TYPE_F32, 16);
    auto * kv = ggml_new_tensor_1d(g.ctx.get(), GGML_TYPE_F32, 16);
    weight->data = external;
    kv->data = external;
    ggml_set_name(weight, "external_weight");
    ggml_set_name(kv, "external_kv_cache");
    auto * output = ggml_add(g.ctx.get(), ggml_add(g.ctx.get(), d, weight), kv);
    g.finish(output);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(&backend.buft));
    std::vector<chunk_snapshot> snapshots;
    ggml_gallocr_set_trace_callback(alloc.get(), collect, &snapshots);
    GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), g.graph));
    GGML_ASSERT(snapshots.size() == 1);
    const auto & peak = snapshots[0];
    GGML_ASSERT(peak.trace.required_bytes == 128);
    GGML_ASSERT(peak.trace.live_bytes == 96 && peak.trace.gap_bytes == 32);
    GGML_ASSERT(peak.trace.peak_node == d && peak.trace.peak_node_index >= 0);
    GGML_ASSERT(contains(peak, c) && contains(peak, d));
    GGML_ASSERT(!contains(peak, a) && !contains(peak, b));
    GGML_ASSERT(!contains(peak, weight) && !contains(peak, kv));
    GGML_ASSERT(ggml_gallocr_get_buffer_size(alloc.get(), 0) == 128);
    const auto generation = ggml_gallocr_get_buffer_generation(alloc.get(), 0);
    GGML_ASSERT(generation == 1);

    // Planning with tracing disabled neither emits entries nor changes sizing.
    snapshots.clear();
    ggml_gallocr_set_trace_callback(alloc.get(), nullptr, nullptr);
    GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), g.graph));
    GGML_ASSERT(snapshots.empty());
    GGML_ASSERT(ggml_gallocr_get_buffer_size(alloc.get(), 0) == 128);
    GGML_ASSERT(ggml_gallocr_get_buffer_generation(alloc.get(), 0) == generation);
}

static void test_view_has_no_storage() {
    trace_backend backend;
    trace_graph g;
    auto * input = g.input(8, "input");
    auto * view = ggml_view_1d(g.ctx.get(), input, 4, 0);
    auto * scaled = ggml_scale(g.ctx.get(), view, 2.0f);
    auto * padded = ggml_pad(g.ctx.get(), scaled, 8, 0, 0, 0);
    g.finish(padded);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(&backend.buft));
    std::vector<chunk_snapshot> snapshots;
    ggml_gallocr_set_trace_callback(alloc.get(), collect, &snapshots);
    GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), g.graph));
    GGML_ASSERT(snapshots.size() == 1);
    const auto & peak = snapshots[0];
    GGML_ASSERT(peak.trace.required_bytes == 96 && peak.trace.live_bytes == 64);
    GGML_ASSERT(peak.trace.gap_bytes == 32 && peak.entries.size() == 2);
    GGML_ASSERT(contains(peak, scaled) && contains(peak, padded));
    GGML_ASSERT(!contains(peak, input) && !contains(peak, view));
}

static void test_inplace_tail_and_owner() {
    trace_backend backend;
    // Backend allocation size can exceed logical tensor size. Reusing this
    // input for SCALE returns its extra aligned tail to the allocator.
    backend.input_padding = 16;
    trace_graph g;
    auto * input = g.input(4, "padded_input");
    auto * scaled = ggml_scale(g.ctx.get(), input, 2.0f);
    auto * padded = ggml_pad(g.ctx.get(), scaled, 8, 0, 0, 0);
    g.finish(padded);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(&backend.buft));
    std::vector<chunk_snapshot> snapshots;
    ggml_gallocr_set_trace_callback(alloc.get(), collect, &snapshots);
    GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), g.graph));
    GGML_ASSERT(snapshots.size() == 1);
    const auto & peak = snapshots[0];
    GGML_ASSERT(peak.trace.required_bytes == 64 && peak.trace.live_bytes == 64);
    GGML_ASSERT(peak.trace.gap_bytes == 0 && peak.entries.size() == 2);
    GGML_ASSERT(contains(peak, scaled) && contains(peak, padded));
    GGML_ASSERT(!contains(peak, input));
    for (const auto & entry : peak.entries) {
        if (entry.tensor == scaled) {
            GGML_ASSERT(entry.offset == 0 && entry.allocation_bytes == 16);
        }
    }
}

static void test_multichunk_shared_owner_and_generations() {
    trace_backend backend(24);
    trace_graph g;
    auto * a = g.input(3, "a"); // 12 logical bytes, 16 allocated bytes
    auto * b = g.input(2, "b");
    auto * output = ggml_concat(g.ctx.get(), a, b, 0);
    g.finish(output);
    ggml_backend_buffer_type_t types[] = {&backend.buft, &backend.buft};
    ggml_gallocr_ptr alloc(ggml_gallocr_new_n(types, 2));
    std::vector<int> node_ids(g.graph->n_nodes, 1);
    std::vector<int> leaf_ids(g.graph->n_leafs, 0);
    std::vector<chunk_snapshot> snapshots;
    ggml_gallocr_set_trace_callback(alloc.get(), collect, &snapshots);
    size_t sizes[2] = {};
    ggml_gallocr_reserve_n_size(alloc.get(), g.graph, node_ids.data(), leaf_ids.data(), sizes);
    GGML_ASSERT(ggml_gallocr_get_buffer_generation(alloc.get(), 0) == 0);
    GGML_ASSERT(ggml_gallocr_get_buffer_size(alloc.get(), 0) == 0);
    GGML_ASSERT(snapshots.size() == 2);
    size_t total = 0;
    bool aligned = false;
    for (const auto & snapshot : snapshots) {
        GGML_ASSERT(snapshot.trace.buffer_id == 0);
        total += snapshot.trace.required_bytes;
        for (const auto & entry : snapshot.entries) {
            if (entry.tensor == a) {
                GGML_ASSERT(entry.allocation_bytes == 16);
                aligned = true;
            }
        }
    }
    GGML_ASSERT(aligned && total == sizes[0] && total == 48);
    snapshots.clear();
    GGML_ASSERT(ggml_gallocr_reserve_n(alloc.get(), g.graph, node_ids.data(), leaf_ids.data()));
    GGML_ASSERT(snapshots.size() == 2);
    GGML_ASSERT(ggml_gallocr_get_buffer_size(alloc.get(), 0) == 48);
    GGML_ASSERT(ggml_gallocr_get_buffer_size(alloc.get(), 1) == 0);
    GGML_ASSERT(ggml_gallocr_get_buffer_generation(alloc.get(), 0) == 1);
    GGML_ASSERT(ggml_gallocr_get_buffer_generation(alloc.get(), 1) == 1);

    trace_backend failing;
    failing.fail_alloc = true;
    ggml_gallocr_ptr failed(ggml_gallocr_new(&failing.buft));
    GGML_ASSERT(!ggml_gallocr_reserve(failed.get(), g.graph));
    GGML_ASSERT(ggml_gallocr_get_buffer_generation(failed.get(), 0) == 1);
    GGML_ASSERT(ggml_gallocr_get_buffer_size(failed.get(), 0) == 0);
}

int main() {
    test_gap_and_external_storage();
    test_view_has_no_storage();
    test_inplace_tail_and_owner();
    test_multichunk_shared_owner_and_generations();
    std::puts("allocator trace tests passed");
    return 0;
}
