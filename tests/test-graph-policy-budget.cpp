#include "llama-graph-budget.h"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #condition); \
    } \
} while (0)

namespace {

llama_graph_node_budget representative_budget() {
    llama_graph_node_budget budget;
    budget.base_nodes = 2040;
    budget.cover_nodes = 336;
    budget.qkv_shard_nodes = 672;
    budget.attn_out_shard_nodes = 448;
    budget.prefill_multiplier = 4;
    budget.weightless_nodes = 56;
    budget.route_nodes = 3360 + 14810 + 112 + 896;
    return budget;
}

llm_graph_runtime_route_flags active_routes() {
    llm_graph_runtime_route_flags routes;
    routes.layer = true;
    routes.ffn = true;
    routes.weighted_norm = true;
    routes.attn_qkv = true;
    routes.attn_out = true;
    return routes;
}

void test_capacity_composition() {
    const auto budget = representative_budget();
    const auto routes = active_routes();

    // Pin each standalone reserve independently, including their combinations.
    // The production helper is also used by llama_context::graph_max_nodes.
    const std::array<uint64_t, 8> expected = {
        21274, 21610, 21946, 22282, 21722, 22058, 22394, 22730,
    };
    for (size_t mask = 0; mask < expected.size(); ++mask) {
        auto selected = routes;
        selected.ffn = (mask & 1) == 0;
        selected.attn_qkv = (mask & 2) == 0;
        selected.attn_out = (mask & 4) == 0;
        REQUIRE(llama_graph_node_capacity(budget, selected, true, true) == expected[mask]);
        REQUIRE(llama_graph_node_capacity(budget, selected, false, true) == 33162);
    }

    // No active prebuilt layer catalog must retain the complete legacy bound.
    auto inactive = routes;
    inactive.layer = false;
    REQUIRE(llama_graph_node_capacity(budget, inactive, true, true) == 33162);

    // Decode has no shard reserve in the caller and must retain the cover even
    // if a caller accidentally leaves compact/prefill route flags selected.
    auto decode = budget;
    decode.qkv_shard_nodes = 0;
    decode.attn_out_shard_nodes = 0;
    REQUIRE(llama_graph_node_capacity(decode, {}, true, false) == 2376);
    REQUIRE(llama_graph_node_capacity(decode, routes, true, false) == 2376);
    REQUIRE(llama_graph_node_capacity(decode, routes, false, false) == 2376);
    REQUIRE(llama_graph_node_capacity({}, routes, true, true) == 0);

    // Capacity depends on the current catalog, rather than a fixed tuned value.
    auto larger = budget;
    larger.route_nodes += 8192;
    REQUIRE(llama_graph_node_capacity(larger, routes, true, true) == 29466);
    REQUIRE(llama_graph_node_capacity(larger, routes, false, true) == 41354);
    larger.weightless_nodes += 28;
    REQUIRE(llama_graph_node_capacity(larger, routes, true, true) == 29494);

    // Composition remains wide until the context applies its uint32_t clamp.
    auto wide = budget;
    wide.base_nodes = std::numeric_limits<uint32_t>::max();
    wide.route_nodes = 0;
    REQUIRE(llama_graph_node_capacity(wide, routes, true, true) ==
            uint64_t(std::numeric_limits<uint32_t>::max()) + 56);
}

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

bool in_arena(const void * pointer, const std::vector<uint8_t> & arena) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    const auto first = reinterpret_cast<uintptr_t>(arena.data());
    return address >= first && address < first + arena.size();
}

void check_metadata_arena(uint64_t capacity, int node_count) {
    constexpr size_t leaf_count = 8;
    constexpr size_t temporary_count = 7;
    REQUIRE(capacity <= uint64_t(std::numeric_limits<int>::max()));
    REQUIRE(uint64_t(node_count) + temporary_count <= capacity);

    // Model weights own their tensor descriptors separately from graph
    // metadata. Neither context allocates tensor data, nor runs any operation.
    std::vector<uint8_t> weight_arena(leaf_count * ggml_tensor_overhead());
    context_ptr weights(ggml_init({weight_arena.size(), weight_arena.data(), true}), ggml_free);
    REQUIRE(weights != nullptr);
    std::array<ggml_tensor *, leaf_count> leaves;
    for (size_t i = 0; i < leaves.size(); ++i) {
        leaves[i] = ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, 4);
        ggml_set_name(leaves[i], ("weight_" + std::to_string(i)).c_str());
        REQUIRE(leaves[i]->data == nullptr);
        REQUIRE(in_arena(leaves[i], weight_arena));
    }

    const size_t allocated = size_t(capacity) * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(size_t(capacity), false);
    std::vector<uint8_t> arena(allocated);
    context_ptr ctx(ggml_init({arena.size(), arena.data(), true}), ggml_free);
    REQUIRE(ctx != nullptr);
    auto * graph = ggml_new_graph_custom(ctx.get(), size_t(capacity), false);
    REQUIRE(ggml_graph_size(graph) == int(capacity));
    REQUIRE(ggml_get_mem_buffer(ctx.get()) == arena.data());
    REQUIRE(in_arena(graph, arena));
    const size_t empty_used = ggml_used_mem(ctx.get());
    REQUIRE(empty_used == ggml_graph_overhead_custom(size_t(capacity), false));

    auto * result = leaves[0];
    for (int i = 0; i < node_count; ++i) {
        result = ggml_add(ctx.get(), result, leaves[1 + size_t(i) % (leaf_count - 1)]);
        // Expand incrementally so even a large graph does not require a deep
        // recursive visit on the host stack. The visited hash shares old nodes.
        const size_t before_expand = ggml_used_mem(ctx.get());
        ggml_build_forward_expand(graph, result);
        REQUIRE(ggml_used_mem(ctx.get()) == before_expand);
    }
    REQUIRE(ggml_graph_n_nodes(graph) == node_count);
    for (int i = 0; i < node_count; ++i) {
        REQUIRE(in_arena(ggml_graph_node(graph, i), arena));
        REQUIRE(ggml_graph_node(graph, i)->data == nullptr);
    }
    for (auto * leaf : leaves) {
        REQUIRE(!in_arena(leaf, arena));
        REQUIRE(ggml_graph_get_tensor(graph, ggml_get_name(leaf)) == leaf);
    }

    // Unreachable temporary descriptors still consume the metadata arena,
    // independently of the executed node count and external weight leaves.
    const size_t before_temporaries = ggml_used_mem(ctx.get());
    for (size_t i = 0; i < temporary_count; ++i) {
        auto * temporary = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        REQUIRE(in_arena(temporary, arena));
        REQUIRE(temporary->data == nullptr);
    }
    REQUIRE(ggml_graph_n_nodes(graph) == node_count);
    REQUIRE(ggml_used_mem(ctx.get()) == before_temporaries + temporary_count * ggml_tensor_overhead());
    REQUIRE(ggml_used_mem(ctx.get()) == empty_used +
            (size_t(node_count) + temporary_count) * ggml_tensor_overhead());
    REQUIRE(ggml_used_mem(ctx.get()) <= arena.size());
    REQUIRE(arena.size() - ggml_used_mem(ctx.get()) ==
            (size_t(capacity) - size_t(node_count) - temporary_count) * ggml_tensor_overhead());
}

void test_metadata_capacity() {
    const auto budget = representative_budget();
    const auto routes = active_routes();
    const uint64_t compact = llama_graph_node_capacity(budget, routes, true, true);
    const uint64_t legacy = llama_graph_node_capacity(budget, routes, false, true);
    check_metadata_arena(compact, 12464);
    check_metadata_arena(legacy, 12464);

    auto larger = budget;
    larger.route_nodes += 8192;
    const uint64_t grown = llama_graph_node_capacity(larger, routes, true, true);
    const int future_nodes = int(compact) + 1;
    REQUIRE(uint64_t(future_nodes) > compact);
    REQUIRE(grown > compact);
    check_metadata_arena(grown, future_nodes);
    // Unsupported compact construction uses a legacy arena, rather than the
    // preceding smaller compact reservation. These are fresh contexts; this
    // does not pretend to exercise llama_context's revision/cache integration.
    REQUIRE(legacy > compact);
    check_metadata_arena(legacy, future_nodes);
}

} // namespace

int main() {
    try {
        test_capacity_composition();
        test_metadata_capacity();
        std::puts("graph policy budget tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "graph policy budget test failed: %s\n", error.what());
        return 1;
    }
}
