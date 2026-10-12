#pragma once

#include "llama-graph.h"

#include <cstdint>

struct llama_graph_node_budget {
    uint64_t base_nodes = 0;
    uint64_t cover_nodes = 0;
    uint64_t qkv_shard_nodes = 0;
    uint64_t attn_out_shard_nodes = 0;
    uint64_t prefill_multiplier = 1;
    uint64_t weightless_nodes = 0;
    uint64_t route_nodes = 0;
};

inline uint64_t llama_graph_node_capacity(
        const llama_graph_node_budget & budget,
        const llm_graph_runtime_route_flags & routes,
        bool compact,
        bool prefill) {
    compact = compact && prefill && routes.layer;
    uint64_t nodes = budget.base_nodes;
    // A prebuilt catalog includes its initial (canonical) topology. Its
    // standalone reserve is needed only when that builder path is active.
    if (!compact || !routes.ffn) {
        nodes += budget.cover_nodes;
    }
    if (!compact || !routes.attn_qkv) {
        nodes += budget.qkv_shard_nodes;
    }
    if (!compact || !routes.attn_out) {
        nodes += budget.attn_out_shard_nodes;
    }
    if (prefill) {
        nodes = compact ? nodes + budget.weightless_nodes : nodes * budget.prefill_multiplier;
        nodes += budget.route_nodes;
    }
    return nodes;
}
