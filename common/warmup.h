#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Extend only when the requested warmup fits in one physical batch. Smaller
// contexts keep their existing warmup instead of generating a partial prefill
// whose graph would still use the small-token fallback. The caller controls
// eligibility through min_tokens; no policy or output selection is changed.
inline void common_warmup_expand_tokens(
        std::vector<llama_token> & tokens,
        uint32_t min_tokens,
        uint32_t n_batch,
        uint32_t n_ubatch) {
    if (tokens.empty() || tokens.size() >= min_tokens ||
            min_tokens > n_batch || min_tokens > n_ubatch) {
        return;
    }
    const size_t seed_count = tokens.size();
    tokens.reserve(min_tokens);
    while (tokens.size() < min_tokens) {
        const llama_token next = tokens[tokens.size() % seed_count];
        tokens.push_back(next);
    }
}
