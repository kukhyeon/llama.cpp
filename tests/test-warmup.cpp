#include "warmup.h"

#include <cstdint>
#include <vector>

#undef NDEBUG
#include <cassert>

int main() {
    const std::vector<llama_token> seed = {11, 29};

    // Non-target contexts retain the existing BOS/EOS warmup.
    auto tokens = seed;
    common_warmup_expand_tokens(tokens, 1, 512, 512);
    assert(tokens == seed);

    // The targeted warmup reaches the existing native GEMM threshold while
    // preserving valid tokens and the default last-token-only output request.
    common_warmup_expand_tokens(tokens, 4, 512, 4);
    assert((tokens == std::vector<llama_token>{11, 29, 11, 29}));
    auto batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    assert(batch.n_tokens == 4);
    assert(batch.logits == nullptr);
    assert(batch.n_seq_id == nullptr && batch.seq_id == nullptr);

    // Either effective batch limit below four keeps the original batch: no
    // truncated extension or multiple microbatches should defeat the threshold.
    for (uint32_t limit : {0u, 1u, 2u, 3u}) {
        tokens = seed;
        common_warmup_expand_tokens(tokens, 4, 512, limit);
        assert(tokens == seed);
        common_warmup_expand_tokens(tokens, 4, limit, 512);
        assert(tokens == seed);
    }

    // Vocabulary fallbacks may provide only one valid token.
    tokens = {0};
    common_warmup_expand_tokens(tokens, 4, 4, 4);
    assert((tokens == std::vector<llama_token>{0, 0, 0, 0}));

    // A recommendation never truncates an existing batch or invents a token
    // when the caller has not supplied any valid vocabulary token.
    tokens = {1, 2, 3, 4, 5};
    const auto original = tokens;
    common_warmup_expand_tokens(tokens, 4, 512, 512);
    assert(tokens == original);
    common_warmup_expand_tokens(tokens, 0, 512, 512);
    assert(tokens == original);
    tokens.clear();
    common_warmup_expand_tokens(tokens, 4, 512, 512);
    assert(tokens.empty());
    return 0;
}
