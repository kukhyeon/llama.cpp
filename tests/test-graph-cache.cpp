#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

void init_tensor_data(ggml_tensor * tensor, void *) {
    if (tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("unexpected tensor type in graph-cache fixture");
    }

    std::vector<float> data(ggml_nelements(tensor));
    for (size_t i = 0; i < data.size(); ++i) {
        // Stable, non-zero values make stale graph inputs observable while
        // keeping the tiny fixture deterministic.
        data[i] = 0.01f * std::sin(float(i + 1));
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

gguf_context_ptr make_tiny_llama_metadata() {
    constexpr uint32_t n_ctx   = 64;
    constexpr uint32_t n_vocab = 128;
    constexpr uint32_t n_embd  = 32;
    constexpr uint32_t n_head  = 4;
    constexpr uint32_t n_ff    = 64;
    constexpr uint32_t n_layer = 1;

    gguf_context_ptr metadata(gguf_init_empty());
    llama_model_saver saver(LLM_ARCH_LLAMA, metadata.get());

    saver.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_LLAMA));
    saver.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    saver.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    saver.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    saver.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    saver.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    saver.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,      n_embd / n_head);
    saver.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1.0e-5f);
    saver.add_kv(LLM_KV_TOKENIZER_MODEL,           "no_vocab");

    return metadata;
}

llama_model_ptr make_model(gguf_context * metadata) {
    llama_model_params params = llama_model_default_params();
    params.use_mmap = false;
    params.use_extra_bufts = false;
    params.progress_callback = [](float, void *) { return true; };

    llama_model_ptr model(llama_model_init_from_user(
            metadata, init_tensor_data, nullptr, params));
    if (!model) {
        throw std::runtime_error("failed to create tiny graph-cache model");
    }
    return model;
}

llama_context_ptr make_context(llama_model * model) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 64;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.no_perf = false;

    llama_context_ptr ctx(llama_init_from_model(model, params));
    if (!ctx) {
        throw std::runtime_error("failed to create graph-cache context");
    }
    return ctx;
}

std::vector<float> run_tokens(
        llama_context * ctx,
        uint32_t n_tokens,
        llama_pos pos_start,
        llama_token token_offset,
        bool clear_memory) {
    if (clear_memory) {
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    llama_batch batch = llama_batch_init(int32_t(n_tokens), 0, 1);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(
                batch,
                llama_token((token_offset + i) % 128),
                pos_start + llama_pos(i),
                { 0 },
                i + 1 == n_tokens);
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed in graph-cache test");
    }

    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        throw std::runtime_error("missing logits in graph-cache test");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    return std::vector<float>(logits, logits + n_vocab);
}

bool equal_logits(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

bool different_logits(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        return true;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    // Four slots are enough for the two prefill topologies plus D1 in this
    // test and ensure that switching away does not evict the first scheduler.
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "4");
    set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);

    try {
        gguf_context_ptr metadata = make_tiny_llama_metadata();
        llama_model_ptr model = make_model(metadata.get());
        llama_context_ptr ctx = make_context(model.get());

        llama_perf_context_reset(ctx.get());

        const auto logits_16_first = run_tokens(ctx.get(), 16, 0, 1, true);
        if (llama_perf_context(ctx.get()).n_reused != 0) {
            fprintf(stderr, "first P16 graph unexpectedly reused a cache entry\n");
            return 1;
        }

        const auto logits_d1_first = run_tokens(ctx.get(), 1, 16, 90, false);
        if (llama_perf_context(ctx.get()).n_reused != 0) {
            fprintf(stderr, "first D1 graph unexpectedly reused a cache entry\n");
            return 1;
        }

        const auto logits_d1_next = run_tokens(ctx.get(), 1, 17, 91, false);
        if (llama_perf_context(ctx.get()).n_reused != 1) {
            fprintf(stderr, "second D1 step did not reuse the active decode graph\n");
            return 1;
        }
        if (!different_logits(logits_d1_first, logits_d1_next)) {
            fprintf(stderr, "D1 graph reuse did not update position/token inputs\n");
            return 1;
        }

        const auto logits_32_first = run_tokens(ctx.get(), 32, 0, 7, true);
        if (llama_perf_context(ctx.get()).n_reused != 1) {
            fprintf(stderr, "first P32 graph unexpectedly reused a cache entry\n");
            return 1;
        }

        // Clearing KV memory creates a new independent query. The P16 graph
        // and its scheduler must still be recoverable from the inactive cache.
        const auto logits_16_replay = run_tokens(ctx.get(), 16, 0, 1, true);
        if (llama_perf_context(ctx.get()).n_reused != 2) {
            fprintf(stderr, "P16 graph was not recovered from the inactive cache\n");
            return 1;
        }
        if (!equal_logits(logits_16_first, logits_16_replay)) {
            fprintf(stderr, "P16 inactive-cache hit produced incorrect logits\n");
            return 1;
        }

        // Recreate the original post-prefill state. The D1 graph was last used
        // at position 17, so this also verifies that can_reuse/set_inputs update
        // the restored graph for position 16 rather than retaining stale state.
        const auto logits_d1_replay = run_tokens(ctx.get(), 1, 16, 90, false);
        if (llama_perf_context(ctx.get()).n_reused != 3) {
            fprintf(stderr, "D1 graph was not recovered from the inactive cache\n");
            return 1;
        }
        if (!equal_logits(logits_d1_first, logits_d1_replay)) {
            fprintf(stderr, "D1 inactive-cache hit produced incorrect logits\n");
            return 1;
        }

        const auto logits_16_updated = run_tokens(ctx.get(), 16, 0, 33, true);
        if (llama_perf_context(ctx.get()).n_reused != 4) {
            fprintf(stderr, "P16 graph was not reused after decode\n");
            return 1;
        }
        if (!different_logits(logits_16_first, logits_16_updated)) {
            fprintf(stderr, "P16 cache hit did not update its token inputs\n");
            return 1;
        }

        const auto logits_16_repeat = run_tokens(ctx.get(), 16, 0, 1, true);
        if (llama_perf_context(ctx.get()).n_reused != 5) {
            fprintf(stderr, "active P16 graph was not reused\n");
            return 1;
        }
        if (!equal_logits(logits_16_first, logits_16_repeat)) {
            fprintf(stderr, "P16 cache hit produced incorrect logits\n");
            return 1;
        }

        const auto logits_32_updated = run_tokens(ctx.get(), 32, 0, 45, true);
        if (llama_perf_context(ctx.get()).n_reused != 6) {
            fprintf(stderr, "P32 graph was not recovered from the inactive cache\n");
            return 1;
        }
        if (!different_logits(logits_32_first, logits_32_updated)) {
            fprintf(stderr, "P32 cache hit did not update its token inputs\n");
            return 1;
        }

        const auto logits_32_repeat = run_tokens(ctx.get(), 32, 0, 7, true);
        if (llama_perf_context(ctx.get()).n_reused != 7) {
            fprintf(stderr, "active P32 graph was not reused\n");
            return 1;
        }
        if (!equal_logits(logits_32_first, logits_32_repeat)) {
            fprintf(stderr, "P32 cache hit produced incorrect logits\n");
            return 1;
        }

        printf("multi-entry graph cache: P16, P32, and D1 entries reusable with updated inputs\n");
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "graph-cache test failed: %s\n", error.what());
        return 1;
    }
}
