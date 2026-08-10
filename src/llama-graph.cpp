#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-backend-policy.h"
#include "llama-batch.h"
#include "llama-cparams.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sstream>
#include <unordered_set>

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static bool tensor_is_allocated(const ggml_tensor * t) {
    return t != nullptr && t->buffer != nullptr;
}

// impl

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d   (ctx, cur, n, ggml_nelements(cur)/n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    }
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_f32_zeros::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);
    if (tensor != nullptr && !data.empty()) {
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size()*sizeof(float));
    }
}

bool llm_graph_input_f32_zeros::can_reuse(const llm_graph_params & params) {
    GGML_UNUSED(params);
    return tensor != nullptr && ggml_nbytes(tensor) == data.size()*sizeof(float);
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL)) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

static void print_mask(const float * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swq_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = data[i * n_kv + j];
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](float * data, int n_swa, llama_swa_type swa_type) {
        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f;
            }
        }
    };

    {
        GGML_ASSERT(self_kq_mask);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));

        float * data = (float *) self_kq_mask->data;

        std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);

        fill_mask(data, 0, LLAMA_SWA_TYPE_NONE);

        if (debug) {
            print_mask(data, n_tokens, n_kv, 0, LLAMA_SWA_TYPE_NONE);
        }
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));

        float * data = (float *) self_kq_mask_swa->data;

        std::fill(data, data + ggml_nelements(self_kq_mask_swa), -INFINITY);

        fill_mask(data, hparams.n_swa, hparams.swa_type);

        if (debug) {
            print_mask(data, n_tokens, n_kv, hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    if (tensor_is_allocated(self_k_idxs)) {
        mctx->set_input_k_idxs(self_k_idxs, ubatch);
    }
    if (tensor_is_allocated(self_v_idxs)) {
        mctx->set_input_v_idxs(self_v_idxs, ubatch);
    }

    if (tensor_is_allocated(self_kq_mask)) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (tensor_is_allocated(self_k_rot)) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (tensor_is_allocated(self_v_rot)) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    if (tensor_is_allocated(self_k_idxs)) {
        mctx->set_input_k_idxs(self_k_idxs, ubatch);
    }

    if (tensor_is_allocated(self_kq_mask)) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);

    mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);

    mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);

    if (self_k_rot) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask,     mctx->get_base(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(),  params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    float * data = (float *) cross_kq_mask->data;

    for (int i = 0; i < n_tokens; ++i) {
        GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
        for (int j = 0; j < n_enc; ++j) {
            float f = -INFINITY;

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = ubatch->seq_id[i][s];

                if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                    f = 0.0f;
                }
            }

            data[i*n_enc + j] = f;
        }
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    if (inp_attn->self_k_rot) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);

        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);
    }

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs() {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    for (auto & [seq_id, t] : t_sampled) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_probs) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_logits) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_candidates) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    runtime_route_candidates(params.runtime_route_candidates),
    runtime_ffn_profile(params.runtime_ffn_profile),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    model            (params.model),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    samplers         (params.samplers),
    cb_func          (params.cb),
    route_subgraph_cb_func(params.route_subgraph_cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il, const char * backend_hint) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il, backend_hint);
    }
}

void llm_graph_context::cb_route(
        ggml_tensor * cur, const char * name, int il, const char * backend_hint) const {
    cb(cur, name, il, backend_hint);
    if (sched == nullptr || backend_hint == nullptr || backend_hint[0] == '\0') {
        GGML_ABORT("runtime route %s layer %d has no concrete backend", name, il);
    }

    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, cur);
    if (backend == nullptr ||
            !llama_backend_policy_backend_matches(backend, backend_hint) ||
            !ggml_backend_supports_op(backend, cur)) {
        GGML_ABORT(
                "runtime route %s layer %d failed to bind requested backend %s",
                name, il, backend_hint);
    }
}

ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il,
                     bool   parallel_projection_names) const {
    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    const bool mark_parallel_projections = parallel_projection_names && layer.wqkv == nullptr;
    auto mark_projection_internals = [&](ggml_tensor * root, const char * branch) {
        if (!mark_parallel_projections) {
            return;
        }

        int internal_index = 0;
        std::unordered_set<ggml_tensor *> visited;
        auto visit = [&](auto && self, ggml_tensor * node) -> void {
            if (node == nullptr || node == root || node == cur || node->op == GGML_OP_NONE ||
                    !visited.insert(node).second) {
                return;
            }
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                self(self, node->src[i]);
            }

            char name[GGML_MAX_NAME];
            snprintf(name, sizeof(name), "attn_qkv.%s.internal%d", branch, internal_index++);
            cb(node, name, il);
        };
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            visit(visit, root->src[i]);
        }
    };

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        mark_projection_internals(Qcur, "q");
        cb(Qcur, mark_parallel_projections ? "attn_qkv.q.proj" : "Qcur", il);
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            cb(Qcur, mark_parallel_projections ? "attn_qkv.q.bias" : "Qcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, mark_parallel_projections ? "attn_qkv.q.clamp" : "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        mark_projection_internals(Kcur, "k");
        cb(Kcur, mark_parallel_projections ? "attn_qkv.k.proj" : "Kcur", il);
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            cb(Kcur, mark_parallel_projections ? "attn_qkv.k.bias" : "Kcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, mark_parallel_projections ? "attn_qkv.k.clamp" : "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        mark_projection_internals(Vcur, "v");
        cb(Vcur, mark_parallel_projections ? "attn_qkv.v.proj" : "Vcur", il);
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            cb(Vcur, mark_parallel_projections ? "attn_qkv.v.bias" : "Vcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, mark_parallel_projections ? "attn_qkv.v.clamp" : "Vcur_clamped", il);
        }
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    }

    cb(Qcur, mark_parallel_projections ? "attn_qkv.q.reshape" : "Qcur", il);
    cb(Kcur, mark_parallel_projections ? "attn_qkv.k.reshape" : "Kcur", il);
    cb(Vcur, mark_parallel_projections ? "attn_qkv.v.reshape" : "Vcur", il);

    return { Qcur, Kcur, Vcur };
}

bool llm_graph_context::build_qkv_shards(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il,
            llm_graph_qkv & out) const {
    out = {};

    llama_backend_policy_attn_qkv_shards policy;
    const bool is_prefill = n_tokens > 1;
    if (!llama_backend_policy_match_attn_qkv_shards(il, is_prefill, policy)) {
        return false;
    }

    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;
    const bool eligible =
        model != nullptr &&
        layer.wqkv == nullptr &&
        layer.wq != nullptr && layer.wk != nullptr && layer.wv != nullptr &&
        (loras == nullptr || loras->empty()) &&
        policy.head_dim == n_embd_head &&
        layer.wq->ne[0] == cur->ne[0] && layer.wq->ne[1] == n_embd_q &&
        layer.wk->ne[0] == cur->ne[0] && layer.wk->ne[1] == n_embd_kv &&
        layer.wv->ne[0] == cur->ne[0] && layer.wv->ne[1] == n_embd_kv;
    if (!eligible) {
        LLAMA_LOG_WARN(
                "backend_policy: attn_qkv_shards layer %d is not eligible "
                "(separate Q/K/V, no LoRA, matching head/output dimensions required); falling back\n",
                il);
        return false;
    }

    struct qkv_lane {
        llama_backend_policy_attn_qkv_shard split;
        ggml_tensor * wq = nullptr;
        ggml_tensor * wk = nullptr;
        ggml_tensor * wv = nullptr;
        ggml_tensor * q = nullptr;
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
    };

    std::vector<qkv_lane> lanes;
    lanes.reserve(policy.splits.size());
    for (const auto & split : policy.splits) {
        qkv_lane lane;
        lane.split = split;
        const std::vector<std::string> backend = { split.backend };
        lane.wq = model->get_backend_policy_residency_tensor(layer.wq, backend);
        lane.wk = model->get_backend_policy_residency_tensor(layer.wk, backend);
        lane.wv = model->get_backend_policy_residency_tensor(layer.wv, backend);
        if (lane.wq == nullptr || lane.wk == nullptr || lane.wv == nullptr) {
            LLAMA_LOG_WARN(
                    "backend_policy: attn_qkv_shards layer %d lane %s is missing resident Q/K/V "
                    "weights for %s; falling back\n",
                    il, split.id.c_str(), split.backend.c_str());
            return false;
        }
        LLAMA_LOG_DEBUG(
                "backend_policy: attn_qkv_shards layer %d lane %s backend %s "
                "q=%" PRId64 "+%" PRId64 " k=%" PRId64 "+%" PRId64
                " v=%" PRId64 "+%" PRId64 "\n",
                il, split.id.c_str(), split.backend.c_str(),
                split.q_start, split.q_size,
                split.k_start, split.k_size,
                split.v_start, split.v_size);
        lanes.push_back(std::move(lane));
    }

    auto validate_projection = [&](char projection, int64_t expected) {
        int64_t covered = 0;
        for (const auto & lane : lanes) {
            int64_t start = 0;
            int64_t size = 0;
            switch (projection) {
                case 'q': start = lane.split.q_start; size = lane.split.q_size; break;
                case 'k': start = lane.split.k_start; size = lane.split.k_size; break;
                case 'v': start = lane.split.v_start; size = lane.split.v_size; break;
                default: GGML_ABORT("invalid QKV projection selector");
            }
            if (start != covered || size <= 0 || start % n_embd_head != 0 || size % n_embd_head != 0) {
                return false;
            }
            covered += size;
        }
        return covered == expected;
    };

    if (lanes.size() != 3 ||
            !validate_projection('q', n_embd_q) ||
            !validate_projection('k', n_embd_kv) ||
            !validate_projection('v', n_embd_kv)) {
        LLAMA_LOG_WARN(
                "backend_policy: attn_qkv_shards layer %d does not exactly cover Q=%" PRId64
                ", K/V=%" PRId64 " on head boundaries; falling back\n",
                il, n_embd_q, n_embd_kv);
        return false;
    }

    auto make_shard = [&](qkv_lane & lane, char projection) {
        ggml_tensor * weight = nullptr;
        int64_t start = 0;
        int64_t size = 0;
        switch (projection) {
            case 'q': weight = lane.wq; start = lane.split.q_start; size = lane.split.q_size; break;
            case 'k': weight = lane.wk; start = lane.split.k_start; size = lane.split.k_size; break;
            case 'v': weight = lane.wv; start = lane.split.v_start; size = lane.split.v_size; break;
            default: GGML_ABORT("invalid QKV projection selector");
        }

        ggml_tensor * shard = ggml_mul_mat_src0_region(
                ctx0, weight, cur,
                0, weight->ne[0], start, size);
        const std::string name = std::string("attn_qkv_shard.") + lane.split.id + "." +
            projection + ".proj";
        cb_route(shard, name.c_str(), il, lane.split.backend.c_str());
        return shard;
    };

    // Emit all three projection shards for one backend before moving to the
    // next lane.  The scheduler therefore sees exactly three pure composite
    // branch splits instead of nine competing calls into three contexts.
    std::unordered_set<ggml_backend_t> lane_backends;
    for (auto & lane : lanes) {
        lane.q = make_shard(lane, 'q');
        ggml_backend_t lane_backend = sched != nullptr
            ? ggml_backend_sched_get_tensor_backend(sched, lane.q)
            : nullptr;
        if (lane_backend == nullptr || !lane_backends.insert(lane_backend).second) {
            GGML_ABORT(
                    "backend_policy: attn_qkv_shards layer %d lane %s did not resolve to a unique backend",
                    il, lane.split.id.c_str());
        }
        lane.k = make_shard(lane, 'k');
        lane.v = make_shard(lane, 'v');
        ggml_build_forward_expand(gf, lane.q);
        ggml_build_forward_expand(gf, lane.k);
        ggml_build_forward_expand(gf, lane.v);
    }

    auto assemble = [&](char projection) {
        std::vector<std::pair<int64_t, ggml_tensor *>> parts;
        parts.reserve(lanes.size());
        for (const auto & lane : lanes) {
            switch (projection) {
                case 'q': parts.emplace_back(lane.split.q_start, lane.q); break;
                case 'k': parts.emplace_back(lane.split.k_start, lane.k); break;
                case 'v': parts.emplace_back(lane.split.v_start, lane.v); break;
                default: GGML_ABORT("invalid QKV projection selector");
            }
        }
        std::sort(parts.begin(), parts.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.first < rhs.first;
        });

        // Right-associated concatenation keeps every branch node before the
        // canonical join even if this helper is later used without the eager
        // branch expansion above.
        ggml_tensor * result = parts.back().second;
        for (size_t i = parts.size() - 1; i > 0; --i) {
            result = ggml_concat(ctx0, parts[i - 1].second, result, 0);
            const std::string name = std::string("attn_qkv_join.") + projection +
                ".concat" + std::to_string(i - 1);
            cb_route(result, name.c_str(), il, policy.assemble_backend.c_str());
        }
        return result;
    };

    ggml_tensor * Qcur = assemble('q');
    ggml_tensor * Kcur = assemble('k');
    ggml_tensor * Vcur = assemble('v');

    // Preserve the ordinary projection's post-matmul ordering exactly:
    // optional scale, bias, clamp, then canonical reshape.
    auto finish_projection = [&](ggml_tensor * value, ggml_tensor * scale, ggml_tensor * bias,
                                 char projection, int64_t heads) {
        const std::string prefix = std::string("attn_qkv_join.") + projection;
        if (scale != nullptr) {
            value = ggml_mul(ctx0, value, scale);
            cb_route(value, (prefix + ".scale").c_str(), il, policy.assemble_backend.c_str());
        }
        if (bias != nullptr) {
            value = ggml_add(ctx0, value, bias);
            cb_route(value, (prefix + ".bias").c_str(), il, policy.assemble_backend.c_str());
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            value = ggml_clamp(ctx0, value, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb_route(value, (prefix + ".clamp").c_str(), il, policy.assemble_backend.c_str());
        }
        value = ggml_reshape_3d(ctx0, value, n_embd_head, heads, n_tokens);
        cb(value, (prefix + ".reshape").c_str(), il, policy.assemble_backend.c_str());
        return value;
    };

    Qcur = finish_projection(Qcur, layer.wq_s, layer.wq_b, 'q', n_head);
    Kcur = finish_projection(Kcur, layer.wk_s, layer.wk_b, 'k', n_head_kv);
    Vcur = finish_projection(Vcur, layer.wv_s, layer.wv_b, 'v', n_head_kv);

    out = { Qcur, Kcur, Vcur };
    return true;
}

bool llm_graph_context::build_attn_out_shards(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * cur,
                int   il,
       const char * final_name,
      ggml_tensor *& out,
        const llama_backend_policy_attn_out_shards * policy_override,
       const char * route_tag,
      ggml_tensor ** out_first) const {
    out = nullptr;
    if (out_first != nullptr) {
        *out_first = nullptr;
    }

    // Keep the ordinary W_O path byte-for-byte unchanged unless every static
    // prerequisite is present. In particular, LoRA corrections are defined on
    // the full projection and must not be silently omitted from a sharded base.
    if (!cparams.attn_out_shards || arch != LLM_ARCH_LLAMA || n_tokens <= 1 ||
            model == nullptr || !model->attn_out_shards_enabled() ||
            wo == nullptr || cur == nullptr || final_name == nullptr || final_name[0] == '\0' ||
            !ggml_is_contiguous_rows(cur) ||
            (loras != nullptr && !loras->empty())) {
        return false;
    }

    llama_backend_policy_attn_out_shards policy = {};
    if (policy_override != nullptr) {
        policy = *policy_override;
    } else {
        if (!llama_backend_policy_match_attn_out_shards(il, true, policy)) {
            return false;
        }
    }
    const bool output_axis = policy.partition_axis == "output";
    // Output-axis lanes feed the complete activation directly to REGION.
    // OpenCL's Q8_0 REGION kernel requires that source to be fully contiguous;
    // input-axis mode instead materializes each slice through ggml_cont().
    if (output_axis && !ggml_is_contiguous(cur)) {
        return false;
    }
    if (policy.splits.size() != 3 || policy.head_dim <= 0 ||
            (policy.partition_axis != "input" && !output_axis) ||
            policy.reduce_backend.empty()) {
        GGML_ABORT(
                "backend_policy: attn_out_shards layer %d matched an invalid policy",
                il);
    }

    const int64_t n_head_dim = hparams.n_embd_head_k((uint32_t) il);
    const int64_t n_in = wo->ne[0];
    const int64_t n_out = wo->ne[1];
    const int64_t block_size = ggml_blck_size(wo->type);
    bool valid = policy.head_dim == n_head_dim &&
        cur->ne[0] == n_in && cur->ne[1] == n_tokens &&
        cur->ne[2] == 1 && cur->ne[3] == 1 &&
        wo->ne[2] == 1 && wo->ne[3] == 1;

    struct attn_out_lane {
        llama_backend_policy_attn_out_shard split;
        llama_backend_policy_resident_cover cover;
        ggml_tensor * view = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * projection = nullptr;
    };

    std::vector<attn_out_lane> lanes;
    lanes.reserve(policy.splits.size());
    for (const auto & split : policy.splits) {
        attn_out_lane lane;
        lane.split = split;
        lanes.push_back(std::move(lane));
    }
    std::sort(lanes.begin(), lanes.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.split.start < rhs.split.start;
    });

    const int64_t partition_width = output_axis ? n_out : n_in;
    const int cover_axis = output_axis ? 1 : 0;
    int64_t covered = 0;
    std::unordered_set<std::string> lane_ids;
    for (auto & lane : lanes) {
        const auto & split = lane.split;
        valid = valid && !split.id.empty() && !split.backend.empty() &&
            lane_ids.insert(split.id).second &&
            split.start == covered && split.size > 0 &&
            split.start % policy.head_dim == 0 && split.size % policy.head_dim == 0 &&
            (output_axis ||
                (split.start % block_size == 0 && split.size % block_size == 0)) &&
            split.start <= partition_width - split.size;
        if (!valid) {
            break;
        }

        lane.cover = model->get_backend_policy_resident_cover(
                wo, split.backend, cover_axis, split.start, split.size);
        valid = lane.cover.tensor != nullptr &&
            lane.cover.tensor->type == wo->type &&
            lane.cover.local_start >= 0 &&
            lane.cover.local_start <= lane.cover.cover_size - split.size &&
            (output_axis || lane.cover.local_start % block_size == 0) &&
            lane.cover.tensor->ne[0] == (output_axis ? n_in : lane.cover.cover_size) &&
            lane.cover.tensor->ne[1] == (output_axis ? lane.cover.cover_size : n_out) &&
            lane.cover.tensor->ne[2] == 1 && lane.cover.tensor->ne[3] == 1;
        if (!valid) {
            break;
        }
        covered += split.size;
    }
    valid = valid && covered == partition_width;
    if (!valid) {
        GGML_ABORT(
                "backend_policy: attn_out_shards layer %d matched policy but has an invalid shape or incomplete three-lane W_O axis-%d cover",
                il, cover_axis);
    }

    // Hexagon's REGION kernel deliberately limits the activation row count to
    // 1024. The full W_O source remains resident, so oversized prefills can
    // safely use the ordinary projection without weakening exact-cover checks.
    if (n_tokens > 1024 && sched != nullptr) {
        const auto is_htp_backend = [](ggml_backend_t backend) {
            if (backend == nullptr || ggml_backend_get_device(backend) == nullptr) {
                return false;
            }
            std::string name = ggml_backend_dev_name(ggml_backend_get_device(backend));
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
                return (char) std::tolower(c);
            });
            return name.find("htp") != std::string::npos ||
                   name.find("hexagon") != std::string::npos;
        };

        for (const auto & lane : lanes) {
            for (int ib = 0; ib < ggml_backend_sched_get_n_backends(sched); ++ib) {
                ggml_backend_t backend = ggml_backend_sched_get_backend(sched, ib);
                if (llama_backend_policy_backend_matches(backend, lane.split.backend) &&
                        is_htp_backend(backend)) {
                    LLAMA_LOG_DEBUG(
                            "backend_policy: attn_out_shards layer %d has n_tokens=%" PRId64
                            " above the HTP REGION limit; using ordinary W_O\n",
                            il, n_tokens);
                    return false;
                }
            }
        }
    }

    std::unordered_set<ggml_backend_t> lane_backends;
    std::vector<ggml_tensor *> partials;
    partials.reserve(lanes.size());

    // Runtime variants need distinct, scheduler-parseable branch names.  The
    // route tag is a short hexadecimal plan identifier supplied by the Llama
    // graph builder; it deliberately contains no '.' because the parallel
    // collector treats the first dot after attn_out_shard as the stage
    // separator. Static v1 names remain byte-for-byte unchanged.
    const std::string route_suffix = route_tag != nullptr && route_tag[0] != '\0'
        ? std::string("-") + route_tag
        : std::string();
    const auto tagged_name = [&](const char * base) {
        return route_suffix.empty() ? std::string(base) : std::string(base) + route_suffix;
    };

    if (!output_axis) {
        // Materialize all metadata-only slices before any executable branch.
        // This keeps VIEW nodes in one dependency prefix instead of attaching
        // the next lane's view to the previous lane's compute split, and makes
        // each branch a pure CONT -> projection unit for cross-device
        // copy/allocation logic.
        for (auto & lane : lanes) {
            const auto & split = lane.split;
            lane.view = ggml_view_2d(
                    ctx0, cur, split.size, cur->ne[1], cur->nb[1],
                    (size_t) split.start * cur->nb[0]);
            ggml_build_forward_expand(gf, lane.view);
        }
    }

    for (auto & lane : lanes) {
        const auto & split = lane.split;
        if (output_axis) {
            // Every output-axis lane consumes the same complete attention
            // result. The scheduler copies that contiguous tensor to remote
            // lane backends; no per-lane VIEW/CONT packing is required.
            lane.input = cur;
            lane.projection = ggml_mul_mat_src0_region(
                    ctx0, lane.cover.tensor, lane.input,
                    0, n_in,
                    lane.cover.local_start, split.size);
        } else {
            lane.input = ggml_cont(ctx0, lane.view);
            const std::string input_name =
                "attn_out_shard." + split.id + route_suffix + ".input";
            cb_route(lane.input, input_name.c_str(), il, split.backend.c_str());

            lane.projection = ggml_mul_mat_src0_region(
                    ctx0, lane.cover.tensor, lane.input,
                    lane.cover.local_start, split.size,
                    0, lane.cover.tensor->ne[1]);
        }

        const std::string projection_name =
            "attn_out_shard." + split.id + route_suffix + ".proj";
        cb_route(lane.projection, projection_name.c_str(), il, split.backend.c_str());

        const ggml_backend_t projection_backend = sched != nullptr
            ? ggml_backend_sched_get_tensor_backend(sched, lane.projection)
            : nullptr;
        const ggml_backend_t input_backend = !output_axis && sched != nullptr
            ? ggml_backend_sched_get_tensor_backend(sched, lane.input)
            : projection_backend;
        if (projection_backend == nullptr || input_backend != projection_backend ||
                !lane_backends.insert(projection_backend).second) {
            GGML_ABORT(
                    "backend_policy: attn_out_shards layer %d lane %s did not resolve to a unique backend",
                    il, split.id.c_str());
        }

        // Finish one lane before emitting the next so the scheduler sees three
        // consecutive, pure projection branches (or CONT+projection branches
        // for the input-axis compatibility path).
        ggml_build_forward_expand(gf, lane.projection);
        if (out_first != nullptr && *out_first == nullptr) {
            *out_first = output_axis ? lane.projection : lane.input;
        }
        partials.push_back(lane.projection);
    }

    // Both joins are right-associated and lanes are already ordered by global
    // start. Input-axis partials overlap and therefore use a fixed ADD order;
    // output-axis partials are disjoint and are concatenated along ne[0].
    ggml_tensor * acc = partials.back();
    for (size_t i = partials.size() - 1; i > 0; --i) {
        acc = output_axis
            ? ggml_concat(ctx0, partials[i - 1], acc, 0)
            : ggml_add(ctx0, partials[i - 1], acc);
        cb_route(
                acc,
                tagged_name(output_axis
                    ? "attn_out_parallel_concat"
                    : "attn_out_parallel_sum").c_str(),
                il, policy.reduce_backend.c_str());
    }
    if (wo_s != nullptr) {
        acc = ggml_mul(ctx0, acc, wo_s);
        cb_route(acc, tagged_name("attn_out_parallel_scale").c_str(),
                il, policy.reduce_backend.c_str());
    }
    if (wo_b != nullptr) {
        acc = ggml_add(ctx0, acc, wo_b);
        cb_route(acc, tagged_name("attn_out_parallel_bias").c_str(),
                il, policy.reduce_backend.c_str());
    }

    // This helper owns both the canonical final name and the strict reduction
    // placement. Calling the ordinary callback again in a caller could allow a
    // generic ops policy to overwrite reduce_backend on this same tensor.
    cb_route(acc, tagged_name(final_name).c_str(), il, policy.reduce_backend.c_str());
    out = acc;
    return true;
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
     llm_ffn_gate_type   type_gate,
                 int   il,
    const llama_backend_policy_ffn_parallel * policy_override,
         const char * route_tag) const {
    llama_backend_policy_ffn_parallel ffn_policy;
    const bool is_prefill = n_tokens > 1;
    const bool has_policy_override = policy_override != nullptr;
    if (has_policy_override) {
        ffn_policy = *policy_override;
    }
    const bool requires_parallel =
        (up   != nullptr && up->buffer   == nullptr) ||
        (gate != nullptr && gate->buffer == nullptr) ||
        (down != nullptr && down->buffer == nullptr);
    auto cb_parallel = [&](ggml_tensor * node, const char * name, const char * backend_hint) {
        if (has_policy_override) {
            cb_route(node, name, il, backend_hint);
        } else {
            cb(node, name, il, backend_hint);
        }
    };
    const bool can_try_parallel =
        model != nullptr &&
        up != nullptr && gate != nullptr && down != nullptr &&
        up_b == nullptr && up_s == nullptr &&
        gate_b == nullptr && gate_s == nullptr &&
        down_b == nullptr && down_s == nullptr &&
        act_scales == nullptr &&
        (loras == nullptr || loras->empty()) &&
        type_op == LLM_FFN_SILU &&
        type_gate == LLM_FFN_PAR &&
        (has_policy_override ||
         llama_backend_policy_match_ffn_parallel(il, is_prefill, ffn_policy));

    if (can_try_parallel) {
        const int64_t n_ff = up->ne[1];
        const int64_t align = std::max<int64_t>(1, ffn_policy.align);
        bool valid = gate->ne[1] == n_ff && down->ne[0] == n_ff;
        std::vector<llama_backend_policy_ffn_split> splits = ffn_policy.splits;
        std::sort(splits.begin(), splits.end(), [](const auto & a, const auto & b) {
            return a.start < b.start;
        });

        int64_t covered = 0;
        for (const auto & split : splits) {
            valid = valid && split.start == covered && split.size > 0;
            valid = valid && (split.start % align) == 0 && (split.size % align) == 0;
            valid = valid && (split.size % ggml_blck_size(up->type)) == 0;
            covered = split.start + split.size;
        }
        valid = valid && covered == n_ff;

        if (valid) {
            std::vector<llama_backend_policy_ffn_split> exec_splits = splits;
            // Keep the HTP/NPU branch after non-HTP branches so that ffn_norm is
            // emitted as a completed prefix split before all parallel branches.
            std::stable_sort(exec_splits.begin(), exec_splits.end(), [](const auto & a, const auto & b) {
                auto is_hybrid_prefix_backend = [](std::string backend) {
                    std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
                        return (char) std::tolower(c);
                    });
                    return backend.find("htp") != std::string::npos ||
                           backend.find("hexagon") != std::string::npos;
                };

                const bool a_prefix = is_hybrid_prefix_backend(a.backend);
                const bool b_prefix = is_hybrid_prefix_backend(b.backend);
                if (a_prefix != b_prefix) {
                    return !a_prefix;
                }
                return a.start < b.start;
            });

            const std::string route_suffix = route_tag != nullptr && route_tag[0] != '\0'
                ? std::string(".") + route_tag : std::string();

            auto reduce_add = [&](ggml_tensor * lhs, ggml_tensor * rhs) {
                ggml_tensor * acc = ggml_add(ctx0, lhs, rhs);
                cb_parallel(acc, ("ffn_parallel_sum" + route_suffix).c_str(),
                        ffn_policy.reduce_backend.empty() ? nullptr : ffn_policy.reduce_backend.c_str());
                return acc;
            };

            auto reduce_branches = [&](const std::vector<ggml_tensor *> & outs) -> ggml_tensor * {
                if (outs.empty()) {
                    return nullptr;
                }
                ggml_tensor * acc = outs.back();
                for (size_t ri = outs.size() - 1; ri > 0; --ri) {
                    acc = reduce_add(outs[ri - 1], acc);
                }
                return acc;
            };

            std::vector<ggml_tensor *> branch_outs;
            branch_outs.reserve(exec_splits.size());

            for (const auto & split : exec_splits) {
                const auto up_cover =
                    model->get_backend_policy_resident_cover(up, split.backend, 1, split.start, split.size);
                const auto gate_cover =
                    model->get_backend_policy_resident_cover(gate, split.backend, 1, split.start, split.size);
                const auto down_cover =
                    model->get_backend_policy_resident_cover(down, split.backend, 0, split.start, split.size);

                if (up_cover.tensor == nullptr || gate_cover.tensor == nullptr || down_cover.tensor == nullptr) {
                    valid = false;
                    LLAMA_LOG_DEBUG("backend_policy: ffn_parallel layer %d shard %s missing a containing resident cover for backend %s; falling back\n",
                            il, split.id.c_str(), split.backend.c_str());
                    break;
                }

                const std::string suffix = route_suffix + "." +
                    (split.id.empty() ? split.backend : split.id);

                const bool covers_consistent =
                    up_cover.cover_start == gate_cover.cover_start &&
                    up_cover.cover_start == down_cover.cover_start &&
                    up_cover.cover_size == gate_cover.cover_size &&
                    up_cover.cover_size == down_cover.cover_size &&
                    up_cover.local_start == gate_cover.local_start &&
                    up_cover.local_start == down_cover.local_start &&
                    up_cover.local_start >= 0 &&
                    up_cover.local_start <= up_cover.cover_size - split.size &&
                    up_cover.tensor->ne[0] == up->ne[0] &&
                    gate_cover.tensor->ne[0] == gate->ne[0] &&
                    up_cover.tensor->ne[1] == up_cover.cover_size &&
                    gate_cover.tensor->ne[1] == gate_cover.cover_size &&
                    down_cover.tensor->ne[0] == down_cover.cover_size &&
                    down_cover.tensor->ne[1] == down->ne[1];
                if (!covers_consistent) {
                    valid = false;
                    LLAMA_LOG_DEBUG("backend_policy: ffn_parallel layer %d shard %s has inconsistent resident cover metadata for backend %s; falling back\n",
                            il, split.id.c_str(), split.backend.c_str());
                    break;
                }

                // The cover tensors stay resident and independently packed for
                // their processor. Only these immutable node-local regions
                // change when a query-boundary clock profile selects a new
                // shard proportion.
                ggml_tensor * up_cur = ggml_mul_mat_src0_region(
                        ctx0, up_cover.tensor, cur,
                        0, up_cover.tensor->ne[0],
                        up_cover.local_start, split.size);
                cb_parallel(up_cur, ("ffn_up" + suffix).c_str(), split.backend.c_str());

                ggml_tensor * gate_cur = ggml_mul_mat_src0_region(
                        ctx0, gate_cover.tensor, cur,
                        0, gate_cover.tensor->ne[0],
                        gate_cover.local_start, split.size);
                cb_parallel(gate_cur, ("ffn_gate" + suffix).c_str(), split.backend.c_str());

                ggml_tensor * act = ggml_swiglu_split(ctx0, gate_cur, up_cur);
                cb_parallel(act, ("ffn_swiglu" + suffix).c_str(), split.backend.c_str());

                ggml_tensor * branch = ggml_mul_mat_src0_region(
                        ctx0, down_cover.tensor, act,
                        down_cover.local_start, split.size,
                        0, down_cover.tensor->ne[1]);
                cb_parallel(branch, ("ffn_down" + suffix).c_str(), split.backend.c_str());

                branch_outs.push_back(branch);
            }

            if (valid && !branch_outs.empty()) {
                ggml_tensor * acc = reduce_branches(branch_outs);
                cb_parallel(acc, ("ffn_parallel_out" + route_suffix).c_str(),
                        ffn_policy.reduce_backend.empty() ? nullptr : ffn_policy.reduce_backend.c_str());
                return acc;
            }
        } else {
            LLAMA_LOG_DEBUG("backend_policy: ffn_parallel layer %d invalid split coverage/alignment for n_ff=%" PRId64 "; falling back\n",
                    il, n_ff);
        }
    }

    if (requires_parallel) {
        GGML_ABORT("backend_policy: ffn_parallel cover-only layer %d cannot fall back to full FFN weights", il);
    }

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                // Step35: HF clamps gate (after SiLU) and up before multiplication
                if (arch == LLM_ARCH_STEP35 && il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                        cb(gate_act, "ffn_silu", il);
                        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                        cb(gate_act, "ffn_silu_clamped", il);

                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        cb(tmp, "ffn_up_clamped", il);

                        cur = ggml_mul(ctx0, gate_act, tmp);
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        // apply per-expert scale2 to merged gate_up (use up_exps_s since gate and up are fused)
        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            gate_up = ggml_mul(ctx0, gate_up, s);
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        // apply per-expert scale2 to up
        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            up = ggml_mul(ctx0, up, s);
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }

        // apply per-expert scale2 to gate
        if (gate_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, gate_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            cur = ggml_mul(ctx0, cur, s);
            cb(cur, "ffn_moe_gate_scaled", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                // Step35: per-layer clamp for routed experts
                if (arch == LLM_ARCH_STEP35 && il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                        cb(gate_act, "ffn_moe_silu", il);
                        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                        cb(gate_act, "ffn_moe_silu_clamped", il);

                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        cur = ggml_mul(ctx0, gate_act, up);
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    // apply per-expert scale2 to down
    if (down_exps_s) {
        ggml_tensor * s = ggml_reshape_3d(ctx0, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
        experts = ggml_mul(ctx0, experts, s);
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_f32_zeros(
        const char * name,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3) const {
    const size_t n_values = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;
    auto inp = std::make_unique<llm_graph_input_f32_zeros>(n_values);

    auto & cur = inp->tensor;
    cur = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_set_input(cur);
    cb(cur, name, -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "attn_scale");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il,
        const char *   backend_hint) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, LLAMA_TENSOR_NAME_FATTN, il, backend_hint);

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il, backend_hint);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il, backend_hint);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il, backend_hint);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il, backend_hint);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il, backend_hint);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il, backend_hint);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il, backend_hint);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il, backend_hint);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il, backend_hint);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il, backend_hint);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il, backend_hint);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il, backend_hint);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il,
            bool    * out_wo_sharded,
       const char *   sharded_wo_final_name) const {
    GGML_UNUSED(n_tokens);
    if (out_wo_sharded != nullptr) {
        *out_wo_sharded = false;
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    ggml_tensor * sharded_out = nullptr;
    const bool sharded_wo = wo != nullptr &&
        build_attn_out_shards(
                wo, wo_b, wo_s, cur, il, sharded_wo_final_name, sharded_out);
    if (sharded_wo) {
        cur = sharded_out;
        if (out_wo_sharded != nullptr) {
            *out_wo_sharded = true;
        }
    } else if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
        cb(cur, "kqv_wo", il);
    }

    if (wo_b && !sharded_wo) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il,
            bool    * out_wo_sharded,
       const char *   sharded_wo_final_name) const {
    GGML_ASSERT(v_mla == nullptr);
    if (out_wo_sharded != nullptr) {
        *out_wo_sharded = false;
    }

    if (inp->self_k_rot) {
        q_cur = ggml_mul_mat_aux(ctx0, q_cur, inp->self_k_rot);
        k_cur = ggml_mul_mat_aux(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = ggml_mul_mat_aux(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = ggml_mul_mat_aux(ctx0, cur, inp->self_v_rot);
    }

    ggml_tensor * sharded_out = nullptr;
    const bool sharded_wo = wo != nullptr &&
        build_attn_out_shards(
                wo, wo_b, wo_s, cur, il, sharded_wo_final_name, sharded_out);
    if (sharded_wo) {
        cur = sharded_out;
        if (out_wo_sharded != nullptr) {
            *out_wo_sharded = true;
        }
    } else if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            cb(cur, "kqv_wo", il);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
            cb(cur, "kqv_wo", il);
        }
    }

    if (wo_b && !sharded_wo) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            cb(cur, "kqv_wo", il);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
            cb(cur, "kqv_wo", il);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = ggml_mul_mat_aux(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = ggml_mul_mat_aux(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = ggml_mul_mat_aux(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = ggml_mul_mat_aux(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->cross_kq_mask, GGML_TYPE_F16) : inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, rs_size);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_attn->self_kq_mask, GGML_TYPE_F16) : inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_attn->self_kq_mask_swa, GGML_TYPE_F16) : inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, int32_t> seq_to_logit_row;
    int32_t logit_row_idx = 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (ubatch.output[i]) {
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            seq_to_logit_row[seq_id] = logit_row_idx;
            logit_row_idx++;
        }
    }

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row of logits
    // this trick makes the graph static, regardless of which samplers are activated
    // this is important in order to minimize graph reallocations
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = seq_to_logit_row.find(seq_id);

        // inactive samplers always work on the first row
        const auto row_idx = it != seq_to_logit_row.end() ? it->second : 0;
        const int i_out    = it != seq_to_logit_row.end() ? 1          : 0;

        ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row_idx * logits_t->nb[1]);
        ggml_format_name(logits_seq, "logits_seq_%d", seq_id);

        struct llama_sampler_data data = {
            /*.logits      =*/ logits_seq,
            /*.probs       =*/ nullptr,
            /*.sampled     =*/ nullptr,
            /*.candidates  =*/ nullptr,
        };

        assert(sampler->iface->backend_apply);
        sampler->iface->backend_apply(sampler, ctx0, gf, &data);

        if (data.sampled != nullptr) {
            res->t_sampled[seq_id] = data.sampled;
            outs[1] = data.sampled;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.probs != nullptr) {
            res->t_sampled_probs[seq_id] = data.probs;
            outs[1] = data.probs;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.logits != nullptr) {
            res->t_sampled_logits[seq_id] = data.logits;
            outs[1] = data.logits;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.candidates != nullptr) {
            res->t_candidates[seq_id] = data.candidates;
            outs[1] = data.candidates;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }
    }

    // TODO: Call llama_sampler_accept_ggml after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        if (auto it = res->t_sampled.find(seq_id); it != res->t_sampled.end()) {
            ggml_tensor * selected_token = it->second;
            if (selected_token != nullptr) {
                llama_sampler_accept_ggml(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
