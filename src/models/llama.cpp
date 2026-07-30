#include "models.h"
#include "llama-backend-policy.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cstdio>

void llama_model_llama::load_arch_hparams(llama_model_loader & ml) {
    const auto n_vocab = vocab.n_tokens();

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (hparams.n_expert == 8) {
        switch (hparams.n_layer) {
            case 32: type = LLM_TYPE_8x7B; break;
            case 56: type = LLM_TYPE_8x22B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    } else {
        switch (hparams.n_layer) {
            case 16: type = LLM_TYPE_1B; break; // Llama 3.2 1B
            case 22: type = LLM_TYPE_1B; break;
            case 26: type = LLM_TYPE_3B; break;
            case 28: type = LLM_TYPE_3B; break; // Llama 3.2 3B
            case 30: type = LLM_TYPE_256M; break; // smoldocling 256M
            // granite uses a vocab with len 49152
            case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
            case 36: type = LLM_TYPE_8B; break; // granite
            case 40: type = LLM_TYPE_13B; break;
            case 48: type = LLM_TYPE_34B; break;
            case 60: type = LLM_TYPE_30B; break;
            case 80: type = hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    }
}

void llama_model_llama::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // optional bias tensors
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        if (n_expert == 0) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(*this, params);
}

template <bool embed>
llama_model_llama::graph<embed>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (params.module_bench_active) {
        const int il_start = std::max<int>(0, cparams.module_bench_layer_start);
        const int il_end = std::min<int>((int) n_layer - 1,
                cparams.module_bench_layer_end < 0 ? (int) n_layer - 1 : cparams.module_bench_layer_end);
        if (il_start > il_end) {
            GGML_ABORT("module-bench invalid layer range");
        }

        ggml_tensor * last = nullptr;
        auto finish = [&](ggml_tensor * out) {
            ggml_build_forward_expand(gf, out);
            res->t_embd = out;
            last = out;
        };

        auto build_module_inp = [&](const char * name, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
            return build_inp_f32_zeros(name, ne0, ne1, ne2, ne3);
        };

        auto module_ffn_tokens = [&](int il) {
            return il == (int) n_layer - 1 ? n_outputs : n_tokens;
        };

        const char * module_backend_hint = cparams.module_bench_backend.empty() ? nullptr : cparams.module_bench_backend.c_str();
        auto cb_module = [&](ggml_tensor * cur, const char * name, int il) {
            cb(cur, name, il, module_backend_hint);
        };

        auto module_resident_norm_weight = [&](ggml_tensor * weight) {
            if (cparams.module_bench_backend.empty()) {
                return weight;
            }
            std::vector<std::string> candidates = { cparams.module_bench_backend };
            if (cparams.module_bench_backend.find("REPACK") == std::string::npos &&
                    cparams.module_bench_backend.find("repack") == std::string::npos) {
                // HTP execution is normally named HTP0 while immutable
                // weighted-MUL operands live in HTP0-REPACK. CPU/OpenCL match
                // the first entry, so this fallback is harmless for them.
                candidates.push_back(cparams.module_bench_backend + "-REPACK");
            }
            return model.get_backend_policy_residency_tensor(weight, candidates);
        };

        auto build_named_rms_norm = [&](ggml_tensor * inp, ggml_tensor * weight, const char * rms_name, const char * out_name, int il) {
            ggml_tensor * out = ggml_rms_norm(ctx0, inp, hparams.f_norm_rms_eps);
            cb_module(out, rms_name, il);
            out = ggml_mul(ctx0, out, weight);
            cb_module(out, out_name, il);
            return out;
        };

        switch (cparams.module_bench_type) {
            case LLAMA_MODULE_BENCH_ATTN_NORM:
                for (int il = il_start; il <= il_end; ++il) {
                    ggml_tensor * inp = build_module_inp("module_attn_norm_inp", n_embd, n_tokens);
                    cb(inp, "module_attn_norm_inp", il);
                    finish(build_named_rms_norm(inp, model.layers[il].attn_norm, "norm", "attn_norm", il));
                }
                break;

            case LLAMA_MODULE_BENCH_ATTN_PROJECTION:
                {
                    ggml_tensor * inp_pos = build_inp_pos();
                    auto * inp_attn = build_attn_inp_kv();
                    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);
                    for (int il = il_start; il <= il_end; ++il) {
                        ggml_tensor * inp = build_module_inp("module_attn_projection_inp", n_embd, n_tokens);
                        cb(inp, "module_attn_projection_inp", il);

                        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
                        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], inp,
                                n_embd_head, n_head, n_head_kv, il);

                        Qcur = ggml_rope_ext(
                                ctx0, Qcur, inp_pos, rope_factors,
                                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow);

                        Kcur = ggml_rope_ext(
                                ctx0, Kcur, inp_pos, rope_factors,
                                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow);

                        cb(Qcur, "Qcur", il);
                        cb(Kcur, "Kcur", il);
                        cb(Vcur, "Vcur", il);

                        ggml_build_forward_expand(gf, Qcur);
                        ggml_build_forward_expand(gf, Vcur);
                        ggml_build_forward_expand(gf, Kcur);

                        const auto & k_idxs = inp_attn->get_k_idxs();
                        const auto & v_idxs = inp_attn->get_v_idxs();
                        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il));
                        ggml_tensor * out = mctx_cur->cpy_v(ctx0, Vcur, v_idxs, il);
                        finish(out);
                    }
                }
                break;

            case LLAMA_MODULE_BENCH_ATTN_SCORE:
                {
                    auto * inp_attn = build_attn_inp_kv();
                    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);
                    for (int il = il_start; il <= il_end; ++il) {
                        ggml_tensor * Qcur = build_module_inp("module_attn_score_q", n_embd_head, n_head, n_tokens);
                        cb_module(Qcur, "module_attn_score_q", il);
                        ggml_tensor * Kcur = mctx_cur->get_k(ctx0, il);
                        ggml_tensor * Vcur = mctx_cur->get_v(ctx0, il);
                        ggml_tensor * out = build_attn_mha(Qcur, Kcur, Vcur, nullptr, inp_attn->get_kq_mask(), nullptr, nullptr,
                                kq_scale, il, module_backend_hint);
                        cb_module(out, "kqv_out", il);
                        finish(out);
                    }
                }
                break;

            case LLAMA_MODULE_BENCH_ATTN_OUT_PROJ:
                for (int il = il_start; il <= il_end; ++il) {
                    ggml_tensor * inp = build_module_inp("module_attn_out_proj_inp", n_embd, n_tokens);
                    cb(inp, "module_attn_out_proj_inp", il);
                    ggml_tensor * out = build_lora_mm(model.layers[il].wo, inp, model.layers[il].wo_s);
                    cb(out, "attn_out", il);
                    if (model.layers[il].wo_b) {
                        out = ggml_add(ctx0, out, model.layers[il].wo_b);
                    }
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_INPUT_GET_ROWS:
                {
                    if (!ubatch.token) {
                        GGML_ABORT("module-bench input_get_rows requires token input");
                    }

                    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp());
                    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
                    cb(inp->tokens, "inp_tokens", -1);
                    ggml_set_input(inp->tokens);
                    res->t_inp_tokens = inp->tokens;

                    ggml_tensor * out = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
                    cb_module(out, "input_get_rows", 0);
                    res->add_input(std::move(inp));
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_FFN_INP:
                for (int il = il_start; il <= il_end; ++il) {
                    const int64_t n_ffn_tokens = module_ffn_tokens(il);
                    ggml_tensor * lhs = build_module_inp("module_ffn_inp_lhs", n_embd, n_ffn_tokens);
                    ggml_tensor * rhs = build_module_inp("module_ffn_inp_rhs", n_embd, n_ffn_tokens);
                    cb_module(lhs, "module_ffn_inp_lhs", il);
                    cb_module(rhs, "module_ffn_inp_rhs", il);
                    ggml_tensor * out = ggml_add(ctx0, lhs, rhs);
                    cb_module(out, "ffn_inp", il);
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_FFN_NORM:
                for (int il = il_start; il <= il_end; ++il) {
                    ggml_tensor * inp = build_module_inp("module_ffn_norm_inp", n_embd, module_ffn_tokens(il));
                    cb(inp, "module_ffn_norm_inp", il);
                    finish(build_named_rms_norm(inp, model.layers[il].ffn_norm, "norm", "ffn_norm", il));
                }
                break;

            case LLAMA_MODULE_BENCH_FFN_CORE:
                for (int il = il_start; il <= il_end; ++il) {
                    if (model.layers[il].ffn_gate_inp != nullptr) {
                        GGML_ABORT("module-bench ffn_core only supports dense FFN layers");
                    }
                    ggml_tensor * inp = build_module_inp("module_ffn_core_inp", n_embd, module_ffn_tokens(il));
                    cb(inp, "ffn_norm", il);
                    ggml_tensor * out = build_ffn(inp,
                            model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                            model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                            model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(out, "ffn_out", il);
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_L_OUT:
                for (int il = il_start; il <= il_end; ++il) {
                    const int64_t n_ffn_tokens = module_ffn_tokens(il);
                    ggml_tensor * lhs = build_module_inp("module_l_out_lhs", n_embd, n_ffn_tokens);
                    ggml_tensor * rhs = build_module_inp("module_l_out_rhs", n_embd, n_ffn_tokens);
                    cb_module(lhs, "module_l_out_lhs", il);
                    cb_module(rhs, "module_l_out_rhs", il);
                    ggml_tensor * out = ggml_add(ctx0, lhs, rhs);
                    cb_module(out, "l_out", il);
                    // This benchmark targets the residual ADD itself. Do not
                    // append the final norm/lm-head when il is the last layer.
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_ATTN_RMS_NORM:
                for (int il = il_start; il <= il_end; ++il) {
                    ggml_tensor * inp = build_module_inp("module_attn_rms_norm_inp", n_embd, n_tokens);
                    cb_module(inp, "module_attn_rms_norm_inp", il);
                    ggml_tensor * out = ggml_rms_norm(ctx0, inp, hparams.f_norm_rms_eps);
                    // Match the raw graph/debug name. The module-bench target
                    // records the semantic attention-side role separately.
                    cb_module(out, "norm", il);
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_ATTN_NORM_MUL:
                for (int il = il_start; il <= il_end; ++il) {
                    ggml_tensor * inp = build_module_inp(
                            "module_attn_norm_mul_inp", n_embd, n_tokens);
                    cb_module(inp, "module_attn_norm_mul_inp", il);
                    ggml_tensor * weight = model.layers[il].attn_norm;
                    if (!cparams.module_bench_backend.empty()) {
                        weight = module_resident_norm_weight(weight);
                        if (weight == nullptr) {
                            GGML_ABORT(
                                    "module-bench attn_norm_mul requires a resident attn_norm.weight on %s (layer %d)",
                                    cparams.module_bench_backend.c_str(), il);
                        }
                    }
                    ggml_tensor * out = ggml_mul(ctx0, inp, weight);
                    cb_module(out, "attn_norm", il);
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_FFN_RMS_NORM:
                for (int il = il_start; il <= il_end; ++il) {
                    ggml_tensor * inp = build_module_inp(
                            "module_ffn_rms_norm_inp", n_embd, module_ffn_tokens(il));
                    cb_module(inp, "module_ffn_rms_norm_inp", il);
                    ggml_tensor * out = ggml_rms_norm(ctx0, inp, hparams.f_norm_rms_eps);
                    // build_norm() also calls the FFN-side RMS node "norm";
                    // the trace's semantic_role field disambiguates it.
                    cb_module(out, "norm", il);
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_FFN_NORM_MUL:
                for (int il = il_start; il <= il_end; ++il) {
                    ggml_tensor * inp = build_module_inp(
                            "module_ffn_norm_mul_inp", n_embd, module_ffn_tokens(il));
                    cb_module(inp, "module_ffn_norm_mul_inp", il);
                    ggml_tensor * weight = model.layers[il].ffn_norm;
                    if (!cparams.module_bench_backend.empty()) {
                        weight = module_resident_norm_weight(weight);
                        if (weight == nullptr) {
                            GGML_ABORT(
                                    "module-bench ffn_norm_mul requires a resident ffn_norm.weight on %s (layer %d)",
                                    cparams.module_bench_backend.c_str(), il);
                        }
                    }
                    ggml_tensor * out = ggml_mul(ctx0, inp, weight);
                    cb_module(out, "ffn_norm", il);
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_FFN_PARALLEL_SUM:
                for (int il = il_start; il <= il_end; ++il) {
                    const int64_t n_ffn_tokens = module_ffn_tokens(il);
                    ggml_tensor * lhs = build_module_inp(
                            "module_ffn_parallel_sum_lhs", n_embd, n_ffn_tokens);
                    ggml_tensor * rhs = build_module_inp(
                            "module_ffn_parallel_sum_rhs", n_embd, n_ffn_tokens);
                    cb_module(lhs, "module_ffn_parallel_sum_lhs", il);
                    cb_module(rhs, "module_ffn_parallel_sum_rhs", il);
                    ggml_tensor * out = ggml_add(ctx0, lhs, rhs);
                    cb_module(out, "ffn_parallel_sum", il);
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_FFN_OUT:
                for (int il = il_start; il <= il_end; ++il) {
                    const int64_t n_ffn_tokens = module_ffn_tokens(il);
                    ggml_tensor * branch = build_module_inp(
                            "module_ffn_out_branch", n_embd, n_ffn_tokens);
                    ggml_tensor * partial_sum = build_module_inp(
                            "module_ffn_out_partial_sum", n_embd, n_ffn_tokens);
                    cb_module(branch, "module_ffn_out_branch", il);
                    cb_module(partial_sum, "module_ffn_out_partial_sum", il);
                    ggml_tensor * out = ggml_add(ctx0, branch, partial_sum);
                    cb_module(out, "ffn_out", il);
                    finish(out);
                }
                break;

            case LLAMA_MODULE_BENCH_OFF:
                GGML_ABORT("module-bench active with off module");
        }

        GGML_ASSERT(last != nullptr);
        return;
    }

    ggml_tensor * cur;
    ggml_tensor * inpL;

    llama_backend_policy_runtime_routes runtime_routes;
    const bool runtime_layer_routes = runtime_route_candidates && n_tokens > 1 &&
        llama_backend_policy_resolve_runtime_routes(true, runtime_routes) &&
        (runtime_routes.mode == "clock" || runtime_routes.mode == "fixed");
    const bool runtime_ffn_routes = runtime_layer_routes &&
        std::find(runtime_routes.candidate_kinds.begin(), runtime_routes.candidate_kinds.end(),
                "ffn_block") != runtime_routes.candidate_kinds.end();
    const bool runtime_weighted_norm_routes = runtime_layer_routes &&
        std::find(runtime_routes.candidate_kinds.begin(), runtime_routes.candidate_kinds.end(),
                "weighted_norm") != runtime_routes.candidate_kinds.end();
    std::vector<std::string> runtime_route_profiles;
    if (runtime_ffn_routes || runtime_weighted_norm_routes) {
        runtime_route_profiles.reserve(runtime_routes.profiles.size());
        runtime_route_profiles.push_back(runtime_routes.initial_profile);
        for (const std::string & profile : runtime_routes.profiles) {
            if (profile != runtime_routes.initial_profile) {
                runtime_route_profiles.push_back(profile);
            }
        }
    }

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Attention-side norm is stateless but its final MUL is weighted. In
        // route mode, prebuild one compact RMS+MUL candidate per clock profile
        // using a backend-resident duplicate of attn_norm.weight. This does not
        // alter the Q/K/V or KV-cache topology; only the norm feeding QKV is
        // selected at the layer's already-latched plan.
        if (runtime_weighted_norm_routes) {
            ggml_tensor * canonical_norm = nullptr;
            for (const std::string & profile : runtime_route_profiles) {
                llama_backend_policy_match rms_match;
                llama_backend_policy_match mul_match;
                const bool have_rms = llama_backend_policy_match_op_for_profile(
                        profile.c_str(), "attn_rms_norm", "attn_rms_norm",
                        GGML_OP_RMS_NORM, il, true, rms_match);
                const bool have_mul = llama_backend_policy_match_op_for_profile(
                        profile.c_str(), "attn_norm", "attn_norm",
                        GGML_OP_MUL, il, true, mul_match);
                if (!have_rms || rms_match.backends.empty() ||
                        !have_mul || mul_match.backends.empty()) {
                    GGML_ABORT(
                            "runtime_routes: profile %s is missing layer %d attention RMS/weighted-MUL policy",
                            profile.c_str(), il);
                }

                ggml_tensor * norm_weight = model.get_backend_policy_residency_tensor(
                        model.layers[il].attn_norm, mul_match.backends);
                if (norm_weight == nullptr) {
                    GGML_ABORT(
                            "runtime_routes: profile %s has no resident attn_norm.weight for layer %d backend %s",
                            profile.c_str(), il, mul_match.backends.front().c_str());
                }

                char route_tag[24];
                snprintf(route_tag, sizeof(route_tag), "%s", profile.c_str());
                ggml_tensor * first = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
                cb_route(first, (std::string("attn_rms_norm.") + route_tag).c_str(), il,
                        rms_match.backends.front().c_str());
                ggml_tensor * weighted = ggml_mul(ctx0, first, norm_weight);
                cb_route(weighted, (std::string("attn_norm.") + route_tag).c_str(), il,
                        mul_match.backends.front().c_str());
                ggml_build_forward_expand(gf, weighted);

                if (route_subgraph_cb_func) {
                    route_subgraph_cb_func(
                            "weighted_norm", profile.c_str(), il, first, weighted);
                }
                if (profile == runtime_routes.initial_profile) {
                    canonical_norm = weighted;
                }
            }
            if (canonical_norm == nullptr) {
                GGML_ABORT("runtime_routes: initial attention norm profile %s was not built",
                        runtime_routes.initial_profile.c_str());
            }
            cur = canonical_norm;
        } else {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
        }

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {
            if (runtime_ffn_routes) {
                ggml_tensor * canonical_out = nullptr;
                for (const std::string & profile : runtime_route_profiles) {
                    llama_backend_policy_ffn_parallel ffn_policy;
                    llama_backend_policy_match rms_match;
                    llama_backend_policy_match mul_match;
                    const bool have_ffn = llama_backend_policy_match_ffn_parallel_for_profile(
                            profile.c_str(), il, true, ffn_policy);
                    const bool have_rms = llama_backend_policy_match_op_for_profile(
                            profile.c_str(), "ffn_rms_norm", "ffn_rms_norm",
                            GGML_OP_RMS_NORM, il, true, rms_match);
                    const bool have_mul = llama_backend_policy_match_op_for_profile(
                            profile.c_str(), "ffn_norm", "ffn_norm",
                            GGML_OP_MUL, il, true, mul_match);
                    if (!have_ffn || !have_rms || rms_match.backends.empty() ||
                            !have_mul || mul_match.backends.empty()) {
                        GGML_ABORT(
                                "runtime_routes: profile %s is missing layer %d FFN/RMS/weighted-MUL policy",
                                profile.c_str(), il);
                    }

                    ggml_tensor * norm_weight = model.get_backend_policy_residency_tensor(
                            model.layers[il].ffn_norm, mul_match.backends);
                    if (norm_weight == nullptr) {
                        GGML_ABORT(
                                "runtime_routes: profile %s has no resident ffn_norm.weight for layer %d backend %s",
                                profile.c_str(), il, mul_match.backends.front().c_str());
                    }

                    char route_tag[24];
                    // Keep the profile visible in scheduler_trace.csv branch
                    // and node names. The scheduler still routes by the stable
                    // numeric plan ID; this tag is observability only.
                    snprintf(route_tag, sizeof(route_tag), "%s", profile.c_str());

                    ggml_tensor * first = ggml_rms_norm(ctx0, ffn_inp, hparams.f_norm_rms_eps);
                    cb_route(first, (std::string("ffn_rms_norm.") + route_tag).c_str(), il,
                            rms_match.backends.front().c_str());

                    ggml_tensor * weighted = ggml_mul(ctx0, first, norm_weight);
                    cb_route(weighted, (std::string("ffn_norm.") + route_tag).c_str(), il,
                            mul_match.backends.front().c_str());

                    ggml_tensor * route_out = build_ffn(weighted,
                            model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                            model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                            model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, il, &ffn_policy, route_tag);
                    cb(route_out, (std::string("ffn_route_out.") + route_tag).c_str(), il);
                    ggml_build_forward_expand(gf, route_out);

                    if (route_subgraph_cb_func) {
                        route_subgraph_cb_func(
                                "ffn_block", profile.c_str(), il, first, route_out);
                    }
                    if (profile == runtime_routes.initial_profile) {
                        canonical_out = route_out;
                    }
                }
                if (canonical_out == nullptr) {
                    GGML_ABORT("runtime_routes: initial FFN profile %s was not built",
                            runtime_routes.initial_profile.c_str());
                }
                cur = canonical_out;
            } else {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                llama_backend_policy_ffn_parallel carried_ffn_policy;
                const bool have_carried_ffn_policy =
                    !runtime_ffn_profile.empty() &&
                    llama_backend_policy_match_ffn_parallel_for_profile(
                            runtime_ffn_profile.c_str(), il, n_tokens > 1,
                            carried_ffn_policy);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il,
                        have_carried_ffn_policy ? &carried_ffn_policy : nullptr,
                        have_carried_ffn_policy ? runtime_ffn_profile.c_str() : nullptr);
                cb(cur, "ffn_out", il);
            }
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

template struct llama_model_llama::graph<false>;
template struct llama_model_llama::graph<true>;
