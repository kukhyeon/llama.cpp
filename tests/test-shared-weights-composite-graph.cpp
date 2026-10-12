#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-shared-weights.h"
#include "llama-backend-policy.h"
#include "llama-graph.h"

#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define REQUIRE(condition) do { if (!(condition)) { \
    throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #condition); \
} } while (0)

namespace {

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) { setenv(name, value, 1); } else { unsetenv(name); }
#endif
}

void check_route_flags(const llm_graph_runtime_route_flags & flags,
        bool layer, bool ffn, bool weighted_norm, bool attn_qkv, bool attn_out) {
    REQUIRE(flags.layer == layer);
    REQUIRE(flags.ffn == ffn);
    REQUIRE(flags.weighted_norm == weighted_norm);
    REQUIRE(flags.attn_qkv == attn_qkv);
    REQUIRE(flags.attn_out == attn_out);
}

llm_graph_runtime_route_flags check_cached_route_flags(
        bool runtime_candidates, uint32_t n_tokens,
        bool qkv_shards, bool out_shards, bool has_lora,
        const llama_backend_policy_runtime_routes & routes) {
    const auto flags = llm_graph_runtime_route_flags_from_catalog(
            runtime_candidates, n_tokens, qkv_shards, out_shards, has_lora, routes);
    const auto available = llm_graph_runtime_route_flags_from_catalog(
            true, 2, true, true, false, routes);
    check_route_flags(llm_graph_filter_runtime_route_flags(
            available, runtime_candidates, n_tokens, qkv_shards, out_shards, has_lora),
            flags.layer, flags.ffn, flags.weighted_norm, flags.attn_qkv, flags.attn_out);
    return flags;
}

void run_runtime_route_flags_tests() {
    llama_backend_policy_runtime_routes routes;
    routes.enabled = true;
    routes.candidate_kinds = {
        "ffn_block", "weighted_norm", "attn_qkv_block", "attn_out_block",
    };
    for (const std::string & mode : {std::string("clock"), std::string("fixed")}) {
        routes.mode = mode;
        for (uint32_t tokens : {0u, 1u, 2u, 1024u, 1025u}) {
            const bool layer = tokens > 1;
            for (bool has_lora : {false, true}) {
                for (bool qkv_shards : {false, true}) {
                    for (bool out_shards : {false, true}) {
                        const auto flags = check_cached_route_flags(
                                true, tokens, qkv_shards, out_shards, has_lora, routes);
                        check_route_flags(flags, layer, layer, layer,
                                layer && tokens <= 1024 && !has_lora && qkv_shards,
                                layer && tokens <= 1024 && !has_lora && out_shards);
                    }
                }
            }
            check_route_flags(check_cached_route_flags(
                    false, tokens, true, true, false, routes), false, false, false, false, false);
        }
    }
    routes.mode = "off";
    check_route_flags(check_cached_route_flags(
            true, 512, true, true, false, routes), false, false, false, false, false);
    routes.mode = "clock";
    routes.enabled = false;
    check_route_flags(check_cached_route_flags(
            true, 512, true, true, false, routes), false, false, false, false, false);
    routes.enabled = true;
    for (const std::string & kind : {
            std::string("ffn_block"), std::string("weighted_norm"),
            std::string("attn_qkv_block"), std::string("attn_out_block"),
            std::string("weightless_stateless")}) {
        routes.candidate_kinds = {kind};
        check_route_flags(check_cached_route_flags(
                true, 512, true, true, false, routes), true,
                kind == "ffn_block", kind == "weighted_norm",
                kind == "attn_qkv_block", kind == "attn_out_block");
    }
    routes.candidate_kinds.clear();
    check_route_flags(check_cached_route_flags(
            true, 512, true, true, false, routes), true, false, false, false, false);

    // These cases short-circuit before reading process-global policy state.
    // A previously populated output catalog must not leak into a fallback.
    routes.profiles = {"stale"};
    check_route_flags(llm_graph_resolve_runtime_route_flags(
            false, 512, true, true, false, routes), false, false, false, false, false);
    REQUIRE(!routes.enabled && routes.profiles.empty());
    routes.enabled = true;
    routes.profiles = {"stale"};
    check_route_flags(llm_graph_resolve_runtime_route_flags(
            true, 1, true, true, false, routes), false, false, false, false, false);
    REQUIRE(!routes.enabled && routes.profiles.empty());
}

// This backend only answers construction-time capability queries. No tensor
// data, CPU ISA support, OpenCL runtime, or accelerator device is required.
struct fixture {
    ggml_context * ctx;
    ggml_backend_buffer_type shared_type {};
    ggml_backend_buffer_type native_type {};
    ggml_backend_buffer shared_buffer {};
    ggml_backend_buffer native_buffer {};
    ggml_backend_device device {};
    ggml_backend backend {};
    int probes = 0;
    bool reject_all = false;
    bool reject_down = false;

    fixture() : ctx(ggml_init({1024 * 1024, nullptr, true})) {
        REQUIRE(ctx != nullptr);
        shared_buffer.buft = &shared_type;
        native_buffer.buft = &native_type;
        ggml_shared_weights_register_buft(&shared_type);
        device.context = this;
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor * op) {
            auto & self = *static_cast<fixture *>(dev->context);
            ++self.probes;
            ggml_mul_mat_src0_composite_params params;
            REQUIRE(ggml_mul_mat_src0_composite_get_params(op, &params));
            REQUIRE(ggml_shared_weights_supports_op(op));
            return !self.reject_all && !(self.reject_down && params.axis == 0);
        };
        backend.device = &device;
    }
    ~fixture() {
        ggml_shared_weights_unregister_buft(&shared_type);
        ggml_free(ctx);
    }
    llm_graph_ffn_segment segment(bool shared, int64_t width,
            int64_t local_start, int64_t size, int64_t global_start,
            int64_t output_width = 3072) {
        auto * buffer = shared ? &shared_buffer : &native_buffer;
        auto * up = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 3072, width);
        auto * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 3072, width);
        auto * down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, output_width);
        up->buffer = gate->buffer = down->buffer = buffer;
        ggml_set_name(up, "weight_up");
        ggml_set_name(gate, "weight_gate");
        ggml_set_name(down, "weight_down");
        return {up, gate, down, local_start, size, global_start};
    }
    std::vector<llm_graph_ffn_segment> segments(bool cpu) {
        if (cpu) {
            return {segment(true, 6656, 5632, 1024, 6592), segment(false, 576, 0, 576, 7616)};
        }
        return {segment(false, 960, 0, 960, 0), segment(true, 6656, 0, 2048, 960)};
    }
};

void check_branch(fixture & f, const std::vector<llm_graph_ffn_segment> & segments,
        int64_t tokens, const std::string & hint, bool composite, bool backend_present = true,
        ggml_type input_type = GGML_TYPE_F32, bool contiguous_input = true) {
    REQUIRE(!segments.empty());
    auto * input = ggml_new_tensor_2d(f.ctx, input_type, 3072, tokens);
    if (!contiguous_input) { input->nb[1] += ggml_type_size(input_type); }
    ggml_set_name(input, "ffn_input");
    const std::string suffix = ".t256-p15-g15-gpu14." + std::string(hint == "CPU_REPACK" ? "cpu" : "gpu");
    std::vector<std::array<unsigned char, sizeof(ggml_tensor)>> original_weights;
    for (const auto & segment : segments) {
        for (auto * weight : {segment.up, segment.gate, segment.down}) {
            original_weights.emplace_back();
            std::memcpy(original_weights.back().data(), weight, sizeof(*weight));
        }
    }
    const int expected_nodes = composite ? 4 : 4 * int(segments.size()) + int(segments.size()) - 1;
    const size_t used_before_prediction = ggml_used_mem(f.ctx);
    const uint64_t predicted_nodes = llm_graph_ffn_segment_nodes(
            input, segments, hint, backend_present ? &f.backend : nullptr);
    REQUIRE(ggml_used_mem(f.ctx) == used_before_prediction);
    REQUIRE(predicted_nodes == uint64_t(expected_nodes));
    std::vector<ggml_tensor *> emitted;
    auto * result = llm_graph_build_ffn_segment_branch(f.ctx, input, segments,
            hint, suffix, backend_present ? &f.backend : nullptr,
            [&](ggml_tensor * node, const char * name, const char * backend_hint) {
                REQUIRE(hint == backend_hint);
                ggml_set_name(node, name);
                emitted.push_back(node);
            });
    REQUIRE(result != nullptr && result->ne[0] == segments[0].down->ne[1] && result->ne[1] == tokens);
    // GGML uses the same capacity for node and leaf arrays. A four-node
    // composite branch still references six distinct weight leaves plus its
    // input, so account for those leaves without a fixed oversized capacity.
    const size_t graph_capacity = std::max<uint64_t>(predicted_nodes, 3 * segments.size() + 1);
    auto * graph = ggml_new_graph_custom(f.ctx, graph_capacity, false);
    ggml_build_forward_expand(graph, result);
    REQUIRE(uint64_t(ggml_graph_n_nodes(graph)) <= predicted_nodes);
    REQUIRE(ggml_graph_n_nodes(graph) == expected_nodes);
    REQUIRE(int(emitted.size()) == expected_nodes);
    int allocated_composite_nodes = 0;
    for (auto * tensor = ggml_get_first_tensor(f.ctx); tensor; tensor = ggml_get_next_tensor(f.ctx, tensor)) {
        allocated_composite_nodes += tensor->op == GGML_OP_MUL_MAT_SRC0_COMPOSITE;
    }
    REQUIRE(allocated_composite_nodes == (composite ? 3 : 0));
    int matmuls = 0, glus = 0, adds = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_GLU) { ++glus; }
        else if (node->op == GGML_OP_ADD) { ++adds; }
        else {
            REQUIRE(node->op == (composite ? GGML_OP_MUL_MAT_SRC0_COMPOSITE : GGML_OP_MUL_MAT_SRC0_REGION));
            ++matmuls;
        }
        REQUIRE(std::string(node->name).find(suffix) != std::string::npos);
        if (composite || segments.size() == 1) {
            REQUIRE(std::string(node->name).find(".segment") == std::string::npos);
        }
    }
    REQUIRE(matmuls == (composite ? 3 : 3 * int(segments.size())));
    REQUIRE(glus == (composite ? 1 : int(segments.size())));
    REQUIRE(adds == (composite ? 0 : int(segments.size()) - 1));
    if (composite) {
        REQUIRE(emitted[0]->src[1] == input && emitted[1]->src[1] == input);
        REQUIRE(emitted[2]->src[0] == emitted[1] && emitted[2]->src[1] == emitted[0]);
        REQUIRE(result->src[1] == emitted[2]);
        for (int stage : {0, 1, 3}) {
            const auto * node = emitted[stage];
            ggml_mul_mat_src0_composite_params params;
            REQUIRE(ggml_mul_mat_src0_composite_get_params(node, &params));
            REQUIRE(params.axis == (stage == 3 ? 0 : 1));
            for (int piece = 0; piece < 2; ++piece) {
                const auto & segment = segments[piece];
                REQUIRE(params.start[piece] == segment.local_start && params.size[piece] == segment.size);
                const auto * weight = stage == 0 ? segment.up : stage == 1 ? segment.gate : segment.down;
                REQUIRE(node->src[piece == 0 ? 0 : 2] == weight);
            }
        }
        REQUIRE(emitted[2]->ne[0] == segments[0].size + segments[1].size);
        REQUIRE(std::string(result->name) == "ffn_down" + suffix);
    } else if (segments.size() > 1) {
        REQUIRE(std::string(result->name) == "ffn_segment_sum" + suffix);
        REQUIRE(std::string(emitted[0]->name) == "ffn_up" + suffix + ".segment0");
        REQUIRE(std::string(emitted[4]->name) == "ffn_up" + suffix + ".segment1");
    }
    size_t original = 0;
    for (const auto & segment : segments) {
        for (const auto * weight : {segment.up, segment.gate, segment.down}) {
            REQUIRE(std::memcmp(weight, original_weights[original++].data(), sizeof(*weight)) == 0);
        }
    }
}

void run_tests(bool repack_disabled) {
    for (bool cpu : {false, true}) {
        const std::string hint = cpu ? "CPU_REPACK" : "OpenCL";
        for (int64_t tokens : {1, 2, 3, 4, 9, 256, 512}) {
            fixture f;
            const bool composite = !repack_disabled && tokens >= 4;
            check_branch(f, f.segments(cpu), tokens, hint, composite);
            REQUIRE(f.probes == (composite ? 6 : 0));
        }
        // Rejecting the final probe must leave a wholly segmented graph with
        // no unreachable composite nodes accidentally emitted by the builder.
        {
            fixture f;
            f.reject_down = true;
            check_branch(f, f.segments(cpu), 256, hint, false);
            REQUIRE(f.probes == (repack_disabled ? 0 : 6));
        }
        {
            fixture f;
            f.reject_all = true;
            check_branch(f, f.segments(cpu), 256, hint, false);
            REQUIRE(f.probes == (repack_disabled ? 0 : 2));
        }
        {
            fixture f;
            check_branch(f, f.segments(cpu), 256, hint, false, false);
            REQUIRE(f.probes == 0);
        }
        for (bool shared : {false, true}) {
            fixture f;
            auto segments = f.segments(cpu);
            for (auto & segment : segments) {
                for (auto * weight : {segment.up, segment.gate, segment.down}) {
                    weight->buffer = shared ? &f.shared_buffer : &f.native_buffer;
                }
            }
            check_branch(f, segments, 256, hint, false);
            REQUIRE(f.probes == 0);
        }
        {
            fixture f;
            auto segments = f.segments(cpu);
            segments.resize(1);
            check_branch(f, segments, 256, hint, false);
            REQUIRE(f.probes == 0);
        }
        {
            fixture f;
            const std::vector<llm_graph_ffn_segment> segments = {
                f.segment(false, 512, 0, 512, 0),
                f.segment(true, 6656, 0, 2048, 512),
                f.segment(false, 448, 0, 448, 2560),
            };
            check_branch(f, segments, 256, hint, false);
            REQUIRE(f.probes == 0);
        }
        // These inputs remain valid for the REGION fallback but violate the
        // narrower composite contract. Prediction must cover the whole graph.
        {
            fixture f;
            check_branch(f, f.segments(cpu), 256, hint, false, true, GGML_TYPE_F16);
            REQUIRE(f.probes == 0);
        }
        {
            fixture f;
            check_branch(f, f.segments(cpu), 256, hint, false, true, GGML_TYPE_F32, false);
            REQUIRE(f.probes == 0);
        }
        {
            fixture f;
            const std::vector<llm_graph_ffn_segment> segments = {
                f.segment(false, 960, 0, 960, 0, 3073),
                f.segment(true, 6656, 0, 2048, 960, 3073),
            };
            check_branch(f, segments, 256, hint, false);
            REQUIRE(f.probes == 0);
        }
    }
    {
        fixture f;
        auto * input = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 3072, 256);
        const size_t used = ggml_used_mem(f.ctx);
        REQUIRE(llm_graph_ffn_segment_nodes(input, {}, "OpenCL", &f.backend) == 0);
        REQUIRE(ggml_used_mem(f.ctx) == used);
        REQUIRE(llm_graph_build_ffn_segment_branch(f.ctx, input, {}, "OpenCL", "", &f.backend,
                [](ggml_tensor *, const char *, const char *) { REQUIRE(false); }) == nullptr);
        REQUIRE(ggml_used_mem(f.ctx) == used);
        REQUIRE(f.probes == 0);
    }
    fixture f;
    check_branch(f, f.segments(false), 256, "HTP0", false);
    REQUIRE(f.probes == 0);
}

} // namespace

int main(int argc, char ** argv) {
    const bool repack_disabled = argc == 2 && std::strcmp(argv[1], "--repack-disabled") == 0;
    if (argc != 1 && !repack_disabled) {
        std::fprintf(stderr, "usage: %s [--repack-disabled]\n", argv[0]);
        return 2;
    }
    set_env("GGML_SHARED_HTP_WEIGHTS", "1");
    set_env("GGML_SHARED_HTP_REPACK", repack_disabled ? "0" : nullptr);
    set_env("GGML_SHARED_WEIGHTS_CSV", nullptr);
    try {
        run_runtime_route_flags_tests();
        run_tests(repack_disabled);
        std::printf("shared FFN composite graph: PASS (repack %s)\n", repack_disabled ? "disabled" : "default");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "shared FFN composite graph: %s\n", error.what());
        return 1;
    }
}
