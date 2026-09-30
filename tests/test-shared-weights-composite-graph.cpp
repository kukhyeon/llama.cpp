#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-shared-weights.h"
#include "llama-graph.h"

#include "../ggml/src/ggml-backend-impl.h"

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
            return !(self.reject_down && params.axis == 0);
        };
        backend.device = &device;
    }
    ~fixture() {
        ggml_shared_weights_unregister_buft(&shared_type);
        ggml_free(ctx);
    }
    llm_graph_ffn_segment segment(bool shared, int64_t width,
            int64_t local_start, int64_t size, int64_t global_start) {
        auto * buffer = shared ? &shared_buffer : &native_buffer;
        auto * up = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 3072, width);
        auto * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 3072, width);
        auto * down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, 3072);
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
        int64_t tokens, const std::string & hint, bool composite, bool backend_present = true) {
    auto * input = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 3072, tokens);
    ggml_set_name(input, "ffn_input");
    const std::string suffix = ".t256-p15-g15-gpu14." + std::string(hint == "CPU_REPACK" ? "cpu" : "gpu");
    std::vector<std::array<unsigned char, sizeof(ggml_tensor)>> original_weights;
    for (const auto & segment : segments) {
        for (auto * weight : {segment.up, segment.gate, segment.down}) {
            original_weights.emplace_back();
            std::memcpy(original_weights.back().data(), weight, sizeof(*weight));
        }
    }
    std::vector<ggml_tensor *> emitted;
    auto * result = llm_graph_build_ffn_segment_branch(f.ctx, input, segments,
            hint, suffix, backend_present ? &f.backend : nullptr,
            [&](ggml_tensor * node, const char * name, const char * backend_hint) {
                REQUIRE(hint == backend_hint);
                ggml_set_name(node, name);
                emitted.push_back(node);
            });
    REQUIRE(result != nullptr && result->ne[0] == 3072 && result->ne[1] == tokens);
    auto * graph = ggml_new_graph_custom(f.ctx, 64, false);
    ggml_build_forward_expand(graph, result);
    const int expected_nodes = composite || segments.size() == 1 ? 4 : 9;
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
    REQUIRE(matmuls == (composite || segments.size() == 1 ? 3 : 6));
    REQUIRE(glus == (composite || segments.size() == 1 ? 1 : 2));
    REQUIRE(adds == (composite || segments.size() == 1 ? 0 : 1));
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
    } else if (segments.size() == 2) {
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
            REQUIRE(f.probes == (composite ? 3 : 0));
        }
        // Rejecting the final probe must leave a wholly segmented graph with
        // no unreachable composite nodes accidentally emitted by the builder.
        {
            fixture f;
            f.reject_down = true;
            check_branch(f, f.segments(cpu), 256, hint, false);
            REQUIRE(f.probes == (repack_disabled ? 0 : 3));
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
        run_tests(repack_disabled);
        std::printf("shared FFN composite graph: PASS (repack %s)\n", repack_disabled ? "disabled" : "default");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "shared FFN composite graph: %s\n", error.what());
        return 1;
    }
}
