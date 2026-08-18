#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-backend-policy.h"
#include "llama-graph.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-add4: %s\n", message);
    }
    return condition;
}

float ordered_add4(float a, float b, float c, float d) {
    // Volatile intermediates make the test oracle independent of compiler
    // reassociation and force one F32 rounding point after every source step.
    volatile float bc = b + c;
    volatile float abc = a + bc;
    volatile float abcd = abc + d;
    return abcd;
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

ggml_backend_meta_split_state mirrored_split_state(
        const ggml_tensor * tensor, void * user_data) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(user_data);
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, { 0 }, 1 };
}

bool test_meta_backend_add4() {
    constexpr int64_t n = 4;
    constexpr size_t graph_size = 8;
    const ggml_init_params params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t seed_cpu = ggml_backend_cpu_init();
    ggml_backend_t meta = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    bool ok = check(ctx != nullptr && seed_cpu != nullptr,
            "meta backend setup failed");

    ggml_tensor * sum = nullptr;
    ggml_cgraph * graph = nullptr;
    if (ok) {
        ggml_backend_dev_t cpu_dev = ggml_backend_get_device(seed_cpu);
        ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
                &cpu_dev, 1, mirrored_split_state, nullptr);
        meta = ggml_backend_dev_init(meta_dev, nullptr);
        ok = check(meta != nullptr, "meta backend initialization failed") && ok;
    }

    ggml_tensor * inputs[4] = {};
    if (ok) {
        for (ggml_tensor *& input : inputs) {
            input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        }
        sum = ggml_add4(ctx, inputs[0], inputs[1], inputs[2], inputs[3]);
        graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, sum);
        ok = check(ggml_backend_supports_op(meta, sum),
                "meta backend rejected ADD4") && ok;
        buffer = ggml_backend_alloc_ctx_tensors(ctx, meta);
        ok = check(buffer != nullptr, "meta tensor allocation failed") && ok;
    }

    const float values[4][n] = {
        { 16777216.0f,  5.0f, -3.0f,  0.25f },
        {        1.0f, -7.0f,  9.0f, -0.50f },
        {-16777216.0f, 11.0f, -2.0f,  0.75f },
        {        0.0f, 13.0f,  4.0f, -1.00f },
    };
    if (ok) {
        for (int i = 0; i < 4; ++i) {
            ggml_backend_tensor_set(inputs[i], values[i], 0, sizeof(values[i]));
        }
        ok = check(ggml_backend_graph_compute(meta, graph) == GGML_STATUS_SUCCESS,
                "meta ADD4 graph compute failed") && ok;
    }

    if (ok) {
        float actual[n] = {};
        ggml_backend_tensor_get(sum, actual, 0, sizeof(actual));
        for (int64_t i = 0; i < n; ++i) {
            const float expected = ordered_add4(
                    values[0][i], values[1][i], values[2][i], values[3][i]);
            if (float_bits(actual[i]) != float_bits(expected)) {
                std::fprintf(stderr,
                        "test-add4: meta mismatch at %lld: got=%g expected=%g\n",
                        (long long) i, actual[i], expected);
                ok = false;
                break;
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(meta);
    ggml_backend_free(seed_cpu);
    ggml_free(ctx);
    return ok;
}

llama_backend_policy_ffn_parallel make_ffn_policy(
        int64_t npu,
        int64_t gpu,
        int64_t cpu,
        const char * reduce_backend) {
    llama_backend_policy_ffn_parallel policy;
    policy.enabled = true;
    policy.phase = "all";
    policy.align = 64;
    policy.reduce_backend = reduce_backend;
    policy.splits = {
        { "npu", 0,         npu, "HTP0-REPACK" },
        { "gpu", npu,       gpu, "OpenCL" },
        { "cpu", npu + gpu, cpu, "CPU_REPACK" },
    };
    return policy;
}

bool test_ffn_add4_eligibility() {
    constexpr int64_t n_ff = 8192;
    constexpr int64_t block = 32;
    const auto p0 = make_ffn_policy(3584, 3008, 1600, "CPU");
    auto p1 = make_ffn_policy(4096, 2048, 2048, "OpenCL");

    // Legacy reduce_backend intentionally does not constrain Step 7. The
    // fused node follows ffn_inp's concrete OpenCL placement instead.
    bool ok = check(
            llm_graph_ffn_add4_policies_eligible({ p0, p1 }, n_ff, block),
            "same three logical lanes were rejected") && true;

    auto zero_lane = p1;
    zero_lane.splits[2].size = 0;
    ok = check(
            !llm_graph_ffn_add4_policies_eligible({ p0, zero_lane }, n_ff, block),
            "zero-sized lane did not force fallback") && ok;

    auto changed_backend = p1;
    changed_backend.splits[1].backend = "CPU_REPACK";
    ok = check(
            !llm_graph_ffn_add4_policies_eligible({ p0, changed_backend }, n_ff, block),
            "changed lane backend did not force fallback") && ok;

    auto changed_id = p1;
    changed_id.splits[1].id = "gpu-other";
    ok = check(
            !llm_graph_ffn_add4_policies_eligible({ p0, changed_id }, n_ff, block),
            "changed logical lane did not force fallback") && ok;

    auto gap = p1;
    gap.splits[1].start += 64;
    ok = check(
            !llm_graph_ffn_add4_policies_eligible({ p0, gap }, n_ff, block),
            "non-exact cover did not force fallback") && ok;

    return ok;
}

bool test_ffn_add4_default_off() {
    const char * current = std::getenv("LLAMA_FFN_FUSED_ADD4");
    const bool had_value = current != nullptr;
    const std::string saved = had_value ? current : "";
#if defined(_WIN32)
    (void) _putenv_s("LLAMA_FFN_FUSED_ADD4", "");
#else
    (void) unsetenv("LLAMA_FFN_FUSED_ADD4");
#endif
    bool ok = check(!llm_graph_ffn_fused_add4_enabled(),
            "ADD4 fusion was not default-off");
#if defined(_WIN32)
    (void) _putenv_s("LLAMA_FFN_FUSED_ADD4", "1");
#else
    (void) setenv("LLAMA_FFN_FUSED_ADD4", "1", 1);
#endif
    ok = check(llm_graph_ffn_fused_add4_enabled(),
            "ADD4 fusion opt-in was ignored") && ok;

#if defined(_WIN32)
    (void) _putenv_s("LLAMA_FFN_FUSED_ADD4", had_value ? saved.c_str() : "");
#else
    if (had_value) {
        (void) setenv("LLAMA_FFN_FUSED_ADD4", saved.c_str(), 1);
    } else {
        (void) unsetenv("LLAMA_FFN_FUSED_ADD4");
    }
#endif
    return ok;
}

} // namespace

int main() {
    constexpr int64_t n = 1027;
    constexpr size_t graph_size = 8;

    const ggml_init_params params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = nullptr;
    bool ok = check(ctx != nullptr && backend != nullptr, "CPU setup failed");

    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * c = nullptr;
    ggml_tensor * d = nullptr;
    ggml_tensor * sum = nullptr;
    ggml_cgraph * graph = nullptr;

    if (ok) {
        a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        sum = ggml_add4(ctx, a, b, c, d);
        graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, sum);

        ok = check(sum->op == GGML_OP_ADD4, "constructor emitted the wrong op") && ok;
        ok = check(std::strcmp(ggml_op_name(sum->op), "ADD4") == 0,
                "op name table is out of sync") && ok;
        ok = check(std::strcmp(ggml_op_symbol(sum->op), "(x+(y+z))+w") == 0,
                "op symbol does not document the reduction order") && ok;
        ok = check(sum->src[0] == a && sum->src[1] == b &&
                   sum->src[2] == c && sum->src[3] == d,
                "constructor did not retain all four inputs") && ok;
        ok = check(ggml_backend_supports_op(backend, sum),
                "CPU backend rejected a valid ADD4") && ok;

        ggml_backend_cpu_set_n_threads(backend, 4);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ok = check(buffer != nullptr, "tensor allocation failed") && ok;
    }

    std::vector<float> av(n);
    std::vector<float> bv(n);
    std::vector<float> cv(n);
    std::vector<float> dv(n);
    std::vector<float> expected(n);
    std::vector<float> actual(n);

    if (ok) {
        for (int64_t i = 0; i < n; ++i) {
            av[i] = 0.25f * (float) (i - 400);
            bv[i] = -0.125f * (float) (i % 113);
            cv[i] = 0.0625f * (float) (i % 71);
            dv[i] = -0.03125f * (float) (i % 37);
        }

        // These two inputs distinguish the required tree from both common
        // reassociations. Element 0 differs from ((a+b)+c)+d and
        // (a+b)+(c+d); element 1 differs from a+((b+c)+d).
        av[0] = 16777216.0f;
        bv[0] = 1.0f;
        cv[0] = -16777216.0f;
        dv[0] = 0.0f;
        av[1] = 1.0e20f;
        bv[1] = -1.0e20f;
        cv[1] = 0.0f;
        dv[1] = 3.25f;

        for (int64_t i = 0; i < n; ++i) {
            expected[i] = ordered_add4(av[i], bv[i], cv[i], dv[i]);
        }

        volatile float left = av[0] + bv[0];
        left = left + cv[0];
        left = left + dv[0];
        volatile float bc_d = bv[1] + cv[1];
        bc_d = bc_d + dv[1];
        volatile float right = av[1] + bc_d;
        ok = check(float_bits(expected[0]) != float_bits(left),
                "left-fold sentinel does not distinguish the required order") && ok;
        ok = check(float_bits(expected[1]) != float_bits(right),
                "right-regroup sentinel does not distinguish the required order") && ok;

        ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
        ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
        ggml_backend_tensor_set(c, cv.data(), 0, cv.size() * sizeof(float));
        ggml_backend_tensor_set(d, dv.data(), 0, dv.size() * sizeof(float));

        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ok = check(status == GGML_STATUS_SUCCESS, "CPU graph compute failed") && ok;
        if (status == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(sum, actual.data(), 0, actual.size() * sizeof(float));
            for (int64_t i = 0; i < n; ++i) {
                if (float_bits(actual[i]) != float_bits(expected[i])) {
                    std::fprintf(stderr,
                            "test-add4: mismatch at %lld: got=%g expected=%g\n",
                            (long long) i, actual[i], expected[i]);
                    ok = false;
                    break;
                }
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    ok = test_meta_backend_add4() && ok;
    ok = test_ffn_add4_eligibility() && ok;
    ok = test_ffn_add4_default_off() && ok;
    return ok ? 0 : 1;
}
