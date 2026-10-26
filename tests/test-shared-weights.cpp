#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-shared-weights.h"
#include "llama.h"
#include "llama-backend-policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#define REQUIRE(x) do { if (!(x)) { throw std::runtime_error(#x); } } while (0)

struct q8_block { ggml_fp16_t d; int8_t q[32]; };
static_assert(sizeof(q8_block) == 34, "Q8 block size");
struct shape { int pk, rows, ks, kn, os, on, tokens; bool full = false; };

static void verify_policy(const char * path) {
    REQUIRE(llama_backend_policy_load(path, true, true));
    llama_backend_policy_residency_plan plan;
    REQUIRE(llama_backend_policy_build_ffn_residency_plan(0, plan));
    REQUIRE(!plan.keep_full_source);
    std::sort(plan.covers.begin(), plan.covers.end(), [](const auto & a, const auto & b) { return a.start < b.start; });
    int64_t end = 0;
    for (const auto & c : plan.covers) { REQUIRE(c.start == end); end += c.size; }
    REQUIRE(end == 8192);
    REQUIRE(plan.covers.size() == 3);
    REQUIRE(plan.covers[0].size == 960);
    REQUIRE(plan.covers[1].start == 960 && plan.covers[1].size == 6656);
    REQUIRE(plan.covers[2].start == 7616 && plan.covers[2].size == 576);
    for (auto p : {LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q,
                   LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_K,
                   LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_V}) {
        REQUIRE(llama_backend_policy_build_attn_qkv_residency_plan(0, p, plan));
        REQUIRE(plan.covers.size() == 1 && plan.covers[0].start == 0);
        REQUIRE(plan.covers[0].size == (p == LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q ? 3072 : 1024));
        REQUIRE(!plan.covers[0].consumers.empty());
    }
    std::vector<llama_backend_policy_ffn_parallel> policies;
    REQUIRE(llama_backend_policy_list_ffn_parallel_load_policies(0, policies));
    for (const auto & p : policies) {
        REQUIRE(p.splits.size() == 3);
        REQUIRE(p.splits[0].backend == "OpenCL");
        REQUIRE(p.splits[1].backend == "HTP0-REPACK");
        REQUIRE(p.splits[2].backend == "CPU_REPACK");
        REQUIRE(p.splits[0].start == 0);
        REQUIRE(p.splits[1].start == p.splits[0].size);
        REQUIRE(p.splits[2].start == p.splits[0].size + p.splits[1].size);
        REQUIRE(p.splits[2].start + p.splits[2].size == 8192);
    }
    ggml_shared_weights_log("validation", nullptr, "policy", -1, 0, 8192, "PASS: unified policy physical coverage and QKV sharing");
    std::printf("PASS unified policy (%zu distinct FFN layouts)\n", policies.size());
}

static ggml_backend_buffer_type_t find_shared_type() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        auto fn = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
        if (fn) {
            for (auto * p = fn(dev); p && *p; ++p) {
                if (ggml_shared_weights_is_buft(*p)) { return *p; }
            }
        }
    }
    return nullptr;
}

static void run_case(const shape & s, ggml_backend_buffer_type_t buft,
        const std::vector<ggml_backend_t> & backends, bool mock, bool native_compare = false) {
    ggml_init_params init = {1024 * 1024, nullptr, true};
    auto * wc = ggml_init(init);
    // Nonzero allocation offset catches readers that accidentally ignore the
    // shared tensor's base offset (in particular the imported OpenCL buffer).
    ggml_new_tensor_1d(wc, GGML_TYPE_Q8_0, 256);
    auto * w = ggml_new_tensor_2d(wc, GGML_TYPE_Q8_0, s.pk, s.rows);
    ggml_set_name(w, "shared_test_weight");
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, buft);
    REQUIRE(wb);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<q8_block> blocks((size_t) s.pk * s.rows / 32);
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i].d = ggml_fp32_to_fp16(0.002f * (1 + (i * 7) % 19));
        for (int j = 0; j < 32; ++j) { blocks[i].q[j] = (int) ((i * 13 + j * 37) % 256) - 128; }
    }
    ggml_backend_buffer_type mock_type = *buft;
    if (mock) {
        // Synthetic HTP layout for host-only coverage; no Hexagon dependency.
        mock_type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        wb->buft = &mock_type;
        ggml_shared_weights_register_buft(&mock_type);
        ggml_shared_weights_register_buffer(wb, ggml_backend_buffer_get_base(wb),
            ggml_backend_buffer_get_size(wb), ggml_backend_buffer_get_size(wb), -1);
        for (int row = 0; row < s.rows; ++row) {
            auto * q = (int8_t *) w->data + row * w->nb[1];
            auto * d = (ggml_fp16_t *) (q + s.pk);
            for (int k = 0; k < s.pk / 32; ++k) {
                const auto & b = blocks[row * (s.pk / 32) + k];
                std::memcpy(q + k * 32, b.q, 32); d[k] = b.d;
            }
        }
    } else { ggml_backend_tensor_set(w, blocks.data(), 0, ggml_nbytes(w)); }
    const auto * identity = ggml_shared_weights_buffer(wb);
    REQUIRE(identity);
    const uint64_t allocation_id = identity->allocation_id;
    REQUIRE(w->data != identity->base);
    std::vector<uint8_t> weight_before(ggml_nbytes(w));
    std::memcpy(weight_before.data(), w->data, weight_before.size());
    for (auto backend : backends) {
        auto dev = ggml_backend_get_device(backend);
        using prepare_t = bool (*)(ggml_backend_dev_t, const ggml_tensor *);
        auto prepare = reinterpret_cast<prepare_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_prepare_shared_weight"));
        if (prepare) { REQUIRE(prepare(dev, w)); }
    }
    std::vector<float> input((size_t) s.kn * s.tokens);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = 0.3f * std::sin((float) ((i * 11) % 997) * 0.031f); }
    std::vector<double> reference((size_t) s.on * s.tokens);
    for (int t = 0; t < s.tokens; ++t) {
        for (int row = 0; row < s.on; ++row) {
            double sum = 0;
            for (int k = 0; k < s.kn; ++k) {
                const auto & b = blocks[(s.os + row) * (s.pk / 32) + (s.ks + k) / 32];
                sum += (double) ggml_fp16_to_fp32(b.d) * b.q[(s.ks + k) % 32] * input[t * s.kn + k];
            }
            reference[t * s.on + row] = sum;
        }
    }
    // Run all device readers concurrently against the SAME physical source.
    std::vector<std::future<void>> runs;
    for (auto backend : backends) {
        runs.push_back(std::async(std::launch::async, [&, backend] {
            auto * ctx = ggml_init(init);
            auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.kn, s.tokens);
            auto * y = s.full ? ggml_mul_mat(ctx, w, x) :
                ggml_mul_mat_src0_region(ctx, w, x, s.ks, s.kn, s.os, s.on);
            REQUIRE(ggml_backend_supports_buft(backend, w->buffer->buft));
            REQUIRE(ggml_backend_supports_op(backend, y));
            auto * graph = ggml_new_graph_custom(ctx, 16, false);
            ggml_build_forward_expand(graph, y);
            auto buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            REQUIRE(buf);
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            const bool cpu = std::strcmp(ggml_backend_name(backend), "CPU") == 0;
            const bool gpu = std::strcmp(ggml_backend_name(backend), "OpenCL") == 0;
            const bool packed = native_compare && s.on % 4 == 0 &&
                ((cpu && s.tokens >= 4) || (gpu && s.tokens > 1));
            std::vector<double> comparison = reference;
            if (packed) {
                // Compare to the existing backend arithmetic, not a relaxed
                // FP64 threshold: native CPU quantizes activations; Adreno's
                // native GEMM accumulates in half precision.
                auto * nc = ggml_init(init);
                auto * nw = ggml_new_tensor_2d(nc, GGML_TYPE_Q8_0, s.kn, s.on);
                auto nwb = ggml_backend_alloc_ctx_tensors(nc, backend);
                REQUIRE(nwb);
                ggml_backend_buffer_set_usage(nwb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                std::vector<q8_block> selected((size_t)s.kn * s.on / 32);
                for (int row = 0; row < s.on; ++row) {
                    std::copy_n(blocks.data() + (s.os + row) * (s.pk / 32) + s.ks / 32,
                        s.kn / 32, selected.data() + row * (s.kn / 32));
                }
                ggml_backend_tensor_set(nw, selected.data(), 0, ggml_nbytes(nw));
                auto * rc = ggml_init(init);
                auto * rx = ggml_new_tensor_2d(rc, GGML_TYPE_F32, s.kn, s.tokens);
                auto * ry = ggml_mul_mat(rc, nw, rx);
                auto * rg = ggml_new_graph_custom(rc, 16, false);
                ggml_build_forward_expand(rg, ry);
                auto rb = ggml_backend_alloc_ctx_tensors(rc, backend);
                REQUIRE(rb);
                ggml_backend_tensor_set(rx, input.data(), 0, ggml_nbytes(rx));
                REQUIRE(ggml_backend_graph_compute(backend, rg) == GGML_STATUS_SUCCESS);
                std::vector<float> native_output(reference.size());
                ggml_backend_tensor_get(ry, native_output.data(), 0, ggml_nbytes(ry));
                comparison.assign(native_output.begin(), native_output.end());
                if (cpu) {
                    using pack_fn = bool (*)(const ggml_tensor *, int, void *, size_t);
                    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                    auto pack = reinterpret_cast<pack_fn>(ggml_backend_reg_get_proc_address(reg,
                        "ggml_backend_cpu_shared_repack_pack"));
                    auto pack_reference = reinterpret_cast<pack_fn>(ggml_backend_reg_get_proc_address(reg,
                        "ggml_backend_cpu_shared_repack_pack_reference"));
                    REQUIRE(pack && pack_reference);
                    for (int interleave : {4, 8}) {
                        const size_t bytes = ggml_nbytes(nw);
                        std::vector<uint8_t> actual_pack(bytes + 128, 0xa5), expected_pack(bytes);
                        REQUIRE(pack(y, interleave, actual_pack.data() + 64, bytes));
                        REQUIRE(pack_reference(nw, interleave, expected_pack.data(), bytes));
                        REQUIRE(std::memcmp(actual_pack.data() + 64, expected_pack.data(), bytes) == 0);
                        REQUIRE(std::all_of(actual_pack.begin(), actual_pack.begin() + 64,
                            [](uint8_t v) { return v == 0xa5; }));
                        REQUIRE(std::all_of(actual_pack.end() - 64, actual_pack.end(),
                            [](uint8_t v) { return v == 0xa5; }));
                        REQUIRE(!pack(y, interleave, actual_pack.data() + 64, bytes - 1));
                    }
                }
                ggml_backend_buffer_free(rb); ggml_free(rc);
                ggml_backend_buffer_free(nwb); ggml_free(nc);
            }
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            if (packed) {
                ggml_shared_weights_scratch_stats first{}, second{};
                REQUIRE(ggml_shared_weights_get_scratch_stats(backend, &first));
                REQUIRE(first.owner);
                REQUIRE(first.weight_request_bytes >= ggml_row_size(GGML_TYPE_Q8_0, s.kn) * s.on);
                REQUIRE(first.allocation_bytes >= first.request_bytes);
                // Repeat on the same backend: the scratch allocation is reused.
                REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                REQUIRE(ggml_shared_weights_get_scratch_stats(backend, &second));
                REQUIRE(first.owner == second.owner && first.allocation_bytes == second.allocation_bytes);
                if (cpu) {
                    auto plan = ggml_backend_graph_plan_create(backend, graph);
                    REQUIRE(plan);
                    ggml_shared_weights_scratch_stats planned{}, freed{};
                    REQUIRE(ggml_shared_weights_get_scratch_stats(backend, &planned));
                    REQUIRE(planned.allocation_bytes > second.allocation_bytes);
                    REQUIRE(ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS);
                    ggml_backend_graph_plan_free(backend, plan);
                    REQUIRE(ggml_shared_weights_get_scratch_stats(backend, &freed));
                    REQUIRE(freed.allocation_bytes == second.allocation_bytes);
                }
            }
            std::vector<float> actual(reference.size());
            ggml_backend_tensor_get(y, actual.data(), 0, ggml_nbytes(y));
            double error = 0, norm = 0, max_error = 0, max_reference = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                REQUIRE(std::isfinite(actual[i]));
                const double delta = actual[i] - comparison[i];
                error += delta * delta; norm += comparison[i] * comparison[i];
                max_error = std::max(max_error, std::abs(delta));
                max_reference = std::max(max_reference, std::abs(comparison[i]));
            }
            const double nmse = error / std::max(norm, 1e-20);
            const bool htp = std::strncmp(ggml_backend_name(backend), "HTP", 3) == 0;
            // Direct readers retain the strict FP64 reference. Native scratch
            // compares strictly to the existing native backend result.
            REQUIRE(nmse < (htp ? 5e-4 : 1e-10));
            if (!htp) { REQUIRE(max_error < 2e-4 * std::max(1.0, max_reference)); }
            REQUIRE(ggml_shared_weights_buffer(wb)->allocation_id == allocation_id);
            char detail[256];
            std::snprintf(detail, sizeof(detail), "PASS nmse=%.9g parent_k=%d k_start=%d k_size=%d out_start=%d out_size=%d tokens=%d full=%d",
                nmse, s.pk, s.ks, s.kn, s.os, s.on, s.tokens, s.full);
            ggml_shared_weights_log("validation", w, ggml_backend_name(backend), 0, s.ks, s.kn, detail);
            std::printf("%s reference=%s %s\n", ggml_backend_name(backend),
                packed ? "native" : "fp64", detail);
            ggml_backend_buffer_free(buf); ggml_free(ctx);
        }));
    }
    for (auto & run : runs) { run.get(); }
    REQUIRE(std::memcmp(weight_before.data(), w->data, weight_before.size()) == 0);
    ggml_backend_buffer_free(wb);
    if (mock) { ggml_shared_weights_unregister_buft(&mock_type); }
    ggml_free(wc);
}

struct route_probe {
    std::vector<std::string> profiles;
    std::set<std::string> observed;
    size_t next = 0;
    int calls = 0;
    bool cycle = false;
    bool valid = true;
    std::string pending;
};

static void probe_layer(llama_context * ctx, int32_t layer, int32_t tokens, void * data) {
    auto & p = *static_cast<route_probe *>(data);
    const char * active = llama_runtime_route_active_profile(ctx);
    if (!p.pending.empty() && (!active || p.pending != active)) { p.valid = false; }
    if (active && *active) { p.observed.insert(active); }
    const auto & next = p.profiles[p.cycle ? p.next++ % p.profiles.size() : 0];
    if (!llama_runtime_route_request_profile(ctx, next.c_str())) { p.valid = false; }
    p.pending = next;
    ++p.calls;
    ggml_shared_weights_log("validation_route", nullptr, active, -1, layer, tokens,
        p.valid ? "PASS: accepted prior boundary request" : "FAIL: route request/acceptance mismatch");
}

// Integration fixture for the supplied Llama 3.2 3B unified policy. Load only
// once, then exercise every profile, token bucket, candidate reuse and decode.
static void verify_model(const char * model_path, const char * policy_path) {
    if (ggml_shared_weights_enabled()) { verify_policy(policy_path); }
    else { REQUIRE(llama_backend_policy_load(policy_path, true, true)); }
    ggml_backend_load_all();
    llama_backend_init();
    auto htp = ggml_backend_dev_by_name("HTP0");
    auto cl = ggml_backend_reg_by_name("OpenCL");
    REQUIRE(htp && cl);
    ggml_backend_dev_t devices[] = {htp, ggml_backend_reg_dev_get(cl, 0), nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices; mp.n_gpu_layers = 99;
    mp.attn_qkv_shards = true; mp.attn_out_shards = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path, mp), llama_model_free);
    REQUIRE(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.attn_qkv_shards = true; cp.attn_out_shards = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    REQUIRE(ctx);
    route_probe probe;
    REQUIRE(llama_backend_policy_list_runtime_route_profiles(true, probe.profiles));
    REQUIRE(llama_runtime_route_set_layer_producer(ctx.get(), probe_layer, &probe));
    const auto * vocab = llama_model_get_vocab(model.get());
    const int nv = llama_vocab_n_tokens(vocab);
    std::string prompt;
    for (int i = 0; i < 64; ++i) {
        prompt += " The river flows through the valley. Explain how water shapes the landscape over time.";
    }
    std::vector<llama_token> tokens(prompt.size() + 8);
    const int count = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, false);
    REQUIRE(count >= 512);
    for (int n : {64, 128, 256, 512}) {
        std::vector<float> reference;
        for (bool cycle : {false, true}) {
            probe.cycle = cycle; probe.calls = 0; probe.pending.clear();
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            auto batch = llama_batch_get_one(tokens.data(), n);
            REQUIRE(llama_decode(ctx.get(), batch) == 0);
            REQUIRE(probe.valid && probe.calls == llama_model_n_layer(model.get()) - 1);
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            REQUIRE(logits);
            double error = 0, norm = 0;
            for (int i = 0; i < nv; ++i) {
                REQUIRE(std::isfinite(logits[i]));
                if (cycle) {
                    const double diff = logits[i] - reference[i];
                    error += diff * diff; norm += (double) reference[i] * reference[i];
                }
            }
            if (!cycle) { reference.assign(logits, logits + nv); continue; }
            const double nmse = error / std::max(norm, 1e-20);
            // Mixed native/HTP arithmetic is not bit-identical; this checks for
            // large graph/route errors, not model-quality equivalence.
            std::fprintf(stderr, "model tokens=%d fixed_vs_routed NMSE=%.9g reference_norm=%.9g error=%.9g\n", n, nmse, norm, error);
            REQUIRE(nmse < 0.01);
            llama_token next = std::max_element(logits, logits + nv) - logits;
            REQUIRE(llama_decode(ctx.get(), llama_batch_get_one(&next, 1)) == 0);
            logits = llama_get_logits_ith(ctx.get(), -1);
            for (int i = 0; i < nv; ++i) { REQUIRE(std::isfinite(logits[i])); }
            char detail[256];
            std::snprintf(detail, sizeof(detail), "PASS tokens=%d fixed_vs_routed_logit_nmse=%.9g boundaries=%d; decode finite; model loaded once", n, nmse, probe.calls);
            ggml_shared_weights_log("validation", nullptr, "model", -1, 0, n, detail);
            std::printf("%s\n", detail); std::fflush(stdout);
        }
    }
    for (const auto & profile : probe.profiles) { REQUIRE(probe.observed.count(profile) != 0); }
    REQUIRE(llama_runtime_route_set_layer_producer(ctx.get(), nullptr, nullptr));
    std::printf("PASS all %zu runtime profiles observed without model reload\n", probe.observed.size());
    ggml_shared_weights_log("validation", nullptr, "model", -1, 0, probe.observed.size(), "PASS: every runtime profile observed without model reload");
}

// GGML_SHARED_REPACK_BREAKDOWN=1 samples every CPU repack: the results must be
// unchanged, and every sampled repack must log its read and copy probes.
static void check_breakdown_rows(const std::string & path) {
    ggml_shared_weights_log("validation", nullptr, "test", -1, 0, 0, "flush"); // flushes buffered rows
    FILE * f = std::fopen(path.c_str(), "r");
    REQUIRE(f);
    char line[4096];
    int repacks = 0, reads = 0, copies = 0;
    while (std::fgets(line, sizeof(line), f)) {
        const std::string row = line;
        if (row.rfind("\"repack\",", 0) == 0) { ++repacks; }
        if (row.rfind("\"repack_probe\",", 0) == 0) {
            reads += row.find(",\"read\",") != std::string::npos;
            copies += row.find(",\"copy\",") != std::string::npos;
        }
    }
    std::fclose(f);
    std::remove(path.c_str());
    REQUIRE(repacks > 0 && reads == repacks && copies == repacks);
    std::printf("PASS repack breakdown: %d sampled repacks with read/copy probes\n", repacks);
}

int main(int argc, char ** argv) {
    const bool native = argc == 4 && std::strcmp(argv[1], "--model-native") == 0;
    const bool disabled = argc == 2 && std::strcmp(argv[1], "--disabled") == 0;
    const bool breakdown = argc == 2 && std::strcmp(argv[1], "--native-repack-breakdown") == 0;
    const bool native_compare = breakdown || (argc == 2 &&
        (std::strcmp(argv[1], "--native-repack") == 0 || std::strcmp(argv[1], "--device-native") == 0));
    std::string breakdown_csv;
#if !defined(_WIN32)
    if (breakdown) {
        const char * tmp = std::getenv("TMPDIR");
        breakdown_csv = std::string(tmp && *tmp ? tmp : "/tmp") + "/test-shared-weights-breakdown-" +
            std::to_string((long long) getpid()) + ".csv";
        std::remove(breakdown_csv.c_str());
        setenv("GGML_SHARED_WEIGHTS_CSV", breakdown_csv.c_str(), 1);
        setenv("GGML_SHARED_REPACK_BREAKDOWN", "1", 1);
        setenv("GGML_SHARED_REPACK_BREAKDOWN_EVICT_MB", "1", 1);
    }
#endif
    // Keep the original direct-reader fixture as an independent regression.
#if defined(_WIN32)
    _putenv_s("GGML_SHARED_HTP_REPACK", native_compare ? "1" : "0");
#else
    if (argc != 4) { setenv("GGML_SHARED_HTP_REPACK", native_compare ? "1" : "0", 1); }
#endif
#if defined(_WIN32)
    _putenv_s("GGML_SHARED_HTP_WEIGHTS", (native || disabled) ? "0" : "1");
#else
    setenv("GGML_SHARED_HTP_WEIGHTS", (native || disabled) ? "0" : "1", 1);
#endif
    llama_log_set([](ggml_log_level level, const char * message, void *) {
        if (level != GGML_LOG_LEVEL_DEBUG) { std::fputs(message, stderr); }
    }, nullptr);
    try {
        if (disabled) {
            auto type = *ggml_backend_cpu_buffer_type();
            ggml_shared_weights_register_buft(&type);
            REQUIRE(!ggml_shared_weights_enabled());
            REQUIRE(!ggml_shared_weights_is_buft(&type));
            ggml_shared_weights_unregister_buft(&type);
            std::printf("PASS shared weight dispatch disabled\n");
            return 0;
        }
        if (argc == 4 && (native || std::strcmp(argv[1], "--model") == 0)) { verify_model(argv[2], argv[3]); return 0; }
        if (argc == 3 && std::strcmp(argv[1], "--policy") == 0) { verify_policy(argv[2]); return 0; }
        ggml_backend_load_all();
        const bool device = argc == 2 &&
            (std::strcmp(argv[1], "--device") == 0 || std::strcmp(argv[1], "--device-native") == 0);
        if (native_compare) {
            auto reg = ggml_backend_reg_by_name("CPU"); REQUIRE(reg);
            using test_mode_fn = void (*)(bool);
            auto force = reinterpret_cast<test_mode_fn>(ggml_backend_reg_get_proc_address(reg,
                "ggml_backend_cpu_shared_repack_test_mode"));
            REQUIRE(force); force(true);
            ggml_shared_weights_set_query_id(7);
            ggml_shared_weights_set_phase(GGML_BACKEND_SCHED_PROFILE_PREFILL);
        }
        auto buft = device ? find_shared_type() : ggml_backend_cpu_buffer_type();
        REQUIRE(buft);
        std::vector<ggml_backend_t> backends;
        backends.push_back(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        REQUIRE(backends.back());
        ggml_backend_cpu_set_n_threads(backends[0], 2);
        if (device) {
            auto opencl = ggml_backend_reg_by_name("OpenCL"); REQUIRE(opencl);
            backends.push_back(ggml_backend_dev_init(ggml_backend_reg_dev_get(opencl, 0), nullptr)); REQUIRE(backends.back());
            backends.push_back(ggml_backend_dev_init(ggml_backend_buft_get_device(buft), nullptr)); REQUIRE(backends.back());
        }
        run_case({3072, 32, 0, 3072, 0, 32, 1, true}, buft, backends, !device, native_compare);
        run_case({3072, 96, 0, 3072, 32, 32, 64}, buft, backends, !device, native_compare);
        for (int n : {1, 64, 128, 256, 512}) {
            run_case({6656, 64, 2688, 1792, 0, 32, n}, buft, backends, !device, native_compare);
        }
        run_case({6656, 64, 64, 256, 32, 32, 64}, buft, backends, !device, native_compare);
        // CPU/OpenCL edge cases: partial 32x32 GPU tiles, partial 2x4 CPU
        // tiles, short K, and unaligned channel starts. These regions are not
        // HTP matmul shapes, though they read the same HTP physical layout.
        auto readers = backends;
        if (device) { readers.pop_back(); }
        for (int n : {1, 3, 7, 8, 9, 31, 32, 33, 65}) {
            run_case({512, 96, 32, 96, 3, 35, n}, buft, readers, !device, native_compare);
        }
        ggml_backend_cpu_set_n_threads(backends[0], 4);
        run_case({256, 32, 224, 32, 31, 1, 5}, buft, readers, !device, native_compare);
        run_case({512, 96, 96, 352, 5, 67, 129}, buft, readers, !device, native_compare);
        ggml_backend_cpu_set_n_threads(backends[0], 1);
        run_case({256, 35, 0, 256, 0, 35, 33, true}, buft, readers, !device, native_compare);
        if (native_compare) {
            ggml_backend_cpu_set_n_threads(backends[0], 4);
            run_case({512, 96, 32, 96, 3, 36, 7}, buft, readers, !device, true);
            run_case({256, 16, 224, 32, 7, 4, 5}, buft, readers, !device, true);
            ggml_backend_cpu_set_n_threads(backends[0], 1);
            run_case({256, 32, 0, 256, 0, 32, 33, true}, buft, readers, !device, true);
        }
        for (auto backend : backends) { ggml_backend_free(backend); }
        if (breakdown) { check_breakdown_rows(breakdown_csv); }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "shared weight validation failed: %s\n", error.what());
        ggml_shared_weights_log("validation", nullptr, "test", -1, 0, 0, error.what());
        return 1;
    }
}
