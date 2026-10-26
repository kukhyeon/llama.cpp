#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-shared-weights.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#define REQUIRE(x) do { if (!(x)) { throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #x); } } while (0)

struct q8_block { ggml_fp16_t d; int8_t q[32]; };
static_assert(sizeof(q8_block) == 34, "Q8 block size");
using pack_fn = bool (*)(const ggml_tensor *, int, void *, size_t);
using layout_fn = bool (*)(ggml_tensor *, int);
using stats_fn = void (*)(uint64_t *, uint64_t *, bool);

struct weight_fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_type shared_type{};
    bool shared;
    std::vector<q8_block> blocks;
    std::vector<uint8_t> original;

    weight_fixture(int k, int m, int seed, bool is_shared, ggml_backend_buffer_type_t type,
            int interleave, pack_fn pack_reference, layout_fn set_layout) : shared(is_shared) {
        ctx = ggml_init({65536, nullptr, true});
        REQUIRE(ctx);
        ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, 256); // Nonzero physical tensor offset.
        tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, type);
        REQUIRE(buffer);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        blocks.resize((size_t) k * m / 32);
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i].d = ggml_fp32_to_fp16(0.002f * (1 + (i * 7 + seed) % 19));
            for (int j = 0; j < 32; ++j) { blocks[i].q[j] = (int)((i * 13 + j * 37 + seed) % 255) - 127; }
        }
        if (shared) {
            shared_type = *type;
            shared_type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
            buffer->buft = &shared_type;
            ggml_shared_weights_register_buft(&shared_type);
            ggml_shared_weights_register_buffer(buffer, ggml_backend_buffer_get_base(buffer),
                ggml_backend_buffer_get_size(buffer), ggml_backend_buffer_get_size(buffer), -1);
            for (int row = 0; row < m; ++row) {
                auto * q = (int8_t *) tensor->data + row * tensor->nb[1];
                auto * d = (ggml_fp16_t *) (q + k);
                for (int b = 0; b < k / 32; ++b) {
                    const auto & block = blocks[row * (k / 32) + b];
                    std::memcpy(q + 32 * b, block.q, 32); d[b] = block.d;
                }
            }
        } else if (interleave) {
            ggml_tensor canonical = *tensor;
            canonical.data = blocks.data(); canonical.extra = nullptr;
            REQUIRE(pack_reference(&canonical, interleave, tensor->data, ggml_nbytes(tensor)));
            REQUIRE(set_layout(tensor, interleave));
        } else {
            ggml_backend_tensor_set(tensor, blocks.data(), 0, ggml_nbytes(tensor));
        }
        original.resize(ggml_nbytes(tensor));
        std::memcpy(original.data(), tensor->data, original.size());
    }
    ~weight_fixture() {
        ggml_backend_buffer_free(buffer);
        if (shared) { ggml_shared_weights_unregister_buft(&shared_type); }
        ggml_free(ctx);
    }
    void unchanged() const { REQUIRE(std::memcmp(original.data(), tensor->data, original.size()) == 0); }
};

static void check_equal(const ggml_tensor * actual, const ggml_tensor * reference) {
    REQUIRE(ggml_nbytes(actual) == ggml_nbytes(reference));
    std::vector<float> a(ggml_nelements(actual)), b(a.size());
    ggml_backend_tensor_get(actual, a.data(), 0, ggml_nbytes(actual));
    ggml_backend_tensor_get(reference, b.data(), 0, ggml_nbytes(reference));
    double error = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::isfinite(a[i]));
        error += (double)(a[i] - b[i]) * (a[i] - b[i]);
        norm += (double)b[i] * b[i];
    }
    REQUIRE(error / std::max(norm, 1e-20) < 1e-10);
}

static std::vector<q8_block> assemble_reference(const weight_fixture & w0, const weight_fixture & w1,
        int axis, const int starts[2], const int sizes[2], int shift = 0) {
    const weight_fixture * sources[] = {&w0, &w1};
    const int k = axis == 0 ? sizes[0] + sizes[1] : w0.tensor->ne[0];
    const int m = axis == 1 ? sizes[0] + sizes[1] : w0.tensor->ne[1];
    std::vector<q8_block> merged((size_t) k * m / 32);
    for (int row = 0; row < m; ++row) {
        for (int block = 0; block < k / 32; ++block) {
            const int position = axis == 0 ? block * 32 : row;
            const int part = position >= sizes[0] ? 1 : 0;
            const int local = starts[part] + shift + position - (part ? sizes[0] : 0);
            const int source_row = axis == 1 ? local : row;
            const int source_block = axis == 0 ? local / 32 : block;
            merged[row * (k / 32) + block] = sources[part]->blocks[
                source_row * (sources[part]->tensor->ne[0] / 32) + source_block];
        }
    }
    return merged;
}

static void check_packed_bytes(const ggml_tensor * op, const ggml_tensor * canonical,
        pack_fn pack, pack_fn pack_reference) {
    for (int interleave : {4, 8}) {
        const size_t bytes = ggml_nbytes(canonical);
        std::vector<uint8_t> got(bytes + 128, 0xa5), expected(bytes);
        REQUIRE(pack(op, interleave, got.data() + 64, bytes));
        REQUIRE(pack_reference(canonical, interleave, expected.data(), bytes));
        REQUIRE(std::memcmp(got.data() + 64, expected.data(), bytes) == 0);
        REQUIRE(std::all_of(got.begin(), got.begin() + 64, [](uint8_t v) { return v == 0xa5; }));
        REQUIRE(std::all_of(got.end() - 64, got.end(), [](uint8_t v) { return v == 0xa5; }));
        const auto previous = got;
        REQUIRE(!pack(op, interleave, got.data() + 64, bytes - 1));
        REQUIRE(got == previous);
    }
}

static void run_case(ggml_backend_t backend, ggml_backend_buffer_type_t native_type,
        int axis, bool shared_first, int native_interleave, int tokens, bool aligned,
        pack_fn pack, pack_fn pack_reference, layout_fn set_layout, stats_fn stats) {
    // Keep unaligned row-concat coverage. Aligned shapes exercise both bulk-copy
    // paths with five row groups, or only three groups for four worker threads.
    const int starts[2] = {axis == 0 ? 32 : (aligned ? 4 : 3), axis == 0 ? 64 : (aligned ? 8 : 5)};
    const int sizes[2] = {axis == 0 ? 64 : (aligned ? 8 : 13), axis == 0 ? 96 : (aligned ? 12 : 19)};
    const int k_concat_rows = aligned ? 12 : 32;
    weight_fixture w0(128, axis == 0 ? k_concat_rows : 24, 1, shared_first,
        shared_first ? ggml_backend_cpu_buffer_type() : native_type, shared_first ? 0 : native_interleave,
        pack_reference, set_layout);
    weight_fixture w1(axis == 0 ? 192 : 128, axis == 0 ? k_concat_rows : 32, 11, !shared_first,
        !shared_first ? ggml_backend_cpu_buffer_type() : native_type, !shared_first ? 0 : native_interleave,
        pack_reference, set_layout);
    const int k = axis == 0 ? sizes[0] + sizes[1] : 128;
    const int m = axis == 1 ? sizes[0] + sizes[1] : k_concat_rows;
    const int up_shift = axis == 0 ? 32 : (aligned ? 4 : 1);
    const auto merged = assemble_reference(w0, w1, axis, starts, sizes);
    const auto merged_up = assemble_reference(w0, w1, axis, starts, sizes, up_shift);
    auto * ctx = ggml_init({1024 * 1024, nullptr, true}); REQUIRE(ctx);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
    auto * y0 = ggml_mul_mat_src0_composite(ctx, w0.tensor, w1.tensor, x, axis,
        starts[0], sizes[0], starts[1], sizes[1]);
    // Reused activation must still be multiplied by freshly assembled weights.
    auto * y1 = ggml_mul_mat_src0_composite(ctx, w0.tensor, w1.tensor, x, axis,
        starts[0] + up_shift, sizes[0], starts[1] + up_shift, sizes[1]);
    ggml_set_name(y0, "ffn_gate.composite"); ggml_set_name(y1, "ffn_up.composite");
    auto * nw = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
    auto * reference = ggml_mul_mat(ctx, nw, x);
    auto * nw_up = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
    auto * reference_up = ggml_mul_mat(ctx, nw_up, x);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, y0); ggml_build_forward_expand(graph, y1);
    auto * rg = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(rg, reference); ggml_build_forward_expand(rg, reference_up);
    auto * barrier = ggml_scale(ctx, x, 0.5f);
    auto * interrupted = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(interrupted, y0);
    ggml_build_forward_expand(interrupted, barrier);
    ggml_build_forward_expand(interrupted, y1);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); REQUIRE(buffer);
    ggml_backend_tensor_set(nw, merged.data(), 0, ggml_nbytes(nw));
    ggml_backend_tensor_set(nw_up, merged_up.data(), 0, ggml_nbytes(nw_up));
    REQUIRE(ggml_backend_supports_op(backend, y0));
    ggml_mul_mat_src0_composite_params params;
    REQUIRE(ggml_mul_mat_src0_composite_get_params(y0, &params));
    auto invalid = *y0;
    params.start[0] = INT64_MAX;
    std::memcpy(invalid.op_params, &params, sizeof(params));
    REQUIRE(!ggml_mul_mat_src0_composite_get_params(&invalid, &params));
    REQUIRE(!ggml_backend_supports_op(backend, &invalid));
    check_packed_bytes(y0, nw, pack, pack_reference);
    check_packed_bytes(y1, nw_up, pack, pack_reference);
    std::vector<float> input(ggml_nelements(x));
    size_t reserved = 0;
    for (int query = 0; query < 3; ++query) {
        for (size_t i = 0; i < input.size(); ++i) { input[i] = std::sin(float(i * 7 + query * 91) * 0.031f); }
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        REQUIRE(ggml_backend_graph_compute(backend, rg) == GGML_STATUS_SUCCESS);
        stats(nullptr, nullptr, true);
        REQUIRE(ggml_backend_graph_compute(backend, query == 2 ? interrupted : graph) == GGML_STATUS_SUCCESS);
        uint64_t prepared = 0, reused = 0; stats(&prepared, &reused, false);
        REQUIRE(prepared == (query == 2 ? 2u : 1u));
        REQUIRE(reused == (query == 2 ? 0u : 1u));
        check_equal(y0, reference); check_equal(y1, reference_up);
        ggml_shared_weights_scratch_stats memory{};
        REQUIRE(ggml_shared_weights_get_scratch_stats(backend, &memory));
        REQUIRE(memory.weight_request_bytes >= ggml_nbytes(nw));
        REQUIRE(memory.allocation_bytes >= memory.request_bytes);
        if (query == 0) { reserved = memory.allocation_bytes; }
        else { REQUIRE(memory.allocation_bytes == reserved); }
    }
    // The third source stays the original physical weight through scheduling.
    ggml_backend_t backends[] = {backend};
    auto sched = ggml_backend_sched_new(backends, nullptr, 1, 32, false, true); REQUIRE(sched);
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    REQUIRE(y0->src[0] == w0.tensor && y0->src[2] == w1.tensor);
    check_equal(y0, reference); check_equal(y1, reference_up);
    ggml_backend_sched_free(sched);
    w0.unchanged(); w1.unchanged();
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static void run_policy_pack_case(ggml_backend_buffer_type_t native_type, int axis,
        bool shared_first, int native_interleave, pack_fn pack, pack_fn pack_reference, layout_fn set_layout) {
    // The CPU policy consumes the last 1024 rows/columns of the 6656-wide HTP
    // cover plus 576 native channels. Validate full-size assembly without GEMM.
    const int shared_extent = 6656, native_extent = 576, other_extent = 3072;
    const int shared_start = 5632, shared_size = 1024;
    weight_fixture shared(axis == 0 ? shared_extent : other_extent,
        axis == 1 ? shared_extent : other_extent, 3, true,
        ggml_backend_cpu_buffer_type(), 0, pack_reference, set_layout);
    weight_fixture native(axis == 0 ? native_extent : other_extent,
        axis == 1 ? native_extent : other_extent, 17, false,
        native_type, native_interleave, pack_reference, set_layout);
    const auto & w0 = shared_first ? shared : native;
    const auto & w1 = shared_first ? native : shared;
    const int starts[2] = {shared_first ? shared_start : 0, shared_first ? 0 : shared_start};
    const int sizes[2] = {shared_first ? shared_size : native_extent,
                         shared_first ? native_extent : shared_size};
    const auto merged = assemble_reference(w0, w1, axis, starts, sizes);
    const int k = axis == 0 ? shared_size + native_extent : other_extent;
    const int m = axis == 1 ? shared_size + native_extent : other_extent;
    auto * ctx = ggml_init({65536, nullptr, true}); REQUIRE(ctx);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 4);
    auto * op = ggml_mul_mat_src0_composite(ctx, w0.tensor, w1.tensor, x, axis,
        starts[0], sizes[0], starts[1], sizes[1]);
    auto * canonical = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
    canonical->data = const_cast<q8_block *>(merged.data());
    check_packed_bytes(op, canonical, pack, pack_reference);
    shared.unchanged(); native.unchanged();
    ggml_free(ctx);
}

// GGML_SHARED_REPACK_BREAKDOWN=1 samples every composite repack, which then
// copies its two parts separately: outputs must match the references, and
// the shared part of each region-path composite must be timed on its own.
static void check_breakdown_rows(const std::string & path) {
    ggml_shared_weights_log("validation", nullptr, "test", -1, 0, 0, "flush"); // flushes buffered rows
    FILE * f = std::fopen(path.c_str(), "r");
    REQUIRE(f);
    char line[4096];
    int repacks = 0, reads = 0, copies = 0, parts = 0;
    while (std::fgets(line, sizeof(line), f)) {
        const std::string row = line;
        if (row.rfind("\"repack\",", 0) == 0) { ++repacks; }
        if (row.rfind("\"repack_probe\",", 0) == 0) {
            reads += row.find(",\"read\",") != std::string::npos;
            copies += row.find(",\"copy\",") != std::string::npos;
            parts += row.find(",\"repack_shared_part\",") != std::string::npos;
        }
    }
    std::fclose(f);
    std::remove(path.c_str());
    // Shared parts whose row count is not a multiple of four are not probed.
    std::printf("composite repack breakdown: %d repacks, %d read, %d copy, %d shared-part probes\n",
        repacks, reads, copies, parts);
    REQUIRE(reads > 0 && reads == copies && reads <= repacks && parts > 0 && parts <= reads);
    std::printf("PASS composite repack breakdown\n");
}

int main(int argc, char ** argv) {
    const bool breakdown = argc == 2 && std::strcmp(argv[1], "--breakdown") == 0;
    if (argc != 1 && !breakdown) {
        std::fprintf(stderr, "usage: %s [--breakdown]\n", argv[0]);
        return 2;
    }
#if defined(_WIN32)
    _putenv_s("GGML_SHARED_HTP_WEIGHTS", "1"); _putenv_s("GGML_SHARED_HTP_REPACK", "1");
#else
    setenv("GGML_SHARED_HTP_WEIGHTS", "1", 1); setenv("GGML_SHARED_HTP_REPACK", "1", 1);
#endif
    std::string breakdown_csv;
#if !defined(_WIN32)
    if (breakdown) {
        const char * tmp = std::getenv("TMPDIR");
        breakdown_csv = std::string(tmp && *tmp ? tmp : "/tmp") + "/test-shared-weights-composite-breakdown-" +
            std::to_string((long long) getpid()) + ".csv";
        std::remove(breakdown_csv.c_str());
        setenv("GGML_SHARED_WEIGHTS_CSV", breakdown_csv.c_str(), 1);
        setenv("GGML_SHARED_REPACK_BREAKDOWN", "1", 1);
        setenv("GGML_SHARED_REPACK_BREAKDOWN_EVICT_MB", "1", 1);
    }
#endif
    try {
        ggml_backend_load_all();
        auto reg = ggml_backend_reg_by_name("CPU"); REQUIRE(reg);
        auto force = (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_shared_repack_test_mode");
        auto pack = (pack_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_shared_repack_pack");
        auto pack_reference = (pack_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_shared_repack_pack_reference");
        auto layout = (layout_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_shared_repack_test_source_layout");
        auto stats = (stats_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_shared_repack_test_input_stats");
        REQUIRE(force && pack && pack_reference && layout && stats); force(true);
        auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr); REQUIRE(backend);
        auto extras = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        ggml_backend_buffer_type_t packed_type = nullptr;
        if (extras) {
            for (auto p = extras(ggml_backend_get_device(backend)); p && *p; ++p) {
                if (std::strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) { packed_type = *p; }
            }
        }
        int cases = 0, policy_cases = 0;
        for (bool aligned : {false, true}) {
            for (int axis : {0, 1}) {
                for (bool shared_first : {false, true}) {
                    for (int interleave : {0, 4, 8}) {
                        if (interleave && !packed_type) { continue; }
                        for (int tokens : {4, 7}) {
                            ggml_backend_cpu_set_n_threads(backend, tokens == 4 ? 1 : 4);
                            run_case(backend, interleave ? packed_type : ggml_backend_cpu_buffer_type(),
                                axis, shared_first, interleave, tokens, aligned, pack, pack_reference, layout, stats);
                            ++cases;
                        }
                    }
                }
            }
        }
        if (packed_type) {
            for (int axis : {0, 1}) {
                for (bool shared_first : {false, true}) {
                    for (int interleave : {4, 8}) {
                        run_policy_pack_case(packed_type, axis, shared_first, interleave, pack, pack_reference, layout);
                        ++policy_cases;
                    }
                }
            }
        }
        force(false); ggml_backend_free(backend);
        if (breakdown) { check_breakdown_rows(breakdown_csv); }
        std::printf("PASS %d composite cases and %d policy packing cases: packed bytes, native outputs, activation reuse/reset, scheduler and immutable sources\n", cases, policy_cases);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "composite validation failed: %s\n", error.what()); return 1;
    }
}
