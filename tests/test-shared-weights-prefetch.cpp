// GGML_FFN_REPACK_PREFETCH on the CPU backend: "ggml_backend_shared_repack_prefetch"
// copies the shared HTP weights of a graph's matmuls into native-layout slots
// before the graph runs. Computing the graph afterwards must use those copies
// and produce bit-identical outputs to the on-demand repack, for region and
// composite matmuls, with any thread count, and must fall back to on-demand
// repacking whenever no slot matches.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-shared-weights.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #x); } } while (0)

namespace {

struct q8_block { ggml_fp16_t d; int8_t q[32]; };
static_assert(sizeof(q8_block) == 34, "Q8 block size");

using prefetch_stats_fn = void (*)(ggml_backend_t, uint64_t *, uint64_t *);
using input_stats_fn = void (*)(uint64_t *, uint64_t *, bool);

// A Q8 weight in a mock shared HTP buffer (rows of quants, then their scales),
// or in an ordinary CPU buffer (canonical blocks).
struct weight {
    ggml_context * ctx = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_type shared_type{};
    bool shared = false;

    weight(int k, int m, int seed, bool is_shared) : shared(is_shared) {
        ctx = ggml_init({65536, nullptr, true});
        REQUIRE(ctx);
        ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, 256); // nonzero tensor offset in its buffer
        tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        REQUIRE(buffer);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<q8_block> blocks((size_t) k * m / 32);
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i].d = ggml_fp32_to_fp16(0.002f * (1 + (i * 7 + seed) % 19));
            for (int j = 0; j < 32; ++j) { blocks[i].q[j] = (int) ((i * 13 + j * 37 + seed) % 255) - 127; }
        }
        if (shared) {
            shared_type = *ggml_backend_cpu_buffer_type();
            shared_type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
            buffer->buft = &shared_type;
            ggml_shared_weights_register_buft(&shared_type);
            ggml_shared_weights_register_buffer(buffer, ggml_backend_buffer_get_base(buffer),
                ggml_backend_buffer_get_size(buffer), ggml_backend_buffer_get_size(buffer), -1);
            for (int row = 0; row < m; ++row) {
                auto * q = (int8_t *) tensor->data + row * tensor->nb[1];
                auto * d = (ggml_fp16_t *) (q + k);
                for (int b = 0; b < k / 32; ++b) {
                    const auto & block = blocks[(size_t) row * (k / 32) + b];
                    std::memcpy(q + 32 * b, block.q, 32);
                    d[b] = block.d;
                }
            }
        } else {
            ggml_backend_tensor_set(tensor, blocks.data(), 0, ggml_nbytes(tensor));
        }
    }

    ~weight() {
        ggml_backend_buffer_free(buffer);
        if (shared) { ggml_shared_weights_unregister_buft(&shared_type); }
        ggml_free(ctx);
    }
};

struct cpu_api {
    ggml_backend_shared_repack_prefetch_t prefetch = nullptr;
    prefetch_stats_fn prefetch_stats = nullptr;
    input_stats_fn input_stats = nullptr;
};

static std::vector<float> read(const ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static void stats(const cpu_api & api, ggml_backend_t backend, uint64_t * copies, uint64_t * hits) {
    api.prefetch_stats(backend, copies, hits);
}

// Computes `graph` once with on-demand repacking and once after a prefetch on
// a fresh backend; every output must be identical. Returns the prefetch count.
struct graph_case {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr;
    std::vector<ggml_tensor *> outputs;

    explicit graph_case(int k, int tokens) {
        ctx = ggml_init({1024 * 1024, nullptr, true});
        REQUIRE(ctx);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
        graph = ggml_new_graph_custom(ctx, 64, false);
    }

    ~graph_case() { ggml_free(ctx); }

    void add(ggml_tensor * y, const char * name) {
        ggml_set_name(y, name);
        ggml_build_forward_expand(graph, y);
        outputs.push_back(y);
    }

    std::vector<std::vector<float>> run(ggml_backend_t backend, int query) {
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);
        std::vector<float> input((size_t) ggml_nelements(x));
        for (size_t i = 0; i < input.size(); ++i) { input[i] = std::sin(float(i * 7 + query * 91) * 0.031f); }
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<std::vector<float>> result;
        for (auto * y : outputs) { result.push_back(read(y)); }
        ggml_backend_buffer_free(buffer);
        // Detach the tensors so the next run allocates them again.
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            t->data = nullptr;
            t->buffer = nullptr;
            t->extra = nullptr;
        }
        return result;
    }
};

static ggml_backend_t new_cpu(int threads) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend);
    ggml_backend_cpu_set_n_threads(backend, threads);
    return backend;
}

// Gate/up-style region matmuls over one shared weight with a shared input,
// then a composite (shared + CPU part) on another input.
static void run_prefetch_equals_on_demand(const cpu_api & api, int prefetch_threads, int compute_threads) {
    const int k = 256, tokens = 9;
    weight shared(k, 96, 1, true);
    weight resident(k, 32, 5, false);
    graph_case g(k, tokens);
    g.add(ggml_mul_mat_src0_region(g.ctx, shared.tensor, g.x, 0, k, 0, 32), "ffn_gate.cpu-0");
    g.add(ggml_mul_mat_src0_region(g.ctx, shared.tensor, g.x, 0, k, 32, 32), "ffn_up.cpu-0");
    g.add(ggml_mul_mat_src0_composite(g.ctx, shared.tensor, resident.tensor, g.x, 1, 64, 16, 8, 16), "ffn_down.cpu-0");

    ggml_backend_t reference = new_cpu(compute_threads);
    api.input_stats(nullptr, nullptr, true);
    const auto expected = g.run(reference, 0);
    uint64_t expected_prepared = 0, expected_reused = 0;
    api.input_stats(&expected_prepared, &expected_reused, false);
    uint64_t copies = 0, hits = 0;
    stats(api, reference, &copies, &hits);
    REQUIRE(copies == 0 && hits == 0 && expected_reused > 0);
    ggml_backend_free(reference);

    ggml_backend_t backend = new_cpu(prefetch_threads);
    REQUIRE(api.prefetch(backend, g.graph) == 3);
    stats(api, backend, &copies, &hits);
    REQUIRE(copies == 3 && hits == 0);
    ggml_backend_cpu_set_n_threads(backend, compute_threads);
    api.input_stats(nullptr, nullptr, true);
    const auto actual = g.run(backend, 0);
    stats(api, backend, &copies, &hits);
    REQUIRE(hits == 3);
    // The matmuls share their input: the prefetched path keeps the reuse.
    uint64_t prepared = 0, reused = 0;
    api.input_stats(&prepared, &reused, false);
    REQUIRE(prepared == expected_prepared && reused == expected_reused);
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::memcmp(expected[i].data(), actual[i].data(), expected[i].size() * sizeof(float)) == 0);
    }
    // Still valid on the next query: copies are reused, nothing is copied again.
    REQUIRE(api.prefetch(backend, g.graph) == 0);
    const auto again = g.run(backend, 1);
    stats(api, backend, &copies, &hits);
    REQUIRE(copies == 3 && hits == 6);
    ggml_backend_t reference2 = new_cpu(compute_threads);
    const auto expected2 = g.run(reference2, 1);
    ggml_backend_free(reference2);
    for (size_t i = 0; i < expected2.size(); ++i) {
        REQUIRE(std::memcmp(expected2[i].data(), again[i].data(), expected2[i].size() * sizeof(float)) == 0);
    }
    ggml_backend_free(backend);
    std::printf("PASS prefetch == on demand (prefetch threads %d, compute threads %d)\n",
        prefetch_threads, compute_threads);
}

// A copy is only used by a matmul that reads the same weight rectangle.
static void run_mismatch_falls_back(const cpu_api & api) {
    const int k = 256, tokens = 5;
    weight shared(k, 96, 3, true);
    weight shared2(k, 96, 4, true);
    weight resident(k, 32, 6, false);
    graph_case prefetched(k, tokens);
    prefetched.add(ggml_mul_mat_src0_region(prefetched.ctx, shared.tensor, prefetched.x, 0, k, 0, 32), "ffn_gate.cpu-0");
    prefetched.add(ggml_mul_mat_src0_composite(prefetched.ctx, shared.tensor, resident.tensor, prefetched.x,
        1, 64, 16, 8, 16), "ffn_down.cpu-0");
    // Same output rectangles, other rows, weight or composite split.
    graph_case other(k, tokens);
    other.add(ggml_mul_mat_src0_region(other.ctx, shared.tensor, other.x, 0, k, 32, 32), "ffn_gate.cpu-0");
    other.add(ggml_mul_mat_src0_region(other.ctx, shared2.tensor, other.x, 0, k, 0, 32), "ffn_up.cpu-0");
    other.add(ggml_mul_mat_src0_composite(other.ctx, shared.tensor, resident.tensor, other.x,
        1, 0, 16, 8, 16), "ffn_down.cpu-0");

    ggml_backend_t reference = new_cpu(2);
    const auto expected = other.run(reference, 0);
    ggml_backend_free(reference);

    ggml_backend_t backend = new_cpu(2);
    REQUIRE(api.prefetch(backend, prefetched.graph) == 2);
    const auto actual = other.run(backend, 0);
    uint64_t copies = 0, hits = 0;
    stats(api, backend, &copies, &hits);
    REQUIRE(copies == 2 && hits == 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::memcmp(expected[i].data(), actual[i].data(), expected[i].size() * sizeof(float)) == 0);
    }
    ggml_backend_free(backend);
    std::printf("PASS a different region or weight repacks on demand\n");
}

// Only the first of two matmuls with a shared input was prefetched. The second
// repacks on demand and reuses the first one's quantized input, which must sit
// where an on-demand matmul leaves it.
static void run_mixed_hit_and_miss(const cpu_api & api) {
    const int k = 256, tokens = 6;
    weight shared(k, 64, 13, true);
    graph_case first_only(k, tokens);
    first_only.add(ggml_mul_mat_src0_region(first_only.ctx, shared.tensor, first_only.x, 0, k, 0, 32), "ffn_gate.cpu-0");
    graph_case both(k, tokens);
    both.add(ggml_mul_mat_src0_region(both.ctx, shared.tensor, both.x, 0, k, 0, 32), "ffn_gate.cpu-0");
    both.add(ggml_mul_mat_src0_region(both.ctx, shared.tensor, both.x, 0, k, 32, 32), "ffn_up.cpu-0");
    ggml_backend_t reference = new_cpu(2);
    const auto expected = both.run(reference, 3);
    ggml_backend_free(reference);
    for (int order = 0; order < 2; ++order) {
        ggml_backend_t backend = new_cpu(2);
        if (order == 0) {
            REQUIRE(api.prefetch(backend, first_only.graph) == 1); // hit, then miss
        } else {
            graph_case second_only(k, tokens);
            second_only.add(ggml_mul_mat_src0_region(second_only.ctx, shared.tensor, second_only.x, 0, k, 32, 32),
                "ffn_up.cpu-0");
            REQUIRE(api.prefetch(backend, second_only.graph) == 1); // miss, then hit
        }
        api.input_stats(nullptr, nullptr, true);
        const auto actual = both.run(backend, 3);
        uint64_t prepared = 0, reused = 0, copies = 0, hits = 0;
        api.input_stats(&prepared, &reused, false);
        stats(api, backend, &copies, &hits);
        REQUIRE(hits == 1 && prepared == 1 && reused == 1);
        for (size_t i = 0; i < expected.size(); ++i) {
            REQUIRE(std::memcmp(expected[i].data(), actual[i].data(), expected[i].size() * sizeof(float)) == 0);
        }
        ggml_backend_free(backend);
    }
    std::printf("PASS a prefetched and an on-demand matmul share one quantized input\n");
}

// More shared matmuls than slots: the first ones are prefetched, the rest
// repack on demand; later prefetches replace the oldest copies.
static void run_slot_limit(const cpu_api & api) {
    const int k = 128, tokens = 4, n = 6;
    weight shared(k, 64 * n, 7, true);
    graph_case g(k, tokens);
    for (int i = 0; i < n; ++i) {
        g.add(ggml_mul_mat_src0_region(g.ctx, shared.tensor, g.x, 0, k, i * 32, 32 + 4 * i),
            ("ffn_gate.cpu-" + std::to_string(i)).c_str());
    }
    ggml_backend_t reference = new_cpu(3);
    const auto expected = g.run(reference, 2);
    ggml_backend_free(reference);

    ggml_backend_t backend = new_cpu(3);
    const int first = api.prefetch(backend, g.graph);
    REQUIRE(first > 0 && first < n);
    const auto actual = g.run(backend, 2);
    uint64_t copies = 0, hits = 0;
    stats(api, backend, &copies, &hits);
    REQUIRE(hits == (uint64_t) first);
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::memcmp(expected[i].data(), actual[i].data(), expected[i].size() * sizeof(float)) == 0);
    }
    // A graph with only the last matmuls replaces older slots; results stay exact.
    graph_case tail(k, tokens);
    for (int i = first; i < n; ++i) {
        tail.add(ggml_mul_mat_src0_region(tail.ctx, shared.tensor, tail.x, 0, k, i * 32, 32 + 4 * i),
            ("ffn_up.cpu-" + std::to_string(i)).c_str());
    }
    REQUIRE(api.prefetch(backend, tail.graph) == n - first);
    const auto tail_actual = tail.run(backend, 2);
    for (int i = first; i < n; ++i) {
        REQUIRE(std::memcmp(expected[(size_t) i].data(), tail_actual[(size_t) (i - first)].data(),
            expected[(size_t) i].size() * sizeof(float)) == 0);
    }
    ggml_backend_free(backend);
    std::printf("PASS %d of %d matmuls prefetched, the rest on demand; slots are replaced\n", first, n);
}

// Nothing to prefetch: ordinary weights, single-token matmuls (no native
// repack), and graphs without matmuls.
static void run_nothing_to_prefetch(const cpu_api & api) {
    const int k = 128;
    weight shared(k, 32, 9, true);
    weight resident(k, 32, 2, false);
    graph_case one_token(k, 1);
    one_token.add(ggml_mul_mat_src0_region(one_token.ctx, shared.tensor, one_token.x, 0, k, 0, 32), "ffn_gate.cpu-0");
    graph_case plain(k, 8);
    plain.add(ggml_mul_mat(plain.ctx, resident.tensor, plain.x), "ffn_gate.cpu-0");
    plain.add(ggml_scale(plain.ctx, plain.x, 2.0f), "ffn_scale.cpu-0");
    ggml_backend_t backend = new_cpu(2);
    REQUIRE(api.prefetch(backend, one_token.graph) == 0);
    REQUIRE(api.prefetch(backend, plain.graph) == 0);
    REQUIRE(api.prefetch(nullptr, plain.graph) == 0);
    uint64_t copies = 0, hits = 0;
    stats(api, backend, &copies, &hits);
    REQUIRE(copies == 0);
    (void) one_token.run(backend, 0);
    (void) plain.run(backend, 0);
    stats(api, backend, &copies, &hits);
    REQUIRE(hits == 0);
    ggml_backend_free(backend);
    std::printf("PASS nothing prefetched for single tokens, ordinary weights or non-matmuls\n");
}

// A slow NPU stand-in that computes on its own CPU backend: the scheduler
// starts the prefetch before its split and waits on it meanwhile.
struct slow_npu {
    ggml_backend_t inner = nullptr;
    ggml_backend_buffer_type_t inner_buft = nullptr;
    ggml_backend_buffer_type buft = {};
    ggml_backend_device device = {};
    ggml_backend_reg reg = {};
};

static slow_npu * npu_of(ggml_backend_t b) { return static_cast<slow_npu *>(b->context); }
static slow_npu * npu_of(ggml_backend_dev_t d) { return static_cast<slow_npu *>(d->context); }
static slow_npu * npu_of(ggml_backend_buffer_type_t t) { return static_cast<slow_npu *>(t->context); }

static ggml_backend_buffer_t npu_alloc(ggml_backend_buffer_type_t t, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(npu_of(t)->inner_buft, size);
    if (buffer) { buffer->buft = t; }
    return buffer;
}

static const ggml_backend_buffer_type_i npu_buft_iface = {
    /* .get_name       = */ [](ggml_backend_buffer_type_t) { return "HTP-SlowTest"; },
    /* .alloc_buffer   = */ npu_alloc,
    /* .get_alignment  = */ [](ggml_backend_buffer_type_t t) { return ggml_backend_buft_get_alignment(npu_of(t)->inner_buft); },
    /* .get_max_size   = */ [](ggml_backend_buffer_type_t t) { return ggml_backend_buft_get_max_size(npu_of(t)->inner_buft); },
    /* .get_alloc_size = */ [](ggml_backend_buffer_type_t t, const ggml_tensor * x) {
        return ggml_backend_buft_get_alloc_size(npu_of(t)->inner_buft, x); },
    /* .is_host        = */ [](ggml_backend_buffer_type_t) { return true; },
};

static const ggml_backend_i npu_iface = {
    /* .get_name                = */ [](ggml_backend_t) { return "HTP-SlowTest"; },
    /* .free                    = */ [](ggml_backend_t b) { ggml_backend_free(npu_of(b)->inner); delete npu_of(b); delete b; },
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ [](ggml_backend_t b, ggml_cgraph * g) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return ggml_backend_graph_compute(npu_of(b)->inner, g); },
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static const ggml_backend_device_i npu_dev_iface = {
    /* .get_name             = */ [](ggml_backend_dev_t) { return "HTP-SlowTest"; },
    /* .get_description      = */ [](ggml_backend_dev_t) { return "HTP-SlowTest"; },
    /* .get_memory           = */ [](ggml_backend_dev_t, size_t * free, size_t * total) { *free = 1u << 30; *total = 1u << 30; },
    /* .get_type             = */ [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; },
    /* .get_props            = */ [](ggml_backend_dev_t, ggml_backend_dev_props * p) {
        p->name = "HTP-SlowTest"; p->description = p->name; p->memory_free = 1u << 30; p->memory_total = 1u << 30;
        p->type = GGML_BACKEND_DEVICE_TYPE_ACCEL; p->device_id = nullptr; p->caps = {}; },
    /* .init_backend         = */ [](ggml_backend_dev_t, const char *) -> ggml_backend_t { return nullptr; },
    /* .get_buffer_type      = */ [](ggml_backend_dev_t d) -> ggml_backend_buffer_type_t { return &npu_of(d)->buft; },
    /* .get_host_buffer_type = */ [](ggml_backend_dev_t) -> ggml_backend_buffer_type_t { return nullptr; },
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ [](ggml_backend_dev_t d, const ggml_tensor * op) { return ggml_backend_supports_op(npu_of(d)->inner, op); },
    /* .supports_buft        = */ [](ggml_backend_dev_t d, ggml_backend_buffer_type_t t) { return t == &npu_of(d)->buft; },
    /* .offload_op           = */ [](ggml_backend_dev_t, const ggml_tensor *) { return false; },
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const ggml_backend_reg_i npu_reg_iface = {
    /* .get_name         = */ [](ggml_backend_reg_t) { return "HTP-SlowTest"; },
    /* .get_device_count = */ [](ggml_backend_reg_t) -> size_t { return 0; },
    /* .get_device       = */ [](ggml_backend_reg_t, size_t) -> ggml_backend_dev_t { return nullptr; },
    /* .get_proc_address = */ [](ggml_backend_reg_t, const char *) -> void * { return nullptr; },
};

static ggml_backend_t make_slow_npu() {
    static ggml_guid guid = { 0x5c, 0x0e, 0x7a, 0x21, 0x98, 0x3d, 0x4f, 0x60, 0xb1, 0x47, 0x2a, 0xc9, 0x0d, 0x6e, 0x83, 0x15 };
    auto * n = new slow_npu;
    n->inner = ggml_backend_cpu_init();
    REQUIRE(n->inner);
    n->inner_buft = ggml_backend_get_default_buffer_type(n->inner);
    n->reg = { GGML_BACKEND_API_VERSION, npu_reg_iface, n };
    n->device = { npu_dev_iface, &n->reg, n };
    n->buft = { npu_buft_iface, &n->device, n };
    return new ggml_backend { &guid, npu_iface, &n->device, n };
}

// The scheduler path end to end on the real CPU backend: attn_out on the slow
// NPU, then an FFN group whose CPU branch multiplies shared weights. With
// GGML_FFN_REPACK_PREFETCH the CPU branch must use the copies made during the
// NPU split and produce exactly the outputs of an on-demand run.
static void run_scheduler_end_to_end(const cpu_api & api) {
    const int k = 256, tokens = 8;
    weight shared(k, 128, 21, true);
    std::vector<float> reference;
    for (const char * env : {static_cast<const char *>(nullptr), "1"}) {
        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        REQUIRE(ctx);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
        ggml_set_name(x, "e2e-x");
        ggml_set_input(x);
        ggml_tensor * attn = ggml_scale(ctx, x, 0.5f);
        ggml_set_name(attn, "attn_out-0");
        ggml_tensor * gate = ggml_mul_mat_src0_region(ctx, shared.tensor, attn, 0, k, 0, 32);
        ggml_set_name(gate, "ffn_gate.cpu-0");
        ggml_tensor * up = ggml_mul_mat_src0_region(ctx, shared.tensor, attn, 0, k, 64, 32);
        ggml_set_name(up, "ffn_up.cpu-0");
        ggml_tensor * cpu_branch = ggml_add(ctx, gate, up);
        ggml_set_name(cpu_branch, "ffn_down.cpu-0");
        ggml_tensor * npu_view = ggml_cont(ctx, ggml_view_2d(ctx, attn, 32, tokens, attn->nb[1], 0));
        ggml_set_name(npu_view, "ffn_gate.npu-0");
        ggml_tensor * npu_branch = ggml_scale(ctx, npu_view, 2.0f);
        ggml_set_name(npu_branch, "ffn_down.npu-0");
        ggml_tensor * out = ggml_add(ctx, cpu_branch, npu_branch);
        ggml_set_name(out, "ffn_parallel_reduce-0");
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, out);

        ggml_backend_t npu = make_slow_npu();
        ggml_backend_t cpu = new_cpu(3);
#if defined(_WIN32)
        _putenv_s("GGML_FFN_REPACK_PREFETCH", env ? env : "");
#else
        if (env) { setenv("GGML_FFN_REPACK_PREFETCH", env, 1); } else { unsetenv("GGML_FFN_REPACK_PREFETCH"); }
#endif
        ggml_backend_t backends[] = { npu, cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, true);
#if defined(_WIN32)
        _putenv_s("GGML_FFN_REPACK_PREFETCH", "");
#else
        unsetenv("GGML_FFN_REPACK_PREFETCH");
#endif
        REQUIRE(sched);
        ggml_backend_sched_set_tensor_backend(sched, x, cpu);
        for (ggml_tensor * t : {attn, npu_view, npu_branch}) { ggml_backend_sched_set_tensor_backend(sched, t, npu); }
        for (ggml_tensor * t : {gate, up, cpu_branch, out}) { ggml_backend_sched_set_tensor_backend(sched, t, cpu); }
        REQUIRE(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> input((size_t) ggml_nelements(x));
        for (size_t i = 0; i < input.size(); ++i) { input[i] = std::cos(float(i * 5) * 0.017f); }
        for (int run = 0; run < 2; ++run) {
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
            const auto result = read(out);
            if (reference.empty()) {
                reference = result;
            } else {
                REQUIRE(std::memcmp(reference.data(), result.data(), result.size() * sizeof(float)) == 0);
            }
        }
        uint64_t copies = 0, hits = 0;
        stats(api, cpu, &copies, &hits);
        if (env) {
            // Copied once; both matmuls of both runs use the copies.
            REQUIRE(copies == 2 && hits == 4);
        } else {
            REQUIRE(copies == 0 && hits == 0);
        }
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_backend_free(npu);
        ggml_free(ctx);
    }
    std::printf("PASS scheduler prefetch on the CPU backend: outputs equal the on-demand run\n");
}

#if defined(__linux__)
static std::vector<int> allowed_cpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    std::vector<int> cpus;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) { cpus.push_back(cpu); }
        }
    }
    return cpus;
}

static bool openmp_build() {
    auto reg = ggml_backend_reg_by_name("CPU");
    auto get_features = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    for (const ggml_backend_feature * f = get_features ? get_features(reg) : nullptr; f && f->name; ++f) {
        if (std::strcmp(f->name, "OPENMP") == 0) { return true; }
    }
    return false;
}

// The scheduler's helper thread runs the CPU prefetch as worker 0 of a pool
// whose workers are pinned to their own cores (`--cpu-strict`): secondary
// workers 1.. take the first cores, worker 0 (the thread that created the
// pool, the scheduler thread) the next one. The helper is not pinned, so it
// can sit on a secondary worker's core, where worker 0 spinning in the graph
// barrier keeps that worker waiting for the CPU. With pinning (the default)
// the prefetch runs thread 0 off the pool's cores, or on worker 0's core when
// the helper may use no other, and gives the helper its affinity back.
static void run_caller_placement(const cpu_api & api, bool pin) {
    const auto cpus = allowed_cpus();
    if (cpus.size() < 4) {
        std::printf("SKIP caller placement: %zu CPUs available, 4 needed\n", cpus.size());
        return;
    }
    auto prefetch_cpus = (void (*)(ggml_backend_t, int *, int)) ggml_backend_reg_get_proc_address(
        ggml_backend_reg_by_name("CPU"), "ggml_backend_cpu_shared_repack_prefetch_cpus");
    REQUIRE(prefetch_cpus);
    const bool openmp = openmp_build();
    const int n_threads = 3;
    const int pool_cpus[3] = { cpus[0], cpus[1], cpus[2] };
    const int free_cpu = cpus[3];
    ggml_threadpool_params tpp;
    ggml_threadpool_params_init(&tpp, n_threads);
    for (int cpu : pool_cpus) { tpp.cpumask[cpu] = true; }
    tpp.strict_cpu = true;
    ggml_threadpool * pool = nullptr;
    // The creating thread is pinned to worker 0's core: create the pool on a
    // throwaway thread so this test's own thread keeps its placement.
    std::thread([&] { pool = ggml_threadpool_new(&tpp); }).join();
    REQUIRE(pool);
    ggml_backend_t backend = new_cpu(n_threads);
    ggml_backend_cpu_set_threadpool(backend, pool);
    // Without OpenMP worker 0 takes the core after the secondary workers'.
    const int worker0_cpu = openmp ? pool_cpus[0] : pool_cpus[2];
    const int worker_cpu[3] = { worker0_cpu, openmp ? pool_cpus[1] : pool_cpus[0], openmp ? pool_cpus[2] : pool_cpus[1] };

    const int k = 256, tokens = 6;
    weight shared(k, 128, 17, true);
    // helper affinity: all CPUs, one pool core plus a free one, only a pool core.
    const std::vector<std::vector<int>> helper_masks = { cpus, { worker_cpu[1], free_cpu }, { worker_cpu[1] } };
    for (size_t round = 0; round < helper_masks.size(); ++round) {
        graph_case r(k, tokens);
        r.add(ggml_mul_mat_src0_region(r.ctx, shared.tensor, r.x, 0, k, 8 * (int) round, 32), "ffn_up.cpu-0");
        int copies = 0;
        bool restored = false;
        long long elapsed_us = 0;
        std::thread helper([&] {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            for (int cpu : helper_masks[round]) { CPU_SET(cpu, &mask); }
            // pid 0: the calling thread (glibc and Android bionic, which lacks pthread_*affinity_np)
            REQUIRE(sched_setaffinity(0, sizeof(mask), &mask) == 0);
            const auto t0 = std::chrono::steady_clock::now();
            copies = api.prefetch(backend, r.graph);
            elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            cpu_set_t after;
            CPU_ZERO(&after);
            REQUIRE(sched_getaffinity(0, sizeof(after), &after) == 0);
            restored = CPU_EQUAL(&after, &mask);
        });
        helper.join();
        REQUIRE(copies == 1);
        int seen[3] = { -1, -1, -1 };
        prefetch_cpus(backend, seen, 3);
        // Informational, not asserted: a caller that shares a pinned worker's
        // core waits for a scheduler time slice. With the last mask the helper
        // also returns to that core afterwards and waits there once more.
        std::printf("  helper mask %zu: thread 0 on CPU %d, prefetch %lld us\n", round, seen[0], elapsed_us);
        REQUIRE(seen[1] == worker_cpu[1] && seen[2] == worker_cpu[2]);
        if (pin) {
            REQUIRE(restored);
        }
        if (openmp) {
            REQUIRE(seen[0] == worker0_cpu); // OpenMP pins every thread itself
            continue;
        }
        const bool on_pool = std::find(pool_cpus, pool_cpus + 3, seen[0]) != pool_cpus + 3;
        if (!pin) {
            // The helper stays within its own mask, wherever that is.
            REQUIRE(std::find(helper_masks[round].begin(), helper_masks[round].end(), seen[0]) != helper_masks[round].end());
        } else if (round < 2) {
            REQUIRE(!on_pool);
        } else {
            REQUIRE(seen[0] == worker0_cpu); // no CPU of its own outside the pool
        }
    }
    ggml_backend_free(backend);
    ggml_threadpool_free(pool);
    std::printf("PASS prefetch caller placement (pin %s%s)\n", pin ? "on" : "off", openmp ? ", OpenMP" : "");
}
#endif

} // namespace

int main(int argc, char ** argv) {
    const bool no_pin = argc == 2 && std::strcmp(argv[1], "--no-pin") == 0;
    if (no_pin) {
#if defined(_WIN32)
        _putenv_s("GGML_FFN_REPACK_PREFETCH_PIN", "0");
#else
        setenv("GGML_FFN_REPACK_PREFETCH_PIN", "0", 1);
#endif
    }
#if defined(_WIN32)
    _putenv_s("GGML_SHARED_HTP_WEIGHTS", "1"); _putenv_s("GGML_SHARED_HTP_REPACK", "1");
#else
    setenv("GGML_SHARED_HTP_WEIGHTS", "1", 1); setenv("GGML_SHARED_HTP_REPACK", "1", 1);
#endif
    try {
        ggml_backend_load_all();
        auto reg = ggml_backend_reg_by_name("CPU");
        REQUIRE(reg);
        // Portable native layout on hosts without the NEON CPU_REPACK kernels.
        auto force = (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_shared_repack_test_mode");
        cpu_api api;
        api.prefetch = (ggml_backend_shared_repack_prefetch_t) ggml_backend_reg_get_proc_address(reg,
            "ggml_backend_shared_repack_prefetch");
        api.prefetch_stats = (prefetch_stats_fn) ggml_backend_reg_get_proc_address(reg,
            "ggml_backend_cpu_shared_repack_prefetch_stats");
        api.input_stats = (input_stats_fn) ggml_backend_reg_get_proc_address(reg,
            "ggml_backend_cpu_shared_repack_test_input_stats");
        REQUIRE(force && api.prefetch && api.prefetch_stats && api.input_stats);
        force(true);
        for (int prefetch_threads : {1, 4}) {
            for (int compute_threads : {1, 3}) {
                run_prefetch_equals_on_demand(api, prefetch_threads, compute_threads);
            }
        }
        run_mismatch_falls_back(api);
        run_mixed_hit_and_miss(api);
        run_slot_limit(api);
        run_nothing_to_prefetch(api);
        run_scheduler_end_to_end(api);
#if defined(__linux__)
        run_caller_placement(api, !no_pin);
#endif
        force(false);
        std::printf("PASS\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "prefetch validation failed: %s\n", error.what());
        return 1;
    }
}
