#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-shared-weights.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
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

struct temporary_csv {
    std::string path = (std::filesystem::temp_directory_path() /
        ("llama-shared-memory-" + std::to_string(ggml_time_us()) + ".csv")).string();
    ~temporary_csv() { std::remove(path.c_str()); }
};

// Exercise the real graph-memory writer with a backend's optional getter.
// A real allocation owned by this fixture gives known capacities independent
// of ISA-specific native repack support. This tests reporting, not repack math.
struct scratch_getter_fixture {
    using get_proc_t = void * (*)(ggml_backend_reg_t, const char *);
    static scratch_getter_fixture * active;
    ggml_backend_reg_t reg;
    get_proc_t original;
    std::unique_ptr<uint8_t[]> allocation;
    ggml_shared_weights_scratch_stats stats {};

    explicit scratch_getter_fixture(ggml_backend_reg_t backend_reg)
        : reg(backend_reg), original(reg->iface.get_proc_address) {
        REQUIRE(active == nullptr);
        stats.owner = this;
        active = this;
        reg->iface.get_proc_address = get_proc;
    }
    ~scratch_getter_fixture() {
        reg->iface.get_proc_address = original;
        active = nullptr;
    }
    void grow(size_t bytes, size_t weights, size_t activations, size_t request) {
        REQUIRE(bytes >= stats.allocation_bytes && request <= bytes);
        if (bytes > stats.allocation_bytes) { allocation.reset(new uint8_t[bytes]); }
        stats = {this, weights, activations, request, bytes};
    }
    static bool get_stats(ggml_backend_t, ggml_shared_weights_scratch_stats * out) {
        *out = active->stats;
        return true;
    }
    static void * get_proc(ggml_backend_reg_t backend_reg, const char * name) {
        if (strcmp(name, "ggml_backend_get_shared_weights_scratch_stats") == 0) {
            return reinterpret_cast<void *>(get_stats);
        }
        return active->original ? active->original(backend_reg, name) : nullptr;
    }
};
scratch_getter_fixture * scratch_getter_fixture::active = nullptr;

void init_tensor_data(ggml_tensor * tensor, void *) {
    REQUIRE(tensor->type == GGML_TYPE_F32);
    std::vector<float> data(ggml_nelements(tensor));
    for (size_t i = 0; i < data.size(); ++i) { data[i] = 0.01f * std::sin(float(i + 1)); }
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

llama_model_ptr make_model() {
    gguf_context_ptr metadata(gguf_init_empty());
    llama_model_saver saver(LLM_ARCH_LLAMA, metadata.get());
    saver.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(LLM_ARCH_LLAMA));
    saver.add_kv(LLM_KV_VOCAB_SIZE, uint32_t(128));
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(64));
    saver.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(32));
    saver.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(1));
    saver.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(64));
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(4));
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(4));
    saver.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, uint32_t(8));
    saver.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1.0e-5f);
    saver.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    params.use_extra_bufts = false;
    params.progress_callback = [](float, void *) { return true; };
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), init_tensor_data, nullptr, params));
    REQUIRE(model != nullptr);
    return model;
}

llama_context_ptr make_context(
        llama_model * model, const std::string & csv_path, uint32_t n_ctx = 64, bool memory_stats = true,
        enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_UNSPECIFIED, uint32_t n_seq_max = 1) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_seq_max = n_seq_max;
    params.n_batch = params.n_ubatch = 64;
    params.n_threads = params.n_threads_batch = 1;
    params.no_perf = false;
    params.offload_kqv = params.op_offload = false;
    params.memory_stats = memory_stats;
    params.embeddings = pooling != LLAMA_POOLING_TYPE_UNSPECIFIED;
    params.pooling_type = pooling;
    params.graph_memory_stats = !csv_path.empty();
    REQUIRE(csv_path.size() < sizeof(params.graph_memory_stats_path));
    std::snprintf(params.graph_memory_stats_path, sizeof(params.graph_memory_stats_path), "%s", csv_path.c_str());
    llama_context_ptr ctx(llama_init_from_model(model, params));
    REQUIRE(ctx != nullptr);
    return ctx;
}

std::vector<float> run_tokens(
        llama_context * ctx, int32_t n_tokens, bool all_outputs = false,
        llama_pos pos_start = 0, bool clear_memory = true) {
    if (clear_memory) { llama_memory_clear(llama_get_memory(ctx), true); }
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, llama_token((pos_start + i + 1) % 128), pos_start + i,
                {0}, all_outputs || i + 1 == n_tokens);
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    REQUIRE(status == 0);
    const float * logits = llama_get_logits_ith(ctx, -1);
    REQUIRE(logits != nullptr);
    std::vector<float> result(logits, logits + 128);
    for (float value : result) { REQUIRE(std::isfinite(value)); }
    return result;
}

void require_same_logits(const std::vector<float> & a, const std::vector<float> & b) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::abs(a[i] - b[i]) <= 1.0e-6f * std::max(1.0f, std::abs(a[i])));
    }
}

struct compute_snapshot {
    size_t reserved = 0;
    size_t base = 0;
    size_t full = 0;
    bool measured = false;
};

compute_snapshot read_compute(llama_context * ctx) {
    compute_snapshot result;
    for (const auto & entry : llama_get_memory_breakdown(ctx)) {
        result.reserved += entry.second.compute;
        result.base += entry.second.compute_graph_base;
        result.full += entry.second.compute_graph_routed;
        result.measured = result.measured || entry.second.compute_graph_measurement_valid;
    }
    return result;
}

void require_actual_compute(const compute_snapshot & actual) {
    REQUIRE(actual.measured);
    REQUIRE(actual.full > 0);
    REQUIRE(actual.reserved >= actual.full);
    // This fixture has one CPU buffer; allow its alignment without accepting
    // a retained maximum-batch or all-logits allocation.
    REQUIRE(actual.reserved - actual.full <= 256);
    REQUIRE(actual.base == actual.full);
}

std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> fields(1);
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                fields.back() += '"';
                ++i;
            } else { quoted = !quoted; }
        } else if (c == ',' && !quoted) { fields.emplace_back(); }
        else { fields.back() += c; }
    }
    REQUIRE(!quoted);
    return fields;
}

using row = std::map<std::string, std::string>;
size_t number(const row & values, const char * key) { return std::stoull(values.at(key)); }

struct csv_snapshot {
    std::vector<row> totals;
    std::vector<row> reserves;
    size_t row_count = 0;
};

csv_snapshot read_and_validate(const std::string & path) {
    std::ifstream input(path);
    REQUIRE(input.good());
    std::string line;
    REQUIRE(bool(std::getline(input, line)));
    const auto header = split_csv(line);
    REQUIRE(header.size() == 48);
    std::map<std::string, std::vector<row>> events;
    csv_snapshot result;
    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        REQUIRE(fields.size() == header.size());
        row values;
        for (size_t i = 0; i < fields.size(); ++i) { values.emplace(header[i], fields[i]); }
        REQUIRE(values.size() == header.size());
        REQUIRE(values.at("schema_version") == "4");
        events[values.at("event_id")].push_back(values);
        if (values.at("row_scope") == "total") {
            (values.at("event") == "reserve" ? result.reserves : result.totals).push_back(values);
        }
        ++result.row_count;
    }
    for (const auto & event : events) {
        size_t count_total = 0, count_scratch = 0;
        size_t weights = 0, activations = 0, requests = 0, allocated = 0;
        const row * total = nullptr;
        for (const auto & values : event.second) {
            const auto & scope = values.at("row_scope");
            if (scope == "backend_scratch") {
                ++count_scratch;
                REQUIRE(values.at("backend") == "CPU");
                REQUIRE(values.at("buffer_type") == "PRIVATE");
                for (const char * key : {"base_planned_peak_bytes", "full_planned_peak_bytes",
                        "active_compute_reserved_bytes", "context_compute_reserved_bytes",
                        "tracked_context_graph_reserved_bytes", "graph_plus_backend_scratch_reserved_bytes"}) {
                    REQUIRE(values.at(key).empty());
                }
                weights += number(values, "weight_request_peak_bytes");
                activations += number(values, "activation_request_peak_bytes");
                requests += number(values, "scratch_request_peak_bytes");
                allocated += number(values, "backend_scratch_reserved_bytes");
            } else if (scope == "total") {
                total = &values;
                ++count_total;
            } else {
                REQUIRE(scope == "buffer");
                for (const char * key : {"weight_request_peak_bytes", "activation_request_peak_bytes",
                        "scratch_request_peak_bytes", "backend_scratch_reserved_bytes",
                        "graph_plus_backend_scratch_reserved_bytes"}) {
                    REQUIRE(values.at(key).empty());
                }
            }
        }
        REQUIRE(count_total == 1 && count_scratch == 1);
        const size_t descriptors = number(*total, "active_scheduler_copy_tensor_count") +
            number(*total, "active_scheduler_input_dependency_count") +
            number(*total, "active_scheduler_route_dependency_count");
        const size_t context_used = number(*total, "active_scheduler_context_used_bytes");
        const size_t context_capacity = number(*total, "active_scheduler_context_capacity_bytes");
        REQUIRE(context_used == descriptors * ggml_tensor_overhead());
        REQUIRE(context_capacity >= context_used);
        REQUIRE(number(*total, "active_scheduler_graph_nodes") >= number(*total, "n_nodes"));
        REQUIRE(number(*total, "active_scheduler_metadata_bytes") >= context_capacity +
            number(*total, "active_scheduler_backend_id_capacity_bytes") +
            number(*total, "active_scheduler_graph_pointer_capacity_bytes"));
        REQUIRE(number(*total, "weight_request_peak_bytes") == weights);
        REQUIRE(number(*total, "activation_request_peak_bytes") == activations);
        REQUIRE(number(*total, "scratch_request_peak_bytes") == requests);
        REQUIRE(number(*total, "backend_scratch_reserved_bytes") == allocated);
        REQUIRE(number(*total, "graph_plus_backend_scratch_reserved_bytes") ==
                number(*total, "tracked_context_graph_reserved_bytes") + allocated);
    }
    return result;
}

void test_output_selection_and_eviction(llama_model * model) {
    // With one cache slot, replacing a large graph must release its compute
    // buffers before a smaller graph is built. Exercise the memory API with
    // CSV logging disabled, as used by memory_stats-only applications.
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
    auto ctx = make_context(model, "");
    REQUIRE(read_compute(ctx.get()).reserved == 0);

    const auto last_logits = run_tokens(ctx.get(), 16);
    const auto last_only = read_compute(ctx.get());
    require_actual_compute(last_only);

    const auto all_logits = run_tokens(ctx.get(), 16, true);
    const auto all_outputs = read_compute(ctx.get());
    require_actual_compute(all_outputs);
    REQUIRE(all_outputs.full > last_only.full);
    require_same_logits(last_logits, all_logits);

    run_tokens(ctx.get(), 64, true);
    const auto largest = read_compute(ctx.get());
    require_actual_compute(largest);
    REQUIRE(largest.reserved > all_outputs.reserved);

    const auto replay = run_tokens(ctx.get(), 16);
    const auto small_again = read_compute(ctx.get());
    require_actual_compute(small_again);
    REQUIRE(small_again.reserved == last_only.reserved);
    REQUIRE(small_again.full == last_only.full);
    require_same_logits(last_logits, replay);

    // Reuse of an unchanged graph keeps its allocation and updates inputs.
    const auto reuse_before = llama_perf_context(ctx.get()).n_reused;
    require_same_logits(last_logits, run_tokens(ctx.get(), 16));
    REQUIRE(llama_perf_context(ctx.get()).n_reused == reuse_before + 1);
    REQUIRE(read_compute(ctx.get()).reserved == small_again.reserved);

    // The first real inference must not inherit a larger warmup allocation.
    llama_set_warmup(ctx.get(), true);
    run_tokens(ctx.get(), 64, true);
    REQUIRE(read_compute(ctx.get()).reserved > last_only.reserved);
    llama_set_warmup(ctx.get(), false);
    require_same_logits(last_logits, run_tokens(ctx.get(), 16));
    const auto after_warmup = read_compute(ctx.get());
    require_actual_compute(after_warmup);
    REQUIRE(after_warmup.reserved == last_only.reserved);

    // Buffer sizing must also work when neither diagnostic mode is enabled.
    auto no_stats = make_context(model, "", 64, false);
    REQUIRE(read_compute(no_stats.get()).reserved == 0);
    require_same_logits(last_logits, run_tokens(no_stats.get(), 16));
    REQUIRE(read_compute(no_stats.get()).reserved == last_only.reserved);
}

void test_rebuild_without_reuse(llama_model * model) {
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "4");
    set_env("LLAMA_GRAPH_REUSE_DISABLE", "1");
    temporary_csv csv;
    auto ctx = make_context(model, csv.path);
    set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
    const auto logits = run_tokens(ctx.get(), 16);
    const auto small = read_compute(ctx.get());
    require_actual_compute(small);
    const auto builds_before = read_and_validate(csv.path).totals.size();

    require_same_logits(logits, run_tokens(ctx.get(), 16));
    REQUIRE(llama_perf_context(ctx.get()).n_reused == 0);
    REQUIRE(read_and_validate(csv.path).totals.size() == builds_before + 1);
    REQUIRE(read_compute(ctx.get()).reserved == small.reserved);
    require_actual_compute(read_compute(ctx.get()));

    run_tokens(ctx.get(), 64, true);
    REQUIRE(read_compute(ctx.get()).reserved > small.reserved);
    require_same_logits(logits, run_tokens(ctx.get(), 16));
    REQUIRE(llama_perf_context(ctx.get()).n_reused == 0);
    require_actual_compute(read_compute(ctx.get()));
    REQUIRE(read_compute(ctx.get()).reserved == small.reserved);
}

std::vector<float> run_embeddings(llama_context * ctx, int32_t n_tokens, int32_t token_offset) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, llama_token((i + token_offset + 1) % 128), i, {0}, true);
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    REQUIRE(status == 0);
    const float * embeddings = llama_get_embeddings_seq(ctx, 0);
    REQUIRE(embeddings != nullptr);
    std::vector<float> result(embeddings, embeddings + 32);
    for (float value : result) { REQUIRE(std::isfinite(value)); }
    return result;
}

void test_compute_compatibility_guards() {
    // An uncached attention graph must be rebuilt, but its mask allocation can
    // survive a rebuild only when the complete shape and attention mode match.
    llm_graph_result result(64);
    llm_graph_params params {};
    params.arch = LLM_ARCH_LLAMA;
    params.ubatch.n_tokens = params.ubatch.n_seq_tokens = 16;
    params.ubatch.n_seqs = params.ubatch.n_seqs_unq = 1;
    params.n_outputs = 1;
    result.set_params(params);
    auto input = std::make_unique<llm_graph_input_attn_no_cache>(params.hparams, params.cparams);
    auto * attention = input.get();
    attention->self_kq_mask = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F32, 16, 16);
    result.add_input(std::move(input));
    REQUIRE(!result.can_reuse(params));
    REQUIRE(result.can_reuse_compute(params));

    auto changed = params;
    changed.ubatch.n_tokens = changed.ubatch.n_seq_tokens = 32;
    REQUIRE(!result.can_reuse_compute(changed));
    changed = params;
    changed.n_outputs = 16;
    REQUIRE(!result.can_reuse_compute(changed));
    changed = params;
    changed.cparams.flash_attn = true;
    REQUIRE(!result.can_reuse_compute(changed));
    changed = params;
    changed.cparams.causal_attn = true;
    REQUIRE(!result.can_reuse_compute(changed));
    changed = params;
    changed.hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    REQUIRE(!result.can_reuse_compute(changed));

    auto * mask = attention->self_kq_mask;
    attention->self_kq_mask = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F32, 8, 32);
    REQUIRE(!result.can_reuse_compute(params)); // Same element count, different layout.
    attention->self_kq_mask = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F16, 16, 16);
    REQUIRE(!result.can_reuse_compute(params));
    attention->self_kq_mask = mask;

    auto swa_params = params;
    swa_params.hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    llm_graph_input_attn_no_cache swa(swa_params.hparams, swa_params.cparams);
    swa.self_kq_mask = mask;
    REQUIRE(!swa.can_reuse_compute(swa_params)); // Missing second mask.
    swa.self_kq_mask_swa = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F32, 16, 16);
    REQUIRE(swa.can_reuse_compute(swa_params));
    swa.self_kq_mask_swa = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F32, 16, 8);
    REQUIRE(!swa.can_reuse_compute(swa_params));

    // Inputs with no implemented compatibility check must continue rejecting
    // compute reuse, even when every known input matches.
    struct unknown_input : llm_graph_input_i {
        void set_input(const llama_ubatch *) override {}
    };
    result.add_input(std::make_unique<unknown_input>());
    REQUIRE(!result.can_reuse_compute(params));
    REQUIRE(!result.can_reuse_compute(params, true));
}

void test_pooling_rebuild_keeps_compute(llama_model * model) {
    // Pooling inputs still require graph rebuilding. Their stable shapes must
    // nevertheless retain compute allocations, including with diagnostics off.
    for (auto pooling : {LLAMA_POOLING_TYPE_MEAN, LLAMA_POOLING_TYPE_CLS}) {
        for (bool disable_reuse : {false, true}) {
            for (bool diagnostics : {false, true}) {
                set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
                set_env("LLAMA_GRAPH_REUSE_DISABLE", disable_reuse ? "1" : nullptr);
                temporary_csv csv;
                auto ctx = make_context(model, diagnostics ? csv.path : "", 64, diagnostics, pooling);
                set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
                REQUIRE(read_compute(ctx.get()).reserved == 0);

                size_t builds = 0;
                const auto check_run = [&](int32_t n_tokens, int32_t token_offset) {
                    const auto embedding = run_embeddings(ctx.get(), n_tokens, token_offset);
                    // A fresh context has no previous graph or buffer bindings.
                    auto fresh = make_context(model, "", 64, false, pooling);
                    require_same_logits(embedding, run_embeddings(fresh.get(), n_tokens, token_offset));
                    REQUIRE(llama_perf_context(ctx.get()).n_reused == 0);
                    const auto snapshot = read_compute(ctx.get());
                    REQUIRE(snapshot.reserved > 0);
                    if (diagnostics) {
                        require_actual_compute(snapshot);
                        const auto rows = read_and_validate(csv.path);
                        const size_t actual_builds = std::count_if(rows.totals.begin(), rows.totals.end(),
                                [](const row & values) { return values.at("event") == "build"; });
                        REQUIRE(actual_builds == ++builds);
                        REQUIRE(number(rows.totals.back(), "cache_entry_count") == 1);
                    }
                    return snapshot;
                };

                const auto small = check_run(16, 0);
                const auto small_scheduler = ctx->get_sched();
                const auto repeated = check_run(16, 3);
                REQUIRE(ctx->get_sched() == small_scheduler);
                REQUIRE(repeated.reserved == small.reserved);

                const auto large = check_run(64, 7);
                REQUIRE(large.reserved > small.reserved);
                const auto large_scheduler = ctx->get_sched();
                const auto large_repeated = check_run(64, 11);
                REQUIRE(ctx->get_sched() == large_scheduler);
                REQUIRE(large_repeated.reserved == large.reserved);

                const auto small_again = check_run(16, 0);
                REQUIRE(small_again.reserved == small.reserved);
                const auto small_again_scheduler = ctx->get_sched();
                const auto replay = check_run(16, 13);
                REQUIRE(ctx->get_sched() == small_again_scheduler);
                REQUIRE(replay.reserved == small.reserved);
            }
        }
    }
}

void shift_prefix(llama_context * ctx) {
    const auto memory = llama_get_memory(ctx);
    REQUIRE(llama_memory_can_shift(memory));
    REQUIRE(llama_memory_seq_rm(memory, 0, 0, 16));
    llama_memory_seq_add(memory, 0, 16, 32, -16);
}

void test_kv_shift(llama_model * model) {
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "4");
    temporary_csv csv;
    auto ctx = make_context(model, csv.path);
    run_tokens(ctx.get(), 16);
    const auto small = read_compute(ctx.get());
    run_tokens(ctx.get(), 32);
    const auto before_shift = read_and_validate(csv.path);
    REQUIRE(number(before_shift.totals.back(), "cache_entry_count") == 2);
    size_t largest_reserve_plan = 0;
    for (const auto & reserved : before_shift.reserves) {
        largest_reserve_plan = std::max(largest_reserve_plan, number(reserved, "full_planned_peak_bytes"));
    }
    REQUIRE(largest_reserve_plan > 0);

    // Applying this pending key-position shift during decode executes the
    // memory-maintenance graph and then restores scheduler topology.
    shift_prefix(ctx.get());
    const auto shifted_logits = run_tokens(ctx.get(), 16, false, 16, false);
    const auto after_shift = read_and_validate(csv.path);
    REQUIRE(after_shift.reserves.size() > before_shift.reserves.size());
    const auto & restored_reserve = after_shift.reserves.back();
    // The maintenance reserve restores topology without measuring its plan;
    // compare its retained allocation with the startup worst-case plan.
    REQUIRE(number(restored_reserve, "active_compute_reserved_bytes") < largest_reserve_plan);
    REQUIRE(number(after_shift.totals.back(), "cache_entry_count") == 1);
    require_actual_compute(read_compute(ctx.get()));
    REQUIRE(read_compute(ctx.get()).reserved == small.reserved);

    // An independent context with graph reuse disabled must produce the same
    // logits after applying the identical KV edit and follow-up decode.
    set_env("LLAMA_GRAPH_REUSE_DISABLE", "1");
    auto fresh = make_context(model, "");
    set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
    run_tokens(fresh.get(), 32);
    shift_prefix(fresh.get());
    require_same_logits(shifted_logits, run_tokens(fresh.get(), 16, false, 16, false));
    require_actual_compute(read_compute(fresh.get()));
}

void test_kv_growth(llama_model * model) {
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "4");
    temporary_csv csv;
    auto cached = make_context(model, csv.path, 512);
    REQUIRE(read_compute(cached.get()).reserved == 0);

    // n_kv is rounded in groups of 256. These two P16 executions have the
    // same token/output counts but need different attention shapes.
    const auto first_logits = run_tokens(cached.get(), 16);
    const auto first = read_and_validate(csv.path).totals.front();
    const auto first_scheduler = cached->get_sched();
    REQUIRE(number(first, "active_compute_reserved_bytes") > number(first, "full_planned_peak_bytes"));
    run_tokens(cached.get(), 64, false, 16, false);
    run_tokens(cached.get(), 64, false, 80, false);
    run_tokens(cached.get(), 64, false, 144, false);
    run_tokens(cached.get(), 32, false, 208, false);
    run_tokens(cached.get(), 16, false, 240, false);
    REQUIRE(cached->get_sched() == first_scheduler);
    const auto before_growth = read_and_validate(csv.path).totals.back();
    const auto reused_before_growth = llama_perf_context(cached.get()).n_reused;
    const auto grown_logits = run_tokens(cached.get(), 16, false, 256, false);
    REQUIRE(llama_perf_context(cached.get()).n_reused == reused_before_growth);
    REQUIRE(cached->get_sched() == first_scheduler);
    const auto grown = read_and_validate(csv.path).totals.back();
    REQUIRE(grown.at("event") == "build");
    REQUIRE(number(grown, "n_tokens") == 16);
    REQUIRE(number(grown, "n_outputs") == 1);
    REQUIRE(number(grown, "full_planned_peak_bytes") > number(first, "full_planned_peak_bytes"));
    REQUIRE(number(grown, "active_compute_reserved_bytes") >= number(grown, "full_planned_peak_bytes"));
    REQUIRE(number(grown, "active_compute_reserved_bytes") - number(grown, "full_planned_peak_bytes") <= 256);
    REQUIRE(number(grown, "active_compute_reserved_bytes") == number(first, "active_compute_reserved_bytes"));
    REQUIRE(number(grown, "context_compute_reserved_bytes") == number(before_growth, "context_compute_reserved_bytes"));
    REQUIRE(number(grown, "cache_entry_count") == 3); // P16/P32/P64, not another entry for KV=512.

    // P32's inactive entry also has a smaller execution KV shape. Recover
    // its full-KV allocation and rebuild in that slot without a fourth entry.
    const auto grown_32_logits = run_tokens(cached.get(), 32, false, 272, false);
    const auto grown_32 = read_and_validate(csv.path).totals.back();
    REQUIRE(grown_32.at("event") == "build");
    REQUIRE(number(grown_32, "cache_entry_count") == 3);
    REQUIRE(number(grown_32, "context_compute_reserved_bytes") == number(grown, "context_compute_reserved_bytes"));

    // A new independent query still computes over the shorter KV range.
    require_same_logits(first_logits, run_tokens(cached.get(), 16));
    REQUIRE(cached->get_sched() == first_scheduler);
    const auto cleared = read_and_validate(csv.path).totals.back();
    REQUIRE(number(cleared, "full_planned_peak_bytes") == number(first, "full_planned_peak_bytes"));
    REQUIRE(number(cleared, "active_compute_reserved_bytes") == number(first, "active_compute_reserved_bytes"));

    // Compare against a context without an inactive graph cache, which
    // rebuilds both P16 shapes, to catch stale cached buffer bindings.
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
    auto fresh = make_context(model, "", 512);
    require_same_logits(first_logits, run_tokens(fresh.get(), 16));
    REQUIRE(read_compute(fresh.get()).reserved == number(first, "active_compute_reserved_bytes"));
    run_tokens(fresh.get(), 64, false, 16, false);
    run_tokens(fresh.get(), 64, false, 80, false);
    run_tokens(fresh.get(), 64, false, 144, false);
    run_tokens(fresh.get(), 32, false, 208, false);
    run_tokens(fresh.get(), 16, false, 240, false);
    const auto fresh_scheduler = fresh->get_sched();
    const auto fresh_reserved = read_compute(fresh.get()).reserved;
    require_same_logits(grown_logits, run_tokens(fresh.get(), 16, false, 256, false));
    REQUIRE(fresh->get_sched() == fresh_scheduler);
    REQUIRE(read_compute(fresh.get()).reserved == fresh_reserved);
    require_actual_compute(read_compute(fresh.get()));
    require_same_logits(grown_32_logits, run_tokens(fresh.get(), 32, false, 272, false));
}

void test_decode_full_kv_reserve(llama_model * model) {
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
    // Exercise the production path with all memory diagnostics disabled.
    auto ctx = make_context(model, "", 512, false);
    const auto fill_prefix = [&](llama_context * target) {
        run_tokens(target, 64);
        run_tokens(target, 64, false, 64, false);
        run_tokens(target, 64, false, 128, false);
        run_tokens(target, 62, false, 192, false);
    };
    fill_prefix(ctx.get());
    run_tokens(ctx.get(), 1, false, 254, false);
    const auto scheduler = ctx->get_sched();
    const auto reserved = read_compute(ctx.get()).reserved;
    REQUIRE(reserved > 0);
    run_tokens(ctx.get(), 1, false, 255, false);
    const auto reused = llama_perf_context(ctx.get()).n_reused;
    const auto logits = run_tokens(ctx.get(), 1, false, 256, false);
    REQUIRE(llama_perf_context(ctx.get()).n_reused == reused);
    REQUIRE(ctx->get_sched() == scheduler);
    REQUIRE(read_compute(ctx.get()).reserved == reserved);

    set_env("LLAMA_GRAPH_REUSE_DISABLE", "1");
    auto fresh = make_context(model, "", 512, false);
    set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
    fill_prefix(fresh.get());
    run_tokens(fresh.get(), 1, false, 254, false);
    run_tokens(fresh.get(), 1, false, 255, false);
    require_same_logits(logits, run_tokens(fresh.get(), 1, false, 256, false));
}

void test_full_kv_batch_shape(llama_model * model) {
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
    auto ctx = make_context(model, "", 512);
    const auto logits = run_tokens(ctx.get(), 16);
    const auto small = read_compute(ctx.get());
    REQUIRE(small.reserved > small.full);
    require_same_logits(logits, run_tokens(ctx.get(), 16, true));
    REQUIRE(read_compute(ctx.get()).reserved > small.reserved);
    run_tokens(ctx.get(), 64, true);
    REQUIRE(read_compute(ctx.get()).reserved > small.reserved);
    require_same_logits(logits, run_tokens(ctx.get(), 16));
    REQUIRE(read_compute(ctx.get()).reserved == small.reserved);

    // More configured streams than active streams: using memory->init_full()
    // here would incorrectly widen the stream dimension of the reserve graph.
    auto multi = make_context(model, "", 2048, true, LLAMA_POOLING_TYPE_UNSPECIFIED, 4);
    require_same_logits(logits, run_tokens(multi.get(), 16));
    REQUIRE(read_compute(multi.get()).reserved == small.reserved);
    const auto scheduler = multi->get_sched();
    require_same_logits(logits, run_tokens(multi.get(), 16));
    REQUIRE(multi->get_sched() == scheduler);
}

} // namespace

int main() {
    set_env("GGML_SHARED_HTP_WEIGHTS", "0");
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "4");
    set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
    try {
        ggml_backend_load_all();
        auto reg = ggml_backend_reg_by_name("CPU");
        REQUIRE(reg != nullptr);
        scratch_getter_fixture scratch(reg);
        temporary_csv csv;
        llama_model_ptr model = make_model();
        llama_context_ptr ctx = make_context(model.get(), csv.path);
        REQUIRE(llama_warmup_min_tokens(ctx.get()) == 1);
        REQUIRE(read_compute(ctx.get()).reserved == 0);
        llama_perf_context_reset(ctx.get());

        const auto logits_16 = run_tokens(ctx.get(), 16);
        const auto compute_16 = read_compute(ctx.get());
        require_actual_compute(compute_16);
        auto initial = read_and_validate(csv.path);
        REQUIRE(initial.reserves.size() >= 3);
        REQUIRE(initial.reserves.front().at("warmup") == "0");
        for (const auto & reserved : initial.reserves) {
            REQUIRE(number(reserved, "n_outputs") == number(reserved, "n_tokens"));
            REQUIRE(number(reserved, "context_metadata_capacity_bytes") ==
                2 * number(reserved, "active_metadata_capacity_bytes"));
            REQUIRE(number(reserved, "active_compute_reserved_bytes") == 0);
            REQUIRE(number(reserved, "context_compute_reserved_bytes") == 0);
        }
        REQUIRE(initial.totals.size() == 2);
        REQUIRE(initial.totals.front().at("event") == "build");
        REQUIRE(initial.totals.back().at("event") == "compute");
        REQUIRE(number(initial.totals.front(), "backend_scratch_reserved_bytes") == 0);
        REQUIRE(number(initial.totals.back(), "backend_scratch_reserved_bytes") == 0);
        REQUIRE(number(initial.totals.front(), "active_compute_reserved_bytes") == compute_16.reserved);
        REQUIRE(number(initial.totals.front(), "full_planned_peak_bytes") == compute_16.full);
        size_t largest_reserve_plan = 0;
        for (const auto & reserved : initial.reserves) {
            largest_reserve_plan = std::max(largest_reserve_plan, number(reserved, "full_planned_peak_bytes"));
        }
        REQUIRE(compute_16.reserved < largest_reserve_plan);

        scratch.grow(256, 64, 48, 128);
        run_tokens(ctx.get(), 16);
        REQUIRE(llama_perf_context(ctx.get()).n_reused == 1);
        auto grown = read_and_validate(csv.path);
        REQUIRE(grown.totals.size() == 3);
        REQUIRE(grown.totals.back().at("event") == "compute");
        REQUIRE(number(grown.totals.back(), "backend_scratch_reserved_bytes") == 256);
        REQUIRE(number(grown.totals.back(), "weight_request_peak_bytes") == 64);
        REQUIRE(number(grown.totals.back(), "activation_request_peak_bytes") == 48);
        REQUIRE(number(grown.totals.back(), "scratch_request_peak_bytes") == 128);

        run_tokens(ctx.get(), 16);
        REQUIRE(llama_perf_context(ctx.get()).n_reused == 2);
        REQUIRE(read_and_validate(csv.path).row_count == grown.row_count);

        // Logical requests can grow without changing allocation capacity.
        scratch.grow(256, 64, 96, 192);
        run_tokens(ctx.get(), 16);
        REQUIRE(llama_perf_context(ctx.get()).n_reused == 3);
        auto changed_request = read_and_validate(csv.path);
        REQUIRE(changed_request.totals.size() == 4);
        REQUIRE(number(changed_request.totals.back(), "scratch_request_peak_bytes") == 192);
        REQUIRE(number(changed_request.totals.back(), "backend_scratch_reserved_bytes") == 256);

        run_tokens(ctx.get(), 32);
        auto cached = read_and_validate(csv.path);
        REQUIRE(cached.totals.size() == 5);
        REQUIRE(cached.totals.back().at("event") == "build");
        REQUIRE(number(cached.totals.back(), "cache_entry_count") >= 2);
        REQUIRE(number(cached.totals.back(), "backend_scratch_reserved_bytes") == 256);
        const size_t compute_32 = number(cached.totals.back(), "active_compute_reserved_bytes");
        REQUIRE(compute_32 > compute_16.reserved);
        REQUIRE(compute_32 >= number(cached.totals.back(), "full_planned_peak_bytes"));
        REQUIRE(compute_32 - number(cached.totals.back(), "full_planned_peak_bytes") <= 256);
        REQUIRE(number(cached.totals.back(), "context_compute_reserved_bytes") == compute_16.reserved + compute_32);
        REQUIRE(read_compute(ctx.get()).reserved == compute_16.reserved + compute_32);
        const size_t cached_full_plan = compute_16.full + number(cached.totals.back(), "full_planned_peak_bytes");
        REQUIRE(read_compute(ctx.get()).full == cached_full_plan);

        scratch.grow(1024, 256, 384, 768);
        require_same_logits(logits_16, run_tokens(ctx.get(), 16));
        REQUIRE(llama_perf_context(ctx.get()).n_reused == 4);
        auto restored = read_and_validate(csv.path);
        REQUIRE(restored.totals.size() == 6);
        const auto & last = restored.totals.back();
        REQUIRE(last.at("event") == "compute");
        REQUIRE(number(last, "cache_entry_count") >= 2);
        REQUIRE(number(last, "backend_scratch_reserved_bytes") == 1024);
        REQUIRE(number(last, "active_compute_reserved_bytes") == compute_16.reserved);
        REQUIRE(number(last, "context_compute_reserved_bytes") == compute_16.reserved + compute_32);
        REQUIRE(read_compute(ctx.get()).full == cached_full_plan);
        for (const char * key : {"metadata_empty_used_bytes", "metadata_build_bytes",
                "base_planned_peak_bytes", "full_planned_peak_bytes"}) {
            REQUIRE(last.at(key) == initial.totals.front().at(key));
        }
        test_output_selection_and_eviction(model.get());
        test_rebuild_without_reuse(model.get());
        test_compute_compatibility_guards();
        test_pooling_rebuild_keeps_compute(model.get());
        test_kv_shift(model.get());
        test_kv_growth(model.get());
        test_decode_full_kv_reserve(model.get());
        test_full_kv_batch_shape(model.get());
        std::puts("PASS graph-memory: actual batch/output sizes, maximum-KV reserve, cache reuse/eviction, pooling rebuilds, warmup, KV shift/growth, scratch schema");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "shared weight memory test failed: %s\n", error.what());
        return 1;
    }
}
