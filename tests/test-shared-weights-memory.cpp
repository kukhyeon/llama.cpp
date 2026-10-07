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
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
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

void require_reserved_compute(llama_context * ctx, ggml_backend_sched_t scheduler, size_t capacity) {
    REQUIRE(ctx->get_sched() == scheduler);
    REQUIRE(read_compute(ctx).reserved == capacity);
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
    std::vector<row> compute_reservations;
    std::vector<row> compute_peaks;
    std::vector<row> compute_peak_tensors;
    size_t row_count = 0;
};

using reservation_key = std::array<std::string, 4>;
using peak_key = std::array<std::string, 5>;

reservation_key reservation_identity(const row & values) {
    return {values.at("scheduler_id"), values.at("reservation_id"),
            values.at("backend"), values.at("buffer_type")};
}

peak_key peak_identity(const row & values) {
    const auto reservation = reservation_identity(values);
    return {reservation[0], reservation[1], reservation[2], reservation[3], values.at("chunk_id")};
}

void validate_compute_rows(const csv_snapshot & snapshot) {
    const std::set<std::string> reasons = {
        "startup_prefill", "startup_decode", "startup_prefill_restore", "runtime_build",
        "warmup_build", "memory_update", "support_check", "runtime_cache_reuse",
        "startup_measurement", "startup_prefill_retry",
    };
    std::map<reservation_key, const row *> reservations;
    std::map<std::string, std::string> reservation_owners;
    for (const auto & values : snapshot.compute_reservations) {
        REQUIRE(number(values, "scheduler_id") > 0);
        REQUIRE(number(values, "reservation_id") > 0);
        REQUIRE(reasons.count(values.at("reserve_reason")) == 1);
        REQUIRE(!values.at("backend").empty());
        REQUIRE(!values.at("buffer_type").empty());
        REQUIRE(reservations.emplace(reservation_identity(values), &values).second);
        const auto owner = reservation_owners.emplace(values.at("reservation_id"), values.at("scheduler_id"));
        REQUIRE(owner.first->second == values.at("scheduler_id"));

        const auto & action = values.at("allocation_action");
        REQUIRE(action == "allocate" || action == "reallocate" || action == "reuse" ||
                action == "measurement_only" || action == "split_only");
        const auto & plan = values.at("plan_kind");
        REQUIRE(plan == "actual" || plan == "full" || plan == "cached_full" || plan == "none");
        REQUIRE(values.at("allocation_success") == "0" || values.at("allocation_success") == "1");
        const size_t before = number(values, "reserved_before_bytes");
        const size_t after = number(values, "reserved_after_bytes");
        if (action == "reuse" || action == "measurement_only" || action == "split_only") {
            REQUIRE(before == after);
        } else if (action == "allocate") {
            REQUIRE(before == 0);
        }
        if (plan == "none") {
            REQUIRE(values.at("planned_required_bytes").empty());
            REQUIRE(values.at("reserved_excess_bytes").empty());
        } else {
            const size_t required = number(values, "planned_required_bytes");
            if (values.at("allocation_success") == "1" && action != "measurement_only") {
                REQUIRE(after >= required);
                REQUIRE(number(values, "reserved_excess_bytes") == after - required);
            } else {
                REQUIRE(values.at("reserved_excess_bytes").empty());
            }
        }
    }

    std::map<peak_key, const row *> peaks;
    std::map<reservation_key, size_t> planned_bytes;
    for (const auto & values : snapshot.compute_peaks) {
        const auto reservation = reservations.find(reservation_identity(values));
        REQUIRE(reservation != reservations.end());
        REQUIRE(values.at("plan_kind") == reservation->second->at("plan_kind"));
        REQUIRE(values.at("plan_kind") != "none");
        REQUIRE(peaks.emplace(peak_identity(values), &values).second);
        REQUIRE(std::stoll(values.at("peak_node_index")) >= -1);
        const size_t span = number(values, "planned_required_bytes");
        const size_t live = number(values, "live_allocated_bytes");
        REQUIRE(live <= span);
        REQUIRE(number(values, "free_gap_bytes") == span - live);
        planned_bytes[reservation_identity(values)] += span;
    }
    for (const auto & entry : planned_bytes) {
        REQUIRE(entry.second == number(*reservations.at(entry.first), "planned_required_bytes"));
    }

    const auto device = ggml_backend_dev_by_name("CPU");
    REQUIRE(device != nullptr);
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(device));
    REQUIRE(alignment > 0);
    std::map<peak_key, size_t> live_bytes;
    std::map<peak_key, std::vector<std::pair<size_t, size_t>>> allocations;
    for (const auto & values : snapshot.compute_peak_tensors) {
        const auto key = peak_identity(values);
        const auto peak = peaks.find(key);
        REQUIRE(peak != peaks.end());
        REQUIRE(values.at("peak_node_index") == peak->second->at("peak_node_index"));
        REQUIRE(!values.at("tensor_op").empty());
        REQUIRE(!values.at("tensor_shape").empty());
        REQUIRE(!values.at("tensor_role").empty());
        const size_t offset = number(values, "tensor_offset_bytes");
        const size_t bytes = number(values, "tensor_allocation_bytes");
        const size_t span = number(*peak->second, "planned_required_bytes");
        REQUIRE(bytes > 0 && bytes % alignment == 0);
        REQUIRE(offset % alignment == 0);
        REQUIRE(offset <= span && bytes <= span - offset);
        live_bytes[key] += bytes;
        allocations[key].emplace_back(offset, offset + bytes);
    }
    for (const auto & entry : peaks) {
        REQUIRE(live_bytes[entry.first] == number(*entry.second, "live_allocated_bytes"));
        auto & ranges = allocations[entry.first];
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            REQUIRE(ranges[i - 1].second <= ranges[i].first);
        }
    }
}

csv_snapshot read_and_validate(const std::string & path) {
    std::ifstream input(path);
    REQUIRE(input.good());
    std::string line;
    REQUIRE(bool(std::getline(input, line)));
    const auto header = split_csv(line);
    const auto expected_header = split_csv(
        "schema_version,event_id,event,phase,warmup,graph_type,n_tokens,n_outputs,n_seqs,"
        "policy_profile,route_mode,cache_entry_count,n_nodes,metadata_empty_used_bytes,"
        "metadata_arena_used_bytes,metadata_build_bytes,active_metadata_capacity_bytes,"
        "context_metadata_capacity_bytes,active_scheduler_metadata_bytes,"
        "context_scheduler_metadata_bytes,active_scheduler_context_used_bytes,"
        "active_scheduler_context_capacity_bytes,active_scheduler_copy_tensor_count,"
        "active_scheduler_input_dependency_count,active_scheduler_route_dependency_count,"
        "active_scheduler_graph_nodes,active_scheduler_graph_leafs,"
        "active_scheduler_backend_id_capacity_bytes,active_scheduler_graph_pointer_capacity_bytes,"
        "active_scheduler_max_split_inputs,active_auxiliary_buffer_bytes,context_auxiliary_buffer_bytes,"
        "reachable_owned_storage_count,logical_unique_storage_bytes,logical_metadata_plus_storage_bytes,"
        "row_scope,backend,buffer_type,base_planned_peak_bytes,full_planned_peak_bytes,"
        "active_compute_reserved_bytes,context_compute_reserved_bytes,tracked_context_graph_reserved_bytes,"
        "weight_request_peak_bytes,activation_request_peak_bytes,scratch_request_peak_bytes,"
        "backend_scratch_reserved_bytes,graph_plus_backend_scratch_reserved_bytes,"
        "scheduler_id,reservation_id,reserve_reason,plan_kind,kv_extent,allocation_action,allocation_success,"
        "planned_required_bytes,reserved_before_bytes,reserved_after_bytes,reserved_excess_bytes,chunk_id,"
        "peak_node_index,peak_node_name,live_allocated_bytes,free_gap_bytes,tensor_name,tensor_op,tensor_shape,"
        "tensor_role,tensor_offset_bytes,tensor_allocation_bytes");
    REQUIRE(header.size() == 70);
    REQUIRE(header == expected_header);
    std::map<std::string, std::vector<row>> events;
    csv_snapshot result;
    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        REQUIRE(fields.size() == header.size());
        row values;
        for (size_t i = 0; i < fields.size(); ++i) { values.emplace(header[i], fields[i]); }
        REQUIRE(values.size() == header.size());
        REQUIRE(values.at("schema_version") == "5");
        ++result.row_count;
        const auto & scope = values.at("row_scope");
        if (scope == "compute_reservation") {
            result.compute_reservations.push_back(values);
            continue;
        }
        if (scope == "compute_peak") {
            result.compute_peaks.push_back(values);
            continue;
        }
        if (scope == "compute_peak_tensor") {
            result.compute_peak_tensors.push_back(values);
            continue;
        }
        events[values.at("event_id")].push_back(values);
        if (scope == "total") {
            (values.at("event") == "reserve" ? result.reserves : result.totals).push_back(values);
        }
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
    validate_compute_rows(result);
    return result;
}

const row & last_reservation(const csv_snapshot & snapshot, const char * reason) {
    const auto found = std::find_if(snapshot.compute_reservations.rbegin(), snapshot.compute_reservations.rend(),
            [&](const row & values) { return values.at("reserve_reason") == reason; });
    REQUIRE(found != snapshot.compute_reservations.rend());
    return *found;
}

std::string require_startup_reservations(const csv_snapshot & snapshot, size_t capacity) {
    const auto & prefill = last_reservation(snapshot, "startup_prefill");
    const auto & restore = last_reservation(snapshot, "startup_prefill_restore");
    const auto & scheduler_id = prefill.at("scheduler_id");
    REQUIRE(prefill.at("allocation_action") == "allocate");
    REQUIRE(number(prefill, "reserved_before_bytes") == 0);
    REQUIRE(number(prefill, "reserved_after_bytes") > 0);
    REQUIRE(number(restore, "reserved_after_bytes") == capacity);
    REQUIRE(!snapshot.compute_peaks.empty());
    REQUIRE(!snapshot.compute_peak_tensors.empty());
    for (const char * reason : {"startup_prefill", "startup_decode", "startup_prefill_restore"}) {
        const auto & values = last_reservation(snapshot, reason);
        REQUIRE(values.at("scheduler_id") == scheduler_id);
        REQUIRE(values.at("plan_kind") == "actual");
        REQUIRE(values.at("allocation_success") == "1");
        REQUIRE(number(values, "planned_required_bytes") > 0);
        REQUIRE(number(values, "kv_extent") > 0);
        REQUIRE(std::any_of(snapshot.compute_peaks.begin(), snapshot.compute_peaks.end(),
                [&](const row & peak) { return reservation_identity(peak) == reservation_identity(values); }));
    }
    return scheduler_id;
}

void require_reused_reservation(
        const csv_snapshot & snapshot, const char * reason, const std::string & scheduler_id, size_t capacity) {
    const auto & values = last_reservation(snapshot, reason);
    REQUIRE(values.at("scheduler_id") == scheduler_id);
    REQUIRE(values.at("allocation_action") == "reuse");
    REQUIRE(values.at("allocation_success") == "1");
    REQUIRE(number(values, "reserved_before_bytes") == capacity);
    REQUIRE(number(values, "reserved_after_bytes") == capacity);
    REQUIRE(number(values, "planned_required_bytes") > 0);
    REQUIRE(number(values, "reserved_excess_bytes") == capacity - number(values, "planned_required_bytes"));
}

void test_eager_compute_retained(llama_model * model) {
    // With one cache slot, startup reserves the maximum batch/all-logits
    // graph once. Graph rebuilding must retain that scheduler and capacity.
    for (bool disable_reuse : {false, true}) {
        for (bool diagnostics : {false, true}) {
            set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
            set_env("LLAMA_GRAPH_REUSE_DISABLE", disable_reuse ? "1" : nullptr);
            temporary_csv csv;
            auto ctx = make_context(model, diagnostics ? csv.path : "", 64, diagnostics);
            set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
            const auto scheduler = ctx->get_sched();
            const size_t capacity = read_compute(ctx.get()).reserved;
            REQUIRE(capacity > 0);
            const std::string scheduler_id = diagnostics
                ? require_startup_reservations(read_and_validate(csv.path), capacity) : "";

            const auto logits = run_tokens(ctx.get(), 16);
            require_reserved_compute(ctx.get(), scheduler, capacity);
            if (diagnostics) {
                const auto initial = read_and_validate(csv.path);
                REQUIRE(capacity > number(initial.totals.front(), "full_planned_peak_bytes"));
                require_reused_reservation(initial, "runtime_build", scheduler_id, capacity);
            }
            require_same_logits(logits, run_tokens(ctx.get(), 16, true));
            require_reserved_compute(ctx.get(), scheduler, capacity);
            run_tokens(ctx.get(), 64, true);
            require_reserved_compute(ctx.get(), scheduler, capacity);
            require_same_logits(logits, run_tokens(ctx.get(), 16));
            require_reserved_compute(ctx.get(), scheduler, capacity);

            const auto reused_before = llama_perf_context(ctx.get()).n_reused;
            require_same_logits(logits, run_tokens(ctx.get(), 16));
            REQUIRE(llama_perf_context(ctx.get()).n_reused == reused_before + (disable_reuse ? 0 : 1));
            require_reserved_compute(ctx.get(), scheduler, capacity);

            // Warmup invalidates graphs but retains the eagerly reserved
            // compute allocation for the subsequent real inference.
            llama_set_warmup(ctx.get(), true);
            run_tokens(ctx.get(), 64, true);
            require_reserved_compute(ctx.get(), scheduler, capacity);
            if (diagnostics) {
                const auto warmup = read_and_validate(csv.path);
                require_reused_reservation(warmup, "warmup_build", scheduler_id, capacity);
                REQUIRE(last_reservation(warmup, "warmup_build").at("warmup") == "1");
            }
            llama_set_warmup(ctx.get(), false);
            require_same_logits(logits, run_tokens(ctx.get(), 16));
            require_reserved_compute(ctx.get(), scheduler, capacity);
            if (diagnostics) {
                const auto restored = read_and_validate(csv.path);
                require_reused_reservation(restored, "runtime_build", scheduler_id, capacity);
                REQUIRE(last_reservation(restored, "runtime_build").at("warmup") == "0");
            }

            auto fresh = make_context(model, "", 64, false);
            require_same_logits(logits, run_tokens(fresh.get(), 16));
        }
    }
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

void test_pooling_rebuild_keeps_compute(llama_model * model) {
    // Pooling inputs require graph rebuilding; both unchanged and changed
    // shapes keep the eagerly reserved compute buffers.
    for (auto pooling : {LLAMA_POOLING_TYPE_MEAN, LLAMA_POOLING_TYPE_CLS}) {
        for (bool disable_reuse : {false, true}) {
            for (bool diagnostics : {false, true}) {
                set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
                set_env("LLAMA_GRAPH_REUSE_DISABLE", disable_reuse ? "1" : nullptr);
                temporary_csv csv;
                auto ctx = make_context(model, diagnostics ? csv.path : "", 64, diagnostics, pooling);
                set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
                const auto scheduler = ctx->get_sched();
                const size_t capacity = read_compute(ctx.get()).reserved;
                REQUIRE(capacity > 0);

                size_t builds = 0;
                const auto check_run = [&](int32_t n_tokens, int32_t token_offset) {
                    const auto embedding = run_embeddings(ctx.get(), n_tokens, token_offset);
                    // A fresh context has no previous graph or buffer bindings.
                    auto fresh = make_context(model, "", 64, false, pooling);
                    require_same_logits(embedding, run_embeddings(fresh.get(), n_tokens, token_offset));
                    REQUIRE(llama_perf_context(ctx.get()).n_reused == 0);
                    const auto snapshot = read_compute(ctx.get());
                    require_reserved_compute(ctx.get(), scheduler, capacity);
                    if (diagnostics) {
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
                REQUIRE(large.reserved == small.reserved);
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
    const size_t startup_capacity = read_compute(ctx.get()).reserved;
    REQUIRE(startup_capacity > 0);
    run_tokens(ctx.get(), 16);
    run_tokens(ctx.get(), 32);
    const auto before_shift = read_and_validate(csv.path);
    REQUIRE(number(before_shift.totals.back(), "cache_entry_count") == 2);

    // KV maintenance invalidates the graph cache and eagerly reserves the
    // maximum graph again before the next ordinary execution.
    shift_prefix(ctx.get());
    const auto shifted_logits = run_tokens(ctx.get(), 16, false, 16, false);
    const auto after_shift = read_and_validate(csv.path);
    REQUIRE(after_shift.reserves.size() > before_shift.reserves.size());
    REQUIRE(last_reservation(after_shift, "memory_update").at("allocation_success") == "1");
    REQUIRE(number(after_shift.reserves.back(), "active_compute_reserved_bytes") == startup_capacity);
    REQUIRE(number(after_shift.totals.back(), "cache_entry_count") == 1);
    REQUIRE(read_compute(ctx.get()).reserved == startup_capacity);

    set_env("LLAMA_GRAPH_REUSE_DISABLE", "1");
    auto fresh = make_context(model, "");
    set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
    run_tokens(fresh.get(), 32);
    shift_prefix(fresh.get());
    require_same_logits(shifted_logits, run_tokens(fresh.get(), 16, false, 16, false));
    REQUIRE(read_compute(fresh.get()).reserved == startup_capacity);
}

void test_kv_growth(llama_model * model) {
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
    temporary_csv csv;
    auto ctx = make_context(model, csv.path, 512);
    const auto scheduler = ctx->get_sched();
    const size_t capacity = read_compute(ctx.get()).reserved;
    REQUIRE(capacity > 0);

    // Crossing the KV padding boundary requires a new execution graph, but
    // the initial maximum-KV, maximum-batch allocation already covers it.
    const auto first_logits = run_tokens(ctx.get(), 16);
    const auto first_snapshot = read_and_validate(csv.path);
    const auto first = first_snapshot.totals.front();
    const auto first_reservation = last_reservation(first_snapshot, "runtime_build");
    run_tokens(ctx.get(), 64, false, 16, false);
    run_tokens(ctx.get(), 64, false, 80, false);
    run_tokens(ctx.get(), 64, false, 144, false);
    run_tokens(ctx.get(), 32, false, 208, false);
    run_tokens(ctx.get(), 16, false, 240, false);
    require_reserved_compute(ctx.get(), scheduler, capacity);
    const auto reused_before = llama_perf_context(ctx.get()).n_reused;
    const auto grown_logits = run_tokens(ctx.get(), 16, false, 256, false);
    REQUIRE(llama_perf_context(ctx.get()).n_reused == reused_before);
    require_reserved_compute(ctx.get(), scheduler, capacity);
    const auto grown_snapshot = read_and_validate(csv.path);
    const auto grown = grown_snapshot.totals.back();
    require_reused_reservation(grown_snapshot, "runtime_build", first_reservation.at("scheduler_id"), capacity);
    REQUIRE(number(last_reservation(grown_snapshot, "runtime_build"), "kv_extent") >
            number(first_reservation, "kv_extent"));
    REQUIRE(grown.at("event") == "build");
    REQUIRE(number(grown, "full_planned_peak_bytes") > number(first, "full_planned_peak_bytes"));
    REQUIRE(number(grown, "full_planned_peak_bytes") <= capacity);
    REQUIRE(number(grown, "cache_entry_count") == 1);

    // A fresh graph on every call provides an independent reference for KV
    // bindings after the larger attention shape is built.
    set_env("LLAMA_GRAPH_REUSE_DISABLE", "1");
    auto fresh = make_context(model, "", 512, false);
    set_env("LLAMA_GRAPH_REUSE_DISABLE", nullptr);
    require_same_logits(first_logits, run_tokens(fresh.get(), 16));
    run_tokens(fresh.get(), 64, false, 16, false);
    run_tokens(fresh.get(), 64, false, 80, false);
    run_tokens(fresh.get(), 64, false, 144, false);
    run_tokens(fresh.get(), 32, false, 208, false);
    run_tokens(fresh.get(), 16, false, 240, false);
    require_same_logits(grown_logits, run_tokens(fresh.get(), 16, false, 256, false));
    require_same_logits(first_logits, run_tokens(ctx.get(), 16));
    require_reserved_compute(ctx.get(), scheduler, capacity);
}

void test_decode_retains_startup_reserve(llama_model * model) {
    set_env("LLAMA_GRAPH_CACHE_CAPACITY", "1");
    // Exercise the production path with all memory diagnostics disabled.
    auto ctx = make_context(model, "", 512, false);
    const auto scheduler = ctx->get_sched();
    const auto reserved = read_compute(ctx.get()).reserved;
    REQUIRE(reserved > 0);
    const auto fill_prefix = [&](llama_context * target) {
        run_tokens(target, 64);
        run_tokens(target, 64, false, 64, false);
        run_tokens(target, 64, false, 128, false);
        run_tokens(target, 62, false, 192, false);
    };
    fill_prefix(ctx.get());
    run_tokens(ctx.get(), 1, false, 254, false);
    require_reserved_compute(ctx.get(), scheduler, reserved);
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
        const size_t startup_capacity = read_compute(ctx.get()).reserved;
        REQUIRE(startup_capacity > 0);
        llama_perf_context_reset(ctx.get());

        const auto logits_16 = run_tokens(ctx.get(), 16);
        const auto compute_16 = read_compute(ctx.get());
        REQUIRE(compute_16.reserved == startup_capacity);
        auto initial = read_and_validate(csv.path);
        const std::string scheduler_16 = require_startup_reservations(initial, startup_capacity);
        require_reused_reservation(initial, "runtime_build", scheduler_16, startup_capacity);
        REQUIRE(initial.reserves.size() >= 3);
        REQUIRE(initial.reserves.front().at("warmup") == "0");
        for (const auto & reserved : initial.reserves) {
            REQUIRE(number(reserved, "n_outputs") == number(reserved, "n_tokens"));
            REQUIRE(number(reserved, "context_metadata_capacity_bytes") ==
                2 * number(reserved, "active_metadata_capacity_bytes"));
        }
        REQUIRE(initial.totals.size() == 2);
        REQUIRE(initial.totals.front().at("event") == "build");
        REQUIRE(initial.totals.back().at("event") == "compute");
        REQUIRE(number(initial.totals.front(), "backend_scratch_reserved_bytes") == 0);
        REQUIRE(number(initial.totals.back(), "backend_scratch_reserved_bytes") == 0);
        REQUIRE(number(initial.totals.front(), "active_compute_reserved_bytes") == compute_16.reserved);
        REQUIRE(number(initial.totals.front(), "full_planned_peak_bytes") < startup_capacity);
        REQUIRE(number(initial.reserves.back(), "active_compute_reserved_bytes") == startup_capacity);
        size_t largest_reserve_plan = 0;
        for (const auto & reserved : initial.reserves) {
            largest_reserve_plan = std::max(largest_reserve_plan, number(reserved, "full_planned_peak_bytes"));
        }
        REQUIRE(largest_reserve_plan > 0);
        REQUIRE(compute_16.reserved >= largest_reserve_plan);

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
        const auto & reservation_32 = last_reservation(cached, "runtime_build");
        REQUIRE(reservation_32.at("scheduler_id") != scheduler_16);
        REQUIRE(reservation_32.at("allocation_action") == "allocate");
        REQUIRE(number(reservation_32, "reserved_before_bytes") == 0);
        REQUIRE(number(reservation_32, "reserved_after_bytes") == compute_32);
        // Additional graph-cache entries keep their own allocation; they do
        // not inherit the first entry's startup maximum reserve.
        REQUIRE(compute_32 > 0);
        REQUIRE(compute_32 >= number(cached.totals.back(), "full_planned_peak_bytes"));
        REQUIRE(number(cached.totals.back(), "context_compute_reserved_bytes") == compute_16.reserved + compute_32);
        REQUIRE(read_compute(ctx.get()).reserved == compute_16.reserved + compute_32);

        scratch.grow(1024, 256, 384, 768);
        require_same_logits(logits_16, run_tokens(ctx.get(), 16));
        REQUIRE(llama_perf_context(ctx.get()).n_reused == 4);
        auto restored = read_and_validate(csv.path);
        require_reused_reservation(restored, "runtime_cache_reuse", scheduler_16, startup_capacity);
        REQUIRE(restored.compute_peaks.size() == cached.compute_peaks.size());
        REQUIRE(restored.compute_peak_tensors.size() == cached.compute_peak_tensors.size());
        REQUIRE(restored.totals.size() == 6);
        const auto & last = restored.totals.back();
        REQUIRE(last.at("event") == "compute");
        REQUIRE(number(last, "cache_entry_count") >= 2);
        REQUIRE(number(last, "backend_scratch_reserved_bytes") == 1024);
        REQUIRE(number(last, "active_compute_reserved_bytes") == compute_16.reserved);
        REQUIRE(number(last, "context_compute_reserved_bytes") == compute_16.reserved + compute_32);
        for (const char * key : {"metadata_empty_used_bytes", "metadata_build_bytes",
                "base_planned_peak_bytes", "full_planned_peak_bytes"}) {
            REQUIRE(last.at(key) == initial.totals.front().at(key));
        }
        test_eager_compute_retained(model.get());
        test_pooling_rebuild_keeps_compute(model.get());
        test_kv_shift(model.get());
        test_kv_growth(model.get());
        test_decode_retains_startup_reserve(model.get());
        std::puts("PASS graph-memory: eager compute reserve, retained capacity, graph cache, pooling rebuilds, warmup, KV shift/growth, scheduler metadata, scratch schema, reservation identity, peak allocations");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "shared weight memory test failed: %s\n", error.what());
        return 1;
    }
}
