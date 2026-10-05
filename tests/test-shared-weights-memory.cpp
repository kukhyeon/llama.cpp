#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-shared-weights.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

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

llama_context_ptr make_context(llama_model * model, const std::string & csv_path) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = params.n_batch = params.n_ubatch = 64;
    params.n_threads = params.n_threads_batch = 1;
    params.no_perf = false;
    params.offload_kqv = params.op_offload = false;
    params.graph_memory_stats = true;
    REQUIRE(csv_path.size() < sizeof(params.graph_memory_stats_path));
    std::snprintf(params.graph_memory_stats_path, sizeof(params.graph_memory_stats_path), "%s", csv_path.c_str());
    llama_context_ptr ctx(llama_init_from_model(model, params));
    REQUIRE(ctx != nullptr);
    return ctx;
}

void run_tokens(llama_context * ctx, int32_t n_tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, llama_token(i + 1), llama_pos(i), {0}, i + 1 == n_tokens);
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    REQUIRE(status == 0);
    REQUIRE(llama_get_logits_ith(ctx, -1) != nullptr);
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
        llama_perf_context_reset(ctx.get());

        run_tokens(ctx.get(), 16);
        auto initial = read_and_validate(csv.path);
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

        scratch.grow(1024, 256, 384, 768);
        run_tokens(ctx.get(), 16);
        REQUIRE(llama_perf_context(ctx.get()).n_reused == 4);
        auto restored = read_and_validate(csv.path);
        REQUIRE(restored.totals.size() == 6);
        const auto & last = restored.totals.back();
        REQUIRE(last.at("event") == "compute");
        REQUIRE(number(last, "cache_entry_count") >= 2);
        REQUIRE(number(last, "backend_scratch_reserved_bytes") == 1024);
        for (const char * key : {"metadata_empty_used_bytes", "metadata_build_bytes",
                "base_planned_peak_bytes", "full_planned_peak_bytes"}) {
            REQUIRE(last.at(key) == initial.totals.front().at(key));
        }
        std::puts("PASS graph-memory schema 4: scheduler descriptors, reserve events, scratch capacities, graph reuse");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "shared weight memory test failed: %s\n", error.what());
        return 1;
    }
}
