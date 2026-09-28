#include "ggml-shared-weights.h"
#include "ggml-impl.h"

#include <cinttypes>
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
struct attachment {
    const void * key;
    void * handle;
    void (*release)(void *);
};
struct allocation {
    ggml_shared_weight_buffer desc;
    std::vector<attachment> imports;
    ~allocation() { for (auto & a : imports) { a.release(a.handle); } }
};
struct registry {
    std::mutex mutex;
    std::unordered_set<ggml_backend_buffer_type_t> types;
    std::unordered_map<ggml_backend_buffer_t, std::unique_ptr<allocation>> buffers;
    // Timestamp-seeded IDs keep appended CSV runs distinct without recording
    // file descriptors or raw host addresses in diagnostic output.
    uint64_t next_id = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
};
// Backend registries may unregister their types during static destruction.
// Keep this small registry alive until process exit to avoid destructor ordering.
registry & state() { static registry * s = new registry; return *s; }
}

bool ggml_shared_weights_enabled(void) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_SHARED_HTP_WEIGHTS");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
namespace {
std::atomic<int> repack_query_id {-1};
std::atomic<int> repack_phase {-1};
bool shared_option(const char * name) {
    const char * value = std::getenv(name);
    return !value || std::strcmp(value, "0") != 0;
}
}
bool ggml_shared_weights_repack_enabled(void) {
    static const bool enabled = shared_option("GGML_SHARED_HTP_REPACK");
    return ggml_shared_weights_enabled() && enabled;
}
bool ggml_shared_weights_timing_enabled(void) {
    static const bool enabled = shared_option("GGML_SHARED_WEIGHTS_REPACK_TIMING");
    const char * path = std::getenv("GGML_SHARED_WEIGHTS_CSV");
    return ggml_shared_weights_repack_enabled() && enabled && path && *path;
}
bool ggml_shared_weights_get_scratch_stats(
        ggml_backend_t backend, ggml_shared_weights_scratch_stats * stats) {
    if (!stats) { return false; }
    *stats = {};
    if (!backend) { return false; }
    auto dev = ggml_backend_get_device(backend);
    if (!dev) { return false; }
    auto fn = reinterpret_cast<ggml_backend_get_shared_weights_scratch_stats_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
            "ggml_backend_get_shared_weights_scratch_stats"));
    return fn && fn(backend, stats);
}
void ggml_shared_weights_set_query_id(int query_id) {
    repack_query_id.store(query_id, std::memory_order_relaxed);
}
void ggml_shared_weights_set_phase(int phase) {
    repack_phase.store(phase, std::memory_order_relaxed);
}
ggml_shared_weights_repack_event ggml_shared_weights_repack_begin(
        const ggml_tensor * op, const char * timing_kind, size_t packed_bytes, size_t scratch_request_bytes) {
    ggml_shared_weights_repack_event event = {};
    event.query_id = repack_query_id.load(std::memory_order_relaxed);
    event.phase = repack_phase.load(std::memory_order_relaxed);
    event.node_name = op ? op->name : "";
    event.tokens = op ? op->ne[1] : 0;
    event.packed_bytes = packed_bytes;
    event.scratch_request_bytes = scratch_request_bytes;
    event.repack_us = -1;
    event.timing_kind = timing_kind;
    if (op && op->src[0]) {
        ggml_mul_mat_src0_region_params r = {
            GGML_MUL_MAT_SRC0_REGION_PARAMS_VERSION, 0, 0, op->src[0]->ne[0], 0, op->src[0]->ne[1]};
        if (op->op == GGML_OP_MUL_MAT_SRC0_REGION) {
            GGML_ASSERT(ggml_mul_mat_src0_region_get_params(op, &r));
        }
        event.k_start = r.k_start; event.k_size = r.k_size;
        event.out_start = r.out_start; event.out_size = r.out_size;
    }
    return event;
}
void ggml_shared_weights_register_buft(ggml_backend_buffer_type_t buft) {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    s.types.insert(buft);
}
void ggml_shared_weights_unregister_buft(ggml_backend_buffer_type_t buft) {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    s.types.erase(buft);
}
bool ggml_shared_weights_is_buft(ggml_backend_buffer_type_t buft) {
    if (!ggml_shared_weights_enabled()) { return false; }
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    return s.types.count(buft) != 0;
}
void ggml_shared_weights_register_buffer(ggml_backend_buffer_t buffer,
        void * base, size_t size, size_t allocation_size, int fd) {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    GGML_ASSERT(size <= allocation_size && s.buffers.count(buffer) == 0);
    auto a = std::unique_ptr<allocation>(new allocation);
    a->desc = {base, size, allocation_size, fd, s.next_id++};
    s.buffers.emplace(buffer, std::move(a));
}
void ggml_shared_weights_unregister_buffer(ggml_backend_buffer_t buffer) {
    if (!ggml_shared_weights_enabled()) { return; }
    // Imported handles must be released before the physical backing memory.
    std::unique_ptr<allocation> retired;
    auto & s = state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.buffers.find(buffer);
        if (it != s.buffers.end()) { retired = std::move(it->second); s.buffers.erase(it); }
    }
}
const ggml_shared_weight_buffer * ggml_shared_weights_buffer(ggml_backend_buffer_t buffer) {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.buffers.find(buffer);
    return it == s.buffers.end() ? nullptr : &it->second->desc;
}
bool ggml_shared_weights_tensor(const ggml_tensor * tensor) {
    return tensor && tensor->buffer && ggml_shared_weights_is_buft(ggml_backend_buffer_get_type(tensor->buffer));
}
bool ggml_shared_weights_supports_op(const ggml_tensor * op) {
    if (!op || (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_SRC0_REGION) ||
            !op->src[0] || !op->src[1] || op->src[0]->type != GGML_TYPE_Q8_0 ||
            op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
            op->src[0]->view_src || op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1 ||
            op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1 ||
            !ggml_is_contiguous(op->src[1]) || !ggml_is_contiguous(op) ||
            op->src[0]->ne[0] % 32 != 0 ||
            op->src[0]->nb[1] != ggml_row_size(GGML_TYPE_Q8_0, op->src[0]->ne[0])) {
        return false;
    }
    if (op->op == GGML_OP_MUL_MAT) {
        return op->src[1]->ne[0] == op->src[0]->ne[0] && op->ne[0] == op->src[0]->ne[1];
    }
    ggml_mul_mat_src0_region_params r;
    return ggml_mul_mat_src0_region_get_params(op, &r) &&
        r.k_start >= 0 && r.k_size > 0 && r.k_start <= op->src[0]->ne[0] - r.k_size &&
        r.out_start >= 0 && r.out_size > 0 && r.out_start <= op->src[0]->ne[1] - r.out_size &&
        r.k_start % 32 == 0 && r.k_size % 32 == 0 &&
        op->src[1]->ne[0] == r.k_size && op->ne[0] == r.out_size;
}
void * ggml_shared_weights_attachment(ggml_backend_buffer_t buffer, const void * key) {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.buffers.find(buffer);
    if (it != s.buffers.end()) {
        for (const auto & a : it->second->imports) { if (a.key == key) { return a.handle; } }
    }
    return nullptr;
}
void ggml_shared_weights_attach(ggml_backend_buffer_t buffer, const void * key,
        void * handle, void (*release)(void *)) {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.buffers.find(buffer);
    GGML_ASSERT(it != s.buffers.end() && handle && release);
    for (const auto & a : it->second->imports) { GGML_ASSERT(a.key != key); }
    it->second->imports.push_back({key, handle, release});
}

static void shared_weights_log_row(const char * event, const ggml_tensor * tensor,
        const char * backend, int axis, int64_t start, int64_t size, const char * detail,
        const ggml_shared_weights_repack_event * timing) {
    const char * path = std::getenv("GGML_SHARED_WEIGHTS_CSV");
    if (!path || !*path) { return; }
    struct csv_writer {
        std::mutex mutex;
        std::string path;
        FILE * file = nullptr;
        unsigned pending = 0;
        bool legacy = false;
        char buffer[64 * 1024];
        ~csv_writer() { if (file) { std::fclose(file); } }
    };
    static csv_writer writer;
    std::lock_guard<std::mutex> lock(writer.mutex);
    static const char legacy_header[] =
        "event,tensor,backend,allocation_id,offset_bytes,tensor_bytes,allocation_bytes,axis,start,size,detail\n";
    static const char header[] =
        "event,tensor,backend,allocation_id,offset_bytes,tensor_bytes,allocation_bytes,axis,start,size,detail,"
        "query_id,phase,node_name,tokens,k_start,k_size,out_start,out_size,packed_bytes,scratch_request_bytes,repack_us,timing_kind\n";
    if (!writer.file || writer.path != path) {
        if (writer.file) {
            if (std::fclose(writer.file) != 0) { GGML_ABORT("cannot close shared weight CSV: %s", writer.path.c_str()); }
            writer.file = nullptr;
        }
        writer.path = path;
        writer.pending = 0;
        writer.legacy = false;
        writer.file = std::fopen(path, "a+");
        if (!writer.file) { GGML_ABORT("cannot open shared weight CSV: %s", path); }
        std::setvbuf(writer.file, writer.buffer, _IOFBF, sizeof(writer.buffer));
        std::rewind(writer.file);
        char first[sizeof(header) + 1];
        if (!std::fgets(first, sizeof(first), writer.file)) { std::fputs(header, writer.file); }
        else if (std::strcmp(first, legacy_header) == 0) {
            // Preserve existing experiment data and its rectangular schema.
            // New files have numeric timing columns; old files retain timings
            // as key/value text in detail without rewriting old rows.
            writer.legacy = true;
            GGML_LOG_WARN("shared weights: appending to legacy CSV; repack timings are in detail: %s\n", path);
        } else if (std::strcmp(first, header) != 0) {
            GGML_ABORT("shared weight CSV schema mismatch: %s", path);
        }
        std::fseek(writer.file, 0, SEEK_END);
    }
    FILE * f = writer.file;
    auto field = [&](const char * str) {
        std::fputc('"', f);
        for (const char * p = str ? str : ""; *p; ++p) {
            if (*p == '"') { std::fputc('"', f); } std::fputc(*p, f);
        }
        std::fputc('"', f);
    };
    const auto * d = tensor ? ggml_shared_weights_buffer(tensor->buffer) : nullptr;
    const size_t offset = d && tensor->data ? (const char *) tensor->data - (const char *) d->base : 0;
    field(event); std::fputc(',', f); field(tensor ? tensor->name : ""); std::fputc(',', f); field(backend);
    std::fprintf(f, ",%" PRIu64 ",%zu,%zu,%zu,%d,%" PRId64 ",%" PRId64 ",",
        d ? d->allocation_id : 0, offset, tensor ? ggml_nbytes(tensor) : 0,
        d ? d->allocation_size : 0, axis, start, size);
    const char * phase = !timing || timing->phase < 0 ? "unknown" :
        timing->phase == GGML_BACKEND_SCHED_PROFILE_PREFILL ? "prefill" : "decode";
    if (writer.legacy && timing) {
        char values[384];
        std::snprintf(values, sizeof(values),
            "query_id=%d phase=%s tokens=%" PRId64 " k_start=%" PRId64 " k_size=%" PRId64
            " out_start=%" PRId64 " out_size=%" PRId64 " packed_bytes=%zu scratch_request_bytes=%zu repack_us=%.3f",
            timing->query_id, phase, timing->tokens, timing->k_start, timing->k_size,
            timing->out_start, timing->out_size, timing->packed_bytes, timing->scratch_request_bytes, timing->repack_us);
        const std::string legacy_detail = std::string(detail ? detail : "") + "; " + values +
            " timing_kind=" + (timing->timing_kind ? timing->timing_kind : "") +
            " node=" + (timing->node_name ? timing->node_name : "");
        field(legacy_detail.c_str());
    } else { field(detail); }
    if (!writer.legacy) {
        if (timing) {
            std::fprintf(f, ",%d,", timing->query_id); field(phase); std::fputc(',', f); field(timing->node_name);
            std::fprintf(f, ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%zu,%zu,%.3f,",
                timing->tokens, timing->k_start, timing->k_size, timing->out_start, timing->out_size,
                timing->packed_bytes, timing->scratch_request_bytes, timing->repack_us);
            field(timing->timing_kind);
        } else { std::fputs(",,,,,,,,,,,,", f); }
    }
    std::fputc('\n', f);
    // Hot-path timing records are buffered just like graph bind records.
    if (++writer.pending >= 256 ||
            (std::strcmp(event ? event : "", "bind") != 0 && !timing)) {
        if (std::fflush(f) != 0) { GGML_ABORT("cannot flush shared weight CSV: %s", path); }
        writer.pending = 0;
    }
    if (std::ferror(f)) { GGML_ABORT("cannot write shared weight CSV: %s", path); }
}
void ggml_shared_weights_log(const char * event, const ggml_tensor * tensor,
        const char * backend, int axis, int64_t start, int64_t size, const char * detail) {
    shared_weights_log_row(event, tensor, backend, axis, start, size, detail, nullptr);
}
void ggml_shared_weights_log_repack(const ggml_tensor * weight, const char * backend,
        const ggml_shared_weights_repack_event * event) {
    if (!event || !ggml_shared_weights_timing_enabled()) { return; }
    const bool k_slice = weight && event->k_size != weight->ne[0];
    shared_weights_log_row("repack", weight, backend, k_slice ? 0 : 1,
        k_slice ? event->k_start : event->out_start,
        k_slice ? event->k_size : event->out_size, "native scratch", event);
}
