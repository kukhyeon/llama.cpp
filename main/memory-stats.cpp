#include "memory-stats.h"

#include "../src/llama-ext.h"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

constexpr size_t CSV_COLUMNS = 17;

struct os_memory_snapshot {
    std::optional<uint64_t> process_rss;
    std::optional<uint64_t> process_pss;
    std::optional<uint64_t> process_peak_rss;
    std::optional<uint64_t> system_available;
    std::optional<uint64_t> system_free;
};

struct processor_memory {
    size_t model = 0;
    size_t context = 0;
    size_t compute = 0;
    size_t partition_added = 0;
    size_t partition_replacement = 0;
    size_t graph_base = 0;
    size_t graph_routed = 0;
    bool graph_measurement_valid = false;
};

enum class processor_kind : size_t {
    cpu = 0,
    gpu = 1,
    npu = 2,
    other = 3,
};

std::unordered_map<std::string, uint64_t> read_proc_kib_values(const char * path) {
    std::unordered_map<std::string, uint64_t> values;
    std::ifstream input(path);
    std::string line;

    while (std::getline(input, line)) {
        std::istringstream stream(line);
        std::string key;
        uint64_t value = 0;
        std::string unit;
        if (!(stream >> key >> value)) {
            continue;
        }
        if (!key.empty() && key.back() == ':') {
            key.pop_back();
        }
        stream >> unit;
        if (unit == "kB") {
            value *= 1024;
        }
        values[key] = value;
    }

    return values;
}

std::optional<uint64_t> find_value(
        const std::unordered_map<std::string, uint64_t> & values,
        const char * key) {
    const auto it = values.find(key);
    return it == values.end() ? std::nullopt : std::optional<uint64_t>(it->second);
}

os_memory_snapshot read_os_memory() {
    os_memory_snapshot result;

#if defined(__linux__) || defined(__ANDROID__)
    const auto meminfo = read_proc_kib_values("/proc/meminfo");
    const auto status = read_proc_kib_values("/proc/self/status");
    const auto smaps = read_proc_kib_values("/proc/self/smaps_rollup");

    result.process_rss = find_value(status, "VmRSS");
    result.process_peak_rss = find_value(status, "VmHWM");
    result.process_pss = find_value(smaps, "Pss");
    result.system_available = find_value(meminfo, "MemAvailable");
    result.system_free = find_value(meminfo, "MemFree");
#endif

    return result;
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

processor_kind classify_buffer(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
    std::string identity = ggml_backend_buft_name(buft);
    if (device != nullptr) {
        identity += " ";
        identity += ggml_backend_dev_name(device);
        identity += " ";
        identity += ggml_backend_dev_description(device);
    }
    identity = lower_copy(std::move(identity));

    // The Hexagon backend currently reports DEVICE_TYPE_GPU, so identify it
    // before consulting the generic device type.
    if (identity.find("htp") != std::string::npos ||
            identity.find("hexagon") != std::string::npos ||
            identity.find("npu") != std::string::npos) {
        return processor_kind::npu;
    }

    if (device != nullptr) {
        switch (ggml_backend_dev_type(device)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                return processor_kind::cpu;
            case GGML_BACKEND_DEVICE_TYPE_GPU:
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                return processor_kind::gpu;
            default:
                break;
        }
    }
    return ggml_backend_buft_is_host(buft)
        ? processor_kind::cpu
        : processor_kind::other;
}

std::string optional_field(const std::optional<uint64_t> & value) {
    return value.has_value() ? std::to_string(*value) : std::string();
}

std::string signed_difference(size_t lhs, size_t rhs) {
    return lhs >= rhs
        ? std::to_string(lhs - rhs)
        : "-" + std::to_string(rhs - lhs);
}

void write_csv_row(std::ofstream & output, const std::array<std::string, CSV_COLUMNS> & row) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (i > 0) {
            output << ',';
        }
        output << row[i];
    }
    output << '\n';
}

} // namespace

memory_stats_writer::memory_stats_writer(bool enabled, std::string path) :
    enabled_(enabled),
    path_(std::move(path)),
    start_(std::chrono::steady_clock::now()) {
    if (!enabled_) {
        return;
    }

    output_.open(path_, std::ios::out | std::ios::trunc);
    if (output_) {
        output_ << "stage,elapsed_ms,processor,backend_tracked_bytes,weight_bytes,context_bytes,"
                   "compute_buffer_reserved_bytes,partition_added_alloc_bytes,partition_replacement_alloc_bytes,"
                   "partition_net_delta_payload_bytes,compute_buffer_base_prefill_bytes,"
                   "compute_buffer_candidate_extra_bytes,"
                   "process_rss_bytes,process_pss_bytes,"
                   "process_peak_rss_bytes,system_mem_available_bytes,system_mem_free_bytes\n";
        output_.flush();
    }
}

bool memory_stats_writer::enabled() const {
    return enabled_;
}

bool memory_stats_writer::ready() const {
    return !enabled_ || output_.good();
}

const std::string & memory_stats_writer::path() const {
    return path_;
}

void memory_stats_writer::record(const char * stage, const llama_context * ctx) {
    if (!enabled_ || !output_ || stage == nullptr) {
        return;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_).count();
    const std::string elapsed_field = std::to_string(elapsed);
    const os_memory_snapshot os = read_os_memory();

    std::array<processor_memory, 4> processors = {};
    std::string partition_net_extra_payload;
    if (ctx != nullptr) {
        const llama_memory_breakdown breakdown = llama_get_memory_breakdown(ctx);
        for (const auto & [buft, memory] : breakdown) {
            auto & dst = processors[static_cast<size_t>(classify_buffer(buft))];
            dst.model += memory.model;
            dst.context += memory.context;
            dst.compute += memory.compute;
            if (memory.compute_graph_measurement_valid) {
                dst.graph_base += memory.compute_graph_base;
                dst.graph_routed += memory.compute_graph_routed;
                dst.graph_measurement_valid = true;
            }
        }

        const llama_model * model = llama_get_model(ctx);
        const llama_partition_weight_breakdown partition =
            llama_get_partition_weight_breakdown(model);
        size_t added_payload = 0;
        size_t replacement_payload = 0;
        bool payload_overflow = false;
        for (const auto & [buft, memory] : partition.buffers) {
            auto & dst = processors[static_cast<size_t>(classify_buffer(buft))];
            dst.partition_added += memory.added_alloc_span;
            dst.partition_replacement += memory.replacement_alloc_span;
            if (memory.added_payload > std::numeric_limits<size_t>::max() - added_payload ||
                    memory.replacement_payload >
                        std::numeric_limits<size_t>::max() - replacement_payload) {
                payload_overflow = true;
            } else {
                added_payload += memory.added_payload;
                replacement_payload += memory.replacement_payload;
            }
        }
        if (partition.valid && !partition.overflow && !payload_overflow &&
                replacement_payload <= std::numeric_limits<size_t>::max() - added_payload) {
            const size_t materialized_payload = added_payload + replacement_payload;
            if (materialized_payload >= partition.omitted_canonical_payload) {
                partition_net_extra_payload = std::to_string(
                        materialized_payload - partition.omitted_canonical_payload);
            } else {
                partition_net_extra_payload = "-" + std::to_string(
                        partition.omitted_canonical_payload - materialized_payload);
            }
        }
    }

    write_csv_row(output_, {
        stage, elapsed_field, "SYSTEM",
        "", "", "", "", "", "",
        partition_net_extra_payload,
        "", "",
        optional_field(os.process_rss),
        optional_field(os.process_pss),
        optional_field(os.process_peak_rss),
        optional_field(os.system_available),
        optional_field(os.system_free),
    });

    if (ctx != nullptr) {
        constexpr std::array<const char *, 4> names = { "CPU", "GPU", "NPU", "OTHER" };
        for (size_t i = 0; i < processors.size(); ++i) {
            const processor_memory & memory = processors[i];
            const size_t owned = memory.model + memory.context + memory.compute;
            if (i == static_cast<size_t>(processor_kind::other) && owned == 0 &&
                    memory.partition_added == 0 && memory.partition_replacement == 0) {
                continue;
            }
            const std::string graph_base = memory.graph_measurement_valid
                ? std::to_string(memory.graph_base)
                : std::string();
            const std::string graph_extra = memory.graph_measurement_valid
                ? signed_difference(memory.graph_routed, memory.graph_base)
                : std::string();
            write_csv_row(output_, {
                stage, elapsed_field, names[i],
                std::to_string(owned),
                std::to_string(memory.model),
                std::to_string(memory.context),
                std::to_string(memory.compute),
                std::to_string(memory.partition_added),
                std::to_string(memory.partition_replacement),
                "",
                graph_base,
                graph_extra,
                "", "", "", "", "",
            });
        }
    }

    output_.flush();
}
