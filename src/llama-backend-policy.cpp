#include "llama-backend-policy.h"

#include "llama-impl.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace {

using json = nlohmann::ordered_json;

// A single selector/action entry from weights.rules or ops.rules.
// Not every field is meaningful for both sections:
// - weights use tensor/pattern/layer selectors and buffer-type backends.
// - ops use name/name_regex/op/phase/layer selectors and execution backends.
struct policy_rule {
    std::string tensor;
    std::string pattern;
    std::string name;
    std::string name_regex;
    std::string op;
    std::string phase;
    int layer = -2;
    int layer_start = -2;
    int layer_end = -2;
    std::vector<std::string> backends;
    std::string source;
};

struct token_range {
    bool configured = false;
    int32_t min = 0;
    int32_t max = 0;
};

struct ffn_clock_axis {
    int index = -1;
    int64_t frequency = 0;
};

struct ffn_clock_point {
    bool configured = false;
    ffn_clock_axis prime;
    ffn_clock_axis gold;
    ffn_clock_axis gpu;
};

struct ffn_parallel_policy {
    bool configured = false;
    bool enabled = false;
    std::string phase;
    int layer_start = -2;
    int layer_end = -2;
    int64_t align = 256;
    std::string reduce_backend;
    int reduce_threads = 0;
    std::vector<llama_backend_policy_ffn_split> splits;
    std::string source;
};

struct profile_policy {
    bool weights_enabled = false;
    bool ops_enabled = false;
    std::vector<std::string> default_weight_backends;
    std::vector<policy_rule> weight_rules;
    std::vector<policy_rule> op_rules;
    ffn_parallel_policy ffn_parallel;
    bool attn_qkv_shards_configured = false;
    llama_backend_policy_attn_qkv_shards attn_qkv_shards;
    bool attn_out_shards_configured = false;
    llama_backend_policy_attn_out_shards attn_out_shards;
    ffn_clock_point clock_point;
    token_range input_tokens;
    token_range ubatch_tokens;
};

// Process-global policy state. The policy is loaded once before model loading
// and then queried from the model loader and graph callback paths.
struct policy_state {
    bool loaded = false;
    bool enabled = false;
    bool weights_enabled = false;
    bool ops_enabled = false;
    std::string path;
    std::vector<std::string> fallback_priority;
    std::vector<std::string> default_weight_backends;
    std::vector<policy_rule> weight_rules;
    std::vector<policy_rule> op_rules;

    bool residency_enabled = false;
    std::vector<policy_rule> residency_rules;

    ffn_parallel_policy ffn_parallel;
    llama_backend_policy_attn_qkv_shards attn_qkv_shards;
    llama_backend_policy_attn_out_shards attn_out_shards;

    bool ffn_clock_switch_enabled = false;
    std::string pending_clock_profile;

    bool stage_switch_enabled = false;
    bool thermal_switch_enabled = false;
    std::string default_profile;
    std::string prefill_profile;
    std::string decode_profile;
    std::string cool_profile;
    std::string hot_profile;
    std::string critical_profile;
    std::string thermal_state_file;
    std::string active_profile;
    uint64_t profile_generation = 0;
    std::map<std::string, profile_policy> profiles;

    llama_backend_policy_runtime_routes runtime_routes;
};

std::mutex g_policy_mutex;
policy_state g_policy;

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

std::string trim(std::string s) {
    auto not_space = [](unsigned char c) {
        return !std::isspace(c);
    };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

bool str_eq_ci(const std::string & a, const std::string & b) {
    return lower(a) == lower(b);
}

bool str_contains_ci(const std::string & haystack, const std::string & needle) {
    return lower(haystack).find(lower(needle)) != std::string::npos;
}

std::vector<std::string> parse_backend_list(const json & obj) {
    std::vector<std::string> result;

    if (obj.contains("backend") && obj["backend"].is_string()) {
        result.push_back(obj["backend"].get<std::string>());
    }

    if (obj.contains("backends") && obj["backends"].is_array()) {
        for (const auto & item : obj["backends"]) {
            if (!item.is_string()) {
                throw std::runtime_error("backend list entries must be strings");
            }
            result.push_back(item.get<std::string>());
        }
    }

    if (obj.contains("copies") && obj["copies"].is_array()) {
        for (const auto & item : obj["copies"]) {
            if (!item.is_string()) {
                throw std::runtime_error("copies entries must be strings");
            }
            result.push_back(item.get<std::string>());
        }
    }

    return result;
}

bool env_enabled(const char * name, bool fallback) {
    const char * value = getenv(name);
    if (!value) {
        return fallback;
    }
    const std::string s = lower(value);
    return s == "1" || s == "true" || s == "on" || s == "yes";
}

std::string env_string(const char * name) {
    const char * value = getenv(name);
    return value ? value : "";
}

std::string read_text_file_first_line(const std::string & path) {
    if (path.empty()) {
        return {};
    }

    std::ifstream file(path);
    if (!file) {
        LLAMA_LOG_DEBUG("backend_policy: failed to read thermal state file '%s'\n", path.c_str());
        return {};
    }

    std::string line;
    std::getline(file, line);
    return trim(line);
}

std::vector<std::string> parse_string_array(const json & obj, const char * key) {
    std::vector<std::string> result;
    if (!obj.contains(key)) {
        return result;
    }
    if (!obj[key].is_array()) {
        throw std::runtime_error(std::string(key) + " must be an array");
    }
    for (const auto & item : obj[key]) {
        if (!item.is_string()) {
            throw std::runtime_error(std::string(key) + " entries must be strings");
        }
        result.push_back(item.get<std::string>());
    }
    return result;
}

void append_unique(std::vector<std::string> & dst, const std::vector<std::string> & src) {
    std::unordered_set<std::string> seen;
    for (const auto & s : dst) {
        seen.insert(lower(s));
    }
    for (const auto & s : src) {
        if (seen.insert(lower(s)).second) {
            dst.push_back(s);
        }
    }
}

std::vector<std::string> with_fallbacks(const std::vector<std::string> & primary, const std::vector<std::string> & fallback) {
    std::vector<std::string> result = primary;
    append_unique(result, fallback);
    return result;
}

// Weight names use "blk.N.*"; graph nodes use callback names with a "-N"
// suffix. Supporting both lets the same layer/layer_range selectors work in
// weights.rules and ops.rules.
int layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }

    int il = -1;
    if (std::sscanf(name, "blk.%d.", &il) == 1) {
        return il;
    }

    const char * dash = std::strrchr(name, '-');
    if (dash != nullptr && dash[1] != '\0') {
        char * end = nullptr;
        const long value = std::strtol(dash + 1, &end, 10);
        if (end != dash + 1 && *end == '\0') {
            return (int) value;
        }
    }

    return -1;
}

bool layer_matches(const policy_rule & rule, int il) {
    if (rule.layer != -2 && il != rule.layer) {
        return false;
    }
    if (rule.layer_start != -2) {
        if (il < rule.layer_start) {
            return false;
        }
        if (rule.layer_end != -2 && il > rule.layer_end) {
            return false;
        }
    }
    return true;
}

bool regex_search_safe(const std::string & value, const std::string & pattern) {
    try {
        return std::regex_search(value, std::regex(pattern));
    } catch (const std::regex_error & e) {
        LLAMA_LOG_WARN("backend_policy: invalid regex '%s': %s\n", pattern.c_str(), e.what());
        return false;
    }
}

bool weight_rule_matches(const policy_rule & rule, const std::string & tensor_name) {
    if (!layer_matches(rule, layer_from_name(tensor_name.c_str()))) {
        return false;
    }

    bool has_selector = false;
    if (!rule.tensor.empty()) {
        has_selector = true;
        if (tensor_name != rule.tensor) {
            return false;
        }
    }

    if (!rule.pattern.empty()) {
        has_selector = true;
        if (!regex_search_safe(tensor_name, rule.pattern)) {
            return false;
        }
    }

    return has_selector || rule.layer != -2 || rule.layer_start != -2;
}

bool phase_matches(const std::string & phase, bool is_prefill) {
    if (phase.empty() || str_eq_ci(phase, "all")) {
        return true;
    }
    if (str_eq_ci(phase, "prefill")) {
        return is_prefill;
    }
    if (str_eq_ci(phase, "decode")) {
        return !is_prefill;
    }
    return false;
}

bool op_rule_matches(
        const policy_rule & rule,
        const std::string & base_name,
        const std::string & tensor_name,
        enum ggml_op op,
        int il,
        bool is_prefill) {
    // The policy intentionally checks both the stable callback name and the
    // final tensor name. This allows broad rules like name="Qcur" and exact
    // layer-aware rules like name_regex="^Qcur-12$".
    if (!phase_matches(rule.phase, is_prefill)) {
        return false;
    }
    if (!layer_matches(rule, il)) {
        return false;
    }

    bool has_selector = false;
    if (!rule.name.empty()) {
        has_selector = true;
        if (base_name != rule.name && tensor_name != rule.name) {
            return false;
        }
    }

    if (!rule.name_regex.empty()) {
        has_selector = true;
        if (!regex_search_safe(base_name, rule.name_regex) && !regex_search_safe(tensor_name, rule.name_regex)) {
            return false;
        }
    }

    if (!rule.op.empty()) {
        has_selector = true;
        const std::string op_name = ggml_op_name(op);
        if (!str_eq_ci(rule.op, op_name) && !str_eq_ci(rule.op, std::string("GGML_OP_") + op_name)) {
            return false;
        }
    }

    return has_selector || rule.layer != -2 || rule.layer_start != -2 || !rule.phase.empty();
}

// Parse one rule while preserving its JSON array location. That source string
// is emitted in debug logs so a log line can be traced back to config quickly.
policy_rule parse_rule(const json & obj, const std::string & source) {
    if (!obj.is_object()) {
        throw std::runtime_error(source + " entries must be objects");
    }

    policy_rule rule;
    rule.source = source;
    if (obj.contains("tensor")) {
        rule.tensor = obj["tensor"].get<std::string>();
    }
    if (obj.contains("pattern")) {
        rule.pattern = obj["pattern"].get<std::string>();
    }
    if (obj.contains("name")) {
        rule.name = obj["name"].get<std::string>();
    }
    if (obj.contains("name_regex")) {
        rule.name_regex = obj["name_regex"].get<std::string>();
    }
    if (obj.contains("op")) {
        rule.op = obj["op"].get<std::string>();
    }
    if (obj.contains("phase")) {
        rule.phase = obj["phase"].get<std::string>();
    }
    if (obj.contains("layer")) {
        rule.layer = obj["layer"].get<int>();
    }
    if (obj.contains("layer_range")) {
        if (!obj["layer_range"].is_array() || obj["layer_range"].size() != 2) {
            throw std::runtime_error(source + ".layer_range must be [start, end]");
        }
        rule.layer_start = obj["layer_range"][0].get<int>();
        rule.layer_end = obj["layer_range"][1].get<int>();
    }

    rule.backends = parse_backend_list(obj);
    append_unique(rule.backends, parse_string_array(obj, "fallback_priority"));
    return rule;
}

void parse_rules(const json & section, const char * section_name, std::vector<policy_rule> & out) {
    if (!section.contains("rules")) {
        return;
    }
    if (!section["rules"].is_array()) {
        throw std::runtime_error(std::string(section_name) + ".rules must be an array");
    }

    int i = 0;
    for (const auto & item : section["rules"]) {
        out.push_back(parse_rule(item, std::string(section_name) + ".rules[" + std::to_string(i) + "]"));
        ++i;
    }
}

void parse_layer_range(const json & obj, const std::string & source, int & layer_start, int & layer_end) {
    if (!obj.contains("layer_range")) {
        return;
    }
    if (!obj["layer_range"].is_array() || obj["layer_range"].size() != 2) {
        throw std::runtime_error(source + ".layer_range must be [start, end]");
    }
    layer_start = obj["layer_range"][0].get<int>();
    layer_end = obj["layer_range"][1].get<int>();
    if (layer_start < 0 || layer_end < layer_start) {
        throw std::runtime_error(source + ".layer_range must contain non-negative [start, end] with start <= end");
    }
}

token_range parse_token_range(const json & root, const char * key, const std::string & source, bool required) {
    token_range out;
    if (!root.contains(key)) {
        if (required) {
            throw std::runtime_error(source + "." + key + " is required");
        }
        return out;
    }

    const auto & value = root[key];
    if (!value.is_array() || value.size() != 2 ||
        !value[0].is_number_integer() || !value[1].is_number_integer()) {
        throw std::runtime_error(source + "." + key + " must be [min, max]");
    }

    const int64_t min = value[0].get<int64_t>();
    const int64_t max = value[1].get<int64_t>();
    if (min <= 0 || max < min || max > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(source + "." + key + " must contain positive token counts with min <= max");
    }

    out.configured = true;
    out.min = (int32_t) min;
    out.max = (int32_t) max;
    return out;
}

ffn_clock_axis parse_clock_axis(
        const json & root,
        const char * name,
        const char * frequency_key,
        const std::string & source) {
    if (!root.contains(name) || !root[name].is_object()) {
        throw std::runtime_error(source + "." + name + " must be an object");
    }

    const auto & axis = root[name];
    if (!axis.contains("index") || !axis["index"].is_number_integer()) {
        throw std::runtime_error(source + "." + name + ".index must be a non-negative integer");
    }
    if (!axis.contains(frequency_key) || !axis[frequency_key].is_number_integer()) {
        throw std::runtime_error(source + "." + name + "." + frequency_key + " must be a positive integer");
    }

    const int64_t index = axis["index"].get<int64_t>();
    const int64_t frequency = axis[frequency_key].get<int64_t>();
    if (index < 0 || index > std::numeric_limits<int>::max()) {
        throw std::runtime_error(source + "." + name + ".index must be a non-negative integer");
    }
    if (frequency <= 0) {
        throw std::runtime_error(source + "." + name + "." + frequency_key + " must be a positive integer");
    }

    ffn_clock_axis out;
    out.index = (int) index;
    out.frequency = frequency;
    return out;
}

ffn_clock_point parse_clock_point(const json & root, const std::string & source) {
    if (!root.is_object()) {
        throw std::runtime_error(source + " must be an object");
    }

    ffn_clock_point out;
    out.configured = true;
    out.prime = parse_clock_axis(root, "prime", "khz", source);
    out.gold = parse_clock_axis(root, "gold", "khz", source);
    out.gpu = parse_clock_axis(root, "gpu", "hz", source);
    return out;
}

ffn_parallel_policy parse_ffn_parallel(const json & root, const std::string & source) {
    ffn_parallel_policy out;
    if (!root.is_object()) {
        throw std::runtime_error(source + " must be an object");
    }

    out.configured = true;
    out.enabled = root.value("enabled", false);
    out.phase = root.value("phase", std::string("prefill"));
    out.align = root.value("align", (int64_t) 256);
    if (!str_eq_ci(out.phase, "all") && !str_eq_ci(out.phase, "prefill") && !str_eq_ci(out.phase, "decode")) {
        throw std::runtime_error(source + ".phase must be all, prefill, or decode");
    }
    if (out.align <= 0) {
        throw std::runtime_error(source + ".align must be a positive integer");
    }
    out.reduce_backend = root.value("reduce_backend", std::string());
    if (root.contains("reduce_threads")) {
        if (!root["reduce_threads"].is_number_integer()) {
            throw std::runtime_error(source + ".reduce_threads must be a non-negative integer");
        }
        const int64_t reduce_threads = root["reduce_threads"].get<int64_t>();
        if (reduce_threads < 0 || reduce_threads > std::numeric_limits<int>::max()) {
            throw std::runtime_error(source + ".reduce_threads must be a non-negative integer");
        }
        out.reduce_threads = (int) reduce_threads;
    }
    out.source = source;
    parse_layer_range(root, source, out.layer_start, out.layer_end);

    int i = 0;
    std::unordered_set<std::string> split_ids;
    if (root.contains("splits")) {
        if (!root["splits"].is_array()) {
            throw std::runtime_error(source + ".splits must be an array");
        }
        if (root.contains("split_layout") || root.contains("split_sizes")) {
            throw std::runtime_error(
                    source + " must use either splits or split_layout/split_sizes, not both");
        }
        for (const auto & item : root["splits"]) {
            if (!item.is_object()) {
                throw std::runtime_error(source + ".splits entries must be objects");
            }
            llama_backend_policy_ffn_split split;
            split.id = item.value("id", std::string("s") + std::to_string(i));
            split.start = item.value("start", (int64_t) -1);
            split.size = item.value("size", (int64_t) -1);
            split.backend = item.value("backend", std::string());
            if (split.start < 0 || split.size < 0 || split.backend.empty()) {
                throw std::runtime_error(source + ".splits[" + std::to_string(i) + "] requires backend, start, and non-negative size");
            }
            if (split.start > std::numeric_limits<int64_t>::max() - split.size) {
                throw std::runtime_error(source + ".splits[" + std::to_string(i) + "] range overflows int64");
            }
            if (split.id.empty() || !split_ids.insert(split.id).second) {
                throw std::runtime_error(source + ".splits ids must be non-empty and unique");
            }
            out.splits.push_back(std::move(split));
            ++i;
        }
    } else {
        // Compact form for a family of clock profiles. split_layout carries
        // the stable backend/id order once (normally through profile_defaults)
        // while each profile supplies only its split_sizes object.
        if (!root.contains("split_layout") || !root["split_layout"].is_array() ||
                !root.contains("split_sizes") || !root["split_sizes"].is_object()) {
            throw std::runtime_error(
                    source + " requires splits or split_layout plus split_sizes");
        }

        int64_t start = 0;
        for (const auto & item : root["split_layout"]) {
            if (!item.is_object()) {
                throw std::runtime_error(source + ".split_layout entries must be objects");
            }
            llama_backend_policy_ffn_split split;
            split.id = item.value("id", std::string());
            split.backend = item.value("backend", std::string());
            if (split.id.empty() || split.backend.empty() || !split_ids.insert(split.id).second) {
                throw std::runtime_error(
                        source + ".split_layout requires unique non-empty id/backend entries");
            }
            if (!root["split_sizes"].contains(split.id) ||
                    !root["split_sizes"][split.id].is_number_integer()) {
                throw std::runtime_error(
                        source + ".split_sizes." + split.id + " must be a non-negative integer");
            }
            split.size = root["split_sizes"][split.id].get<int64_t>();
            split.start = start;
            if (split.size < 0 || start > std::numeric_limits<int64_t>::max() - split.size) {
                throw std::runtime_error(
                        source + ".split_sizes." + split.id + " must be a non-negative non-overflowing integer");
            }
            start += split.size;
            out.splits.push_back(std::move(split));
            ++i;
        }

        if ((int) root["split_sizes"].size() != i) {
            throw std::runtime_error(
                    source + ".split_sizes contains an id not declared by split_layout");
        }
    }

    if (out.splits.empty()) {
        throw std::runtime_error(source + ".splits must not be empty");
    }

    // The graph requires one contiguous, aligned partition starting at zero.
    // Reject gaps/overlaps during policy loading so an invalid cover-only
    // policy cannot allocate weights successfully and abort much later.
    auto sorted_splits = out.splits;
    std::sort(sorted_splits.begin(), sorted_splits.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.start != rhs.start) {
            return lhs.start < rhs.start;
        }
        // A disabled lane and the following active lane share one boundary.
        // Visit the zero-width interval first so both observe the same covered
        // prefix without making explicit split order significant.
        if ((lhs.size == 0) != (rhs.size == 0)) {
            return lhs.size == 0;
        }
        return lhs.id < rhs.id;
    });
    int64_t covered = 0;
    size_t active_splits = 0;
    for (const auto & split : sorted_splits) {
        if (split.start != covered) {
            throw std::runtime_error(source + ".splits must form a contiguous partition starting at 0");
        }
        if ((split.start % out.align) != 0 || (split.size % out.align) != 0) {
            throw std::runtime_error(
                    source + ".splits start and size must be multiples of align=" +
                    std::to_string(out.align));
        }
        covered += split.size;
        active_splits += split.size > 0 ? 1 : 0;
    }
    if (active_splits == 0) {
        throw std::runtime_error(source + ".splits must contain at least one positive processor lane");
    }

    return out;
}

bool parse_env_ffn_splits(std::vector<llama_backend_policy_ffn_split> & splits) {
    const std::string spec = trim(env_string("LLAMA_FFN_PARALLEL_SPLITS"));
    if (spec.empty()) {
        return false;
    }

    splits.clear();
    int64_t start = 0;
    std::stringstream ss(spec);
    std::string item;
    int idx = 0;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            continue;
        }

        const size_t colon = item.find(':');
        if (colon == std::string::npos) {
            LLAMA_LOG_WARN("backend_policy: invalid LLAMA_FFN_PARALLEL_SPLITS item '%s'; expected backend:size\n", item.c_str());
            splits.clear();
            return false;
        }

        llama_backend_policy_ffn_split split;
        split.id = std::string("env") + std::to_string(idx);
        split.backend = trim(item.substr(0, colon));
        const std::string size_str = trim(item.substr(colon + 1));
        char * end = nullptr;
        const long long size = std::strtoll(size_str.c_str(), &end, 10);
        if (split.backend.empty() || end == size_str.c_str() || *end != '\0' || size < 0 ||
                start > std::numeric_limits<int64_t>::max() - size) {
            LLAMA_LOG_WARN("backend_policy: invalid LLAMA_FFN_PARALLEL_SPLITS item '%s'; expected backend:non_negative_size\n", item.c_str());
            splits.clear();
            return false;
        }

        split.start = start;
        split.size = size;
        start += split.size;
        splits.push_back(std::move(split));
        ++idx;
    }

    const size_t active_splits = std::count_if(
            splits.begin(), splits.end(), [](const auto & split) { return split.size > 0; });
    if (active_splits == 0) {
        LLAMA_LOG_WARN(
                "backend_policy: LLAMA_FFN_PARALLEL_SPLITS requires at least one positive size\n");
        splits.clear();
        return false;
    }

    return true;
}

llama_backend_policy_ffn_parallel materialize_ffn_policy(const ffn_parallel_policy & policy) {
    llama_backend_policy_ffn_parallel out;
    out.enabled = policy.enabled;
    out.phase = policy.phase;
    out.layer_start = policy.layer_start;
    out.layer_end = policy.layer_end;
    out.align = policy.align;
    out.reduce_backend = policy.reduce_backend;
    out.reduce_threads = policy.reduce_threads;
    out.splits = policy.splits;
    out.source = policy.source;

    if (const std::string align_env = trim(env_string("LLAMA_FFN_PARALLEL_ALIGN")); !align_env.empty()) {
        char * end = nullptr;
        const long long align = std::strtoll(align_env.c_str(), &end, 10);
        if (end != align_env.c_str() && *end == '\0' && align > 0) {
            out.align = align;
        }
    }

    if (const std::string reduce_env = trim(env_string("LLAMA_FFN_PARALLEL_REDUCE_BACKEND")); !reduce_env.empty()) {
        out.reduce_backend = reduce_env;
    }

    std::vector<llama_backend_policy_ffn_split> env_splits;
    if (parse_env_ffn_splits(env_splits)) {
        out.splits = std::move(env_splits);
        out.source = "LLAMA_FFN_PARALLEL_SPLITS";
    }

    return out;
}

bool attn_qkv_shard_id_is_safe(const std::string & id) {
    return !id.empty() && std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
}

bool attn_qkv_shard_backend_needs_htp_alignment(const std::string & backend) {
    const std::string normalized = lower(backend);
    return normalized.find("htp") != std::string::npos ||
           normalized.find("hexagon") != std::string::npos;
}

llama_backend_policy_attn_qkv_shards parse_attn_qkv_shards(
        const json & root,
        const std::string & source) {
    if (!root.is_object()) {
        throw std::runtime_error(source + " must be an object");
    }

    llama_backend_policy_attn_qkv_shards out;
    out.enabled = root.value("enabled", false);
    out.phase = root.value("phase", std::string("prefill"));
    out.assemble_backend = root.value("assemble_backend", std::string("CPU"));
    out.source = source;

    // Preserve the existing schema even though the graph builder currently
    // attempts Q/K/V sharding only for prefill. A decode-only policy remains
    // graph-inert; "all" contributes its prefill half to cover planning.
    if (!str_eq_ci(out.phase, "all") && !str_eq_ci(out.phase, "prefill") &&
            !str_eq_ci(out.phase, "decode")) {
        throw std::runtime_error(source + ".phase must be all, prefill, or decode");
    }
    out.phase = lower(out.phase);
    if (out.assemble_backend.empty()) {
        throw std::runtime_error(source + ".assemble_backend must be a non-empty string");
    }
    if (root.contains("head_dim")) {
        if (!root["head_dim"].is_number_integer()) {
            throw std::runtime_error(source + ".head_dim must be a positive integer");
        }
        out.head_dim = root["head_dim"].get<int64_t>();
    }
    if (out.head_dim <= 0) {
        throw std::runtime_error(source + ".head_dim must be a positive integer");
    }
    parse_layer_range(root, source, out.layer_start, out.layer_end);

    if (!root.contains("split_layout") || !root["split_layout"].is_array() ||
            root["split_layout"].size() != 3) {
        throw std::runtime_error(source + ".split_layout must contain exactly 3 entries");
    }
    if (!root.contains("split_sizes") || !root["split_sizes"].is_object()) {
        throw std::runtime_error(source + ".split_sizes must be an object");
    }

    struct layout_entry {
        std::string id;
        std::string backend;
        int64_t lane_align = 1;
    };
    std::vector<layout_entry> layout;
    layout.reserve(3);
    std::unordered_set<std::string> split_ids;
    std::unordered_set<std::string> split_backends;

    for (size_t i = 0; i < root["split_layout"].size(); ++i) {
        const auto & item = root["split_layout"][i];
        const std::string item_source =
            source + ".split_layout[" + std::to_string(i) + "]";
        if (!item.is_object()) {
            throw std::runtime_error(item_source + " must be an object");
        }

        layout_entry entry;
        entry.id = item.value("id", std::string());
        entry.backend = item.value("backend", std::string());
        if (!attn_qkv_shard_id_is_safe(entry.id)) {
            throw std::runtime_error(
                    item_source + ".id must contain only ASCII letters, digits, '_' or '-'");
        }
        if (entry.backend.empty()) {
            throw std::runtime_error(item_source + ".backend must be a non-empty string");
        }
        if (!split_ids.insert(lower(entry.id)).second) {
            throw std::runtime_error(source + ".split_layout ids must be unique ignoring case");
        }
        if (!split_backends.insert(lower(entry.backend)).second) {
            throw std::runtime_error(source + ".split_layout backends must be unique ignoring case");
        }

        if (item.contains("align")) {
            if (!item["align"].is_number_integer()) {
                throw std::runtime_error(item_source + ".align must be a positive integer");
            }
            entry.lane_align = item["align"].get<int64_t>();
            if (entry.lane_align <= 0) {
                throw std::runtime_error(item_source + ".align must be a positive integer");
            }
        }
        if (attn_qkv_shard_backend_needs_htp_alignment(entry.backend)) {
            entry.lane_align = std::max<int64_t>(entry.lane_align, 256);
        }

        llama_backend_policy_attn_qkv_shard split;
        split.id = entry.id;
        split.backend = entry.backend;
        out.splits.push_back(std::move(split));
        layout.push_back(std::move(entry));
    }

    const auto & split_sizes = root["split_sizes"];
    if (split_sizes.size() != 3 || !split_sizes.contains("q") ||
            !split_sizes.contains("k") || !split_sizes.contains("v")) {
        throw std::runtime_error(source + ".split_sizes must contain exactly q, k, and v");
    }

    auto populate_projection = [&](const char * projection, auto start_member, auto size_member) {
        const auto & sizes = split_sizes[projection];
        const std::string projection_source = source + ".split_sizes." + projection;
        if (!sizes.is_object() || sizes.size() != layout.size()) {
            throw std::runtime_error(
                    projection_source + " must contain exactly the split_layout ids");
        }

        int64_t start = 0;
        for (size_t i = 0; i < layout.size(); ++i) {
            const auto & entry = layout[i];
            if (!sizes.contains(entry.id) || !sizes[entry.id].is_number_integer()) {
                throw std::runtime_error(
                        projection_source + "." + entry.id + " must be a non-negative integer");
            }
            const int64_t size = sizes[entry.id].get<int64_t>();
            if (size < 0 || start > std::numeric_limits<int64_t>::max() - size) {
                throw std::runtime_error(
                        projection_source + "." + entry.id +
                        " must be a non-negative non-overflowing integer");
            }
            if (size > 0 && ((start % out.head_dim) != 0 || (size % out.head_dim) != 0 ||
                    (start % entry.lane_align) != 0 || (size % entry.lane_align) != 0)) {
                throw std::runtime_error(
                        projection_source + "." + entry.id +
                        " start/size must be divisible by head_dim=" +
                        std::to_string(out.head_dim) + " and lane align=" +
                        std::to_string(entry.lane_align));
            }

            out.splits[i].*start_member = start;
            out.splits[i].*size_member = size;
            start += size;
        }

        const size_t active_splits = std::count_if(
                out.splits.begin(), out.splits.end(), [&](const auto & split) {
                    return split.*size_member > 0;
                });
        if (active_splits < 1 || active_splits > 3) {
            throw std::runtime_error(
                    projection_source + " must contain 1 to 3 positive processor lanes");
        }
    };

    populate_projection(
            "q", &llama_backend_policy_attn_qkv_shard::q_start,
            &llama_backend_policy_attn_qkv_shard::q_size);
    populate_projection(
            "k", &llama_backend_policy_attn_qkv_shard::k_start,
            &llama_backend_policy_attn_qkv_shard::k_size);
    populate_projection(
            "v", &llama_backend_policy_attn_qkv_shard::v_start,
            &llama_backend_policy_attn_qkv_shard::v_size);

    // A QKV processor lane is one composite scheduler branch. Supporting a
    // projection-local zero would leave that branch structurally different
    // across Q/K/V and make lane exclusion ambiguous. A lane is therefore
    // either active for all three projections or inactive for all three.
    for (const auto & split : out.splits) {
        const int active_projections =
            (split.q_size > 0 ? 1 : 0) +
            (split.k_size > 0 ? 1 : 0) +
            (split.v_size > 0 ? 1 : 0);
        if (active_projections != 0 && active_projections != 3) {
            throw std::runtime_error(
                    source + ".split_sizes lane '" + split.id +
                    "' must be zero or positive for Q, K, and V together");
        }
    }

    return out;
}

llama_backend_policy_attn_out_shards parse_attn_out_shards(
        const json & root,
        const std::string & source) {
    if (!root.is_object()) {
        throw std::runtime_error(source + " must be an object");
    }

    llama_backend_policy_attn_out_shards out;
    out.enabled = root.value("enabled", false);
    out.phase = root.value("phase", std::string("prefill"));
    out.partition_axis = root.value("partition_axis", std::string("input"));
    out.reduce_backend = root.value("reduce_backend", std::string("CPU"));
    out.source = source;

    // The first version deliberately leaves decode and stateful attention on
    // the ordinary path. Accepting "all" here would also make cover-only
    // residency tempting even though that fallback is still required.
    if (!str_eq_ci(out.phase, "prefill")) {
        throw std::runtime_error(source + ".phase currently supports only prefill");
    }
    out.phase = "prefill";
    if (!str_eq_ci(out.partition_axis, "input") &&
            !str_eq_ci(out.partition_axis, "output")) {
        throw std::runtime_error(source + ".partition_axis must be input or output");
    }
    out.partition_axis = lower(out.partition_axis);
    if (out.reduce_backend.empty()) {
        throw std::runtime_error(source + ".reduce_backend must be a non-empty string");
    }
    if (root.contains("head_dim")) {
        if (!root["head_dim"].is_number_integer()) {
            throw std::runtime_error(source + ".head_dim must be a positive integer");
        }
        out.head_dim = root["head_dim"].get<int64_t>();
    }
    if (out.head_dim <= 0) {
        throw std::runtime_error(source + ".head_dim must be a positive integer");
    }
    parse_layer_range(root, source, out.layer_start, out.layer_end);

    if (!root.contains("split_layout") || !root["split_layout"].is_array() ||
            root["split_layout"].size() != 3) {
        throw std::runtime_error(source + ".split_layout must contain exactly 3 entries");
    }
    if (!root.contains("split_sizes") || !root["split_sizes"].is_object() ||
            root["split_sizes"].size() != 3) {
        throw std::runtime_error(
                source + ".split_sizes must contain exactly the 3 split_layout ids");
    }

    struct layout_entry {
        std::string id;
        std::string backend;
        int64_t align = 1;
    };
    std::vector<layout_entry> layout;
    layout.reserve(3);
    std::unordered_set<std::string> split_ids;
    std::unordered_set<std::string> split_backends;

    // ggml tensor names are limited to GGML_MAX_NAME (64 including NUL).
    // Reserve enough room for the longest emitted branch name:
    // "attn_out_shard." + id + ".input-" + a 10-digit layer id.
    static constexpr size_t max_layer_digits = std::numeric_limits<int>::digits10 + 1;
    static constexpr size_t branch_fixed_length =
        sizeof("attn_out_shard.") - 1 + sizeof(".input-") - 1 + max_layer_digits;
    static_assert(GGML_MAX_NAME - 1 > branch_fixed_length);
    static constexpr size_t max_split_id_length =
        GGML_MAX_NAME - 1 - branch_fixed_length;

    for (size_t i = 0; i < root["split_layout"].size(); ++i) {
        const auto & item = root["split_layout"][i];
        const std::string item_source =
            source + ".split_layout[" + std::to_string(i) + "]";
        if (!item.is_object()) {
            throw std::runtime_error(item_source + " must be an object");
        }

        layout_entry entry;
        entry.id = item.value("id", std::string());
        entry.backend = item.value("backend", std::string());
        if (!attn_qkv_shard_id_is_safe(entry.id) ||
                entry.id.size() > max_split_id_length) {
            throw std::runtime_error(
                    item_source +
                    ".id must be 1.." + std::to_string(max_split_id_length) +
                    " ASCII letters, digits, '_' or '-' so graph names are not truncated");
        }
        if (entry.backend.empty()) {
            throw std::runtime_error(item_source + ".backend must be a non-empty string");
        }
        if (!split_ids.insert(lower(entry.id)).second) {
            throw std::runtime_error(source + ".split_layout ids must be unique ignoring case");
        }
        if (!split_backends.insert(lower(entry.backend)).second) {
            throw std::runtime_error(source + ".split_layout backends must be unique ignoring case");
        }

        if (item.contains("align")) {
            if (!item["align"].is_number_integer()) {
                throw std::runtime_error(item_source + ".align must be a positive integer");
            }
            entry.align = item["align"].get<int64_t>();
            if (entry.align <= 0) {
                throw std::runtime_error(item_source + ".align must be a positive integer");
            }
        }
        if (attn_qkv_shard_backend_needs_htp_alignment(entry.backend)) {
            entry.align = std::max<int64_t>(entry.align, 256);
        }
        layout.push_back(std::move(entry));
    }

    const auto & sizes = root["split_sizes"];
    int64_t start = 0;
    for (const auto & entry : layout) {
        if (!sizes.contains(entry.id) || !sizes[entry.id].is_number_integer()) {
            throw std::runtime_error(
                source + ".split_sizes." + entry.id + " must be a non-negative integer");
        }
        const int64_t size = sizes[entry.id].get<int64_t>();
        if (size < 0 || start > std::numeric_limits<int64_t>::max() - size) {
            throw std::runtime_error(
                    source + ".split_sizes." + entry.id +
                    " must be a non-negative non-overflowing integer");
        }
        if (size > 0 && ((start % out.head_dim) != 0 || (size % out.head_dim) != 0 ||
                (start % entry.align) != 0 || (size % entry.align) != 0)) {
            throw std::runtime_error(
                    source + ".split_sizes." + entry.id +
                    " start/size must be divisible by head_dim=" +
                    std::to_string(out.head_dim) + " and lane align=" +
                    std::to_string(entry.align));
        }

        llama_backend_policy_attn_out_shard split;
        split.id = entry.id;
        split.backend = entry.backend;
        split.start = start;
        split.size = size;
        out.splits.push_back(std::move(split));
        start += size;
    }

    const size_t active_splits = std::count_if(
            out.splits.begin(), out.splits.end(), [](const auto & split) {
                return split.size > 0;
            });
    if (active_splits < 1 || active_splits > 3) {
        throw std::runtime_error(
                source + ".split_sizes must contain 1, 2, or 3 positive processor lanes");
    }

    return out;
}

bool effective_ffn_policy(llama_backend_policy_ffn_parallel & out) {
    ffn_parallel_policy policy = g_policy.ffn_parallel;
    std::string selected_profile = g_policy.active_profile;
    if (selected_profile.empty() && g_policy.runtime_routes.enabled &&
            std::find(
                g_policy.runtime_routes.candidate_kinds.begin(),
                g_policy.runtime_routes.candidate_kinds.end(),
                "ffn_block") != g_policy.runtime_routes.candidate_kinds.end()) {
        selected_profile = g_policy.runtime_routes.initial_profile;
    }
    if (!selected_profile.empty()) {
        const auto it = g_policy.profiles.find(selected_profile);
        if (it != g_policy.profiles.end() && it->second.ffn_parallel.configured) {
            policy = it->second.ffn_parallel;
        }
    }

    std::vector<llama_backend_policy_ffn_split> env_splits;
    const bool has_env_splits = parse_env_ffn_splits(env_splits);
    if (!policy.configured && has_env_splits) {
        policy.configured = true;
        policy.enabled = true;
        policy.phase = "prefill";
        policy.align = 256;
        policy.source = "LLAMA_FFN_PARALLEL_SPLITS";
        policy.splits = env_splits;
    }

    if (!policy.configured) {
        out = {};
        return false;
    }

    policy.enabled = env_enabled("LLAMA_FFN_PARALLEL", policy.enabled);
    out = materialize_ffn_policy(policy);
    out.enabled = policy.enabled;

    if (!out.enabled || out.splits.empty()) {
        out = {};
        return false;
    }

    return true;
}

void parse_profile_section(
        const json & root,
        const json & policy_root,
        const std::string & profile_name,
        bool policy_enabled,
        bool enable_weights,
        bool enable_ops,
        profile_policy & out) {
    if (root.contains("weights") && root["weights"].is_object()) {
        const auto & weights = root["weights"];
        out.weights_enabled = policy_enabled && enable_weights && weights.value("enabled", true);
        if (weights.contains("default")) {
            if (weights["default"].is_string()) {
                out.default_weight_backends.push_back(weights["default"].get<std::string>());
            } else if (weights["default"].is_array()) {
                for (const auto & item : weights["default"]) {
                    out.default_weight_backends.push_back(item.get<std::string>());
                }
            } else {
                throw std::runtime_error("profiles." + profile_name + ".weights.default must be a string or an array");
            }
        }
        parse_rules(weights, ("profiles." + profile_name + ".weights").c_str(), out.weight_rules);
    }

    if (root.contains("ops") && root["ops"].is_object()) {
        const auto & ops = root["ops"];
        out.ops_enabled = policy_enabled && enable_ops && ops.value("enabled", true);
        parse_rules(ops, ("profiles." + profile_name + ".ops").c_str(), out.op_rules);
    }

    const std::string profile_source = "profiles." + profile_name;
    if (root.contains("ffn_parallel")) {
        out.ffn_parallel = parse_ffn_parallel(root["ffn_parallel"], profile_source + ".ffn_parallel");
        out.ffn_parallel.enabled = policy_enabled && out.ffn_parallel.enabled;
    }

    if (root.contains("attn_qkv_shards")) {
        const auto & override = root["attn_qkv_shards"];
        const std::string source = profile_source + ".attn_qkv_shards";
        if (!override.is_object()) {
            throw std::runtime_error(source + " must be an object");
        }
        if (override.size() != 1 || !override.contains("split_sizes")) {
            throw std::runtime_error(
                    source + " may override only split_sizes; topology is inherited from root attn_qkv_shards");
        }
        if (!policy_root.contains("attn_qkv_shards") ||
                !policy_root["attn_qkv_shards"].is_object()) {
            throw std::runtime_error(source + " requires a root attn_qkv_shards policy");
        }

        const auto & sizes = override["split_sizes"];
        const auto & root_sizes = policy_root["attn_qkv_shards"]["split_sizes"];
        if (!sizes.is_object() || sizes.size() != 3 ||
                !sizes.contains("q") || !sizes.contains("k") || !sizes.contains("v") ||
                !root_sizes.is_object() || root_sizes.size() != 3 ||
                !root_sizes.contains("q") || !root_sizes.contains("k") || !root_sizes.contains("v")) {
            throw std::runtime_error(
                    source + ".split_sizes must contain exactly q, k, and v");
        }
        for (const char * projection : { "q", "k", "v" }) {
            const auto & projection_sizes = sizes[projection];
            const auto & root_projection_sizes = root_sizes[projection];
            if (!projection_sizes.is_object() || !root_projection_sizes.is_object() ||
                    projection_sizes.size() != root_projection_sizes.size()) {
                throw std::runtime_error(
                        source + ".split_sizes." + projection +
                        " must contain every root split id");
            }
            for (auto it = root_projection_sizes.begin(); it != root_projection_sizes.end(); ++it) {
                if (!projection_sizes.contains(it.key()) ||
                        !projection_sizes[it.key()].is_number_integer()) {
                    throw std::runtime_error(
                            source + ".split_sizes." + projection + "." + it.key() +
                            " must be an integer");
                }
            }
        }

        json effective = policy_root["attn_qkv_shards"];
        effective.merge_patch(override);
        out.attn_qkv_shards = parse_attn_qkv_shards(effective, source);
        out.attn_qkv_shards.enabled = policy_enabled && out.attn_qkv_shards.enabled;
        out.attn_qkv_shards_configured = true;
    }

    if (root.contains("attn_out_shards")) {
        const auto & override = root["attn_out_shards"];
        const std::string source = profile_source + ".attn_out_shards";
        if (!override.is_object()) {
            throw std::runtime_error(source + " must be an object");
        }
        if (override.size() != 1 || !override.contains("split_sizes")) {
            throw std::runtime_error(
                    source + " may override only split_sizes; topology is inherited from root attn_out_shards");
        }
        if (!policy_root.contains("attn_out_shards") ||
                !policy_root["attn_out_shards"].is_object()) {
            throw std::runtime_error(
                    source + " requires a root attn_out_shards policy");
        }

        const auto & sizes = override["split_sizes"];
        const auto & root_sizes = policy_root["attn_out_shards"]["split_sizes"];
        if (!sizes.is_object() || !root_sizes.is_object() ||
                sizes.size() != root_sizes.size()) {
            throw std::runtime_error(
                    source + ".split_sizes must contain every root split id");
        }
        for (auto it = root_sizes.begin(); it != root_sizes.end(); ++it) {
            if (!sizes.contains(it.key()) || !sizes[it.key()].is_number_integer()) {
                throw std::runtime_error(
                        source + ".split_sizes." + it.key() + " must be an integer");
            }
        }

        json effective = policy_root["attn_out_shards"];
        effective.merge_patch(override);
        out.attn_out_shards = parse_attn_out_shards(effective, source);
        out.attn_out_shards.enabled = policy_enabled && out.attn_out_shards.enabled;
        out.attn_out_shards_configured = true;
    }

    if (root.contains("applicability")) {
        if (!root["applicability"].is_object()) {
            throw std::runtime_error(profile_source + ".applicability must be an object");
        }
        const auto & applicability = root["applicability"];
        out.input_tokens = parse_token_range(
                applicability, "input_tokens", profile_source + ".applicability", true);
        out.ubatch_tokens = parse_token_range(
                applicability, "ubatch_tokens", profile_source + ".applicability", false);
    }

    if (root.contains("clock_point")) {
        out.clock_point = parse_clock_point(root["clock_point"], profile_source + ".clock_point");
        if (!out.input_tokens.configured) {
            throw std::runtime_error(profile_source + ".clock_point requires applicability.input_tokens");
        }
    }
}

bool match_weight_rules(
        const std::vector<std::string> & default_backends,
        const std::vector<policy_rule> & rules,
        const std::vector<std::string> & fallback_priority,
        const std::string & tensor_name,
        const char * default_source,
        llama_backend_policy_match & out) {
    out = {};

    if (!default_backends.empty()) {
        out.matched = true;
        out.backends = default_backends;
        out.source = default_source;
    }

    // Later matching rules override earlier ones. This mirrors override-tensor
    // style use cases where a broad default can be followed by precise
    // exceptions for a tensor family or layer range.
    for (const auto & rule : rules) {
        if (weight_rule_matches(rule, tensor_name)) {
            out.matched = true;
            out.backends = rule.backends;
            out.source = rule.source;
        }
    }

    if (!out.matched || out.backends.empty()) {
        out = {};
        return false;
    }

    out.backends = with_fallbacks(out.backends, fallback_priority);
    return true;
}

bool match_op_rules(
        const std::vector<policy_rule> & rules,
        const std::vector<std::string> & fallback_priority,
        const std::string & base,
        const std::string & name,
        enum ggml_op op,
        int il,
        bool is_prefill,
        llama_backend_policy_match & out) {
    out = {};

    // Later matching rules override earlier ones, so configs can combine broad
    // op rules with narrower name/layer exceptions.
    for (const auto & rule : rules) {
        if (op_rule_matches(rule, base, name, op, il, is_prefill)) {
            out.matched = true;
            out.backends = rule.backends;
            out.source = rule.source;
        }
    }

    if (!out.matched || out.backends.empty()) {
        out = {};
        return false;
    }

    out.backends = with_fallbacks(out.backends, fallback_priority);
    return true;
}

bool name_matches_backend_type(ggml_backend_dev_t dev, const std::string & backend_name) {
    if (dev == nullptr) {
        return false;
    }

    const auto type = ggml_backend_dev_type(dev);
    if (str_eq_ci(backend_name, "cpu") && type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true;
    }
    if (str_eq_ci(backend_name, "gpu") && type == GGML_BACKEND_DEVICE_TYPE_GPU) {
        return true;
    }
    if (str_eq_ci(backend_name, "accel") && type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        return true;
    }
    if (str_eq_ci(backend_name, "htp") && str_contains_ci(ggml_backend_dev_name(dev), "htp")) {
        return true;
    }

    return false;
}

bool extra_buft_matches(ggml_backend_dev_t dev, const std::string & backend_name) {
    if (dev == nullptr) {
        return false;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return false;
    }

    auto get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts == nullptr) {
        return false;
    }

    ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts(dev);
    while (extra_bufts && *extra_bufts) {
        if (str_eq_ci(backend_name, ggml_backend_buft_name(*extra_bufts))) {
            return true;
        }
        ++extra_bufts;
    }

    return false;
}

bool token_range_contains(const token_range & range, int32_t value) {
    return !range.configured || (value >= range.min && value <= range.max);
}

bool token_ranges_overlap(const token_range & lhs, const token_range & rhs) {
    if (!lhs.configured || !rhs.configured) {
        return true;
    }
    return lhs.min <= rhs.max && rhs.min <= lhs.max;
}

bool same_clock_frequencies(const ffn_clock_point & lhs, const ffn_clock_point & rhs) {
    return lhs.prime.frequency == rhs.prime.frequency &&
           lhs.gold.frequency == rhs.gold.frequency &&
           lhs.gpu.frequency == rhs.gpu.frequency;
}

struct clock_profile_selection {
    bool matched = false;
    std::string profile;
    double distance = -1.0;
};

clock_profile_selection select_clock_profile_locked(
        const std::vector<std::string> * candidates,
        bool require_enabled_ffn,
        int32_t input_tokens,
        int32_t ubatch_tokens,
        int64_t gold_khz,
        int64_t prime_khz,
        int64_t gpu_hz) {
    clock_profile_selection result;
    if (input_tokens <= 0 || ubatch_tokens <= 0 ||
        gold_khz <= 0 || prime_khz <= 0 || gpu_hz <= 0) {
        return result;
    }

    const profile_policy * best = nullptr;
    std::string best_name;
    long double best_distance = std::numeric_limits<long double>::infinity();
    long double best_total_khz = std::numeric_limits<long double>::infinity();
    constexpr long double tie_epsilon = 1.0e-15L;

    const auto consider = [&](const std::string & profile_name, const profile_policy & profile) {
        if (!profile.clock_point.configured ||
            (require_enabled_ffn &&
                (!profile.ffn_parallel.configured ||
                 !env_enabled("LLAMA_FFN_PARALLEL", profile.ffn_parallel.enabled))) ||
            !token_range_contains(profile.input_tokens, input_tokens) ||
            !token_range_contains(profile.ubatch_tokens, ubatch_tokens)) {
            return;
        }

        const auto relative_error = [](int64_t actual, int64_t target) {
            return std::fabs((long double) actual - (long double) target) /
                (long double) target;
        };
        const long double distance =
            relative_error(prime_khz, profile.clock_point.prime.frequency) +
            relative_error(gold_khz, profile.clock_point.gold.frequency) +
            relative_error(gpu_hz, profile.clock_point.gpu.frequency);
        const long double total_khz =
            (long double) profile.clock_point.prime.frequency +
            (long double) profile.clock_point.gold.frequency +
            (long double) profile.clock_point.gpu.frequency / 1000.0L;

        const bool distance_better = distance + tie_epsilon < best_distance;
        const bool distance_tied = std::fabs(distance - best_distance) <= tie_epsilon;
        const bool lower_frequency = total_khz + tie_epsilon < best_total_khz;
        const bool fully_tied = distance_tied &&
            std::fabs(total_khz - best_total_khz) <= tie_epsilon;
        if (best == nullptr || distance_better ||
            (distance_tied && lower_frequency) ||
            (fully_tied && profile_name < best_name)) {
            best = &profile;
            best_name = profile_name;
            best_distance = distance;
            best_total_khz = total_khz;
        }
    };

    if (candidates == nullptr) {
        for (const auto & kv : g_policy.profiles) {
            consider(kv.first, kv.second);
        }
    } else {
        for (const auto & profile_name : *candidates) {
            const auto it = g_policy.profiles.find(profile_name);
            if (it != g_policy.profiles.end()) {
                consider(it->first, it->second);
            }
        }
    }

    if (best != nullptr) {
        result.matched = true;
        result.profile = std::move(best_name);
        result.distance = (double) best_distance;
    }
    return result;
}

bool ffn_policy_matches_layer(const llama_backend_policy_ffn_parallel & policy, int il) {
    if (policy.layer_start == -2) {
        return true;
    }
    return il >= policy.layer_start && (policy.layer_end == -2 || il <= policy.layer_end);
}

bool same_ffn_residency_requirements(
        const llama_backend_policy_ffn_parallel & lhs,
        const llama_backend_policy_ffn_parallel & rhs) {
    // Phase also carries whether the full source tensor must be retained for
    // fallback. Keeping an "all" representative for a duplicate "prefill"
    // policy could otherwise make the loader discard weights needed by decode.
    if (!str_eq_ci(lhs.phase, rhs.phase)) {
        return false;
    }
    if (lhs.splits.size() != rhs.splits.size()) {
        return false;
    }

    auto lhs_splits = lhs.splits;
    auto rhs_splits = rhs.splits;
    const auto by_range = [](const llama_backend_policy_ffn_split & a, const llama_backend_policy_ffn_split & b) {
        if (a.start != b.start) {
            return a.start < b.start;
        }
        if (a.size != b.size) {
            return a.size < b.size;
        }
        return lower(a.backend) < lower(b.backend);
    };
    std::sort(lhs_splits.begin(), lhs_splits.end(), by_range);
    std::sort(rhs_splits.begin(), rhs_splits.end(), by_range);

    for (size_t i = 0; i < lhs_splits.size(); ++i) {
        if (lhs_splits[i].start != rhs_splits[i].start ||
            lhs_splits[i].size != rhs_splits[i].size ||
            !str_eq_ci(lhs_splits[i].backend, rhs_splits[i].backend)) {
            return false;
        }
    }
    return true;
}

void validate_profile_reference(
        const policy_state & state,
        const std::string & profile,
        const std::string & source) {
    if (!profile.empty() && state.profiles.find(profile) == state.profiles.end()) {
        throw std::runtime_error(source + " references unknown profile '" + profile + "'");
    }
}

const llama_backend_policy_attn_qkv_shards * effective_attn_qkv_policy_for_profile(
        const policy_state & state,
        const std::string & profile_name) {
    const auto it = state.profiles.find(profile_name);
    if (it == state.profiles.end()) {
        return nullptr;
    }
    return it->second.attn_qkv_shards_configured
        ? &it->second.attn_qkv_shards
        : &state.attn_qkv_shards;
}

bool attn_qkv_policy_matches_layer(
        const llama_backend_policy_attn_qkv_shards & policy,
        int il) {
    return policy.layer_start == -2 ||
        (il >= policy.layer_start &&
         (policy.layer_end == -2 || il <= policy.layer_end));
}

bool attn_qkv_policy_width(
        const llama_backend_policy_attn_qkv_shards & policy,
        llama_backend_policy_attn_qkv_projection projection,
        int64_t & width) {
    width = 0;
    if (policy.splits.size() != 3) {
        return false;
    }

    for (const auto & split : policy.splits) {
        int64_t start = 0;
        int64_t size = 0;
        switch (projection) {
            case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q:
                start = split.q_start;
                size = split.q_size;
                break;
            case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_K:
                start = split.k_start;
                size = split.k_size;
                break;
            case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_V:
                start = split.v_start;
                size = split.v_size;
                break;
            default:
                return false;
        }
        if (start != width || size < 0 ||
                width > std::numeric_limits<int64_t>::max() - size) {
            return false;
        }
        width += size;
    }
    return true;
}

const llama_backend_policy_attn_out_shards * effective_attn_out_policy_for_profile(
        const policy_state & state,
        const std::string & profile_name) {
    const auto it = state.profiles.find(profile_name);
    if (it == state.profiles.end()) {
        return nullptr;
    }
    return it->second.attn_out_shards_configured
        ? &it->second.attn_out_shards
        : &state.attn_out_shards;
}

bool attn_out_policy_matches_layer(
        const llama_backend_policy_attn_out_shards & policy,
        int il) {
    return policy.layer_start == -2 ||
        (il >= policy.layer_start &&
         (policy.layer_end == -2 || il <= policy.layer_end));
}

bool attn_out_policy_width(
        const llama_backend_policy_attn_out_shards & policy,
        int64_t & width) {
    width = 0;
    if (policy.splits.size() != 3) {
        return false;
    }
    for (const auto & split : policy.splits) {
        if (split.start != width || split.size < 0 ||
                width > std::numeric_limits<int64_t>::max() - split.size) {
            return false;
        }
        width += split.size;
    }
    return true;
}

const llama_backend_policy_runtime_route_transition * find_runtime_route_transition(
        const llama_backend_policy_runtime_routes & routes,
        const std::string & from_profile) {
    for (const auto & transition : routes.transitions) {
        if (transition.from_profile == from_profile) {
            return &transition;
        }
    }
    return nullptr;
}

void parse_runtime_routes(const json & root, policy_state & state) {
    if (!root.contains("runtime_routes")) {
        return;
    }
    if (!root["runtime_routes"].is_object()) {
        throw std::runtime_error("runtime_routes must be an object");
    }

    const auto & section = root["runtime_routes"];
    const bool requested_enabled = section.value("enabled", false);
    if (!requested_enabled) {
        // Disabled catalogs are deliberately inert. This keeps a staged or
        // partially authored section from affecting the legacy path.
        return;
    }

    llama_backend_policy_runtime_routes routes;
    routes.enabled = state.enabled;
    routes.mode = lower(trim(section.value("mode", std::string("fixed"))));
    if (routes.mode != "off" && routes.mode != "prepare" && routes.mode != "fixed" && routes.mode != "clock") {
        throw std::runtime_error("runtime_routes.mode must be off, prepare, fixed, or clock");
    }
    if (!routes.enabled || routes.mode == "off") {
        routes.enabled = false;
        routes.mode = "off";
        state.runtime_routes = std::move(routes);
        return;
    }

    routes.phase = lower(trim(section.value("phase", std::string("prefill"))));
    routes.initial_profile = section.value("initial_profile", std::string());
    routes.profiles = parse_string_array(section, "profiles");
    routes.candidate_kinds = parse_string_array(section, "candidate_kinds");
    routes.output_mode = lower(trim(section.value("output_mode", std::string("canonical"))));
    routes.strict = section.value("strict", false);

    if (!str_eq_ci(routes.phase, "all") &&
        !str_eq_ci(routes.phase, "prefill") &&
        !str_eq_ci(routes.phase, "decode")) {
        throw std::runtime_error("runtime_routes.phase must be all, prefill, or decode");
    }
    if (routes.mode == "clock" && routes.phase != "prefill") {
        throw std::runtime_error(
                "runtime_routes.mode=clock currently reuses the prefill FFN clock snapshot and requires phase=prefill");
    }
    if (routes.initial_profile.empty()) {
        throw std::runtime_error("runtime_routes.initial_profile must be a non-empty profile name");
    }
    if (routes.profiles.empty()) {
        throw std::runtime_error("runtime_routes.profiles must be a non-empty array");
    }

    std::unordered_set<std::string> allowed_profiles;
    for (const auto & profile : routes.profiles) {
        if (profile.empty()) {
            throw std::runtime_error("runtime_routes.profiles entries must be non-empty");
        }
        if (!allowed_profiles.insert(profile).second) {
            throw std::runtime_error("runtime_routes.profiles contains duplicate profile '" + profile + "'");
        }
        validate_profile_reference(state, profile, "runtime_routes.profiles");
        if (routes.mode == "clock") {
            const auto & policy = state.profiles.at(profile);
            if (!policy.clock_point.configured || !policy.input_tokens.configured) {
                throw std::runtime_error(
                        "runtime_routes clock profile '" + profile +
                        "' requires clock_point and applicability.input_tokens");
            }
        }
    }
    if (allowed_profiles.find(routes.initial_profile) == allowed_profiles.end()) {
        throw std::runtime_error(
                "runtime_routes.initial_profile '" + routes.initial_profile +
                "' is not in runtime_routes.profiles");
    }

    if (routes.candidate_kinds.empty()) {
        routes.candidate_kinds.push_back("weightless_stateless");
    }
    std::unordered_set<std::string> candidate_kinds;
    for (std::string & kind : routes.candidate_kinds) {
        kind = lower(trim(kind));
        if (kind != "weightless_stateless" && kind != "weighted_norm" &&
                kind != "ffn_block" && kind != "attn_qkv_block" &&
                kind != "attn_out_block") {
            throw std::runtime_error(
                    "runtime_routes.candidate_kinds supports only weightless_stateless, weighted_norm, ffn_block, attn_qkv_block, and attn_out_block");
        }
        if (!candidate_kinds.insert(kind).second) {
            throw std::runtime_error("runtime_routes.candidate_kinds contains duplicate '" + kind + "'");
        }
    }

    if (section.contains("boundary")) {
        if (!section["boundary"].is_object()) {
            throw std::runtime_error("runtime_routes.boundary must be an object");
        }
        const auto & boundary = section["boundary"];
        routes.boundary_node = trim(boundary.value("node", routes.boundary_node));
        routes.boundary_backend = trim(boundary.value("backend", routes.boundary_backend));
        routes.boundary_granularity = lower(trim(
                boundary.value("granularity", routes.boundary_granularity)));
    }
    // Accept a top-level granularity as a convenience for early policy files,
    // but reject conflicting declarations rather than silently choosing one.
    if (section.contains("granularity")) {
        const std::string granularity = lower(trim(section["granularity"].get<std::string>()));
        if (section.contains("boundary") &&
            section["boundary"].contains("granularity") &&
            granularity != routes.boundary_granularity) {
            throw std::runtime_error(
                    "runtime_routes.granularity conflicts with runtime_routes.boundary.granularity");
        }
        routes.boundary_granularity = granularity;
    }
    if (routes.boundary_node.empty()) {
        throw std::runtime_error("runtime_routes.boundary.node must be non-empty");
    }
    if (routes.mode == "clock" && routes.boundary_node != "l_out") {
        throw std::runtime_error(
                "runtime_routes.mode=clock requires boundary.node=l_out so a plan "
                "observed after layer N can only affect layer N+1");
    }
    if (routes.boundary_backend.empty()) {
        throw std::runtime_error("runtime_routes.boundary.backend must be non-empty");
    }
    if (str_eq_ci(routes.boundary_backend, "auto")) {
        routes.boundary_backend = "auto";
    }
    if (routes.boundary_granularity != "layer") {
        throw std::runtime_error("runtime_routes boundary granularity currently supports only layer");
    }
    if (routes.output_mode != "canonical") {
        throw std::runtime_error("runtime_routes.output_mode currently supports only canonical");
    }

    if (section.contains("min_dwell_layers")) {
        if (!section["min_dwell_layers"].is_number_integer()) {
            throw std::runtime_error("runtime_routes.min_dwell_layers must be a non-negative integer");
        }
        const int64_t min_dwell_layers = section["min_dwell_layers"].get<int64_t>();
        if (min_dwell_layers < 0 || min_dwell_layers > std::numeric_limits<int>::max()) {
            throw std::runtime_error("runtime_routes.min_dwell_layers must be a non-negative integer");
        }
        routes.min_dwell_layers = (int) min_dwell_layers;
    }

    if (!section.contains("transitions")) {
        throw std::runtime_error("runtime_routes.transitions is required");
    }
    if (section["transitions"].is_string()) {
        const std::string shorthand = lower(trim(section["transitions"].get<std::string>()));
        if (shorthand != "complete" && shorthand != "all") {
            throw std::runtime_error(
                    "runtime_routes.transitions string must be complete or all");
        }
        for (const std::string & from_profile : routes.profiles) {
            llama_backend_policy_runtime_route_transition transition;
            transition.from_profile = from_profile;
            for (const std::string & destination : routes.profiles) {
                if (destination != from_profile) {
                    transition.to_profiles.push_back(destination);
                }
            }
            routes.transitions.push_back(std::move(transition));
        }
    } else if (section["transitions"].is_object()) {
        for (auto it = section["transitions"].begin(); it != section["transitions"].end(); ++it) {
            const std::string from_profile = it.key();
            if (allowed_profiles.find(from_profile) == allowed_profiles.end()) {
                throw std::runtime_error(
                        "runtime_routes.transitions references source profile '" +
                        from_profile + "' outside runtime_routes.profiles");
            }
            if (!it.value().is_array()) {
                throw std::runtime_error(
                        "runtime_routes.transitions." + from_profile + " must be an array");
            }

            llama_backend_policy_runtime_route_transition transition;
            transition.from_profile = from_profile;
            std::unordered_set<std::string> destinations;
            for (const auto & item : it.value()) {
                if (!item.is_string()) {
                    throw std::runtime_error(
                            "runtime_routes.transitions." + from_profile + " entries must be strings");
                }
                const std::string destination = item.get<std::string>();
                if (allowed_profiles.find(destination) == allowed_profiles.end()) {
                    throw std::runtime_error(
                            "runtime_routes.transitions." + from_profile + " references profile '" +
                            destination + "' outside runtime_routes.profiles");
                }
                if (!destinations.insert(destination).second) {
                    throw std::runtime_error(
                            "runtime_routes.transitions." + from_profile + " contains duplicate profile '" +
                            destination + "'");
                }
                transition.to_profiles.push_back(destination);
            }
            routes.transitions.push_back(std::move(transition));
        }
    } else {
        throw std::runtime_error(
                "runtime_routes.transitions must be an object or complete/all string");
    }

    auto require_profile_op = [&](const std::string & kind,
                                  const char * semantic_name,
                                  const char * op_name) {
        for (const std::string & profile_name : routes.profiles) {
            const auto & profile = state.profiles.at(profile_name);
            const bool found = profile.ops_enabled &&
                std::any_of(profile.op_rules.begin(), profile.op_rules.end(),
                        [&](const policy_rule & rule) {
                            return rule.name == semantic_name &&
                                str_eq_ci(rule.op, op_name) &&
                                (rule.phase.empty() || str_eq_ci(rule.phase, "all") ||
                                 str_eq_ci(rule.phase, "prefill")) &&
                                !rule.backends.empty();
                        });
            if (!found) {
                throw std::runtime_error(
                        "runtime_routes " + kind + " profile '" + profile_name +
                        "' requires an enabled exact " + semantic_name +
                        " " + op_name + " prefill op rule");
            }
        }
    };

    if (candidate_kinds.find("weighted_norm") != candidate_kinds.end()) {
        require_profile_op("weighted_norm", "attn_rms_norm", "RMS_NORM");
        require_profile_op("weighted_norm", "attn_norm", "MUL");
    }

    if (candidate_kinds.find("ffn_block") != candidate_kinds.end()) {
        if (state.ffn_clock_switch_enabled) {
            throw std::runtime_error(
                    "runtime_routes ffn_block cannot be combined with legacy ffn_clock_switch; "
                    "the layer plan must be the only FFN clock authority");
        }
        int expected_reduce_threads = -1;
        for (const std::string & profile : routes.profiles) {
            const auto & ffn = state.profiles.at(profile).ffn_parallel;
            if (!ffn.configured || !ffn.enabled || ffn.splits.empty()) {
                throw std::runtime_error(
                        "runtime_routes ffn_block profile '" + profile +
                        "' requires an enabled ffn_parallel section");
            }
            if (expected_reduce_threads < 0) {
                expected_reduce_threads = ffn.reduce_threads;
            } else if (ffn.reduce_threads != expected_reduce_threads) {
                throw std::runtime_error(
                        "runtime_routes ffn_block profiles must use the same reduce_threads");
            }
        }
        require_profile_op("ffn_block", "ffn_rms_norm", "RMS_NORM");
        require_profile_op("ffn_block", "ffn_norm", "MUL");
    }

    if (candidate_kinds.find("attn_qkv_block") != candidate_kinds.end()) {
        if (!state.attn_qkv_shards.enabled ||
                !phase_matches(state.attn_qkv_shards.phase, true)) {
            throw std::runtime_error(
                    "runtime_routes attn_qkv_block requires an enabled root prefill attn_qkv_shards policy");
        }
        if (!str_eq_ci(routes.phase, "prefill")) {
            throw std::runtime_error(
                    "runtime_routes attn_qkv_block currently requires phase=prefill");
        }

        // Routed graph names append a 16-hex plan id, projection role, op
        // role, and layer suffix. Keep the inherited lane id short enough to
        // remain unique within GGML_MAX_NAME; static QKV policies retain the
        // original unrestricted safe-character compatibility.
        constexpr size_t routed_qkv_split_id_max = 12;
        for (const auto & split : state.attn_qkv_shards.splits) {
            if (split.id.size() > routed_qkv_split_id_max) {
                throw std::runtime_error(
                        "runtime_routes attn_qkv_block split id '" + split.id +
                        "' must be at most " +
                        std::to_string(routed_qkv_split_id_max) +
                        " bytes so routed graph names are not truncated");
            }
        }

        int64_t root_q_width = 0;
        int64_t root_k_width = 0;
        int64_t root_v_width = 0;
        if (!attn_qkv_policy_width(
                    state.attn_qkv_shards,
                    LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q,
                    root_q_width) ||
                !attn_qkv_policy_width(
                    state.attn_qkv_shards,
                    LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_K,
                    root_k_width) ||
                !attn_qkv_policy_width(
                    state.attn_qkv_shards,
                    LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_V,
                    root_v_width)) {
            throw std::runtime_error(
                    "runtime_routes attn_qkv_block root policy must contain complete Q/K/V partitions with 1 to 3 active lanes");
        }

        for (const std::string & profile_name : routes.profiles) {
            const auto * policy = effective_attn_qkv_policy_for_profile(state, profile_name);
            int64_t q_width = 0;
            int64_t k_width = 0;
            int64_t v_width = 0;
            if (policy == nullptr || !policy->enabled ||
                    !attn_qkv_policy_width(
                        *policy, LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q, q_width) ||
                    !attn_qkv_policy_width(
                        *policy, LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_K, k_width) ||
                    !attn_qkv_policy_width(
                        *policy, LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_V, v_width) ||
                    q_width != root_q_width ||
                    k_width != root_k_width ||
                    v_width != root_v_width) {
                throw std::runtime_error(
                        "runtime_routes attn_qkv_block profile '" + profile_name +
                        "' requires complete Q/K/V splits with the root projection widths");
            }
        }
    }

    if (candidate_kinds.find("attn_out_block") != candidate_kinds.end()) {
        if (!state.attn_out_shards.enabled ||
                state.attn_out_shards.partition_axis != "output") {
            throw std::runtime_error(
                    "runtime_routes attn_out_block requires an enabled root output-axis attn_out_shards policy");
        }

        int64_t root_width = 0;
        if (!attn_out_policy_width(state.attn_out_shards, root_width)) {
            throw std::runtime_error(
                    "runtime_routes attn_out_block root policy must contain a complete partition with 1, 2, or 3 active lanes");
        }
        for (const std::string & profile_name : routes.profiles) {
            const auto * policy = effective_attn_out_policy_for_profile(state, profile_name);
            int64_t profile_width = 0;
            if (policy == nullptr || !policy->enabled ||
                    policy->partition_axis != "output" ||
                    !attn_out_policy_width(*policy, profile_width) ||
                    profile_width != root_width) {
                throw std::runtime_error(
                        "runtime_routes attn_out_block profile '" + profile_name +
                        "' requires an enabled output-axis split with 1, 2, or 3 active lanes and the root partition width");
            }
        }
    }

    if ((candidate_kinds.find("weighted_norm") != candidate_kinds.end() ||
         candidate_kinds.find("ffn_block") != candidate_kinds.end()) &&
            (!state.residency_enabled || state.residency_rules.empty())) {
        throw std::runtime_error(
                "runtime_routes weighted_norm/ffn_block requires enabled norm-weight residency rules");
    }

    // The allowlist is also the model-load/cache memory contract. Requiring
    // every entry to be reachable prevents a typo from silently expanding the
    // resident union with candidates that can never be selected.
    std::unordered_set<std::string> reachable;
    std::vector<std::string> worklist = { routes.initial_profile };
    while (!worklist.empty()) {
        std::string profile = std::move(worklist.back());
        worklist.pop_back();
        if (!reachable.insert(profile).second) {
            continue;
        }
        const auto * transition = find_runtime_route_transition(routes, profile);
        if (transition == nullptr) {
            continue;
        }
        for (const auto & destination : transition->to_profiles) {
            if (reachable.find(destination) == reachable.end()) {
                worklist.push_back(destination);
            }
        }
    }
    if (reachable.size() != routes.profiles.size()) {
        for (const auto & profile : routes.profiles) {
            if (reachable.find(profile) == reachable.end()) {
                throw std::runtime_error(
                        "runtime_routes profile '" + profile +
                        "' is not reachable from initial_profile '" + routes.initial_profile + "'");
            }
        }
    }

    if (routes.mode == "clock") {
        // Normal clocks must be able to walk every throttled route back to the
        // initial profile through direct layer-safe transitions.
        for (const auto & start : routes.profiles) {
            std::unordered_set<std::string> seen;
            std::vector<std::string> pending = { start };
            while (!pending.empty()) {
                std::string profile = std::move(pending.back());
                pending.pop_back();
                if (!seen.insert(profile).second || profile == routes.initial_profile) {
                    continue;
                }
                const auto * transition = find_runtime_route_transition(routes, profile);
                if (transition != nullptr) {
                    pending.insert(
                            pending.end(), transition->to_profiles.begin(), transition->to_profiles.end());
                }
            }
            if (seen.find(routes.initial_profile) == seen.end()) {
                throw std::runtime_error(
                        "runtime_routes clock profile '" + start +
                        "' has no transition path back to initial_profile '" +
                        routes.initial_profile + "'");
            }
        }

        // Equal clock points with overlapping token applicability cannot be
        // distinguished by the reused snapshot selector.
        for (size_t i = 0; i < routes.profiles.size(); ++i) {
            const auto & lhs = state.profiles.at(routes.profiles[i]);
            for (size_t j = i + 1; j < routes.profiles.size(); ++j) {
                const auto & rhs = state.profiles.at(routes.profiles[j]);
                if (same_clock_frequencies(lhs.clock_point, rhs.clock_point) &&
                    token_ranges_overlap(lhs.input_tokens, rhs.input_tokens) &&
                    token_ranges_overlap(lhs.ubatch_tokens, rhs.ubatch_tokens)) {
                    throw std::runtime_error(
                            "runtime_routes profiles '" + routes.profiles[i] + "' and '" +
                            routes.profiles[j] +
                            "' have an indistinguishable clock/token selector");
                }
            }
        }
    }

    state.runtime_routes = std::move(routes);
}

} // namespace

bool llama_backend_policy_load(const char * path, bool enable_weights, bool enable_ops) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);

    policy_state next;
    next.path = path ? path : "";

    try {
        // Build a new state first and publish it only after full validation.
        // This avoids leaving a partially parsed policy active on parse errors.
        std::ifstream file(next.path);
        if (!file) {
            LLAMA_LOG_ERROR("backend_policy: failed to open '%s'\n", next.path.c_str());
            return false;
        }

        json root;
        file >> root;

        next.loaded = true;
        next.enabled = root.value("enabled", true);

        if (root.contains("devices") && root["devices"].is_object()) {
            next.fallback_priority = parse_string_array(root["devices"], "fallback_priority");
        }
        if (next.fallback_priority.empty()) {
            next.fallback_priority = parse_string_array(root, "fallback_priority");
        }
        if (next.fallback_priority.empty()) {
            next.fallback_priority = {"CPU"};
        }

        if (root.contains("weights") && root["weights"].is_object()) {
            const auto & weights = root["weights"];
            next.weights_enabled = next.enabled && enable_weights && weights.value("enabled", true);
            if (weights.contains("default")) {
                if (weights["default"].is_string()) {
                    next.default_weight_backends.push_back(weights["default"].get<std::string>());
                } else if (weights["default"].is_array()) {
                    for (const auto & item : weights["default"]) {
                        next.default_weight_backends.push_back(item.get<std::string>());
                    }
                } else {
                    throw std::runtime_error("weights.default must be a string or an array");
                }
            }
            parse_rules(weights, "weights", next.weight_rules);
        }

        if (root.contains("ops") && root["ops"].is_object()) {
            const auto & ops = root["ops"];
            next.ops_enabled = next.enabled && enable_ops && ops.value("enabled", true);
            parse_rules(ops, "ops", next.op_rules);
        }

        if (root.contains("residency") && root["residency"].is_object()) {
            const auto & residency = root["residency"];
            next.residency_enabled = next.enabled && residency.value("enabled", false);
            parse_rules(residency, "residency", next.residency_rules);
        }

        if (root.contains("ffn_parallel")) {
            next.ffn_parallel = parse_ffn_parallel(root["ffn_parallel"], "ffn_parallel");
            next.ffn_parallel.enabled = next.enabled && next.ffn_parallel.enabled;
        }

        if (root.contains("attn_qkv_shards")) {
            next.attn_qkv_shards =
                parse_attn_qkv_shards(root["attn_qkv_shards"], "attn_qkv_shards");
            next.attn_qkv_shards.enabled = next.enabled && next.attn_qkv_shards.enabled;
        }

        if (root.contains("attn_out_shards")) {
            next.attn_out_shards =
                parse_attn_out_shards(root["attn_out_shards"], "attn_out_shards");
            next.attn_out_shards.enabled = next.enabled && next.attn_out_shards.enabled;
        }

        if (root.contains("ffn_clock_switch")) {
            if (!root["ffn_clock_switch"].is_object()) {
                throw std::runtime_error("ffn_clock_switch must be an object");
            }
            next.ffn_clock_switch_enabled =
                next.enabled && root["ffn_clock_switch"].value("enabled", false);
        }

        if (root.contains("runtime") && root["runtime"].is_object()) {
            const auto & runtime = root["runtime"];
            next.default_profile = runtime.value("default_profile", std::string());
        }

        if (root.contains("stage_switch") && root["stage_switch"].is_object()) {
            const auto & stage = root["stage_switch"];
            next.stage_switch_enabled = next.enabled && stage.value("enabled", false);
            next.prefill_profile = stage.value("prefill_profile", std::string());
            next.decode_profile = stage.value("decode_profile", std::string());
        }

        if (root.contains("thermal_switch") && root["thermal_switch"].is_object()) {
            const auto & thermal = root["thermal_switch"];
            next.thermal_switch_enabled = next.enabled && thermal.value("enabled", false);
            next.cool_profile = thermal.value("cool_profile", std::string());
            next.hot_profile = thermal.value("hot_profile", std::string());
            next.critical_profile = thermal.value("critical_profile", std::string());
            next.thermal_state_file = thermal.value("state_file", std::string());
        }

        json profile_defaults = json::object();
        if (root.contains("profile_defaults")) {
            if (!root["profile_defaults"].is_object()) {
                throw std::runtime_error("profile_defaults must be an object");
            }
            profile_defaults = root["profile_defaults"];
        }

        if (root.contains("profiles") && root["profiles"].is_object()) {
            for (auto it = root["profiles"].begin(); it != root["profiles"].end(); ++it) {
                if (!it.value().is_object()) {
                    throw std::runtime_error("profiles entries must be objects");
                }
                if (it.key().empty() || it.key().size() >= LLAMA_BACKEND_POLICY_PROFILE_NAME_MAX) {
                    throw std::runtime_error(
                            "profile names must be non-empty and shorter than " +
                            std::to_string(LLAMA_BACKEND_POLICY_PROFILE_NAME_MAX) + " bytes");
                }
                // RFC 7396 merge-patch semantics make profile_defaults useful
                // without changing the materialized policy representation:
                // nested objects merge and arrays such as rules/splits replace.
                json effective_profile = profile_defaults;
                effective_profile.merge_patch(it.value());
                profile_policy profile;
                parse_profile_section(
                        effective_profile, root, it.key(), next.enabled,
                        enable_weights, enable_ops, profile);
                next.profiles.emplace(it.key(), std::move(profile));
            }
        }

        validate_profile_reference(next, next.default_profile, "runtime.default_profile");
        validate_profile_reference(next, next.prefill_profile, "stage_switch.prefill_profile");
        validate_profile_reference(next, next.decode_profile, "stage_switch.decode_profile");
        validate_profile_reference(next, next.cool_profile, "thermal_switch.cool_profile");
        validate_profile_reference(next, next.hot_profile, "thermal_switch.hot_profile");
        validate_profile_reference(next, next.critical_profile, "thermal_switch.critical_profile");

        parse_runtime_routes(root, next);

        if (next.ffn_clock_switch_enabled) {
            size_t selectable_profiles = 0;
            for (const auto & kv : next.profiles) {
                if (kv.second.clock_point.configured &&
                    kv.second.input_tokens.configured &&
                    kv.second.ffn_parallel.configured &&
                    env_enabled("LLAMA_FFN_PARALLEL", kv.second.ffn_parallel.enabled)) {
                    ++selectable_profiles;
                }
            }
            if (selectable_profiles == 0) {
                throw std::runtime_error(
                        "ffn_clock_switch.enabled requires at least one enabled profile with "
                        "clock_point, applicability.input_tokens, and ffn_parallel");
            }

            // Two profiles with the same measured clock point and overlapping
            // token applicability would be indistinguishable to the selector.
            // Treat that as a configuration error instead of silently choosing
            // whichever profile name happens to sort first.
            for (auto lhs = next.profiles.begin(); lhs != next.profiles.end(); ++lhs) {
                if (!lhs->second.clock_point.configured ||
                    !lhs->second.ffn_parallel.configured ||
                    !env_enabled("LLAMA_FFN_PARALLEL", lhs->second.ffn_parallel.enabled)) {
                    continue;
                }
                for (auto rhs = std::next(lhs); rhs != next.profiles.end(); ++rhs) {
                    if (!rhs->second.clock_point.configured ||
                        !rhs->second.ffn_parallel.configured ||
                        !env_enabled("LLAMA_FFN_PARALLEL", rhs->second.ffn_parallel.enabled)) {
                        continue;
                    }
                    if (same_clock_frequencies(lhs->second.clock_point, rhs->second.clock_point) &&
                        token_ranges_overlap(lhs->second.input_tokens, rhs->second.input_tokens) &&
                        token_ranges_overlap(lhs->second.ubatch_tokens, rhs->second.ubatch_tokens)) {
                        throw std::runtime_error(
                                "ffn clock profiles '" + lhs->first + "' and '" + rhs->first +
                                "' have an indistinguishable clock/token selector");
                    }
                }
            }
        }

        if (!root.contains("weights")) {
            next.weights_enabled = false;
        }
        if (!root.contains("ops")) {
            next.ops_enabled = false;
        }

        g_policy = std::move(next);

        LLAMA_LOG_INFO("backend_policy: loaded '%s' (weights=%s, weight_rules=%zu, ops=%s, op_rules=%zu, residency=%s, residency_rules=%zu, ffn_parallel=%s, attn_qkv_shards=%s, attn_out_shards=%s, ffn_clock_switch=%s, profiles=%zu)\n",
                g_policy.path.c_str(),
                g_policy.weights_enabled ? "on" : "off", g_policy.weight_rules.size(),
                g_policy.ops_enabled ? "on" : "off", g_policy.op_rules.size(),
                g_policy.residency_enabled ? "on" : "off", g_policy.residency_rules.size(),
                g_policy.ffn_parallel.enabled ? "on" : "off",
                g_policy.attn_qkv_shards.enabled ? "on" : "off",
                g_policy.attn_out_shards.enabled ? "on" : "off",
                g_policy.ffn_clock_switch_enabled ? "on" : "off",
                g_policy.profiles.size());
        if (g_policy.runtime_routes.enabled) {
            LLAMA_LOG_INFO(
                    "backend_policy: runtime routes enabled (mode=%s, phase=%s, initial=%s, profiles=%zu)\n",
                    g_policy.runtime_routes.mode.c_str(),
                    g_policy.runtime_routes.phase.c_str(),
                    g_policy.runtime_routes.initial_profile.c_str(),
                    g_policy.runtime_routes.profiles.size());
        }
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("backend_policy: failed to parse '%s': %s\n", next.path.c_str(), e.what());
        return false;
    }
}

void llama_backend_policy_clear(void) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    g_policy = {};
}

bool llama_backend_policy_weights_enabled() {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    return g_policy.loaded && g_policy.weights_enabled;
}

bool llama_backend_policy_ops_enabled() {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    if (!g_policy.loaded) {
        return false;
    }
    if (g_policy.ops_enabled) {
        return true;
    }
    for (const auto & kv : g_policy.profiles) {
        if (kv.second.ops_enabled) {
            return true;
        }
    }
    return false;
}

bool llama_backend_policy_residency_enabled() {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    return g_policy.loaded && env_enabled("LLAMA_BACKEND_POLICY_RESIDENCY", g_policy.residency_enabled);
}

bool llama_backend_policy_ffn_parallel_enabled() {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    llama_backend_policy_ffn_parallel policy;
    return g_policy.loaded && effective_ffn_policy(policy);
}

bool llama_backend_policy_attn_qkv_shards_enabled() {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    return g_policy.loaded && g_policy.attn_qkv_shards.enabled;
}

bool llama_backend_policy_attn_out_shards_enabled() {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    return g_policy.loaded && g_policy.attn_out_shards.enabled;
}

int llama_backend_policy_ffn_parallel_reduce_threads() {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    llama_backend_policy_ffn_parallel policy;
    if (!g_policy.loaded || !effective_ffn_policy(policy)) {
        return 0;
    }
    const size_t active_splits = std::count_if(
            policy.splits.begin(), policy.splits.end(), [](const auto & split) {
                return split.size > 0;
            });
    return active_splits >= 2 ? policy.reduce_threads : 0;
}

bool llama_backend_policy_ffn_clock_switch_enabled(void) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    const bool layer_plan_owns_ffn = g_policy.runtime_routes.enabled &&
        g_policy.runtime_routes.mode != "off" &&
        std::find(g_policy.runtime_routes.candidate_kinds.begin(),
                g_policy.runtime_routes.candidate_kinds.end(),
                "ffn_block") != g_policy.runtime_routes.candidate_kinds.end();
    return g_policy.loaded && g_policy.enabled &&
        !layer_plan_owns_ffn &&
        env_enabled("LLAMA_BACKEND_POLICY_FFN_CLOCK_SWITCH", g_policy.ffn_clock_switch_enabled);
}

bool llama_backend_policy_reset_ffn_clock_profile(void) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    const bool changed = !g_policy.pending_clock_profile.empty();
    g_policy.pending_clock_profile.clear();
    return changed;
}

struct llama_backend_policy_ffn_clock_result llama_backend_policy_select_ffn_clock_profile(
        int32_t input_tokens,
        int32_t ubatch_tokens,
        int64_t gold_khz,
        int64_t prime_khz,
        int64_t gpu_hz) {
    llama_backend_policy_ffn_clock_result result = {};
    result.distance = -1.0;

    std::lock_guard<std::mutex> lock(g_policy_mutex);
    const bool layer_plan_owns_ffn = g_policy.runtime_routes.enabled &&
        g_policy.runtime_routes.mode != "off" &&
        std::find(g_policy.runtime_routes.candidate_kinds.begin(),
                g_policy.runtime_routes.candidate_kinds.end(),
                "ffn_block") != g_policy.runtime_routes.candidate_kinds.end();
    result.enabled = g_policy.loaded && g_policy.enabled && !layer_plan_owns_ffn &&
        env_enabled("LLAMA_BACKEND_POLICY_FFN_CLOCK_SWITCH", g_policy.ffn_clock_switch_enabled);
    if (!result.enabled) {
        return result;
    }

    // A failed sysfs read must not evict a previously valid pending profile.
    // The caller can retry at the next query boundary.
    if (input_tokens <= 0 || ubatch_tokens <= 0 ||
        gold_khz <= 0 || prime_khz <= 0 || gpu_hz <= 0) {
        return result;
    }

    const auto selection = select_clock_profile_locked(
            nullptr, /* require_enabled_ffn = */ true,
            input_tokens, ubatch_tokens, gold_khz, prime_khz, gpu_hz);
    if (!selection.matched) {
        if (!g_policy.pending_clock_profile.empty()) {
            g_policy.pending_clock_profile.clear();
            result.pending_changed = true;
        }
        return result;
    }

    result.matched = true;
    result.distance = selection.distance;
    std::snprintf(result.profile, sizeof(result.profile), "%s", selection.profile.c_str());
    result.pending_changed = selection.profile != g_policy.pending_clock_profile;
    g_policy.pending_clock_profile = selection.profile;
    return result;
}

bool llama_backend_policy_update_runtime_profile(bool is_prefill) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);

    if (!g_policy.loaded || !g_policy.enabled) {
        return false;
    }

    // A layer-plan graph prebuilds every FFN topology and atomically latches
    // one plan at l_out. Never let the legacy query-boundary global profile
    // rebuild that graph or select a different FFN partition authority, even
    // when an older launcher still exports FFN_CLOCK_SWITCH=on.
    if (g_policy.runtime_routes.enabled && g_policy.runtime_routes.mode != "off" &&
            std::find(g_policy.runtime_routes.candidate_kinds.begin(),
                    g_policy.runtime_routes.candidate_kinds.end(),
                    "ffn_block") != g_policy.runtime_routes.candidate_kinds.end()) {
        return false;
    }

    std::string next_profile = env_string("LLAMA_BACKEND_POLICY_ACTIVE_PROFILE");

    const bool clock_switch_enabled =
        env_enabled("LLAMA_BACKEND_POLICY_FFN_CLOCK_SWITCH", g_policy.ffn_clock_switch_enabled);
    if (next_profile.empty() && clock_switch_enabled && !g_policy.pending_clock_profile.empty()) {
        next_profile = g_policy.pending_clock_profile;
    }

    const bool thermal_enabled = env_enabled("LLAMA_BACKEND_POLICY_THERMAL_SWITCH", g_policy.thermal_switch_enabled);
    if (next_profile.empty() && thermal_enabled) {
        std::string state = trim(env_string("LLAMA_BACKEND_POLICY_THERMAL_STATE"));
        if (state.empty()) {
            std::string state_file = env_string("LLAMA_BACKEND_POLICY_THERMAL_STATE_FILE");
            if (state_file.empty()) {
                state_file = g_policy.thermal_state_file;
            }
            state = read_text_file_first_line(state_file);
        }
        state = lower(state);
        if ((state == "critical" || state == "throttle" || state == "throttled" || state == "throttling") && !g_policy.critical_profile.empty()) {
            next_profile = g_policy.critical_profile;
        } else if ((state == "hot" || state == "warm") && !g_policy.hot_profile.empty()) {
            next_profile = g_policy.hot_profile;
        } else if (!g_policy.cool_profile.empty()) {
            next_profile = g_policy.cool_profile;
        }
    }

    const bool stage_enabled = env_enabled("LLAMA_BACKEND_POLICY_STAGE_SWITCH", g_policy.stage_switch_enabled);
    if (next_profile.empty() && stage_enabled) {
        next_profile = is_prefill ? g_policy.prefill_profile : g_policy.decode_profile;
    }

    if (next_profile.empty()) {
        next_profile = g_policy.default_profile;
    }

    if (next_profile == g_policy.active_profile) {
        return false;
    }

    LLAMA_LOG_DEBUG("backend_policy: active profile changed '%s' -> '%s'\n",
            g_policy.active_profile.c_str(), next_profile.c_str());
    g_policy.active_profile = std::move(next_profile);
    g_policy.profile_generation++;
    return true;
}

const char * llama_backend_policy_active_profile() {
    static thread_local std::string profile;
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    profile = g_policy.active_profile;
    return profile.c_str();
}

bool llama_backend_policy_match_weight(const char * tensor_name, llama_backend_policy_match & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || !g_policy.weights_enabled || tensor_name == nullptr) {
        return false;
    }

    return match_weight_rules(
            g_policy.default_weight_backends,
            g_policy.weight_rules,
            g_policy.fallback_priority,
            tensor_name,
            "weights.default",
            out);
}

bool llama_backend_policy_match_residency(const char * tensor_name, llama_backend_policy_match & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || !env_enabled("LLAMA_BACKEND_POLICY_RESIDENCY", g_policy.residency_enabled) || tensor_name == nullptr) {
        return false;
    }

    return match_weight_rules(
            {},
            g_policy.residency_rules,
            {},
            tensor_name,
            "residency.default",
            out);
}

bool llama_backend_policy_match_profile_weight(
        const char * tensor_name,
        bool is_prefill,
        llama_backend_policy_match & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || tensor_name == nullptr) {
        return false;
    }

    std::string profile_name = g_policy.active_profile;
    if (profile_name.empty()) {
        const bool stage_enabled = env_enabled("LLAMA_BACKEND_POLICY_STAGE_SWITCH", g_policy.stage_switch_enabled);
        if (stage_enabled) {
            profile_name = is_prefill ? g_policy.prefill_profile : g_policy.decode_profile;
        } else {
            profile_name = g_policy.default_profile;
        }
    }

    if (profile_name.empty()) {
        return false;
    }

    auto it = g_policy.profiles.find(profile_name);
    if (it == g_policy.profiles.end() || !it->second.weights_enabled) {
        return false;
    }

    return match_weight_rules(
            it->second.default_weight_backends,
            it->second.weight_rules,
            g_policy.fallback_priority,
            tensor_name,
            ("profiles." + profile_name + ".weights.default").c_str(),
            out);
}

bool llama_backend_policy_match_op(
        const char * base_name,
        const char * tensor_name,
        enum ggml_op op,
        int il,
        bool is_prefill,
        llama_backend_policy_match & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded) {
        return false;
    }

    const std::string base = base_name ? base_name : "";
    const std::string name = tensor_name ? tensor_name : "";

    if (!g_policy.active_profile.empty()) {
        auto it = g_policy.profiles.find(g_policy.active_profile);
        if (it != g_policy.profiles.end() && it->second.ops_enabled) {
            return match_op_rules(it->second.op_rules, g_policy.fallback_priority, base, name, op, il, is_prefill, out);
        }
    }

    if (!g_policy.ops_enabled) {
        return false;
    }

    return match_op_rules(g_policy.op_rules, g_policy.fallback_priority, base, name, op, il, is_prefill, out);
}

bool llama_backend_policy_resolve_runtime_routes(
        bool is_prefill,
        llama_backend_policy_runtime_routes & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || !g_policy.enabled || !g_policy.runtime_routes.enabled ||
        !phase_matches(g_policy.runtime_routes.phase, is_prefill)) {
        return false;
    }

    out = g_policy.runtime_routes;
    return true;
}

bool llama_backend_policy_list_runtime_route_profiles(
        bool is_prefill,
        std::vector<std::string> & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out.clear();

    if (!g_policy.loaded || !g_policy.enabled || !g_policy.runtime_routes.enabled ||
        !phase_matches(g_policy.runtime_routes.phase, is_prefill)) {
        return false;
    }

    out = g_policy.runtime_routes.profiles;
    return true;
}

bool llama_backend_policy_list_runtime_route_next_profiles(
        const char * current_profile,
        bool is_prefill,
        std::vector<std::string> & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out.clear();

    const auto & routes = g_policy.runtime_routes;
    if (!g_policy.loaded || !g_policy.enabled || !routes.enabled ||
        !phase_matches(routes.phase, is_prefill)) {
        return false;
    }

    const std::string current = current_profile ? current_profile : "";
    if (current.empty()) {
        out = routes.profiles;
        return true;
    }
    if (std::find(routes.profiles.begin(), routes.profiles.end(), current) == routes.profiles.end()) {
        return false;
    }

    // Remaining on the current route is always legal. Transitions describe
    // additional candidates, not an obligation to switch at every boundary.
    out.push_back(current);
    const auto * transition = find_runtime_route_transition(routes, current);
    if (transition != nullptr) {
        for (const auto & destination : transition->to_profiles) {
            if (std::find(out.begin(), out.end(), destination) == out.end()) {
                out.push_back(destination);
            }
        }
    }
    return true;
}

bool llama_backend_policy_match_op_for_profile(
        const char * profile_name,
        const char * base_name,
        const char * tensor_name,
        enum ggml_op op,
        int il,
        bool is_prefill,
        llama_backend_policy_match & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || profile_name == nullptr || profile_name[0] == '\0') {
        return false;
    }
    const auto it = g_policy.profiles.find(profile_name);
    if (it == g_policy.profiles.end() || !it->second.ops_enabled) {
        return false;
    }

    return match_op_rules(
            it->second.op_rules,
            g_policy.fallback_priority,
            base_name ? base_name : "",
            tensor_name ? tensor_name : "",
            op,
            il,
            is_prefill,
            out);
}

bool llama_backend_policy_select_runtime_route_profile(
        const char * current_profile,
        bool is_prefill,
        int32_t input_tokens,
        int32_t ubatch_tokens,
        int64_t gold_khz,
        int64_t prime_khz,
        int64_t gpu_hz,
        llama_backend_policy_runtime_route_selection & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    const auto & routes = g_policy.runtime_routes;
    out.enabled = g_policy.loaded && g_policy.enabled && routes.enabled &&
        routes.mode == "clock" && phase_matches(routes.phase, is_prefill);
    if (!out.enabled) {
        return false;
    }

    const std::string current = current_profile ? current_profile : "";
    if (!current.empty() &&
            std::find(routes.profiles.begin(), routes.profiles.end(), current) == routes.profiles.end()) {
        return false;
    }

    // Pick the globally best clock profile first. A greedy choice restricted
    // to current+successors can become permanently stuck when a legal recovery
    // path contains a temporarily worse intermediate clock point.
    const auto target = select_clock_profile_locked(
            &routes.profiles, /* require_enabled_ffn = */ false,
            input_tokens, ubatch_tokens, gold_khz, prime_khz, gpu_hz);
    if (!target.matched) {
        return false;
    }

    std::string selected_profile = target.profile;
    if (!current.empty() && current != target.profile) {
        // runtime_routes validation makes the catalog strongly connected for
        // clock mode (initial reaches every profile and every profile reaches
        // initial). Return only the first edge of a shortest legal path so a
        // layer boundary never jumps across an unprepared transition.
        std::vector<std::string> queue = { current };
        std::map<std::string, std::string> predecessor;
        predecessor.emplace(current, std::string());
        const auto profile_applies_to_shape = [&](const std::string & profile_name) {
            const auto profile_it = g_policy.profiles.find(profile_name);
            return profile_it != g_policy.profiles.end() &&
                token_range_contains(profile_it->second.input_tokens, input_tokens) &&
                token_range_contains(profile_it->second.ubatch_tokens, ubatch_tokens);
        };
        for (size_t head = 0; head < queue.size() && predecessor.find(target.profile) == predecessor.end(); ++head) {
            // Keep a value copy: queue growth below may reallocate and would
            // invalidate a reference to queue[head].
            const std::string from = queue[head];
            const auto * transition = find_runtime_route_transition(routes, from);
            if (transition == nullptr) {
                continue;
            }
            for (const std::string & destination : transition->to_profiles) {
                // Do not route through an unprofiled graph shape merely because
                // it gives a shorter topological path. A longer path whose
                // intermediate plans apply to this ubatch remains eligible.
                if (!profile_applies_to_shape(destination)) {
                    continue;
                }
                if (predecessor.emplace(destination, from).second) {
                    queue.push_back(destination);
                }
            }
        }
        if (predecessor.find(target.profile) == predecessor.end()) {
            return false;
        }

        selected_profile = target.profile;
        while (predecessor.at(selected_profile) != current) {
            selected_profile = predecessor.at(selected_profile);
        }
    }

    // The steady-state path is sampled at every layer boundary, so avoid a
    // temporary candidate vector and second lookup when the globally closest
    // profile is already the direct request.
    if (selected_profile == target.profile) {
        out.matched = true;
        out.profile = target.profile;
        out.distance = target.distance;
        return true;
    }

    // Report distance for the intermediate profile actually requested at this
    // boundary, not the eventual terminal target. This also refuses an
    // intermediate hop whose token applicability does not cover the current
    // graph shape.
    const std::vector<std::string> selected = { selected_profile };
    const auto hop = select_clock_profile_locked(
            &selected, /* require_enabled_ffn = */ false,
            input_tokens, ubatch_tokens, gold_khz, prime_khz, gpu_hz);
    if (!hop.matched) {
        return false;
    }

    out.matched = true;
    out.profile = hop.profile;
    out.distance = hop.distance;
    return true;
}

bool llama_backend_policy_runtime_route_clock_enabled(void) {
    llama_backend_policy_runtime_routes routes;
    if (llama_backend_policy_resolve_runtime_routes(true, routes) && routes.mode == "clock") {
        return true;
    }
    return llama_backend_policy_resolve_runtime_routes(false, routes) && routes.mode == "clock";
}

struct llama_backend_policy_runtime_route_result llama_backend_policy_select_layer_route_profile(
        const char * current_profile,
        bool is_prefill,
        int32_t input_tokens,
        int32_t ubatch_tokens,
        int64_t gold_khz,
        int64_t prime_khz,
        int64_t gpu_hz) {
    llama_backend_policy_runtime_route_result result = {};
    result.distance = -1.0;

    llama_backend_policy_runtime_route_selection selection;
    llama_backend_policy_select_runtime_route_profile(
            current_profile, is_prefill, input_tokens, ubatch_tokens,
            gold_khz, prime_khz, gpu_hz, selection);
    result.enabled = selection.enabled;
    result.matched = selection.matched;
    result.distance = selection.distance;
    if (selection.matched) {
        std::snprintf(result.profile, sizeof(result.profile), "%s", selection.profile.c_str());
    }
    return result;
}

bool llama_backend_policy_match_ffn_parallel(
        int il,
        bool is_prefill,
        llama_backend_policy_ffn_parallel & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || !effective_ffn_policy(out)) {
        return false;
    }

    if (!phase_matches(out.phase, is_prefill)) {
        out = {};
        return false;
    }
    if (out.layer_start != -2) {
        if (il < out.layer_start || (out.layer_end != -2 && il > out.layer_end)) {
            out = {};
            return false;
        }
    }

    return true;
}

bool llama_backend_policy_match_attn_qkv_shards(
        int il,
        bool is_prefill,
        llama_backend_policy_attn_qkv_shards & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || !g_policy.attn_qkv_shards.enabled) {
        return false;
    }

    const auto & policy = g_policy.attn_qkv_shards;
    if (!phase_matches(policy.phase, is_prefill)) {
        return false;
    }
    if (policy.layer_start != -2 &&
            (il < policy.layer_start || (policy.layer_end != -2 && il > policy.layer_end))) {
        return false;
    }

    out = policy;
    return true;
}

bool llama_backend_policy_match_attn_qkv_shards_for_profile(
        const char * profile_name,
        int il,
        bool is_prefill,
        llama_backend_policy_attn_qkv_shards & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || profile_name == nullptr || profile_name[0] == '\0' ||
            !is_prefill) {
        return false;
    }

    const auto * policy = effective_attn_qkv_policy_for_profile(g_policy, profile_name);
    if (policy == nullptr || !policy->enabled ||
            !phase_matches(policy->phase, is_prefill) ||
            !attn_qkv_policy_matches_layer(*policy, il)) {
        return false;
    }

    out = *policy;
    return true;
}

bool llama_backend_policy_match_attn_out_shards(
        int il,
        bool is_prefill,
        llama_backend_policy_attn_out_shards & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || !g_policy.attn_out_shards.enabled || !is_prefill) {
        return false;
    }

    const auto & policy = g_policy.attn_out_shards;
    if (policy.layer_start != -2 &&
            (il < policy.layer_start || (policy.layer_end != -2 && il > policy.layer_end))) {
        return false;
    }

    out = policy;
    return true;
}

bool llama_backend_policy_match_attn_out_shards_for_profile(
        const char * profile_name,
        int il,
        bool is_prefill,
        llama_backend_policy_attn_out_shards & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || profile_name == nullptr || profile_name[0] == '\0' ||
            !is_prefill) {
        return false;
    }

    const auto * policy = effective_attn_out_policy_for_profile(g_policy, profile_name);
    if (policy == nullptr || !policy->enabled ||
            !phase_matches(policy->phase, is_prefill) ||
            !attn_out_policy_matches_layer(*policy, il)) {
        return false;
    }

    out = *policy;
    return true;
}

bool llama_backend_policy_match_ffn_parallel_for_profile(
        const char * profile_name,
        int il,
        bool is_prefill,
        llama_backend_policy_ffn_parallel & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || profile_name == nullptr || profile_name[0] == '\0') {
        return false;
    }
    const auto it = g_policy.profiles.find(profile_name);
    if (it == g_policy.profiles.end() || !it->second.ffn_parallel.configured ||
            !it->second.ffn_parallel.enabled) {
        return false;
    }

    out = materialize_ffn_policy(it->second.ffn_parallel);
    if (!out.enabled || !phase_matches(out.phase, is_prefill)) {
        out = {};
        return false;
    }
    if (out.layer_start != -2 &&
            (il < out.layer_start || (out.layer_end != -2 && il > out.layer_end))) {
        out = {};
        return false;
    }
    return true;
}

uint64_t llama_backend_policy_runtime_route_plan_id(const char * profile_name) {
    if (profile_name == nullptr || profile_name[0] == '\0') {
        return 0;
    }
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char * ch = (const unsigned char *) profile_name; *ch != '\0'; ++ch) {
        hash ^= *ch;
        hash *= 1099511628211ULL;
    }
    return hash == 0 ? 1 : hash;
}

bool llama_backend_policy_match_ffn_parallel_layer(
        int il,
        llama_backend_policy_ffn_parallel & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || !effective_ffn_policy(out)) {
        return false;
    }
    if (out.layer_start != -2) {
        if (il < out.layer_start || (out.layer_end != -2 && il > out.layer_end)) {
            out = {};
            return false;
        }
    }

    return true;
}

bool llama_backend_policy_list_ffn_parallel_load_policies(
        int il,
        std::vector<llama_backend_policy_ffn_parallel> & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out.clear();

    if (!g_policy.loaded || !g_policy.enabled) {
        return false;
    }

    const auto append_policy = [&](ffn_parallel_policy policy) {
        if (!policy.configured) {
            return;
        }
        policy.enabled = env_enabled("LLAMA_FFN_PARALLEL", policy.enabled);
        if (!policy.enabled) {
            return;
        }

        llama_backend_policy_ffn_parallel candidate = materialize_ffn_policy(policy);
        candidate.enabled = true;
        if (candidate.splits.empty() || !ffn_policy_matches_layer(candidate, il)) {
            return;
        }
        for (const auto & existing : out) {
            if (same_ffn_residency_requirements(existing, candidate)) {
                return;
            }
        }
        out.push_back(std::move(candidate));
    };

    ffn_parallel_policy base = g_policy.ffn_parallel;
    if (!base.configured) {
        std::vector<llama_backend_policy_ffn_split> env_splits;
        if (parse_env_ffn_splits(env_splits)) {
            base.configured = true;
            base.enabled = true;
            base.phase = "prefill";
            base.align = 256;
            base.source = "LLAMA_FFN_PARALLEL_SPLITS";
            base.splits = std::move(env_splits);
        }
    }
    append_policy(std::move(base));

    const bool layer_routed_ffn = g_policy.runtime_routes.enabled &&
        std::find(g_policy.runtime_routes.candidate_kinds.begin(),
                g_policy.runtime_routes.candidate_kinds.end(),
                "ffn_block") != g_policy.runtime_routes.candidate_kinds.end();
    if (layer_routed_ffn) {
        // runtime_routes.profiles is the explicit resident-candidate
        // allowlist.  Profiles kept in the catalog for another experiment
        // must not silently increase the connected FFN weight-cover union.
        for (const std::string & profile_name : g_policy.runtime_routes.profiles) {
            const auto it = g_policy.profiles.find(profile_name);
            GGML_ASSERT(it != g_policy.profiles.end());
            append_policy(it->second.ffn_parallel);
        }
    } else {
        // Preserve the legacy query-boundary behavior when FFN blocks are not
        // owned by the layer route plan.
        for (const auto & kv : g_policy.profiles) {
            append_policy(kv.second.ffn_parallel);
        }
    }

    return !out.empty();
}

bool llama_backend_policy_build_ffn_residency_plan(
        int il,
        llama_backend_policy_residency_plan & out) {
    out = {};

    std::vector<llama_backend_policy_ffn_parallel> policies;
    if (!llama_backend_policy_list_ffn_parallel_load_policies(il, policies)) {
        return false;
    }

    struct backend_intervals {
        std::string backend;
        std::vector<std::pair<int64_t, int64_t>> ranges;
    };

    std::map<std::string, backend_intervals> grouped;
    out.enabled = true;
    out.keep_full_source = false;
    out.source = "ffn_parallel connected cover union";
    bool has_all_phase_root = false;

    for (const auto & policy : policies) {
        if (policy.source == "ffn_parallel" && str_eq_ci(policy.phase, "all")) {
            has_all_phase_root = true;
        }
        if (!str_eq_ci(policy.phase, "all")) {
            out.keep_full_source = true;
        }

        for (const auto & split : policy.splits) {
            if (split.size == 0) {
                continue;
            }
            const std::string key = lower(split.backend);
            auto & entry = grouped[key];
            if (entry.backend.empty()) {
                entry.backend = split.backend;
            }
            entry.ranges.emplace_back(split.start, split.start + split.size);
        }
    }

    // Before the first clock selection, and whenever no profile is selected,
    // graph construction uses the root policy. A profile-only residency union
    // therefore cannot safely discard the full source tensor.
    if (!has_all_phase_root) {
        out.keep_full_source = true;
    }

    // A manually/default-selected profile whose layer range excludes this
    // layer falls back to the ordinary FFN path. Preserve the full source for
    // that case even if every residency policy that did match was phase=all.
    {
        std::lock_guard<std::mutex> lock(g_policy_mutex);
        for (const auto & kv : g_policy.profiles) {
            ffn_parallel_policy profile = kv.second.ffn_parallel;
            if (!profile.configured) {
                continue;
            }
            profile.enabled = env_enabled("LLAMA_FFN_PARALLEL", profile.enabled);
            if (!profile.enabled) {
                continue;
            }
            const auto candidate = materialize_ffn_policy(profile);
            if (!ffn_policy_matches_layer(candidate, il)) {
                out.keep_full_source = true;
                break;
            }
        }
    }

    for (const auto & kv : grouped) {
        const backend_intervals & entry = kv.second;
        std::vector<std::pair<int64_t, int64_t>> ranges = entry.ranges;
        std::sort(ranges.begin(), ranges.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

        int64_t cover_start = -1;
        int64_t cover_end = -1;
        const auto append_cover = [&]() {
            if (cover_start < 0 || cover_end <= cover_start) {
                return;
            }

            llama_backend_policy_residency_cover cover;
            cover.id = "cover." + kv.first + "." + std::to_string(cover_start) + "." +
                std::to_string(cover_end - cover_start);
            cover.start = cover_start;
            cover.size = cover_end - cover_start;
            cover.backend = entry.backend;
            out.covers.push_back(std::move(cover));
        };

        for (const auto & range : ranges) {
            if (range.second <= range.first) {
                continue;
            }
            if (cover_start < 0) {
                cover_start = range.first;
                cover_end = range.second;
                continue;
            }

            // Merge both overlapping and touching intervals. A gap, however,
            // must remain a separate allocation rather than being filled by a
            // convex hull that no runtime profile requested.
            if (range.first <= cover_end) {
                cover_end = std::max(cover_end, range.second);
                continue;
            }

            append_cover();
            cover_start = range.first;
            cover_end = range.second;
        }
        append_cover();
    }

    return !out.covers.empty();
}

bool llama_backend_policy_build_attn_qkv_residency_plan(
        int il,
        llama_backend_policy_attn_qkv_projection projection,
        llama_backend_policy_residency_plan & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    const char * projection_name = nullptr;
    switch (projection) {
        case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q:
            projection_name = "q";
            break;
        case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_K:
            projection_name = "k";
            break;
        case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_V:
            projection_name = "v";
            break;
        default:
            return false;
    }

    if (!g_policy.loaded || !g_policy.attn_qkv_shards.enabled ||
            !phase_matches(g_policy.attn_qkv_shards.phase, true) ||
            !attn_qkv_policy_matches_layer(g_policy.attn_qkv_shards, il)) {
        return false;
    }

    struct backend_intervals {
        std::string key;
        std::string backend;
        std::vector<std::pair<int64_t, int64_t>> ranges;
    };
    std::vector<backend_intervals> grouped;
    int64_t root_width = 0;
    if (!attn_qkv_policy_width(g_policy.attn_qkv_shards, projection, root_width)) {
        return false;
    }

    const auto append_policy = [&](const llama_backend_policy_attn_qkv_shards & policy) {
        int64_t width = 0;
        if (!policy.enabled || !phase_matches(policy.phase, true) ||
                !attn_qkv_policy_matches_layer(policy, il) ||
                !attn_qkv_policy_width(policy, projection, width) ||
                width != root_width) {
            return false;
        }

        for (const auto & split : policy.splits) {
            int64_t start = 0;
            int64_t size = 0;
            switch (projection) {
                case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q:
                    start = split.q_start;
                    size = split.q_size;
                    break;
                case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_K:
                    start = split.k_start;
                    size = split.k_size;
                    break;
                case LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_V:
                    start = split.v_start;
                    size = split.v_size;
                    break;
                default:
                    return false;
            }

            if (size == 0) {
                continue;
            }

            const std::string key = lower(split.backend);
            auto it = std::find_if(grouped.begin(), grouped.end(), [&](const auto & entry) {
                return entry.key == key;
            });
            if (it == grouped.end()) {
                grouped.push_back({ key, split.backend, {} });
                it = std::prev(grouped.end());
            }
            it->ranges.emplace_back(start, start + size);
        }
        return true;
    };

    if (!append_policy(g_policy.attn_qkv_shards)) {
        return false;
    }

    const bool routed_attn_qkv = g_policy.runtime_routes.enabled &&
        std::find(
                g_policy.runtime_routes.candidate_kinds.begin(),
                g_policy.runtime_routes.candidate_kinds.end(),
                "attn_qkv_block") != g_policy.runtime_routes.candidate_kinds.end();
    if (routed_attn_qkv) {
        for (const std::string & profile_name : g_policy.runtime_routes.profiles) {
            const auto * policy = effective_attn_qkv_policy_for_profile(g_policy, profile_name);
            if (policy == nullptr || !append_policy(*policy)) {
                out = {};
                return false;
            }
        }
    }

    for (auto & entry : grouped) {
        std::sort(entry.ranges.begin(), entry.ranges.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

        int64_t cover_start = -1;
        int64_t cover_end = -1;
        const auto append_cover = [&]() {
            if (cover_start < 0 || cover_end <= cover_start) {
                return;
            }
            llama_backend_policy_residency_cover cover;
            cover.id = std::string("cover.") + projection_name + "." + entry.key + "." +
                std::to_string(cover_start) + "." +
                std::to_string(cover_end - cover_start);
            cover.start = cover_start;
            cover.size = cover_end - cover_start;
            cover.backend = entry.backend;
            out.covers.push_back(std::move(cover));
        };

        for (const auto & range : entry.ranges) {
            if (cover_start < 0) {
                cover_start = range.first;
                cover_end = range.second;
            } else if (range.first <= cover_end) {
                cover_end = std::max(cover_end, range.second);
            } else {
                append_cover();
                cover_start = range.first;
                cover_end = range.second;
            }
        }
        append_cover();
    }

    out.enabled = !out.covers.empty();
    out.keep_full_source = true;
    out.source = std::string("attn_qkv_shards ") + projection_name + " axis-1 covers";
    return out.enabled;
}

bool llama_backend_policy_build_attn_out_residency_plan(
        int il,
        llama_backend_policy_residency_plan & out) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    out = {};

    if (!g_policy.loaded || !g_policy.attn_out_shards.enabled ||
            !attn_out_policy_matches_layer(g_policy.attn_out_shards, il)) {
        return false;
    }

    struct backend_intervals {
        std::string key;
        std::string backend;
        std::vector<std::pair<int64_t, int64_t>> ranges;
    };
    std::vector<backend_intervals> grouped;

    const auto append_policy = [&](const llama_backend_policy_attn_out_shards & policy) {
        int64_t width = 0;
        if (!policy.enabled || !attn_out_policy_matches_layer(policy, il) ||
                !attn_out_policy_width(policy, width)) {
            return false;
        }

        for (const auto & split : policy.splits) {
            if (split.size == 0) {
                continue;
            }
            const std::string key = lower(split.backend);
            auto it = std::find_if(grouped.begin(), grouped.end(), [&](const auto & entry) {
                return entry.key == key;
            });
            if (it == grouped.end()) {
                grouped.push_back({ key, split.backend, {} });
                it = std::prev(grouped.end());
            }
            it->ranges.emplace_back(split.start, split.start + split.size);
        }
        return true;
    };

    if (!append_policy(g_policy.attn_out_shards)) {
        return false;
    }

    const bool routed_attn_out = g_policy.runtime_routes.enabled &&
        std::find(
                g_policy.runtime_routes.candidate_kinds.begin(),
                g_policy.runtime_routes.candidate_kinds.end(),
                "attn_out_block") != g_policy.runtime_routes.candidate_kinds.end();
    if (routed_attn_out) {
        for (const std::string & profile_name : g_policy.runtime_routes.profiles) {
            const auto * policy = effective_attn_out_policy_for_profile(g_policy, profile_name);
            if (policy == nullptr || !append_policy(*policy)) {
                out = {};
                return false;
            }
        }
    }

    for (auto & entry : grouped) {
        std::sort(entry.ranges.begin(), entry.ranges.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

        int64_t cover_start = -1;
        int64_t cover_end = -1;
        const auto append_cover = [&]() {
            if (cover_start < 0 || cover_end <= cover_start) {
                return;
            }
            llama_backend_policy_residency_cover cover;
            cover.id = "cover." + entry.key + "." +
                std::to_string(cover_start) + "." +
                std::to_string(cover_end - cover_start);
            cover.start = cover_start;
            cover.size = cover_end - cover_start;
            cover.backend = entry.backend;
            out.covers.push_back(std::move(cover));
        };

        for (const auto & range : entry.ranges) {
            if (cover_start < 0) {
                cover_start = range.first;
                cover_end = range.second;
            } else if (range.first <= cover_end) {
                cover_end = std::max(cover_end, range.second);
            } else {
                append_cover();
                cover_start = range.first;
                cover_end = range.second;
            }
        }
        append_cover();
    }

    out.enabled = !out.covers.empty();
    out.keep_full_source = true;
    out.source = g_policy.attn_out_shards.partition_axis == "output"
        ? "attn_out_shards axis-1 covers"
        : "attn_out_shards axis-0 covers";
    return out.enabled;
}

bool llama_backend_policy_buft_matches(
        ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft,
        const std::string & backend_name) {
    if (buft == nullptr || backend_name.empty()) {
        return false;
    }

    if (str_eq_ci(backend_name, ggml_backend_buft_name(buft))) {
        return true;
    }

    // CPU/GPU/HTP aliases intentionally map only to each device's default
    // buffer. Extra buffer types such as HTP0-REPACK must match by exact name.
    if (dev != nullptr) {
        if (str_eq_ci(backend_name, ggml_backend_dev_name(dev))) {
            return buft == ggml_backend_dev_buffer_type(dev);
        }
        if (name_matches_backend_type(dev, backend_name)) {
            return buft == ggml_backend_dev_buffer_type(dev);
        }
    } else if (str_eq_ci(backend_name, "cpu")) {
        return buft == ggml_backend_cpu_buffer_type();
    }

    return false;
}

bool llama_backend_policy_backend_matches(
        ggml_backend_t backend,
        const std::string & backend_name) {
    if (backend == nullptr || backend_name.empty()) {
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }

    if (str_eq_ci(backend_name, ggml_backend_dev_name(dev))) {
        return true;
    }
    if (name_matches_backend_type(dev, backend_name)) {
        return true;
    }

    // Allow op rules to refer to an extra buffer name such as HTP0-REPACK and
    // still resolve to the owning execution backend when needed.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (buft != nullptr && str_eq_ci(backend_name, ggml_backend_buft_name(buft))) {
        return true;
    }

    return extra_buft_matches(dev, backend_name);
}
