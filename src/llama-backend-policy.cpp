#include "llama-backend-policy.h"

#include "llama-impl.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

struct profile_policy {
    bool weights_enabled = false;
    bool ops_enabled = false;
    std::vector<std::string> default_weight_backends;
    std::vector<policy_rule> weight_rules;
    std::vector<policy_rule> op_rules;
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

    if (!root.contains("splits") || !root["splits"].is_array()) {
        throw std::runtime_error(source + ".splits must be an array");
    }

    int i = 0;
    for (const auto & item : root["splits"]) {
        if (!item.is_object()) {
            throw std::runtime_error(source + ".splits entries must be objects");
        }
        llama_backend_policy_ffn_split split;
        split.id = item.value("id", std::string("s") + std::to_string(i));
        split.start = item.value("start", (int64_t) -1);
        split.size = item.value("size", (int64_t) -1);
        split.backend = item.value("backend", std::string());
        if (split.start < 0 || split.size <= 0 || split.backend.empty()) {
            throw std::runtime_error(source + ".splits[" + std::to_string(i) + "] requires backend, start, and positive size");
        }
        out.splits.push_back(std::move(split));
        ++i;
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
        if (split.backend.empty() || end == size_str.c_str() || *end != '\0' || size <= 0) {
            LLAMA_LOG_WARN("backend_policy: invalid LLAMA_FFN_PARALLEL_SPLITS item '%s'; expected backend:positive_size\n", item.c_str());
            splits.clear();
            return false;
        }

        split.start = start;
        split.size = size;
        start += split.size;
        splits.push_back(std::move(split));
        ++idx;
    }

    return !splits.empty();
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

bool effective_ffn_policy(llama_backend_policy_ffn_parallel & out) {
    ffn_parallel_policy policy = g_policy.ffn_parallel;

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

        if (root.contains("profiles") && root["profiles"].is_object()) {
            for (auto it = root["profiles"].begin(); it != root["profiles"].end(); ++it) {
                if (!it.value().is_object()) {
                    throw std::runtime_error("profiles entries must be objects");
                }
                profile_policy profile;
                parse_profile_section(it.value(), it.key(), next.enabled, enable_weights, enable_ops, profile);
                next.profiles.emplace(it.key(), std::move(profile));
            }
        }

        if (!root.contains("weights")) {
            next.weights_enabled = false;
        }
        if (!root.contains("ops")) {
            next.ops_enabled = false;
        }

        g_policy = std::move(next);

        LLAMA_LOG_INFO("backend_policy: loaded '%s' (weights=%s, weight_rules=%zu, ops=%s, op_rules=%zu, residency=%s, residency_rules=%zu, ffn_parallel=%s, profiles=%zu)\n",
                g_policy.path.c_str(),
                g_policy.weights_enabled ? "on" : "off", g_policy.weight_rules.size(),
                g_policy.ops_enabled ? "on" : "off", g_policy.op_rules.size(),
                g_policy.residency_enabled ? "on" : "off", g_policy.residency_rules.size(),
                g_policy.ffn_parallel.enabled ? "on" : "off",
                g_policy.profiles.size());
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

int llama_backend_policy_ffn_parallel_reduce_threads() {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    llama_backend_policy_ffn_parallel policy;
    return g_policy.loaded && effective_ffn_policy(policy) && policy.splits.size() >= 2
        ? policy.reduce_threads
        : 0;
}

bool llama_backend_policy_update_runtime_profile(bool is_prefill) {
    std::lock_guard<std::mutex> lock(g_policy_mutex);

    if (!g_policy.loaded || !g_policy.enabled) {
        return false;
    }

    std::string next_profile = env_string("LLAMA_BACKEND_POLICY_ACTIVE_PROFILE");

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
