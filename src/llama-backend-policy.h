#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <string>
#include <vector>

struct llama_backend_policy_match {
    bool matched = false;
    // Ordered backend/buffer names to try. The first entries are rule-specific,
    // followed by global fallback_priority from the policy file.
    std::vector<std::string> backends;
    // Human-readable rule origin, e.g. "weights.rules[3]", for debug logs.
    std::string source;
};

bool llama_backend_policy_weights_enabled();
bool llama_backend_policy_ops_enabled();

// Match a model weight tensor name against weights.default / weights.rules.
// The returned backend names are buffer-type oriented, e.g. HTP0-REPACK or CPU.
bool llama_backend_policy_match_weight(
        const char * tensor_name,
        llama_backend_policy_match & out);

// Match a graph op against ops.rules. base_name is the callback name before
// layer suffixing ("Qcur"), while tensor_name is the final graph node name
// ("Qcur-12"). The returned backend names are execution backends, e.g. HTP0.
bool llama_backend_policy_match_op(
        const char * base_name,
        const char * tensor_name,
        enum ggml_op op,
        int il,
        bool is_prefill,
        llama_backend_policy_match & out);

// Match a requested policy name to a backend buffer type. This is used during
// model loading, where weights are assigned to buffer types rather than to
// already-created backend instances.
bool llama_backend_policy_buft_matches(
        ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft,
        const std::string & backend_name);

// Match a requested policy name to a concrete backend instance. This is used
// during graph construction for op scheduling hints.
bool llama_backend_policy_backend_matches(
        ggml_backend_t backend,
        const std::string & backend_name);
