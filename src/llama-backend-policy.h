#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
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

struct llama_backend_policy_ffn_split {
    std::string id;
    int64_t start = 0;
    int64_t size = 0;
    std::string backend;
};

struct llama_backend_policy_ffn_parallel {
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

bool llama_backend_policy_weights_enabled();
bool llama_backend_policy_ops_enabled();
bool llama_backend_policy_residency_enabled();
bool llama_backend_policy_ffn_parallel_enabled();
int  llama_backend_policy_ffn_parallel_reduce_threads();

// Update the process-global runtime profile from the current execution phase
// and optional environment/config driven switches. Returns true when the active
// profile changed and any reusable graph should be rebuilt.
bool llama_backend_policy_update_runtime_profile(bool is_prefill);
const char * llama_backend_policy_active_profile();

// Match a model weight tensor name against weights.default / weights.rules.
// The returned backend names are buffer-type oriented, e.g. HTP0-REPACK or CPU.
bool llama_backend_policy_match_weight(
        const char * tensor_name,
        llama_backend_policy_match & out);

// Match a tensor against residency.rules. The returned backend names are the
// copies that should be created at model-load time, e.g. CPU and HTP0-REPACK.
bool llama_backend_policy_match_residency(
        const char * tensor_name,
        llama_backend_policy_match & out);

// Match the active profile's weight rules for graph-time source selection.
// This does not affect the primary tensor placement chosen at model load.
bool llama_backend_policy_match_profile_weight(
        const char * tensor_name,
        bool is_prefill,
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

bool llama_backend_policy_match_ffn_parallel(
        int il,
        bool is_prefill,
        llama_backend_policy_ffn_parallel & out);

bool llama_backend_policy_match_ffn_parallel_layer(
        int il,
        llama_backend_policy_ffn_parallel & out);

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
