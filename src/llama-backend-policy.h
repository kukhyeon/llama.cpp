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

// Model-load residency for every FFN policy that can become active at runtime.
// Each entry is one connected union cover for a backend. Overlapping or
// touching profile intervals are merged, while genuinely disjoint intervals
// remain separate covers. The cover tensor is independently packed by its
// backend and runtime profiles select a logical sub-region with
// GGML_OP_MUL_MAT_SRC0_REGION.
struct llama_backend_policy_ffn_residency_plan {
    bool enabled = false;
    bool keep_full_source = true;
    std::vector<llama_backend_policy_ffn_split> covers;
    std::string source;
};

// Optional catalog for pre-built, layer-boundary backend routes. The catalog
// is inert unless runtime_routes.enabled is true in the policy. Profiles are
// kept in JSON order, while transitions contain only profiles reachable from
// initial_profile after load-time validation.
struct llama_backend_policy_runtime_route_transition {
    std::string from_profile;
    std::vector<std::string> to_profiles;
};

struct llama_backend_policy_runtime_routes {
    bool enabled = false;
    std::string mode = "off";
    std::string phase = "prefill";
    std::string initial_profile;
    std::vector<std::string> profiles;
    std::vector<llama_backend_policy_runtime_route_transition> transitions;
    std::vector<std::string> candidate_kinds;
    std::string boundary_node = "l_out";
    std::string boundary_backend = "auto";
    std::string boundary_granularity = "layer";
    std::string output_mode = "canonical";
    int min_dwell_layers = 1;
    bool strict = false;
};

struct llama_backend_policy_runtime_route_selection {
    bool enabled = false;
    bool matched = false;
    std::string profile;
    double distance = -1.0;
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

// Return the enabled runtime-route catalog when it applies to this phase.
// With no runtime_routes section, or with enabled=false, these APIs return
// false and leave their output empty so the legacy query-boundary path is
// unchanged.
bool llama_backend_policy_resolve_runtime_routes(
        bool is_prefill,
        llama_backend_policy_runtime_routes & out);

bool llama_backend_policy_list_runtime_route_profiles(
        bool is_prefill,
        std::vector<std::string> & out);

// List the current route and its direct outgoing transitions. If current is
// null/empty, the whole validated reachable catalog is returned.
bool llama_backend_policy_list_runtime_route_next_profiles(
        const char * current_profile,
        bool is_prefill,
        std::vector<std::string> & out);

// Match ops against a named profile without mutating the process-global active
// profile. This is used while preparing multiple candidate graphs.
bool llama_backend_policy_match_op_for_profile(
        const char * profile_name,
        const char * base_name,
        const char * tensor_name,
        enum ggml_op op,
        int il,
        bool is_prefill,
        llama_backend_policy_match & out);

// Select the nearest clock-point route from the current route plus its direct
// successors. Passing no current route selects from the full reachable
// catalog. Unlike the legacy FFN selector, profiles do not need ffn_parallel.
bool llama_backend_policy_select_runtime_route_profile(
        const char * current_profile,
        bool is_prefill,
        int32_t input_tokens,
        int32_t ubatch_tokens,
        int64_t gold_khz,
        int64_t prime_khz,
        int64_t gpu_hz,
        llama_backend_policy_runtime_route_selection & out);

bool llama_backend_policy_match_ffn_parallel(
        int il,
        bool is_prefill,
        llama_backend_policy_ffn_parallel & out);

// Resolve one prepared profile without mutating the legacy process-global
// active profile. Layer-boundary plans use this while constructing all FFN
// variants up front.
bool llama_backend_policy_match_ffn_parallel_for_profile(
        const char * profile_name,
        int il,
        bool is_prefill,
        llama_backend_policy_ffn_parallel & out);

// Stable non-zero identifier shared by graph construction, scheduler route
// registration, and runtime profile publication.
uint64_t llama_backend_policy_runtime_route_plan_id(const char * profile_name);

bool llama_backend_policy_match_ffn_parallel_layer(
        int il,
        llama_backend_policy_ffn_parallel & out);

// Return every enabled base/profile FFN policy whose split ranges contribute
// to resident covers for this layer. Policies with identical ranges/backends
// are deduplicated because their model-load residency requirements are equal.
bool llama_backend_policy_list_ffn_parallel_load_policies(
        int il,
        std::vector<llama_backend_policy_ffn_parallel> & out);

// Build the union residency plan for one layer. Intervals are grouped by
// backend and merged into connected covers, while uncovered gaps are not
// allocated. keep_full_source is false only when every applicable policy has
// phase="all".
bool llama_backend_policy_build_ffn_residency_plan(
        int il,
        llama_backend_policy_ffn_residency_plan & out);

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
