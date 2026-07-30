#include "llama.h"
#include "llama-backend-policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class scoped_environment {
public:
    scoped_environment(const char * name, const char * value) : name_(name) {
        const char * previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        set(value);
    }

    ~scoped_environment() {
        if (had_previous_) {
            set(previous_.c_str());
        } else {
            unset();
        }
    }

    scoped_environment(const scoped_environment &) = delete;
    scoped_environment & operator=(const scoped_environment &) = delete;

private:
    void set(const char * value) {
#if defined(_WIN32)
        if (_putenv_s(name_.c_str(), value) != 0) {
            throw std::runtime_error("failed to set environment variable " + name_);
        }
#else
        if (setenv(name_.c_str(), value, 1) != 0) {
            throw std::runtime_error("failed to set environment variable " + name_);
        }
#endif
    }

    void unset() {
#if defined(_WIN32)
        (void) _putenv_s(name_.c_str(), "");
#else
        (void) unsetenv(name_.c_str());
#endif
    }

    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

class temporary_policy_file {
public:
    temporary_policy_file(const char * name, const char * contents) : path_(name) {
        std::ofstream file(path_, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("failed to create temporary policy: " + path_);
        }
        file << contents;
        if (!file) {
            throw std::runtime_error("failed to write temporary policy: " + path_);
        }
    }

    ~temporary_policy_file() {
        (void) std::remove(path_.c_str());
    }

    temporary_policy_file(const temporary_policy_file &) = delete;
    temporary_policy_file & operator=(const temporary_policy_file &) = delete;

    const char * path() const {
        return path_.c_str();
    }

private:
    std::string path_;
};

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << __FILE__ << ":" << __LINE__ \
                      << ": check failed: " #condition << std::endl; \
            return false; \
        } \
    } while (false)

bool selected_profile(
        const llama_backend_policy_ffn_clock_result & result,
        const char * expected,
        double expected_distance = -1.0) {
    CHECK(result.enabled);
    CHECK(result.matched);
    CHECK(std::string(result.profile) == expected);
    CHECK(result.distance >= 0.0);
    if (expected_distance >= 0.0) {
        CHECK(std::fabs(result.distance - expected_distance) < 1.0e-12);
    }
    return true;
}

bool test_clock_profile_selection(const char * policy_path) {
    llama_backend_policy_clear();

    CHECK(!llama_backend_policy_ffn_clock_switch_enabled());
    const auto before_load = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, 1000000);
    CHECK(!before_load.enabled);
    CHECK(!before_load.matched);
    CHECK(!before_load.pending_changed);

    CHECK(llama_backend_policy_load(policy_path, false, false));
    CHECK(llama_backend_policy_ffn_clock_switch_enabled());

    // Exact matches establish the selected profile and have zero distance.
    const auto exact_slow = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, 1000000);
    CHECK(selected_profile(exact_slow, "slow", 0.0));
    CHECK(exact_slow.pending_changed);

    // Repeating the same boundary observation must not invalidate the graph.
    const auto repeated_slow = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, 1000000);
    CHECK(selected_profile(repeated_slow, "slow", 0.0));
    CHECK(!repeated_slow.pending_changed);

    // A non-throttled query explicitly returns to the root/base FFN policy.
    CHECK(llama_backend_policy_reset_ffn_clock_profile());
    CHECK(!llama_backend_policy_reset_ffn_clock_profile());
    const auto slow_after_reset = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, 1000000);
    CHECK(selected_profile(slow_after_reset, "slow", 0.0));
    CHECK(slow_after_reset.pending_changed);

    // Invalid clock readings represent a transient sysfs failure. They must
    // leave the last pending selection intact so the next query can retry.
    const auto invalid_clock = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, -1);
    CHECK(invalid_clock.enabled);
    CHECK(!invalid_clock.matched);
    CHECK(!invalid_clock.pending_changed);
    const auto after_invalid_clock = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, 1000000);
    CHECK(selected_profile(after_invalid_clock, "slow", 0.0));
    CHECK(!after_invalid_clock.pending_changed);

    // The nearest point is selected using the sum of relative errors, so GPU
    // Hz and CPU kHz contribute on equal normalized terms.
    const auto near_fast = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 2800, 2800, 2800000);
    CHECK(selected_profile(near_fast, "fast"));
    CHECK(near_fast.pending_changed);

    // Prime and Gold are part of one joint point, not independently snapped
    // to frequency tables that could form a profile that was never measured.
    const auto prime_fast = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1100, 2900, 2900000);
    CHECK(selected_profile(prime_fast, "prime_fast"));
    CHECK(prime_fast.pending_changed);

    const auto gold_fast = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 2900, 1100, 2900000);
    CHECK(selected_profile(gold_fast, "gold_fast"));
    CHECK(gold_fast.pending_changed);

    // At equal normalized distance, prefer the lower total clock point. The
    // profile names deliberately sort in the opposite order (fast < slow).
    const auto tied = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1500, 1500, 1500000);
    CHECK(selected_profile(tied, "slow", 1.5));
    CHECK(tied.pending_changed);

    // A profile without an ubatch range accepts any positive ubatch, while
    // input token applicability still isolates it from the 10-token profiles.
    const auto optional_ubatch = llama_backend_policy_select_ffn_clock_profile(
            32, 128, 1500, 1500, 1500000);
    CHECK(selected_profile(optional_ubatch, "token_32", 0.0));
    CHECK(optional_ubatch.pending_changed);

    // A valid observation with no token-compatible profile clears the pending
    // clock selection, allowing the normal runtime fallback to take effect.
    const auto no_token_match = llama_backend_policy_select_ffn_clock_profile(
            10, 9, 1000, 1000, 1000000);
    CHECK(no_token_match.enabled);
    CHECK(!no_token_match.matched);
    CHECK(no_token_match.pending_changed);
    CHECK(no_token_match.distance < 0.0);
    CHECK(no_token_match.profile[0] == '\0');

    const auto reselect_after_clear = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, 1000000);
    CHECK(selected_profile(reselect_after_clear, "slow", 0.0));
    CHECK(reselect_after_clear.pending_changed);

    return true;
}

bool test_failed_load_is_transactional(const char * invalid_policy_path) {
    // A malformed replacement must not discard the already loaded policy.
    CHECK(!llama_backend_policy_load(invalid_policy_path, false, false));
    CHECK(llama_backend_policy_ffn_clock_switch_enabled());

    const auto retained = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, 1000000);
    CHECK(selected_profile(retained, "slow", 0.0));
    CHECK(!retained.pending_changed);

    llama_backend_policy_clear();
    CHECK(!llama_backend_policy_ffn_clock_switch_enabled());
    const auto after_clear = llama_backend_policy_select_ffn_clock_profile(
            10, 4, 1000, 1000, 1000000);
    CHECK(!after_clear.enabled);
    CHECK(!after_clear.matched);
    return true;
}

bool test_cover_residency_plan(const char * union_policy_path) {
    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(union_policy_path, false, false));

    llama_backend_policy_ffn_residency_plan plan;
    CHECK(llama_backend_policy_build_ffn_residency_plan(0, plan));
    CHECK(plan.enabled);
    CHECK(!plan.keep_full_source);

    auto backend_covers = [&](const char * backend) {
        std::vector<llama_backend_policy_ffn_split> result;
        for (const auto & cover : plan.covers) {
            if (cover.backend == backend) {
                result.push_back(cover);
            }
        }
        std::sort(result.begin(), result.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.start < rhs.start;
        });
        return result;
    };

    auto check_contiguous_union = [&](const char * backend, int64_t expected_start, int64_t expected_end, int64_t & payload) {
        const auto covers = backend_covers(backend);
        CHECK(!covers.empty());
        int64_t covered = expected_start;
        payload = 0;
        for (const auto & cover : covers) {
            CHECK(cover.start == covered);
            CHECK(cover.size > 0);
            covered += cover.size;
            payload += cover.size;
        }
        CHECK(covered == expected_end);
        return true;
    };

    int64_t npu_payload = 0;
    int64_t gpu_payload = 0;
    int64_t cpu_payload = 0;
    CHECK(check_contiguous_union("HTP0-REPACK", 0, 5888, npu_payload));
    CHECK(check_contiguous_union("OpenCL", 4096, 6976, gpu_payload));
    CHECK(check_contiguous_union("CPU_REPACK", 5504, 8192, cpu_payload));
    CHECK(npu_payload == 5888);
    CHECK(gpu_payload == 2880);
    CHECK(cpu_payload == 2688);
    CHECK(npu_payload + gpu_payload + cpu_payload == 11456);
    CHECK(std::fabs((double) (npu_payload + gpu_payload + cpu_payload) / 8192.0 - 1.3984375) < 1.0e-12);

    // All overlapping/touching profile ranges for a processor collapse into
    // one independently packed cover.
    const auto npu_covers = backend_covers("HTP0-REPACK");
    const auto gpu_covers = backend_covers("OpenCL");
    const auto cpu_covers = backend_covers("CPU_REPACK");
    CHECK(npu_covers.size() == 1);
    CHECK(npu_covers[0].start == 0 && npu_covers[0].size == 5888);
    CHECK(gpu_covers.size() == 1);
    CHECK(gpu_covers[0].start == 4096 && gpu_covers[0].size == 2880);
    CHECK(cpu_covers.size() == 1);
    CHECK(cpu_covers[0].start == 5504 && cpu_covers[0].size == 2688);

    // A union is not a convex hull. Preserve the intentionally uncovered gap.
    const auto disjoint = backend_covers("DISJOINT");
    CHECK(disjoint.size() == 2);
    CHECK(disjoint[0].start == 0 && disjoint[0].size == 64);
    CHECK(disjoint[1].start == 8128 && disjoint[1].size == 64);

    llama_backend_policy_ffn_parallel active;
    CHECK(llama_backend_policy_match_ffn_parallel(0, true, active));
    CHECK(llama_backend_policy_match_ffn_parallel(0, false, active));
    CHECK(!llama_backend_policy_build_ffn_residency_plan(1, plan));

    llama_backend_policy_clear();
    return true;
}

bool test_profile_only_residency_keeps_full_source(const char * policy_path) {
    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(policy_path, false, false));

    llama_backend_policy_ffn_residency_plan plan;
    CHECK(llama_backend_policy_build_ffn_residency_plan(0, plan));
    CHECK(plan.enabled);
    CHECK(plan.keep_full_source);

    llama_backend_policy_clear();
    return true;
}

bool test_runtime_route_catalog(const char * policy_path) {
    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(policy_path, false, true));

    llama_backend_policy_runtime_routes routes;
    CHECK(llama_backend_policy_resolve_runtime_routes(true, routes));
    CHECK(routes.enabled);
    CHECK(routes.mode == "clock");
    CHECK(routes.phase == "prefill");
    CHECK(routes.initial_profile == "cool");
    CHECK(routes.profiles == std::vector<std::string>({ "cool", "warm", "hot" }));
    CHECK(routes.candidate_kinds == std::vector<std::string>({ "weightless_stateless" }));
    CHECK(routes.boundary_node == "l_out");
    CHECK(routes.boundary_backend == "CPU");
    CHECK(routes.boundary_granularity == "layer");
    CHECK(routes.output_mode == "canonical");
    CHECK(routes.min_dwell_layers == 2);
    CHECK(routes.strict);
    CHECK(!llama_backend_policy_resolve_runtime_routes(false, routes));
    CHECK(!routes.enabled);

    std::vector<std::string> profiles;
    CHECK(llama_backend_policy_list_runtime_route_profiles(true, profiles));
    CHECK(profiles == std::vector<std::string>({ "cool", "warm", "hot" }));
    CHECK(!llama_backend_policy_list_runtime_route_profiles(false, profiles));
    CHECK(profiles.empty());

    CHECK(llama_backend_policy_list_runtime_route_next_profiles("cool", true, profiles));
    CHECK(profiles == std::vector<std::string>({ "cool", "warm" }));
    CHECK(llama_backend_policy_list_runtime_route_next_profiles("warm", true, profiles));
    CHECK(profiles == std::vector<std::string>({ "warm", "cool", "hot" }));
    CHECK(llama_backend_policy_list_runtime_route_next_profiles(nullptr, true, profiles));
    CHECK(profiles == std::vector<std::string>({ "cool", "warm", "hot" }));
    CHECK(!llama_backend_policy_list_runtime_route_next_profiles("unknown", true, profiles));
    CHECK(profiles.empty());

    llama_backend_policy_match op_match;
    CHECK(llama_backend_policy_match_op_for_profile(
            "warm", "attn_rms_norm", "attn_rms_norm-1",
            GGML_OP_RMS_NORM, 1, true, op_match));
    CHECK(op_match.matched);
    CHECK(!op_match.backends.empty());
    CHECK(op_match.backends.front() == "HTP0");
    CHECK(!llama_backend_policy_match_op_for_profile(
            "warm", "other", "other-1", GGML_OP_ADD, 1, true, op_match));
    CHECK(!llama_backend_policy_match_op_for_profile(
            "warm", "ffn_rms_norm", "ffn_rms_norm-1",
            GGML_OP_RMS_NORM, 1, true, op_match));
    CHECK(!llama_backend_policy_match_op_for_profile(
            "unknown", "", "", GGML_OP_RMS_NORM, 1, true, op_match));

    const std::string active_before = llama_backend_policy_active_profile();
    llama_backend_policy_runtime_route_selection selection;

    // The globally closest hot profile is not a direct transition from cool,
    // so the selector advances only one edge to warm.
    CHECK(llama_backend_policy_select_runtime_route_profile(
            "cool", true, 10, 4, 3000, 3000, 3000000, selection));
    CHECK(selection.enabled);
    CHECK(selection.matched);
    CHECK(selection.profile == "warm");

    CHECK(llama_backend_policy_select_runtime_route_profile(
            "warm", true, 10, 4, 3000, 3000, 3000000, selection));
    CHECK(selection.profile == "hot");
    CHECK(std::fabs(selection.distance) < 1.0e-12);

    // Recovery follows the direct transition graph instead of attempting an
    // illegal hot -> cool reset in one step. warm=4000 is deliberately farther
    // from the cool=1000 sample than hot=3000, proving selection targets the
    // global optimum and takes its shortest-path hop rather than getting stuck
    // in a local greedy minimum.
    CHECK(llama_backend_policy_select_runtime_route_profile(
            "hot", true, 10, 4, 1000, 1000, 1000000, selection));
    CHECK(selection.profile == "warm");
    CHECK(llama_backend_policy_select_runtime_route_profile(
            "warm", true, 10, 4, 1000, 1000, 1000000, selection));
    CHECK(selection.profile == "cool");

    CHECK(llama_backend_policy_select_runtime_route_profile(
            nullptr, true, 10, 4, 1000, 1000, 1000000, selection));
    CHECK(selection.profile == "cool");

    CHECK(!llama_backend_policy_select_runtime_route_profile(
            "cool", true, 10, 4, -1, 1000, 1000000, selection));
    CHECK(selection.enabled);
    CHECK(!selection.matched);

    CHECK(!llama_backend_policy_select_runtime_route_profile(
            "cool", false, 10, 4, 1000, 1000, 1000000, selection));
    CHECK(!selection.enabled);

    // Candidate selection is intentionally stateless: it must not reuse the
    // legacy pending/active FFN profile state.
    CHECK(std::string(llama_backend_policy_active_profile()) == active_before);

    llama_backend_policy_clear();
    return true;
}

bool test_disabled_runtime_routes_are_inert(const char * policy_path) {
    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(policy_path, false, false));

    llama_backend_policy_runtime_routes routes;
    CHECK(!llama_backend_policy_resolve_runtime_routes(true, routes));
    CHECK(!routes.enabled);

    std::vector<std::string> profiles;
    CHECK(!llama_backend_policy_list_runtime_route_profiles(true, profiles));
    CHECK(profiles.empty());

    llama_backend_policy_clear();
    return true;
}

bool test_fixed_runtime_routes_do_not_clock_select(const char * policy_path) {
    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(policy_path, false, false));

    llama_backend_policy_runtime_routes routes;
    CHECK(llama_backend_policy_resolve_runtime_routes(true, routes));
    CHECK(routes.enabled);
    CHECK(routes.mode == "fixed");
    CHECK(routes.initial_profile == "only");

    llama_backend_policy_runtime_route_selection selection;
    CHECK(!llama_backend_policy_select_runtime_route_profile(
            "only", true, 10, 4, 1000, 1000, 1000000, selection));
    CHECK(!selection.enabled);
    CHECK(!selection.matched);

    llama_backend_policy_clear();
    return true;
}

bool test_invalid_runtime_routes_are_transactional(
        const char * valid_policy_path,
        const std::vector<const char *> & invalid_policy_paths) {
    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(valid_policy_path, false, true));

    for (const char * invalid_policy_path : invalid_policy_paths) {
        CHECK(!llama_backend_policy_load(invalid_policy_path, false, true));

        llama_backend_policy_runtime_routes routes;
        CHECK(llama_backend_policy_resolve_runtime_routes(true, routes));
        CHECK(routes.initial_profile == "cool");
        CHECK(routes.profiles.size() == 3);
    }

    llama_backend_policy_clear();
    return true;
}

bool test_compact_unified_ffn_routes(const char * policy_path) {
    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(policy_path, false, true));

    llama_backend_policy_runtime_routes routes;
    CHECK(llama_backend_policy_resolve_runtime_routes(true, routes));
    CHECK(routes.initial_profile == "cool");
    CHECK(routes.profiles == std::vector<std::string>({ "cool", "hot" }));
    CHECK(routes.candidate_kinds ==
            std::vector<std::string>({ "weightless_stateless", "weighted_norm", "ffn_block" }));

    std::vector<std::string> next;
    CHECK(llama_backend_policy_list_runtime_route_next_profiles("cool", true, next));
    CHECK(next == std::vector<std::string>({ "cool", "hot" }));
    CHECK(llama_backend_policy_list_runtime_route_next_profiles("hot", true, next));
    CHECK(next == std::vector<std::string>({ "hot", "cool" }));

    llama_backend_policy_ffn_parallel cool;
    llama_backend_policy_ffn_parallel hot;
    CHECK(llama_backend_policy_match_ffn_parallel_for_profile("cool", 0, true, cool));
    CHECK(llama_backend_policy_match_ffn_parallel_for_profile("hot", 0, true, hot));
    CHECK(cool.splits.size() == 2 && hot.splits.size() == 2);
    CHECK(cool.splits[0].id == "npu" && cool.splits[0].start == 0 && cool.splits[0].size == 64);
    CHECK(cool.splits[1].id == "gpu" && cool.splits[1].start == 64 && cool.splits[1].size == 128);
    CHECK(hot.splits[0].id == "npu" && hot.splits[0].start == 0 && hot.splits[0].size == 128);
    CHECK(hot.splits[1].id == "gpu" && hot.splits[1].start == 128 && hot.splits[1].size == 64);
    CHECK(cool.reduce_threads == 2 && hot.reduce_threads == 2);

    llama_backend_policy_match match;
    CHECK(llama_backend_policy_match_op_for_profile(
            "cool", "ffn_rms_norm", "ffn_rms_norm-0",
            GGML_OP_RMS_NORM, 0, true, match));
    CHECK(match.backends.front() == "OpenCL");
    CHECK(llama_backend_policy_match_op_for_profile(
            "hot", "ffn_rms_norm", "ffn_rms_norm-0",
            GGML_OP_RMS_NORM, 0, true, match));
    CHECK(match.backends.front() == "HTP0");
    CHECK(llama_backend_policy_match_op_for_profile(
            "hot", "ffn_norm", "ffn_norm-0",
            GGML_OP_MUL, 0, true, match));
    CHECK(match.backends.front() == "HTP0-REPACK");
    CHECK(llama_backend_policy_match_op_for_profile(
            "cool", "attn_norm", "attn_norm-0",
            GGML_OP_MUL, 0, true, match));
    CHECK(match.backends.front() == "CPU");
    CHECK(llama_backend_policy_match_op_for_profile(
            "hot", "attn_norm", "attn_norm-0",
            GGML_OP_MUL, 0, true, match));
    CHECK(match.backends.front() == "HTP0-REPACK");
    CHECK(llama_backend_policy_match_op_for_profile(
            "cool", "ffn_inp", "ffn_inp-0",
            GGML_OP_ADD, 0, true, match));
    CHECK(match.backends.front() == "CPU");
    CHECK(llama_backend_policy_match_op_for_profile(
            "hot", "ffn_inp", "ffn_inp-0",
            GGML_OP_ADD, 0, true, match));
    CHECK(match.backends.front() == "HTP0");
    CHECK(!llama_backend_policy_match_op_for_profile(
            "hot", "unrelated_mul", "unrelated_mul-0",
            GGML_OP_MUL, 0, true, match));

    CHECK(llama_backend_policy_match_residency("blk.0.ffn_norm.weight", match));
    CHECK(match.backends == std::vector<std::string>({ "CPU", "OpenCL", "HTP0-REPACK" }));
    CHECK(llama_backend_policy_match_residency("blk.0.attn_norm.weight", match));
    CHECK(match.backends == std::vector<std::string>({ "CPU", "OpenCL", "HTP0-REPACK" }));

    llama_backend_policy_ffn_residency_plan plan;
    CHECK(llama_backend_policy_build_ffn_residency_plan(0, plan));
    CHECK(!plan.keep_full_source);

    std::vector<llama_backend_policy_ffn_parallel> load_policies;
    CHECK(llama_backend_policy_list_ffn_parallel_load_policies(0, load_policies));
    for (const auto & policy : load_policies) {
        int64_t total = 0;
        for (const auto & split : policy.splits) {
            CHECK(split.start == total);
            total += split.size;
        }
        // The authored-but-unlisted profile below has a 256-wide cover. It
        // must not participate in the layer-route residency union.
        CHECK(total == 192);
    }

    const uint64_t cool_id = llama_backend_policy_runtime_route_plan_id("cool");
    const uint64_t hot_id = llama_backend_policy_runtime_route_plan_id("hot");
    CHECK(cool_id != 0 && hot_id != 0 && cool_id != hot_id);

    // The layer plan is the sole FFN profile authority. An old launcher may
    // still export FFN_CLOCK_SWITCH=1, but it must not reactivate the legacy
    // query-boundary graph rebuild path.
    CHECK(!llama_backend_policy_ffn_clock_switch_enabled());
    const auto legacy = llama_backend_policy_select_ffn_clock_profile(
            8, 8, 1000, 1000, 1000000);
    CHECK(!legacy.enabled && !legacy.matched);
    CHECK(!llama_backend_policy_update_runtime_profile(true));

    llama_backend_policy_clear();
    return true;
}

} // namespace

int main() {
    static const char * valid_policy = R"JSON(
{
  "enabled": true,
  "ffn_clock_switch": {
    "enabled": true
  },
  "profiles": {
    "slow": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 0, "hz": 1000000 }
      },
      "applicability": {
        "input_tokens": [8, 16],
        "ubatch_tokens": [4, 8]
      },
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "cpu", "backend": "CPU", "start": 0, "size": 64 }
        ]
      }
    },
    "fast": {
      "clock_point": {
        "prime": { "index": 1, "khz": 3000 },
        "gold":  { "index": 1, "khz": 3000 },
        "gpu":   { "index": 1, "hz": 3000000 }
      },
      "applicability": {
        "input_tokens": [8, 16],
        "ubatch_tokens": [4, 8]
      },
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "cpu", "backend": "CPU", "start": 0, "size": 64 }
        ]
      }
    },
    "prime_fast": {
      "clock_point": {
        "prime": { "index": 2, "khz": 3000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 1, "hz": 3000000 }
      },
      "applicability": {
        "input_tokens": [8, 16],
        "ubatch_tokens": [4, 8]
      },
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "cpu", "backend": "CPU", "start": 0, "size": 64 }
        ]
      }
    },
    "gold_fast": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 2, "khz": 3000 },
        "gpu":   { "index": 1, "hz": 3000000 }
      },
      "applicability": {
        "input_tokens": [8, 16],
        "ubatch_tokens": [4, 8]
      },
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "cpu", "backend": "CPU", "start": 0, "size": 64 }
        ]
      }
    },
    "token_32": {
      "clock_point": {
        "prime": { "index": 3, "khz": 1500 },
        "gold":  { "index": 3, "khz": 1500 },
        "gpu":   { "index": 3, "hz": 1500000 }
      },
      "applicability": {
        "input_tokens": [32, 32]
      },
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "cpu", "backend": "CPU", "start": 0, "size": 64 }
        ]
      }
    }
  }
}
)JSON";

    static const char * invalid_policy = R"JSON(
{
  "enabled": true,
  "ffn_parallel": {
    "enabled": true,
    "phase": "prefill",
    "align": 64,
    "splits": [
      { "id": "first",  "backend": "CPU", "start": 0,   "size": 64 },
      { "id": "gapped", "backend": "CPU", "start": 128, "size": 64 }
    ]
  }
}
)JSON";

    static const char * union_policy = R"JSON(
{
  "enabled": true,
  "ffn_parallel": {
    "enabled": true,
    "phase": "all",
    "layer_range": [0, 0],
    "align": 64,
    "splits": [
      { "id": "npu", "backend": "HTP0-REPACK", "start": 0,    "size": 4096 },
      { "id": "gpu", "backend": "OpenCL",     "start": 4096, "size": 2048 },
      { "id": "cpu", "backend": "CPU_REPACK", "start": 6144, "size": 2048 }
    ]
  },
  "profiles": {
    "cover-left": {
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "npu", "backend": "HTP0-REPACK", "start": 0,    "size": 4864 },
          { "id": "gpu", "backend": "OpenCL",     "start": 4864, "size": 640 },
          { "id": "cpu", "backend": "CPU_REPACK", "start": 5504, "size": 2688 }
        ]
      }
    },
    "cover-right": {
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "npu", "backend": "HTP0-REPACK", "start": 0,    "size": 5888 },
          { "id": "gpu", "backend": "OpenCL",     "start": 5888, "size": 832 },
          { "id": "cpu", "backend": "CPU_REPACK", "start": 6720, "size": 1472 }
        ]
      }
    },
    "gpu-right": {
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "npu", "backend": "HTP0-REPACK", "start": 0,    "size": 5120 },
          { "id": "gpu", "backend": "OpenCL",     "start": 5120, "size": 1856 },
          { "id": "cpu", "backend": "CPU_REPACK", "start": 6976, "size": 1216 }
        ]
      }
    },
    "disjoint-prefix": {
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "disjoint", "backend": "DISJOINT", "start": 0,  "size": 64 },
          { "id": "other",    "backend": "OTHER",    "start": 64, "size": 8128 }
        ]
      }
    },
    "disjoint-suffix": {
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "other",    "backend": "OTHER",    "start": 0,    "size": 8128 },
          { "id": "disjoint", "backend": "DISJOINT", "start": 8128, "size": 64 }
        ]
      }
    }
  }
}
)JSON";

    static const char * profile_only_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "only": {
      "ffn_parallel": {
        "enabled": true,
        "phase": "all",
        "layer_range": [0, 0],
        "align": 64,
        "splits": [
          { "id": "cpu", "backend": "CPU", "start": 0, "size": 64 }
        ]
      }
    }
  }
}
)JSON";

    static const char * runtime_routes_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "cool": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 0, "hz": 1000000 }
      },
      "applicability": {
        "input_tokens": [8, 16],
        "ubatch_tokens": [4, 8]
      },
      "ops": {
        "enabled": true,
        "rules": [
          { "op": "RMS_NORM", "phase": "prefill", "backend": "CPU" }
        ]
      }
    },
    "warm": {
      "clock_point": {
        "prime": { "index": 1, "khz": 4000 },
        "gold":  { "index": 1, "khz": 4000 },
        "gpu":   { "index": 1, "hz": 4000000 }
      },
      "applicability": {
        "input_tokens": [8, 16],
        "ubatch_tokens": [4, 8]
      },
      "ops": {
        "enabled": true,
        "rules": [
          { "name": "attn_rms_norm", "op": "RMS_NORM", "phase": "prefill", "backend": "HTP0" }
        ]
      }
    },
    "hot": {
      "clock_point": {
        "prime": { "index": 2, "khz": 3000 },
        "gold":  { "index": 2, "khz": 3000 },
        "gpu":   { "index": 2, "hz": 3000000 }
      },
      "applicability": {
        "input_tokens": [8, 16],
        "ubatch_tokens": [4, 8]
      },
      "ops": {
        "enabled": true,
        "rules": [
          { "op": "RMS_NORM", "phase": "prefill", "backend": "OpenCL" }
        ]
      }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "clock",
    "phase": "prefill",
    "initial_profile": "cool",
    "profiles": ["cool", "warm", "hot"],
    "transitions": {
      "cool": ["warm"],
      "warm": ["cool", "hot"],
      "hot": ["warm"]
    },
    "candidate_kinds": ["weightless_stateless"],
    "boundary": {
      "node": "l_out",
      "backend": "CPU",
      "granularity": "layer"
    },
    "output_mode": "canonical",
    "min_dwell_layers": 2,
    "strict": true
  }
}
)JSON";

    static const char * disabled_runtime_routes_policy = R"JSON(
{
  "enabled": true,
  "runtime_routes": {
    "enabled": false,
    "profiles": "this disabled section is intentionally incomplete"
  }
}
)JSON";

    static const char * fixed_runtime_routes_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "only": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 0, "hz": 1000000 }
      },
      "applicability": {
        "input_tokens": [8, 16],
        "ubatch_tokens": [4, 8]
      }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "initial_profile": "only",
    "profiles": ["only"],
    "transitions": {}
  }
}
)JSON";

    static const char * unknown_runtime_route_profile_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "known": {}
  },
  "runtime_routes": {
    "enabled": true,
    "initial_profile": "known",
    "profiles": ["known", "missing"],
    "transitions": {
      "known": ["missing"]
    }
  }
}
)JSON";

    static const char * unreachable_runtime_route_profile_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "initial": {},
    "reachable": {},
    "orphan": {}
  },
  "runtime_routes": {
    "enabled": true,
    "initial_profile": "initial",
    "profiles": ["initial", "reachable", "orphan"],
    "transitions": {
      "initial": ["reachable"]
    }
  }
}
)JSON";

    static const char * unsupported_runtime_route_kind_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "only": {}
  },
  "runtime_routes": {
    "enabled": true,
    "initial_profile": "only",
    "profiles": ["only"],
    "transitions": {},
    "candidate_kinds": ["resident_weighted"]
  }
}
)JSON";

    static const char * sticky_runtime_route_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "cool": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 0, "hz": 1000000 }
      },
      "applicability": { "input_tokens": [8, 16] }
    },
    "hot": {
      "clock_point": {
        "prime": { "index": 1, "khz": 2000 },
        "gold":  { "index": 1, "khz": 2000 },
        "gpu":   { "index": 1, "hz": 2000000 }
      },
      "applicability": { "input_tokens": [8, 16] }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "clock",
    "phase": "prefill",
    "initial_profile": "cool",
    "profiles": ["cool", "hot"],
    "transitions": { "cool": ["hot"], "hot": [] }
  }
}
)JSON";

    static const char * decode_clock_runtime_route_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "only": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 0, "hz": 1000000 }
      },
      "applicability": { "input_tokens": [8, 16] }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "clock",
    "phase": "decode",
    "initial_profile": "only",
    "profiles": ["only"],
    "transitions": {}
  }
}
)JSON";

    static const char * ambiguous_runtime_route_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "a": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 0, "hz": 1000000 }
      },
      "applicability": { "input_tokens": [8, 16] }
    },
    "b": {
      "clock_point": {
        "prime": { "index": 1, "khz": 1000 },
        "gold":  { "index": 1, "khz": 1000 },
        "gpu":   { "index": 1, "hz": 1000000 }
      },
      "applicability": { "input_tokens": [12, 20] }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "clock",
    "phase": "prefill",
    "initial_profile": "a",
    "profiles": ["a", "b"],
    "transitions": { "a": ["b"], "b": ["a"] }
  }
}
)JSON";

    static const char * non_layer_boundary_clock_route_policy = R"JSON(
{
  "enabled": true,
  "profiles": {
    "cool": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 0, "hz": 1000000 }
      },
      "applicability": { "input_tokens": [8, 16] }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "clock",
    "phase": "prefill",
    "initial_profile": "cool",
    "profiles": ["cool"],
    "transitions": {},
    "boundary": { "node": "ffn_out", "granularity": "layer" }
  }
}
)JSON";

    static const char * compact_unified_ffn_route_policy = R"JSON(
{
  "enabled": true,
  "ffn_clock_switch": { "enabled": false },
  "ffn_parallel": {
    "enabled": true,
    "phase": "all",
    "layer_range": [0, 0],
    "align": 64,
    "reduce_backend": "CPU",
    "reduce_threads": 2,
    "split_layout": [
      { "id": "npu", "backend": "HTP0-REPACK" },
      { "id": "gpu", "backend": "OpenCL" }
    ],
    "split_sizes": { "npu": 64, "gpu": 128 }
  },
  "profile_defaults": {
    "applicability": {
      "input_tokens": [8, 8],
      "ubatch_tokens": [8, 8]
    },
    "ffn_parallel": {
      "enabled": true,
      "phase": "all",
      "layer_range": [0, 0],
      "align": 64,
      "reduce_backend": "CPU",
      "reduce_threads": 2,
      "split_layout": [
        { "id": "npu", "backend": "HTP0-REPACK" },
        { "id": "gpu", "backend": "OpenCL" }
      ]
    },
    "ops": {
      "enabled": true,
      "rules": [
        { "name": "attn_rms_norm", "op": "RMS_NORM", "phase": "prefill", "backend": "HTP0" },
        { "name": "attn_norm", "op": "MUL", "phase": "prefill", "backend": "HTP0-REPACK" },
        { "name": "ffn_rms_norm", "op": "RMS_NORM", "phase": "prefill", "backend": "HTP0" },
        { "name": "ffn_norm", "op": "MUL", "phase": "prefill", "backend": "HTP0-REPACK" },
        { "name": "ffn_inp", "op": "ADD", "phase": "prefill", "backend": "HTP0" }
      ]
    }
  },
  "profiles": {
    "cool": {
      "clock_point": {
        "prime": { "index": 0, "khz": 1000 },
        "gold":  { "index": 0, "khz": 1000 },
        "gpu":   { "index": 0, "hz": 1000000 }
      },
      "ffn_parallel": { "split_sizes": { "npu": 64, "gpu": 128 } },
      "ops": {
        "rules": [
          { "name": "attn_rms_norm", "op": "RMS_NORM", "phase": "prefill", "backend": "CPU" },
          { "name": "attn_norm", "op": "MUL", "phase": "prefill", "backend": "CPU" },
          { "name": "ffn_rms_norm", "op": "RMS_NORM", "phase": "prefill", "backend": "OpenCL" },
          { "name": "ffn_norm", "op": "MUL", "phase": "prefill", "backend": "HTP0-REPACK" },
          { "name": "ffn_inp", "op": "ADD", "phase": "prefill", "backend": "CPU" }
        ]
      }
    },
    "hot": {
      "clock_point": {
        "prime": { "index": 1, "khz": 2000 },
        "gold":  { "index": 1, "khz": 2000 },
        "gpu":   { "index": 1, "hz": 2000000 }
      },
      "ffn_parallel": { "split_sizes": { "npu": 128, "gpu": 64 } }
    },
    "unused": {
      "ffn_parallel": { "split_sizes": { "npu": 128, "gpu": 128 } }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "clock",
    "phase": "prefill",
    "initial_profile": "cool",
    "profiles": ["cool", "hot"],
    "transitions": "complete",
    "candidate_kinds": ["weightless_stateless", "weighted_norm", "ffn_block"],
    "boundary": { "node": "l_out", "backend": "auto", "granularity": "layer" },
    "output_mode": "canonical",
    "strict": true
  },
  "residency": {
    "enabled": true,
    "rules": [
      {
        "pattern": "^blk\\.[0-9]+\\.(attn|ffn)_norm\\.weight$",
        "copies": ["CPU", "OpenCL", "HTP0-REPACK"]
      }
    ]
  }
}
)JSON";

    try {
        scoped_environment clock_switch("LLAMA_BACKEND_POLICY_FFN_CLOCK_SWITCH", "1");
        scoped_environment ffn_parallel("LLAMA_FFN_PARALLEL", "1");
        temporary_policy_file valid_file("test-backend-policy-clock-valid.json", valid_policy);
        temporary_policy_file invalid_file("test-backend-policy-clock-invalid.json", invalid_policy);
        temporary_policy_file union_file("test-backend-policy-clock-union.json", union_policy);
        temporary_policy_file profile_only_file(
                "test-backend-policy-clock-profile-only.json", profile_only_policy);
        temporary_policy_file runtime_routes_file(
                "test-backend-policy-runtime-routes-valid.json", runtime_routes_policy);
        temporary_policy_file disabled_runtime_routes_file(
                "test-backend-policy-runtime-routes-disabled.json", disabled_runtime_routes_policy);
        temporary_policy_file fixed_runtime_routes_file(
                "test-backend-policy-runtime-routes-fixed.json", fixed_runtime_routes_policy);
        temporary_policy_file unknown_runtime_route_profile_file(
                "test-backend-policy-runtime-routes-unknown-profile.json", unknown_runtime_route_profile_policy);
        temporary_policy_file unreachable_runtime_route_profile_file(
                "test-backend-policy-runtime-routes-unreachable.json", unreachable_runtime_route_profile_policy);
        temporary_policy_file unsupported_runtime_route_kind_file(
                "test-backend-policy-runtime-routes-kind.json", unsupported_runtime_route_kind_policy);
        temporary_policy_file sticky_runtime_route_file(
                "test-backend-policy-runtime-routes-sticky.json", sticky_runtime_route_policy);
        temporary_policy_file decode_clock_runtime_route_file(
                "test-backend-policy-runtime-routes-decode-clock.json", decode_clock_runtime_route_policy);
        temporary_policy_file ambiguous_runtime_route_file(
                "test-backend-policy-runtime-routes-ambiguous.json", ambiguous_runtime_route_policy);
        temporary_policy_file non_layer_boundary_clock_route_file(
                "test-backend-policy-runtime-routes-non-layer-boundary.json",
                non_layer_boundary_clock_route_policy);
        temporary_policy_file compact_unified_ffn_route_file(
                "test-backend-policy-runtime-routes-compact-ffn.json",
                compact_unified_ffn_route_policy);

        if (!test_clock_profile_selection(valid_file.path())) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_failed_load_is_transactional(invalid_file.path())) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_cover_residency_plan(union_file.path())) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_profile_only_residency_keeps_full_source(profile_only_file.path())) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_runtime_route_catalog(runtime_routes_file.path())) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_disabled_runtime_routes_are_inert(disabled_runtime_routes_file.path())) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_fixed_runtime_routes_do_not_clock_select(fixed_runtime_routes_file.path())) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_invalid_runtime_routes_are_transactional(
                runtime_routes_file.path(),
                {
                    unknown_runtime_route_profile_file.path(),
                    unreachable_runtime_route_profile_file.path(),
                    unsupported_runtime_route_kind_file.path(),
                    sticky_runtime_route_file.path(),
                    decode_clock_runtime_route_file.path(),
                    ambiguous_runtime_route_file.path(),
                    non_layer_boundary_clock_route_file.path(),
                })) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_compact_unified_ffn_routes(compact_unified_ffn_route_file.path())) {
            llama_backend_policy_clear();
            return 1;
        }
    } catch (const std::exception & error) {
        llama_backend_policy_clear();
        std::cerr << "test-backend-policy-clock: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
