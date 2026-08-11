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

bool test_attn_qkv_shards_policy() {
    static constexpr char valid_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "phase": "prefill",
    "layer_range": [2, 4],
    "head_dim": 128,
    "assemble_backend": "OpenCL",
    "split_layout": [
      { "id": "npu", "backend": "HTP0-REPACK" },
      { "id": "gpu", "backend": "OpenCL", "align": 64 },
      { "id": "cpu", "backend": "CPU", "align": 64 }
    ],
    "split_sizes": {
      "q": { "npu": 512, "gpu": 256, "cpu": 256 },
      "k": { "npu": 256, "gpu": 128, "cpu": 128 },
      "v": { "npu": 256, "gpu": 128, "cpu": 128 }
    }
  }
}
)JSON";

    // The HTP lane is implicitly at least 256-wide, even without an explicit
    // layout align. A 128-wide HTP shard must therefore fail transactionally.
    static constexpr char invalid_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "npu", "backend": "HTP0-REPACK" },
      { "id": "gpu", "backend": "OpenCL" },
      { "id": "cpu", "backend": "CPU" }
    ],
    "split_sizes": {
      "q": { "npu": 128, "gpu": 128, "cpu": 128 },
      "k": { "npu": 128, "gpu": 128, "cpu": 128 },
      "v": { "npu": 128, "gpu": 128, "cpu": 128 }
    }
  }
}
)JSON";

    temporary_policy_file valid_file("test-backend-policy-attn-qkv-valid.json", valid_policy);
    temporary_policy_file invalid_file("test-backend-policy-attn-qkv-invalid.json", invalid_policy);

    llama_backend_policy_clear();
    CHECK(!llama_backend_policy_attn_qkv_shards_enabled());
    CHECK(llama_backend_policy_load(valid_file.path(), false, false));
    CHECK(llama_backend_policy_attn_qkv_shards_enabled());

    llama_backend_policy_attn_qkv_shards policy;
    CHECK(!llama_backend_policy_match_attn_qkv_shards(1, true, policy));
    CHECK(!llama_backend_policy_match_attn_qkv_shards(2, false, policy));
    CHECK(llama_backend_policy_match_attn_qkv_shards(2, true, policy));
    CHECK(policy.enabled);
    CHECK(policy.phase == "prefill");
    CHECK(policy.layer_start == 2);
    CHECK(policy.layer_end == 4);
    CHECK(policy.head_dim == 128);
    CHECK(policy.assemble_backend == "OpenCL");
    CHECK(policy.splits.size() == 3);
    CHECK(policy.splits[0].id == "npu");
    CHECK(policy.splits[0].backend == "HTP0-REPACK");
    CHECK(policy.splits[0].q_start == 0 && policy.splits[0].q_size == 512);
    CHECK(policy.splits[1].q_start == 512 && policy.splits[1].q_size == 256);
    CHECK(policy.splits[2].q_start == 768 && policy.splits[2].q_size == 256);
    CHECK(policy.splits[1].k_start == 256 && policy.splits[1].k_size == 128);
    CHECK(policy.splits[2].v_start == 384 && policy.splits[2].v_size == 128);

    CHECK(!llama_backend_policy_load(invalid_file.path(), false, false));
    CHECK(llama_backend_policy_attn_qkv_shards_enabled());
    CHECK(llama_backend_policy_match_attn_qkv_shards(3, true, policy));
    CHECK(policy.splits[0].q_size == 512);

    llama_backend_policy_clear();
    CHECK(!llama_backend_policy_attn_qkv_shards_enabled());
    return true;
}

bool test_profile_attn_qkv_shards_policy() {
    static constexpr char routed_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "phase": "prefill",
    "layer_range": [0, 27],
    "head_dim": 128,
    "assemble_backend": "OpenCL",
    "split_layout": [
      { "id": "cpu", "backend": "CPU", "align": 128 },
      { "id": "gpu", "backend": "OpenCL", "align": 128 },
      { "id": "npu", "backend": "HTP0-REPACK" }
    ],
    "split_sizes": {
      "q": { "cpu": 256, "gpu": 256, "npu": 1024 },
      "k": { "cpu": 256, "gpu": 256, "npu": 512 },
      "v": { "cpu": 256, "gpu": 256, "npu": 512 }
    }
  },
  "profiles": {
    "shifted": {
      "attn_qkv_shards": {
        "split_sizes": {
          "q": { "cpu": 1024, "gpu": 256, "npu": 256 },
          "k": { "cpu": 512, "gpu": 256, "npu": 256 },
          "v": { "cpu": 256, "gpu": 512, "npu": 256 }
        }
      }
    },
    "root-fallback": {}
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "fixed",
    "phase": "prefill",
    "initial_profile": "shifted",
    "profiles": ["shifted", "root-fallback"],
    "transitions": "complete",
    "candidate_kinds": ["attn_qkv_block"],
    "boundary": { "node": "l_out", "backend": "auto", "granularity": "layer" },
    "output_mode": "canonical",
    "strict": true
  }
}
)JSON";

    static constexpr char topology_override_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "cpu", "backend": "CPU" },
      { "id": "gpu", "backend": "OpenCL" },
      { "id": "npu", "backend": "HTP0-REPACK" }
    ],
    "split_sizes": {
      "q": { "cpu": 256, "gpu": 256, "npu": 256 },
      "k": { "cpu": 128, "gpu": 128, "npu": 256 },
      "v": { "cpu": 128, "gpu": 128, "npu": 256 }
    }
  },
  "profiles": {
    "invalid": {
      "attn_qkv_shards": {
        "assemble_backend": "CPU",
        "split_sizes": {
          "q": { "cpu": 256, "gpu": 256, "npu": 256 },
          "k": { "cpu": 128, "gpu": 128, "npu": 256 },
          "v": { "cpu": 128, "gpu": 128, "npu": 256 }
        }
      }
    }
  }
}
)JSON";

    static constexpr char incomplete_override_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "cpu", "backend": "CPU" },
      { "id": "gpu", "backend": "OpenCL" },
      { "id": "npu", "backend": "HTP0-REPACK" }
    ],
    "split_sizes": {
      "q": { "cpu": 256, "gpu": 256, "npu": 256 },
      "k": { "cpu": 128, "gpu": 128, "npu": 256 },
      "v": { "cpu": 128, "gpu": 128, "npu": 256 }
    }
  },
  "profiles": {
    "invalid": {
      "attn_qkv_shards": {
        "split_sizes": {
          "q": { "cpu": 256, "gpu": 256, "npu": 256 },
          "k": { "cpu": 128, "gpu": 128, "npu": 256 }
        }
      }
    }
  }
}
)JSON";

    static constexpr char mismatched_width_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "cpu", "backend": "CPU", "align": 128 },
      { "id": "gpu", "backend": "OpenCL", "align": 128 },
      { "id": "npu", "backend": "HTP0-REPACK" }
    ],
    "split_sizes": {
      "q": { "cpu": 256, "gpu": 256, "npu": 1024 },
      "k": { "cpu": 128, "gpu": 128, "npu": 256 },
      "v": { "cpu": 128, "gpu": 128, "npu": 256 }
    }
  },
  "profiles": {
    "invalid": {
      "attn_qkv_shards": {
        "split_sizes": {
          "q": { "cpu": 768, "gpu": 256, "npu": 256 },
          "k": { "cpu": 128, "gpu": 128, "npu": 256 },
          "v": { "cpu": 128, "gpu": 128, "npu": 256 }
        }
      }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "fixed",
    "phase": "prefill",
    "initial_profile": "invalid",
    "profiles": ["invalid"],
    "transitions": "complete",
    "candidate_kinds": ["attn_qkv_block"]
  }
}
)JSON";

    static constexpr char decode_phase_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "phase": "decode",
    "split_layout": [
      { "id": "cpu", "backend": "CPU" },
      { "id": "gpu", "backend": "OpenCL" },
      { "id": "npu", "backend": "HTP0-REPACK" }
    ],
    "split_sizes": {
      "q": { "cpu": 256, "gpu": 256, "npu": 256 },
      "k": { "cpu": 128, "gpu": 128, "npu": 256 },
      "v": { "cpu": 128, "gpu": 128, "npu": 256 }
    }
  }
}
)JSON";

    static constexpr char routed_long_id_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "cpu_lane_long", "backend": "CPU" },
      { "id": "gpu", "backend": "OpenCL" },
      { "id": "npu", "backend": "HTP0-REPACK" }
    ],
    "split_sizes": {
      "q": { "cpu_lane_long": 256, "gpu": 256, "npu": 256 },
      "k": { "cpu_lane_long": 128, "gpu": 128, "npu": 256 },
      "v": { "cpu_lane_long": 128, "gpu": 128, "npu": 256 }
    }
  },
  "profiles": { "only": {} },
  "runtime_routes": {
    "enabled": true,
    "mode": "fixed",
    "phase": "prefill",
    "initial_profile": "only",
    "profiles": ["only"],
    "transitions": "complete",
    "candidate_kinds": ["attn_qkv_block"]
  }
}
)JSON";

    // The routed limit must not break the existing static QKV path, whose
    // shorter node format can still carry the same safe-character lane id.
    static constexpr char static_long_id_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_qkv_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "cpu_lane_long", "backend": "CPU" },
      { "id": "gpu", "backend": "OpenCL" },
      { "id": "npu", "backend": "HTP0-REPACK" }
    ],
    "split_sizes": {
      "q": { "cpu_lane_long": 256, "gpu": 256, "npu": 256 },
      "k": { "cpu_lane_long": 128, "gpu": 128, "npu": 256 },
      "v": { "cpu_lane_long": 128, "gpu": 128, "npu": 256 }
    }
  }
}
)JSON";

    temporary_policy_file routed_file(
            "test-backend-policy-attn-qkv-profile-routed.json", routed_policy);
    temporary_policy_file topology_file(
            "test-backend-policy-attn-qkv-profile-topology.json", topology_override_policy);
    temporary_policy_file incomplete_file(
            "test-backend-policy-attn-qkv-profile-incomplete.json", incomplete_override_policy);
    temporary_policy_file mismatched_width_file(
            "test-backend-policy-attn-qkv-profile-width.json", mismatched_width_policy);
    temporary_policy_file decode_phase_file(
            "test-backend-policy-attn-qkv-decode.json", decode_phase_policy);
    temporary_policy_file routed_long_id_file(
            "test-backend-policy-attn-qkv-routed-long-id.json", routed_long_id_policy);
    temporary_policy_file static_long_id_file(
            "test-backend-policy-attn-qkv-static-long-id.json", static_long_id_policy);

    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(routed_file.path(), false, false));

    llama_backend_policy_runtime_routes routes;
    CHECK(llama_backend_policy_resolve_runtime_routes(true, routes));
    CHECK(std::find(routes.candidate_kinds.begin(), routes.candidate_kinds.end(),
                    "attn_qkv_block") != routes.candidate_kinds.end());

    llama_backend_policy_attn_qkv_shards policy;
    CHECK(llama_backend_policy_match_attn_qkv_shards_for_profile(
            "shifted", 0, true, policy));
    CHECK(policy.source == "profiles.shifted.attn_qkv_shards");
    CHECK(policy.splits.size() == 3);
    CHECK(policy.splits[0].id == "cpu" && policy.splits[0].backend == "CPU");
    CHECK(policy.splits[0].q_start == 0 && policy.splits[0].q_size == 1024);
    CHECK(policy.splits[1].q_start == 1024 && policy.splits[1].q_size == 256);
    CHECK(policy.splits[2].q_start == 1280 && policy.splits[2].q_size == 256);
    CHECK(policy.splits[0].k_size == 512 && policy.splits[2].k_start == 768);
    CHECK(policy.splits[0].v_size == 256 && policy.splits[2].v_start == 768);

    CHECK(llama_backend_policy_match_attn_qkv_shards_for_profile(
            "root-fallback", 27, true, policy));
    CHECK(policy.source == "attn_qkv_shards");
    CHECK(policy.splits[0].q_size == 256);
    CHECK(!llama_backend_policy_match_attn_qkv_shards_for_profile(
            "unknown", 0, true, policy));
    CHECK(!llama_backend_policy_match_attn_qkv_shards_for_profile(
            "shifted", 28, true, policy));
    CHECK(!llama_backend_policy_match_attn_qkv_shards_for_profile(
            "shifted", 0, false, policy));

    // Root matching stays static and is not mutated by profile resolution.
    CHECK(llama_backend_policy_match_attn_qkv_shards(0, true, policy));
    CHECK(policy.source == "attn_qkv_shards");
    CHECK(policy.splits[0].q_size == 256);

    llama_backend_policy_residency_plan plan;
    CHECK(llama_backend_policy_build_attn_qkv_residency_plan(
            0, LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q, plan));
    CHECK(plan.enabled);
    CHECK(plan.keep_full_source);
    CHECK(plan.source == "attn_qkv_shards q axis-1 covers");
    CHECK(plan.covers.size() == 4);

    std::vector<llama_backend_policy_residency_cover> gpu_covers;
    for (const auto & cover : plan.covers) {
        if (cover.backend == "OpenCL") {
            gpu_covers.push_back(cover);
        }
    }
    CHECK(gpu_covers.size() == 2);
    CHECK(gpu_covers[0].start == 256 && gpu_covers[0].size == 256);
    CHECK(gpu_covers[1].start == 1024 && gpu_covers[1].size == 256);

    CHECK(llama_backend_policy_build_attn_qkv_residency_plan(
            0, LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_K, plan));
    CHECK(plan.source == "attn_qkv_shards k axis-1 covers");
    CHECK(plan.covers.size() == 3);
    CHECK(!llama_backend_policy_build_attn_qkv_residency_plan(
            28, LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q, plan));

    for (const char * invalid : {
            topology_file.path(), incomplete_file.path(),
            mismatched_width_file.path(), routed_long_id_file.path() }) {
        CHECK(!llama_backend_policy_load(invalid, false, false));
        // Failed profile/root loads are transactional.
        CHECK(llama_backend_policy_match_attn_qkv_shards_for_profile(
                "shifted", 0, true, policy));
        CHECK(policy.splits[0].q_size == 1024);
    }

    // Decode remains schema-compatible with the original static policy, but
    // it is deliberately inert for the current prefill-only graph/cover path.
    CHECK(llama_backend_policy_load(decode_phase_file.path(), false, false));
    CHECK(!llama_backend_policy_match_attn_qkv_shards(0, true, policy));
    CHECK(llama_backend_policy_match_attn_qkv_shards(0, false, policy));
    CHECK(!llama_backend_policy_build_attn_qkv_residency_plan(
            0, LLAMA_BACKEND_POLICY_ATTN_QKV_PROJECTION_Q, plan));

    CHECK(llama_backend_policy_load(static_long_id_file.path(), false, false));
    CHECK(llama_backend_policy_match_attn_qkv_shards(0, true, policy));
    CHECK(policy.splits[0].id == "cpu_lane_long");

    llama_backend_policy_clear();
    return true;
}

bool test_attn_out_shards_policy() {
    static constexpr char valid_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "phase": "prefill",
    "layer_range": [2, 4],
    "head_dim": 128,
    "reduce_backend": "OpenCL",
    "split_layout": [
      { "id": "npu", "backend": "HTP0-REPACK" },
      { "id": "gpu", "backend": "OpenCL", "align": 128 },
      { "id": "cpu", "backend": "CPU", "align": 128 }
    ],
    "split_sizes": { "npu": 512, "gpu": 512, "cpu": 512 }
  }
}
)JSON";

    static constexpr char valid_output_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "partition_axis": "OUTPUT",
    "head_dim": 128,
    "reduce_backend": "OpenCL",
    "split_layout": [
      { "id": "npu", "backend": "HTP0-REPACK" },
      { "id": "gpu", "backend": "OpenCL" },
      { "id": "cpu", "backend": "CPU" }
    ],
    "split_sizes": { "npu": 512, "gpu": 512, "cpu": 512 }
  }
}
)JSON";

    static constexpr char invalid_htp_align[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "head_dim": 128,
    "reduce_backend": "CPU",
    "split_layout": [
      { "id": "npu", "backend": "HTP0-REPACK" },
      { "id": "gpu", "backend": "OpenCL" },
      { "id": "cpu", "backend": "CPU" }
    ],
    "split_sizes": { "npu": 128, "gpu": 128, "cpu": 128 }
  }
}
)JSON";

    static constexpr char invalid_duplicate_backend[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "a", "backend": "CPU" },
      { "id": "b", "backend": "cpu" },
      { "id": "c", "backend": "OpenCL" }
    ],
    "split_sizes": { "a": 128, "b": 128, "c": 128 }
  }
}
)JSON";

    static constexpr char invalid_duplicate_id[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "lane", "backend": "CPU" },
      { "id": "LANE", "backend": "OpenCL" },
      { "id": "other", "backend": "GPU" }
    ],
    "split_sizes": { "lane": 128, "LANE": 128, "other": 128 }
  }
}
)JSON";

    static constexpr char invalid_head_align[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "head_dim": 128,
    "split_layout": [
      { "id": "a", "backend": "CPU" },
      { "id": "b", "backend": "OpenCL" },
      { "id": "c", "backend": "GPU" }
    ],
    "split_sizes": { "a": 192, "b": 128, "c": 128 }
  }
}
)JSON";

    static constexpr char invalid_phase[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "phase": "decode",
    "split_layout": [
      { "id": "a", "backend": "CPU" },
      { "id": "b", "backend": "OpenCL" },
      { "id": "c", "backend": "GPU" }
    ],
    "split_sizes": { "a": 128, "b": 128, "c": 128 }
  }
}
)JSON";

    static constexpr char invalid_partition_axis[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "partition_axis": "heads",
    "split_layout": [
      { "id": "a", "backend": "CPU" },
      { "id": "b", "backend": "OpenCL" },
      { "id": "c", "backend": "GPU" }
    ],
    "split_sizes": { "a": 128, "b": 128, "c": 128 }
  }
}
)JSON";

    static constexpr char invalid_lane_count[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "a", "backend": "CPU" },
      { "id": "b", "backend": "OpenCL" }
    ],
    "split_sizes": { "a": 128, "b": 128 }
  }
}
)JSON";

    static constexpr char invalid_long_id[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "split_layout": [
      { "id": "abcdefghijklmnopqrstuvwxyz123456", "backend": "CPU" },
      { "id": "b", "backend": "OpenCL" },
      { "id": "c", "backend": "GPU" }
    ],
    "split_sizes": { "abcdefghijklmnopqrstuvwxyz123456": 128, "b": 128, "c": 128 }
  }
}
)JSON";

    temporary_policy_file valid_file("test-backend-policy-attn-out-valid.json", valid_policy);
    temporary_policy_file valid_output_file("test-backend-policy-attn-out-valid-output.json", valid_output_policy);
    temporary_policy_file invalid_htp_file("test-backend-policy-attn-out-htp.json", invalid_htp_align);
    temporary_policy_file invalid_backend_file("test-backend-policy-attn-out-backend.json", invalid_duplicate_backend);
    temporary_policy_file invalid_id_file("test-backend-policy-attn-out-id.json", invalid_duplicate_id);
    temporary_policy_file invalid_head_file("test-backend-policy-attn-out-head.json", invalid_head_align);
    temporary_policy_file invalid_phase_file("test-backend-policy-attn-out-phase.json", invalid_phase);
    temporary_policy_file invalid_axis_file("test-backend-policy-attn-out-axis.json", invalid_partition_axis);
    temporary_policy_file invalid_count_file("test-backend-policy-attn-out-count.json", invalid_lane_count);
    temporary_policy_file invalid_long_id_file("test-backend-policy-attn-out-long-id.json", invalid_long_id);

    llama_backend_policy_clear();
    CHECK(!llama_backend_policy_attn_out_shards_enabled());
    CHECK(llama_backend_policy_load(valid_file.path(), false, false));
    CHECK(llama_backend_policy_attn_out_shards_enabled());

    llama_backend_policy_attn_out_shards policy;
    CHECK(!llama_backend_policy_match_attn_out_shards(1, true, policy));
    CHECK(!llama_backend_policy_match_attn_out_shards(2, false, policy));
    CHECK(llama_backend_policy_match_attn_out_shards(2, true, policy));
    CHECK(policy.enabled);
    CHECK(policy.phase == "prefill");
    CHECK(policy.partition_axis == "input");
    CHECK(policy.layer_start == 2 && policy.layer_end == 4);
    CHECK(policy.head_dim == 128);
    CHECK(policy.reduce_backend == "OpenCL");
    CHECK(policy.splits.size() == 3);
    CHECK(policy.splits[0].id == "npu");
    CHECK(policy.splits[0].backend == "HTP0-REPACK");
    CHECK(policy.splits[0].start == 0 && policy.splits[0].size == 512);
    CHECK(policy.splits[1].start == 512 && policy.splits[1].size == 512);
    CHECK(policy.splits[2].start == 1024 && policy.splits[2].size == 512);

    llama_backend_policy_residency_plan plan;
    CHECK(llama_backend_policy_build_attn_out_residency_plan(2, plan));
    CHECK(plan.enabled);
    CHECK(plan.keep_full_source);
    CHECK(plan.source == "attn_out_shards axis-0 covers");
    CHECK(plan.covers.size() == 3);
    CHECK(plan.covers[0].id == "cover.htp0-repack.0.512");
    CHECK(plan.covers[0].backend == "HTP0-REPACK");
    CHECK(plan.covers[1].start == 512 && plan.covers[1].size == 512);
    CHECK(plan.covers[2].start == 1024 && plan.covers[2].size == 512);
    CHECK(!llama_backend_policy_build_attn_out_residency_plan(1, plan));

    for (const char * invalid : {
            invalid_htp_file.path(), invalid_backend_file.path(), invalid_id_file.path(),
            invalid_head_file.path(), invalid_phase_file.path(), invalid_axis_file.path(), invalid_count_file.path(),
            invalid_long_id_file.path() }) {
        CHECK(!llama_backend_policy_load(invalid, false, false));
        // Failed loads are transactional: the valid static plan remains live.
        CHECK(llama_backend_policy_attn_out_shards_enabled());
        CHECK(llama_backend_policy_match_attn_out_shards(3, true, policy));
        CHECK(policy.splits[0].size == 512);
    }

    CHECK(llama_backend_policy_load(valid_output_file.path(), false, false));
    CHECK(llama_backend_policy_match_attn_out_shards(0, true, policy));
    CHECK(policy.partition_axis == "output");
    CHECK(llama_backend_policy_build_attn_out_residency_plan(0, plan));
    CHECK(plan.enabled);
    CHECK(plan.keep_full_source);
    CHECK(plan.source == "attn_out_shards axis-1 covers");
    CHECK(plan.covers.size() == 3);
    CHECK(plan.covers[0].start == 0 && plan.covers[0].size == 512);
    CHECK(plan.covers[1].start == 512 && plan.covers[1].size == 512);
    CHECK(plan.covers[2].start == 1024 && plan.covers[2].size == 512);

    llama_backend_policy_clear();
    CHECK(!llama_backend_policy_attn_out_shards_enabled());
    return true;
}

bool test_profile_attn_out_shards_policy() {
    static constexpr char routed_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "phase": "prefill",
    "partition_axis": "output",
    "layer_range": [0, 27],
    "head_dim": 128,
    "reduce_backend": "OpenCL",
    "split_layout": [
      { "id": "cpu", "backend": "CPU", "align": 256 },
      { "id": "gpu", "backend": "OpenCL", "align": 256 },
      { "id": "npu", "backend": "HTP0-REPACK", "align": 256 }
    ],
    "split_sizes": { "cpu": 1024, "gpu": 1024, "npu": 1024 }
  },
  "profiles": {
    "p15-g15-gpu14": {
      "attn_out_shards": {
        "split_sizes": { "cpu": 768, "gpu": 512, "npu": 1792 }
      }
    },
    "p4-g8-gpu14": {
      "attn_out_shards": {
        "split_sizes": { "cpu": 512, "gpu": 512, "npu": 2048 }
      }
    },
    "root-fallback": {}
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "fixed",
    "phase": "prefill",
    "initial_profile": "p15-g15-gpu14",
    "profiles": ["p15-g15-gpu14", "p4-g8-gpu14", "root-fallback"],
    "transitions": "complete",
    "candidate_kinds": ["attn_out_block"],
    "boundary": { "node": "l_out", "backend": "auto", "granularity": "layer" },
    "output_mode": "canonical",
    "strict": true
  }
}
)JSON";

    static constexpr char topology_override_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "partition_axis": "output",
    "split_layout": [
      { "id": "cpu", "backend": "CPU", "align": 256 },
      { "id": "gpu", "backend": "OpenCL", "align": 256 },
      { "id": "npu", "backend": "HTP0-REPACK", "align": 256 }
    ],
    "split_sizes": { "cpu": 1024, "gpu": 1024, "npu": 1024 }
  },
  "profiles": {
    "invalid": {
      "attn_out_shards": {
        "reduce_backend": "CPU",
        "split_sizes": { "cpu": 512, "gpu": 512, "npu": 2048 }
      }
    }
  }
}
)JSON";

    static constexpr char missing_root_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "profiles": {
    "invalid": {
      "attn_out_shards": {
        "split_sizes": { "cpu": 512, "gpu": 512, "npu": 2048 }
      }
    }
  }
}
)JSON";

    static constexpr char input_axis_route_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "partition_axis": "input",
    "split_layout": [
      { "id": "cpu", "backend": "CPU", "align": 256 },
      { "id": "gpu", "backend": "OpenCL", "align": 256 },
      { "id": "npu", "backend": "HTP0-REPACK", "align": 256 }
    ],
    "split_sizes": { "cpu": 1024, "gpu": 1024, "npu": 1024 }
  },
  "profiles": { "only": {} },
  "runtime_routes": {
    "enabled": true,
    "mode": "fixed",
    "initial_profile": "only",
    "profiles": ["only"],
    "transitions": "complete",
    "candidate_kinds": ["attn_out_block"]
  }
}
)JSON";

    static constexpr char mismatched_width_policy[] = R"JSON(
{
  "version": 1,
  "enabled": true,
  "attn_out_shards": {
    "enabled": true,
    "partition_axis": "output",
    "split_layout": [
      { "id": "cpu", "backend": "CPU", "align": 256 },
      { "id": "gpu", "backend": "OpenCL", "align": 256 },
      { "id": "npu", "backend": "HTP0-REPACK", "align": 256 }
    ],
    "split_sizes": { "cpu": 1024, "gpu": 1024, "npu": 1024 }
  },
  "profiles": {
    "invalid": {
      "attn_out_shards": {
        "split_sizes": { "cpu": 512, "gpu": 512, "npu": 1792 }
      }
    }
  },
  "runtime_routes": {
    "enabled": true,
    "mode": "fixed",
    "initial_profile": "invalid",
    "profiles": ["invalid"],
    "transitions": "complete",
    "candidate_kinds": ["attn_out_block"]
  }
}
)JSON";

    temporary_policy_file routed_file(
            "test-backend-policy-attn-out-profile-routed.json", routed_policy);
    temporary_policy_file topology_file(
            "test-backend-policy-attn-out-profile-topology.json", topology_override_policy);
    temporary_policy_file missing_root_file(
            "test-backend-policy-attn-out-profile-no-root.json", missing_root_policy);
    temporary_policy_file input_axis_file(
            "test-backend-policy-attn-out-profile-input-axis.json", input_axis_route_policy);
    temporary_policy_file mismatched_width_file(
            "test-backend-policy-attn-out-profile-width.json", mismatched_width_policy);

    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(routed_file.path(), false, false));

    llama_backend_policy_runtime_routes routes;
    CHECK(llama_backend_policy_resolve_runtime_routes(true, routes));
    CHECK(std::find(routes.candidate_kinds.begin(), routes.candidate_kinds.end(),
                    "attn_out_block") != routes.candidate_kinds.end());

    llama_backend_policy_attn_out_shards policy;
    CHECK(llama_backend_policy_match_attn_out_shards_for_profile(
            "p15-g15-gpu14", 0, true, policy));
    CHECK(policy.source == "profiles.p15-g15-gpu14.attn_out_shards");
    CHECK(policy.splits.size() == 3);
    CHECK(policy.splits[0].id == "cpu");
    CHECK(policy.splits[0].start == 0 && policy.splits[0].size == 768);
    CHECK(policy.splits[1].start == 768 && policy.splits[1].size == 512);
    CHECK(policy.splits[2].start == 1280 && policy.splits[2].size == 1792);

    CHECK(llama_backend_policy_match_attn_out_shards_for_profile(
            "p4-g8-gpu14", 27, true, policy));
    CHECK(policy.source == "profiles.p4-g8-gpu14.attn_out_shards");
    CHECK(policy.splits[0].start == 0 && policy.splits[0].size == 512);
    CHECK(policy.splits[1].start == 512 && policy.splits[1].size == 512);
    CHECK(policy.splits[2].start == 1024 && policy.splits[2].size == 2048);

    CHECK(llama_backend_policy_match_attn_out_shards_for_profile(
            "root-fallback", 0, true, policy));
    CHECK(policy.source == "attn_out_shards");
    CHECK(policy.splits[0].start == 0 && policy.splits[0].size == 1024);
    CHECK(policy.splits[1].start == 1024 && policy.splits[1].size == 1024);
    CHECK(policy.splits[2].start == 2048 && policy.splits[2].size == 1024);
    CHECK(!llama_backend_policy_match_attn_out_shards_for_profile(
            "unknown", 0, true, policy));
    CHECK(!llama_backend_policy_match_attn_out_shards_for_profile(
            "p15-g15-gpu14", 28, true, policy));
    CHECK(!llama_backend_policy_match_attn_out_shards_for_profile(
            "p15-g15-gpu14", 0, false, policy));

    // The root remains the static fallback and is not mutated by a profile match.
    CHECK(llama_backend_policy_match_attn_out_shards(0, true, policy));
    CHECK(policy.source == "attn_out_shards");
    CHECK(policy.splits[0].size == 1024);

    llama_backend_policy_residency_plan plan;
    CHECK(llama_backend_policy_build_attn_out_residency_plan(0, plan));
    CHECK(plan.enabled);
    CHECK(plan.keep_full_source);
    CHECK(plan.source == "attn_out_shards axis-1 covers");
    CHECK(plan.covers.size() == 3);
    CHECK(plan.covers[0].backend == "CPU");
    CHECK(plan.covers[0].start == 0 && plan.covers[0].size == 1024);
    CHECK(plan.covers[1].backend == "OpenCL");
    CHECK(plan.covers[1].start == 512 && plan.covers[1].size == 1536);
    CHECK(plan.covers[2].backend == "HTP0-REPACK");
    CHECK(plan.covers[2].start == 1024 && plan.covers[2].size == 2048);

    for (const char * invalid : {
            topology_file.path(), missing_root_file.path(),
            input_axis_file.path(), mismatched_width_file.path() }) {
        CHECK(!llama_backend_policy_load(invalid, false, false));
        // Failed profile loads are transactional.
        CHECK(llama_backend_policy_match_attn_out_shards_for_profile(
                "p15-g15-gpu14", 0, true, policy));
        CHECK(policy.splits[0].size == 768);
    }

    llama_backend_policy_clear();
    return true;
}

bool test_cover_residency_plan(const char * union_policy_path) {
    llama_backend_policy_clear();
    CHECK(llama_backend_policy_load(union_policy_path, false, false));

    llama_backend_policy_residency_plan plan;
    CHECK(llama_backend_policy_build_ffn_residency_plan(0, plan));
    CHECK(plan.enabled);
    CHECK(!plan.keep_full_source);
    CHECK(plan.source == "ffn_parallel connected cover union");

    auto backend_covers = [&](const char * backend) {
        std::vector<llama_backend_policy_residency_cover> result;
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
    CHECK(npu_covers[0].id == "cover.htp0-repack.0.5888");
    CHECK(npu_covers[0].start == 0 && npu_covers[0].size == 5888);
    CHECK(gpu_covers.size() == 1);
    CHECK(gpu_covers[0].id == "cover.opencl.4096.2880");
    CHECK(gpu_covers[0].start == 4096 && gpu_covers[0].size == 2880);
    CHECK(cpu_covers.size() == 1);
    CHECK(cpu_covers[0].id == "cover.cpu_repack.5504.2688");
    CHECK(cpu_covers[0].start == 5504 && cpu_covers[0].size == 2688);

    // A union is not a convex hull. Preserve the intentionally uncovered gap.
    const auto disjoint = backend_covers("DISJOINT");
    CHECK(disjoint.size() == 2);
    CHECK(disjoint[0].id == "cover.disjoint.0.64");
    CHECK(disjoint[0].start == 0 && disjoint[0].size == 64);
    CHECK(disjoint[1].id == "cover.disjoint.8128.64");
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

    llama_backend_policy_residency_plan plan;
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

    llama_backend_policy_residency_plan plan;
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
        if (!test_attn_qkv_shards_policy()) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_profile_attn_qkv_shards_policy()) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_attn_out_shards_policy()) {
            llama_backend_policy_clear();
            return 1;
        }
        if (!test_profile_attn_out_shards_policy()) {
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
