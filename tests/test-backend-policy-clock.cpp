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

    try {
        scoped_environment clock_switch("LLAMA_BACKEND_POLICY_FFN_CLOCK_SWITCH", "1");
        scoped_environment ffn_parallel("LLAMA_FFN_PARALLEL", "1");
        temporary_policy_file valid_file("test-backend-policy-clock-valid.json", valid_policy);
        temporary_policy_file invalid_file("test-backend-policy-clock-invalid.json", invalid_policy);
        temporary_policy_file union_file("test-backend-policy-clock-union.json", union_policy);
        temporary_policy_file profile_only_file(
                "test-backend-policy-clock-profile-only.json", profile_only_policy);

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
    } catch (const std::exception & error) {
        llama_backend_policy_clear();
        std::cerr << "test-backend-policy-clock: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
