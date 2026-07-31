#pragma once

#include <chrono>
#include <cstdint>

struct common_query_pacing_decision {
    std::chrono::steady_clock::time_point scheduled_start;
    std::chrono::steady_clock::duration wait;
    std::chrono::steady_clock::duration lateness;
    bool deadline_miss = false;
};

// Return the absolute start target for a zero-based query index. Deriving every
// target from the original epoch prevents logging and scheduler jitter from
// accumulating as drift between queries.
inline common_query_pacing_decision common_query_pacing_decide(
        std::chrono::steady_clock::time_point epoch,
        uint64_t                              zero_based_query_index,
        std::chrono::milliseconds             period,
        std::chrono::steady_clock::time_point now) {
    if (period <= std::chrono::milliseconds::zero()) {
        return {now, std::chrono::steady_clock::duration::zero(),
                std::chrono::steady_clock::duration::zero(), false};
    }

    const auto scheduled_start = epoch +
        period * static_cast<std::chrono::milliseconds::rep>(zero_based_query_index);

    if (now > scheduled_start) {
        return {scheduled_start, std::chrono::steady_clock::duration::zero(),
                now - scheduled_start, true};
    }

    return {scheduled_start, scheduled_start - now,
            std::chrono::steady_clock::duration::zero(), false};
}
