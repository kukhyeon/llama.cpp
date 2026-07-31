#include "query-pacing.h"

#include <chrono>
#include <cstdint>

#undef NDEBUG
#include <cassert>

int main(void) {
    using clock = std::chrono::steady_clock;
    using std::chrono::milliseconds;

    const clock::time_point epoch{};
    const milliseconds period(60000);

    // The first query starts at the schedule epoch.
    {
        const auto decision = common_query_pacing_decide(epoch, 0, period, epoch);
        assert(decision.scheduled_start == epoch);
        assert(decision.wait == clock::duration::zero());
        assert(decision.lateness == clock::duration::zero());
        assert(!decision.deadline_miss);
    }

    // A query that completes after 20 seconds leaves 40 seconds of cooling
    // before the next fixed-period start.
    {
        const auto now = epoch + milliseconds(20000);
        const auto decision = common_query_pacing_decide(epoch, 1, period, now);
        assert(decision.scheduled_start == epoch + period);
        assert(decision.wait == milliseconds(40000));
        assert(decision.lateness == clock::duration::zero());
        assert(!decision.deadline_miss);
    }

    // Reaching the scheduled start exactly is not a deadline miss.
    {
        const auto now = epoch + period;
        const auto decision = common_query_pacing_decide(epoch, 1, period, now);
        assert(decision.scheduled_start == now);
        assert(decision.wait == clock::duration::zero());
        assert(decision.lateness == clock::duration::zero());
        assert(!decision.deadline_miss);
    }

    // Starting after the fixed-period boundary reports lateness and never
    // requests a negative wait.
    {
        const auto now = epoch + milliseconds(61000);
        const auto decision = common_query_pacing_decide(epoch, 1, period, now);
        assert(decision.scheduled_start == epoch + period);
        assert(decision.wait == clock::duration::zero());
        assert(decision.lateness == milliseconds(1000));
        assert(decision.deadline_miss);
    }

    // Every target is derived from the original epoch, so a late or early
    // previous query cannot accumulate schedule drift.
    {
        const auto now = epoch + milliseconds(70000);
        const auto decision = common_query_pacing_decide(epoch, 2, period, now);
        assert(decision.scheduled_start == epoch + milliseconds(120000));
        assert(decision.wait == milliseconds(50000));
        assert(decision.lateness == clock::duration::zero());
        assert(!decision.deadline_miss);
    }

    // A zero period disables pacing regardless of query index.
    {
        const auto now = epoch + milliseconds(1234);
        const auto decision = common_query_pacing_decide(epoch, 9, milliseconds::zero(), now);
        assert(decision.scheduled_start == now);
        assert(decision.wait == clock::duration::zero());
        assert(decision.lateness == clock::duration::zero());
        assert(!decision.deadline_miss);
    }

    return 0;
}
