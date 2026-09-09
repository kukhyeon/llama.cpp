#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from profile_pruner import (  # noqa: E402
    ClockProfileCandidate,
    LinearLatencyModel,
    ObservedProfileState,
    prune_clock_profiles,
    relative_frequency_distance,
    select_runtime_profile,
)
from trace_parser import ClockPoint  # noqa: E402


def _point(prime: int) -> ClockPoint:
    return ClockPoint(prime_khz=prime, gold_khz=100, gpu_hz=100_000)


def _candidate(name: str, prime: int, cpu: int, gpu: int) -> ClockProfileCandidate:
    return ClockProfileCandidate(name, _point(prime), {"cpu": cpu, "gpu": gpu})


def _state(name: str, prime: int, cpu_slope: float, gpu_slope: float, split: dict[str, int]) -> ObservedProfileState:
    return ObservedProfileState(
        name=name,
        point=_point(prime),
        weight=1.0,
        latency_models={
            "cpu": LinearLatencyModel(cpu_slope),
            "gpu": LinearLatencyModel(gpu_slope),
        },
        optimal_split=split,
    )


class RuntimeSelectorTests(unittest.TestCase):
    def test_distance_is_sum_of_target_relative_errors(self) -> None:
        actual = ClockPoint(150, 120, 120_000)
        target = ClockPoint(100, 100, 100_000)
        self.assertAlmostEqual(relative_frequency_distance(actual, target), 0.9)

    def test_ties_prefer_lower_frequency_then_lexical_name(self) -> None:
        lower = _candidate("z-lower", 100, 500, 500)
        higher = _candidate("a-higher", 300, 500, 500)
        selected, _ = select_runtime_profile(_point(150), [higher, lower])
        self.assertEqual(selected.name, "z-lower")

        same_z = _candidate("z-same", 200, 500, 500)
        same_a = _candidate("a-same", 200, 500, 500)
        selected, _ = select_runtime_profile(_point(200), [same_z, same_a])
        self.assertEqual(selected.name, "a-same")


class ProfilePruningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fast_cpu_split = {"cpu": 800, "gpu": 200}
        self.fast_gpu_split = {"cpu": 200, "gpu": 800}
        self.candidates = [
            _candidate("a", 180, 800, 200),
            _candidate("b", 200, 800, 200),
            _candidate("c", 300, 200, 800),
        ]
        self.states = [
            _state("state-a", 180, 1.0, 4.0, self.fast_cpu_split),
            _state("state-b", 200, 1.0, 4.0, self.fast_cpu_split),
            _state("state-c", 300, 4.0, 1.0, self.fast_gpu_split),
        ]

    def test_greedy_pruning_removes_redundant_profile_only(self) -> None:
        result = prune_clock_profiles(self.states, self.candidates)

        self.assertEqual(result.removed_profiles, ("a",))
        self.assertEqual(result.retained_profiles, ("b", "c"))
        self.assertEqual(
            result.state_to_profile,
            {"state-a": "b", "state-b": "b", "state-c": "c"},
        )
        self.assertAlmostEqual(result.max_predicted_regret, 0.0)
        self.assertAlmostEqual(result.weighted_predicted_regret, 0.0)
        json.dumps(result.to_dict())

    def test_named_anchor_is_never_removed(self) -> None:
        result = prune_clock_profiles(self.states, self.candidates, preserve_anchors=("a",))

        self.assertEqual(result.removed_profiles, ("b",))
        self.assertEqual(result.retained_profiles, ("a", "c"))
        self.assertIn("a", result.preserved_anchors)
        self.assertEqual(result.state_to_profile["state-b"], "a")

    def test_removal_is_rejected_when_any_state_exceeds_regret_bound(self) -> None:
        result = prune_clock_profiles(
            [self.states[0], self.states[2]],
            [self.candidates[0], self.candidates[2]],
            max_predicted_regret=0.03,
        )
        self.assertEqual(result.removed_profiles, ())
        self.assertEqual(result.retained_profiles, ("a", "c"))


if __name__ == "__main__":
    unittest.main()
