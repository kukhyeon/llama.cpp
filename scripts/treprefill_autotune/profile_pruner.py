#!/usr/bin/env python3
"""Greedily prune TrePrefill clock profiles under a predicted-regret bound.

Profile selection intentionally reproduces ``select_clock_profile_locked()`` in
``src/llama-backend-policy.cpp``:

1. Sum ``abs(actual - target) / target`` for Prime, Gold and GPU clocks.
2. On a distance tie, prefer the lower total target frequency (GPU converted
   from Hz to kHz).
3. On a full tie, prefer the lexicographically smaller profile name.

For each observed state, a candidate profile's FFN split is evaluated with the
state-specific latency models.  A profile is removed only when runtime
selection among the remaining candidates keeps every observed state at or
below ``max_predicted_regret``.  The greedy process stops at a locally minimal
profile set; it does not claim a globally minimum set-cover solution.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, Sequence

try:
    from .trace_parser import ClockPoint
except ImportError:  # Direct module execution/import from this directory.
    from trace_parser import ClockPoint


_TIE_EPSILON = 1.0e-15


class LatencyPredictor(Protocol):
    def predict_us(self, size: int) -> float: ...


@dataclass(frozen=True)
class LinearLatencyModel:
    """Minimal serializable latency predictor accepted by the pruner."""

    slope_us_per_element: float
    intercept_us: float = 0.0

    def predict_us(self, size: int) -> float:
        if size <= 0:
            return 0.0
        return max(0.0, self.intercept_us + self.slope_us_per_element * size)


@dataclass(frozen=True)
class ObservedProfileState:
    name: str
    point: ClockPoint
    weight: float
    latency_models: Mapping[str, LatencyPredictor]
    optimal_split: Mapping[str, int]
    optimal_makespan_us: float | None = None


@dataclass(frozen=True)
class ClockProfileCandidate:
    name: str
    point: ClockPoint
    split_sizes: Mapping[str, int]


@dataclass(frozen=True)
class StateProfileAssignment:
    state: str
    selected_profile: str
    selector_distance: float
    baseline_makespan_us: float
    predicted_makespan_us: float
    predicted_regret: float


@dataclass(frozen=True)
class PruningStep:
    removed_profile: str
    retained_profiles: tuple[str, ...]
    max_predicted_regret: float
    weighted_predicted_regret: float


@dataclass(frozen=True)
class ProfilePruningResult:
    retained_profiles: tuple[str, ...]
    removed_profiles: tuple[str, ...]
    preserved_anchors: tuple[str, ...]
    state_to_profile: Mapping[str, str]
    assignments: tuple[StateProfileAssignment, ...]
    max_predicted_regret: float
    weighted_predicted_regret: float
    steps: tuple[PruningStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "retained_profiles": list(self.retained_profiles),
            "removed_profiles": list(self.removed_profiles),
            "preserved_anchors": list(self.preserved_anchors),
            "state_to_profile": dict(self.state_to_profile),
            "assignments": [asdict(assignment) for assignment in self.assignments],
            "max_predicted_regret": self.max_predicted_regret,
            "weighted_predicted_regret": self.weighted_predicted_regret,
            "steps": [asdict(step) for step in self.steps],
        }


def relative_frequency_distance(actual: ClockPoint, target: ClockPoint) -> float:
    """Return the exact three-term relative distance used by the runtime."""

    if target.prime_khz <= 0 or target.gold_khz <= 0 or target.gpu_hz <= 0:
        raise ValueError("target frequencies must be positive")
    if actual.prime_khz <= 0 or actual.gold_khz <= 0 or actual.gpu_hz <= 0:
        raise ValueError("actual frequencies must be positive")
    return (
        abs(actual.prime_khz - target.prime_khz) / target.prime_khz
        + abs(actual.gold_khz - target.gold_khz) / target.gold_khz
        + abs(actual.gpu_hz - target.gpu_hz) / target.gpu_hz
    )


def select_runtime_profile(
    actual: ClockPoint,
    candidates: Sequence[ClockProfileCandidate],
) -> tuple[ClockProfileCandidate, float]:
    """Select a profile with the runtime's distance and tie-break rules."""

    if not candidates:
        raise ValueError("at least one profile candidate is required")
    best: ClockProfileCandidate | None = None
    best_distance = math.inf
    best_total_khz = math.inf
    for candidate in candidates:
        distance = relative_frequency_distance(actual, candidate.point)
        total_khz = (
            candidate.point.prime_khz
            + candidate.point.gold_khz
            + candidate.point.gpu_hz / 1000.0
        )
        distance_better = distance + _TIE_EPSILON < best_distance
        distance_tied = abs(distance - best_distance) <= _TIE_EPSILON
        lower_frequency = total_khz + _TIE_EPSILON < best_total_khz
        fully_tied = distance_tied and abs(total_khz - best_total_khz) <= _TIE_EPSILON
        if (
            best is None
            or distance_better
            or (distance_tied and lower_frequency)
            or (fully_tied and candidate.name < best.name)
        ):
            best = candidate
            best_distance = distance
            best_total_khz = total_khz
    assert best is not None
    return best, best_distance


def prune_clock_profiles(
    observed_states: Sequence[ObservedProfileState],
    candidates: Sequence[ClockProfileCandidate],
    *,
    max_predicted_regret: float = 0.03,
    preserve_anchors: Sequence[str] = (),
) -> ProfilePruningResult:
    """Greedily remove clock profiles while every state's regret stays bounded.

    At each iteration all legal single-profile removals are evaluated.  The
    removal with the lowest weighted regret is chosen; max regret and profile
    name are deterministic tie-breakers.  Iteration ends when no additional
    profile can be removed safely.
    """

    _validate_inputs(observed_states, candidates, max_predicted_regret, preserve_anchors)
    by_name = {candidate.name: candidate for candidate in candidates}
    anchors = frozenset(preserve_anchors)
    active = set(by_name)
    removed: list[str] = []
    steps: list[PruningStep] = []

    initial = _evaluate_profile_set(observed_states, [by_name[name] for name in sorted(active)])
    if not _evaluation_is_feasible(initial, max_predicted_regret):
        raise ValueError(
            "the full candidate set already exceeds max_predicted_regret; "
            "cannot perform safe pruning"
        )

    while len(active) > 1:
        feasible: list[
            tuple[float, float, str, set[str], tuple[StateProfileAssignment, ...]]
        ] = []
        for name in sorted(active - anchors):
            trial = active - {name}
            if not trial:
                continue
            assignments = _evaluate_profile_set(
                observed_states,
                [by_name[item] for item in sorted(trial)],
            )
            if not _evaluation_is_feasible(assignments, max_predicted_regret):
                continue
            max_regret, weighted_regret = _regret_metrics(observed_states, assignments)
            feasible.append((weighted_regret, max_regret, name, trial, assignments))

        if not feasible:
            break
        weighted_regret, max_regret, name, active, _ = min(
            feasible,
            key=lambda item: (item[0], item[1], item[2]),
        )
        removed.append(name)
        steps.append(
            PruningStep(
                removed_profile=name,
                retained_profiles=tuple(sorted(active)),
                max_predicted_regret=max_regret,
                weighted_predicted_regret=weighted_regret,
            )
        )

    assignments = _evaluate_profile_set(
        observed_states,
        [by_name[name] for name in sorted(active)],
    )
    max_regret, weighted_regret = _regret_metrics(observed_states, assignments)
    mapping = {assignment.state: assignment.selected_profile for assignment in assignments}
    return ProfilePruningResult(
        retained_profiles=tuple(sorted(active)),
        removed_profiles=tuple(removed),
        preserved_anchors=tuple(sorted(anchors)),
        state_to_profile=mapping,
        assignments=assignments,
        max_predicted_regret=max_regret,
        weighted_predicted_regret=weighted_regret,
        steps=tuple(steps),
    )


def evaluate_profile_mapping(
    observed_states: Sequence[ObservedProfileState],
    candidates: Sequence[ClockProfileCandidate],
) -> tuple[StateProfileAssignment, ...]:
    """Evaluate a fixed candidate set without pruning it."""

    _validate_inputs(observed_states, candidates, math.inf, ())
    return _evaluate_profile_set(observed_states, candidates)


def _evaluate_profile_set(
    states: Sequence[ObservedProfileState],
    candidates: Sequence[ClockProfileCandidate],
) -> tuple[StateProfileAssignment, ...]:
    assignments: list[StateProfileAssignment] = []
    for state in states:
        selected, distance = select_runtime_profile(state.point, candidates)
        baseline = state.optimal_makespan_us
        if baseline is None:
            baseline = _predict_makespan(state.latency_models, state.optimal_split)
        predicted = _predict_makespan(state.latency_models, selected.split_sizes)
        regret = max(0.0, predicted / baseline - 1.0) if math.isfinite(predicted) else math.inf
        assignments.append(
            StateProfileAssignment(
                state=state.name,
                selected_profile=selected.name,
                selector_distance=distance,
                baseline_makespan_us=baseline,
                predicted_makespan_us=predicted,
                predicted_regret=regret,
            )
        )
    return tuple(assignments)


def _predict_makespan(
    models: Mapping[str, LatencyPredictor],
    split_sizes: Mapping[str, int],
) -> float:
    latencies: list[float] = []
    for device, raw_size in split_sizes.items():
        size = int(raw_size)
        if size < 0:
            raise ValueError(f"negative shard size for {device}: {size}")
        if size == 0:
            continue
        model = models.get(device)
        if model is None:
            return math.inf
        latency = float(model.predict_us(size))
        if not math.isfinite(latency) or latency < 0:
            return math.inf
        latencies.append(latency)
    return max(latencies, default=0.0)


def _regret_metrics(
    states: Sequence[ObservedProfileState],
    assignments: Sequence[StateProfileAssignment],
) -> tuple[float, float]:
    if not assignments:
        return 0.0, 0.0
    max_regret = max(assignment.predicted_regret for assignment in assignments)
    total_weight = sum(state.weight for state in states)
    weighted = sum(
        state.weight * assignment.predicted_regret
        for state, assignment in zip(states, assignments)
    ) / total_weight
    return max_regret, weighted


def _evaluation_is_feasible(
    assignments: Sequence[StateProfileAssignment],
    max_predicted_regret: float,
) -> bool:
    return all(
        math.isfinite(assignment.predicted_regret)
        and assignment.predicted_regret <= max_predicted_regret + _TIE_EPSILON
        for assignment in assignments
    )


def _validate_inputs(
    states: Sequence[ObservedProfileState],
    candidates: Sequence[ClockProfileCandidate],
    max_regret: float,
    anchors: Sequence[str],
) -> None:
    if not states:
        raise ValueError("at least one observed state is required")
    if not candidates:
        raise ValueError("at least one profile candidate is required")
    if max_regret < 0 or math.isnan(max_regret):
        raise ValueError("max_predicted_regret must be non-negative")
    state_names = [state.name for state in states]
    profile_names = [candidate.name for candidate in candidates]
    if len(set(state_names)) != len(state_names):
        raise ValueError("observed state names must be unique")
    if len(set(profile_names)) != len(profile_names):
        raise ValueError("candidate profile names must be unique")
    missing_anchors = set(anchors) - set(profile_names)
    if missing_anchors:
        raise ValueError(f"unknown anchor profiles: {', '.join(sorted(missing_anchors))}")

    for state in states:
        if not state.name:
            raise ValueError("observed state name must not be empty")
        if not math.isfinite(state.weight) or state.weight <= 0:
            raise ValueError(f"state weight must be positive: {state.name}")
        baseline = state.optimal_makespan_us
        if baseline is None:
            baseline = _predict_makespan(state.latency_models, state.optimal_split)
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError(f"invalid optimal makespan for state {state.name}")
    for candidate in candidates:
        if not candidate.name:
            raise ValueError("candidate profile name must not be empty")
        if not candidate.split_sizes or sum(int(size) for size in candidate.split_sizes.values()) <= 0:
            raise ValueError(f"candidate {candidate.name} has an empty split")

