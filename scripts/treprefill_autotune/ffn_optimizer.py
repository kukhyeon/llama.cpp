#!/usr/bin/env python3
"""Aligned FFN shard optimizer for TrePrefill CPU/GPU/NPU partitions.

The optimizer uses measured ``(shard size, latency)`` pairs.  It fits one small
latency model per device and enumerates only allocations that exactly cover
``n_ff`` while respecting the backend constraints:

* CPU: 64 elements
* GPU/OpenCL: 64 elements
* NPU/HTP Q8 path: 256 elements

The default minimum is one alignment quantum per measured device, preserving a
parallel lane on every supplied device.  Pass an explicit minimum of zero when
the tuner is allowed to drop a device.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from .trace_parser import FfnBranchSample, read_ffn_branch_samples
except ImportError:  # Direct execution: python ffn_optimizer.py ...
    from trace_parser import FfnBranchSample, read_ffn_branch_samples


DEVICE_ORDER = ("cpu", "gpu", "npu")
DEFAULT_ALIGNMENTS: Mapping[str, int] = {"cpu": 64, "gpu": 64, "npu": 256}


@dataclass(frozen=True)
class SizeLatencySample:
    size: int
    latency_us: float

    @classmethod
    def from_value(cls, value: "SizeLatencySample | Sequence[float] | Mapping[str, float]") -> "SizeLatencySample":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(size=int(value["size"]), latency_us=float(value["latency_us"]))
        if len(value) != 2:
            raise ValueError("size/latency sample must contain exactly two values")
        return cls(size=int(value[0]), latency_us=float(value[1]))


@dataclass(frozen=True)
class DeviceLatencyModel:
    device: str
    slope_us_per_element: float
    intercept_us: float
    raw_samples: int
    distinct_sizes: int
    rmse_us: float

    def predict_us(self, size: int) -> float:
        if size <= 0:
            return 0.0
        return max(0.0, self.intercept_us + self.slope_us_per_element * size)


@dataclass(frozen=True)
class ShardCandidate:
    split_sizes: Mapping[str, int]
    predicted_latency_us: Mapping[str, float]
    predicted_makespan_us: float
    predicted_imbalance_us: float

    def to_dict(self) -> dict[str, object]:
        return {
            "split_sizes": dict(self.split_sizes),
            "predicted_latency_us": dict(self.predicted_latency_us),
            "predicted_makespan_us": self.predicted_makespan_us,
            "predicted_imbalance_us": self.predicted_imbalance_us,
        }


@dataclass(frozen=True)
class OptimizationResult:
    n_ff: int
    alignments: Mapping[str, int]
    minimums: Mapping[str, int]
    maximums: Mapping[str, int]
    models: Mapping[str, DeviceLatencyModel]
    feasible_candidates: int
    best: ShardCandidate
    alternatives: tuple[ShardCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_ff": self.n_ff,
            "alignments": dict(self.alignments),
            "minimums": dict(self.minimums),
            "maximums": dict(self.maximums),
            "models": {device: asdict(model) for device, model in self.models.items()},
            "feasible_candidates": self.feasible_candidates,
            "best": self.best.to_dict(),
            "alternatives": [candidate.to_dict() for candidate in self.alternatives],
        }


@dataclass(frozen=True)
class FfnObservation:
    input_tokens: int
    actual_state: str
    active_plan: str
    device: str
    shard_size: int
    latency_us: float
    query_id: int | None
    graph_id: int | None
    layer: int | None
    source: str


@dataclass(frozen=True)
class FfnObservationGroup:
    input_tokens: int
    actual_state: str
    active_plan: str
    observations: tuple[FfnObservation, ...]

    def size_latency_samples(self) -> dict[str, list[SizeLatencySample]]:
        grouped: dict[str, list[SizeLatencySample]] = defaultdict(list)
        for observation in self.observations:
            grouped[observation.device].append(
                SizeLatencySample(observation.shard_size, observation.latency_us)
            )
        return dict(grouped)


@dataclass(frozen=True)
class FfnObservationCatalog:
    groups: tuple[FfnObservationGroup, ...]
    scheduler_files: tuple[str, ...]
    skipped_without_shard_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "scheduler_files": list(self.scheduler_files),
            "skipped_without_shard_size": self.skipped_without_shard_size,
            "groups": [
                {
                    "input_tokens": group.input_tokens,
                    "actual_state": group.actual_state,
                    "active_plan": group.active_plan,
                    "observations": [asdict(observation) for observation in group.observations],
                }
                for group in self.groups
            ],
        }


def fit_device_latency_model(
    device: str,
    samples: Iterable[SizeLatencySample | Sequence[float] | Mapping[str, float]],
    *,
    fit_intercept: bool = False,
) -> DeviceLatencyModel:
    """Fit a non-negative latency model after collapsing repeats by median.

    Through-origin fitting is the safe default because most traces contain only
    one shard size per state.  An intercept is fitted only when requested and at
    least two distinct positive sizes exist.
    """

    parsed = [SizeLatencySample.from_value(sample) for sample in samples]
    if not parsed:
        raise ValueError(f"no latency samples for {device}")
    for sample in parsed:
        if sample.size <= 0 or not math.isfinite(sample.latency_us) or sample.latency_us <= 0:
            raise ValueError(f"invalid latency sample for {device}: {sample}")

    by_size: dict[int, list[float]] = defaultdict(list)
    for sample in parsed:
        by_size[sample.size].append(sample.latency_us)
    points = sorted((size, statistics.median(values)) for size, values in by_size.items())

    intercept = 0.0
    if fit_intercept and len(points) >= 2:
        mean_x = statistics.fmean(x for x, _ in points)
        mean_y = statistics.fmean(y for _, y in points)
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
        intercept = mean_y - slope * mean_x
        # Negative launch overhead is not physically useful for extrapolation.
        if slope <= 0 or intercept < 0:
            intercept = 0.0
            slope = sum(x * y for x, y in points) / sum(x * x for x, _ in points)
    else:
        slope = sum(x * y for x, y in points) / sum(x * x for x, _ in points)

    if not math.isfinite(slope) or slope <= 0:
        raise ValueError(f"non-positive fitted slope for {device}")
    residuals = [y - (intercept + slope * x) for x, y in points]
    rmse = math.sqrt(statistics.fmean(value * value for value in residuals))
    return DeviceLatencyModel(
        device=device,
        slope_us_per_element=slope,
        intercept_us=intercept,
        raw_samples=len(parsed),
        distinct_sizes=len(points),
        rmse_us=rmse,
    )


def fit_device_latency_models(
    samples: Mapping[str, Iterable[SizeLatencySample | Sequence[float] | Mapping[str, float]]],
    *,
    fit_intercept: bool = False,
) -> dict[str, DeviceLatencyModel]:
    devices = _ordered_devices(samples)
    if not devices:
        raise ValueError("at least one device sample set is required")
    return {
        device: fit_device_latency_model(device, samples[device], fit_intercept=fit_intercept)
        for device in devices
    }


def optimize_ffn_shards(
    n_ff: int,
    samples: Mapping[str, Iterable[SizeLatencySample | Sequence[float] | Mapping[str, float]]],
    *,
    alignments: Mapping[str, int] | None = None,
    minimums: Mapping[str, int] | None = None,
    maximums: Mapping[str, int] | None = None,
    fit_intercept: bool = False,
    max_alternatives: int = 8,
    max_relative_slowdown: float = 0.05,
    min_l1_distance: int = 0,
) -> OptimizationResult:
    """Fit device models and find an exactly aligned minimum-makespan split."""

    if n_ff <= 0:
        raise ValueError("n_ff must be positive")
    if max_alternatives < 0:
        raise ValueError("max_alternatives must be non-negative")
    if max_relative_slowdown < 0:
        raise ValueError("max_relative_slowdown must be non-negative")

    models = fit_device_latency_models(samples, fit_intercept=fit_intercept)
    devices = _ordered_devices(models)
    aligns = _normalized_constraint(alignments, devices, DEFAULT_ALIGNMENTS, "alignment")
    mins = _normalized_constraint(
        minimums,
        devices,
        {device: aligns[device] for device in devices},
        "minimum",
    )
    maxes = _normalized_constraint(
        maximums,
        devices,
        {device: n_ff for device in devices},
        "maximum",
    )
    for device in devices:
        alignment = aligns[device]
        if mins[device] < 0 or maxes[device] < mins[device]:
            raise ValueError(f"invalid bounds for {device}: {mins[device]}..{maxes[device]}")
        if mins[device] % alignment or maxes[device] % alignment:
            raise ValueError(f"bounds for {device} must be multiples of {alignment}")

    allocations = _enumerate_allocations(n_ff, devices, aligns, mins, maxes)
    candidates = [_score_allocation(allocation, models) for allocation in allocations]
    if not candidates:
        raise ValueError("no aligned allocation exactly covers n_ff under the supplied bounds")
    candidates.sort(key=_candidate_sort_key)
    best = candidates[0]
    alternatives = tuple(
        prune_shard_candidates(
            candidates[1:],
            best=best,
            max_candidates=max_alternatives,
            max_relative_slowdown=max_relative_slowdown,
            min_l1_distance=min_l1_distance,
        )
    )
    return OptimizationResult(
        n_ff=n_ff,
        alignments=aligns,
        minimums=mins,
        maximums=maxes,
        models=models,
        feasible_candidates=len(candidates),
        best=best,
        alternatives=alternatives,
    )


def prune_shard_candidates(
    candidates: Iterable[ShardCandidate],
    *,
    best: ShardCandidate,
    max_candidates: int,
    max_relative_slowdown: float,
    min_l1_distance: int = 0,
) -> list[ShardCandidate]:
    """Keep only useful near-optimal, sufficiently distinct alternatives."""

    if max_candidates <= 0:
        return []
    threshold = best.predicted_makespan_us * (1.0 + max_relative_slowdown)
    kept: list[ShardCandidate] = []
    anchors = [best]
    for candidate in sorted(candidates, key=_candidate_sort_key):
        if candidate.predicted_makespan_us > threshold:
            break
        if min_l1_distance and any(
            _allocation_distance(candidate.split_sizes, anchor.split_sizes) < min_l1_distance
            for anchor in anchors
        ):
            continue
        kept.append(candidate)
        anchors.append(candidate)
        if len(kept) >= max_candidates:
            break
    return kept


def parse_ffn_observations(
    paths: str | Path | Iterable[str | Path],
    *,
    policy: str | Path | Mapping[str, object] | None = None,
    template: str | Path | Mapping[str, object] | None = None,
    include_warmup: bool = False,
    token_override: int | None = None,
    prefill_only: bool = True,
) -> FfnObservationCatalog:
    """Join scheduler FFN rows with policy shard sizes.

    Observations are grouped by ``(actual clock state, active graph plan)``.
    When no policy is supplied, a run directory's ``config.txt`` is checked for
    ``POLICY_FILE=...``.  ``template`` is a fallback source for profile/root
    split sizes and does not override the primary policy.
    """

    scheduler_files = _discover_scheduler_files(paths)
    explicit_policy = _load_json_mapping(policy) if policy is not None else None
    template_mapping = _load_json_mapping(template) if template is not None else None
    if token_override is not None and token_override <= 0:
        raise ValueError("token_override must be positive")
    grouped: dict[tuple[int, str, str], list[FfnObservation]] = defaultdict(list)
    skipped = 0

    for scheduler_file in scheduler_files:
        run_policy = explicit_policy or _policy_from_run_directory(scheduler_file.parent)
        allocation_map: dict[str, dict[str, int]] = {}
        root_allocation: dict[str, int] | None = None
        for document in (template_mapping, run_policy):
            if document:
                doc_profiles, doc_root = _policy_allocations(document)
                allocation_map.update(doc_profiles)
                if doc_root:
                    root_allocation = doc_root

        for sample in read_ffn_branch_samples(
            scheduler_file,
            include_warmup=include_warmup,
            prefill_only=prefill_only,
        ):
            actual_state = sample.clock_profile or sample.plan_profile
            active_plan = sample.plan_profile or sample.clock_profile
            if not actual_state or not active_plan:
                skipped += 1
                continue
            input_tokens = token_override or sample.n_tokens
            if input_tokens is None or input_tokens <= 0:
                skipped += 1
                continue
            allocation = allocation_map.get(active_plan) or allocation_map.get(actual_state) or root_allocation
            shard_size = allocation.get(sample.device) if allocation else None
            if shard_size is None or shard_size <= 0:
                skipped += 1
                continue
            grouped[(input_tokens, actual_state, active_plan)].append(
                FfnObservation(
                    input_tokens=input_tokens,
                    actual_state=actual_state,
                    active_plan=active_plan,
                    device=sample.device,
                    shard_size=shard_size,
                    latency_us=sample.latency_us,
                    query_id=sample.query_id,
                    graph_id=sample.graph_id,
                    layer=sample.layer,
                    source=sample.source,
                )
            )

    groups = tuple(
        FfnObservationGroup(tokens, actual, plan, tuple(observations))
        for (tokens, actual, plan), observations in sorted(grouped.items())
    )
    return FfnObservationCatalog(
        groups=groups,
        scheduler_files=tuple(str(path) for path in scheduler_files),
        skipped_without_shard_size=skipped,
    )


def optimize_observation_group(
    n_ff: int,
    group: FfnObservationGroup,
    **optimizer_options: object,
) -> OptimizationResult:
    """Convenience adapter from parsed trace observations to the optimizer."""

    return optimize_ffn_shards(n_ff, group.size_latency_samples(), **optimizer_options)


def _enumerate_allocations(
    total: int,
    devices: Sequence[str],
    alignments: Mapping[str, int],
    minimums: Mapping[str, int],
    maximums: Mapping[str, int],
) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []

    def visit(index: int, remaining: int, current: dict[str, int]) -> None:
        device = devices[index]
        if index == len(devices) - 1:
            value = remaining
            if minimums[device] <= value <= maximums[device] and value % alignments[device] == 0:
                result.append({**current, device: value})
            return

        later = devices[index + 1 :]
        later_min = sum(minimums[item] for item in later)
        later_max = sum(maximums[item] for item in later)
        low = max(minimums[device], remaining - later_max)
        high = min(maximums[device], remaining - later_min)
        alignment = alignments[device]
        low = ((low + alignment - 1) // alignment) * alignment
        for value in range(low, high + 1, alignment):
            current[device] = value
            visit(index + 1, remaining - value, current)
        current.pop(device, None)

    visit(0, total, {})
    return result


def _score_allocation(
    allocation: Mapping[str, int],
    models: Mapping[str, DeviceLatencyModel],
) -> ShardCandidate:
    latencies = {device: models[device].predict_us(size) for device, size in allocation.items()}
    active = [latency for device, latency in latencies.items() if allocation[device] > 0]
    makespan = max(active, default=0.0)
    imbalance = makespan - min(active, default=0.0)
    return ShardCandidate(dict(allocation), latencies, makespan, imbalance)


def _candidate_sort_key(candidate: ShardCandidate) -> tuple[object, ...]:
    widths = tuple(candidate.split_sizes.get(device, 0) for device in DEVICE_ORDER)
    return (candidate.predicted_makespan_us, candidate.predicted_imbalance_us, widths)


def _allocation_distance(left: Mapping[str, int], right: Mapping[str, int]) -> int:
    return sum(abs(left.get(device, 0) - right.get(device, 0)) for device in set(left) | set(right))


def _ordered_devices(mapping: Mapping[str, object]) -> tuple[str, ...]:
    unknown = sorted(set(mapping) - set(DEVICE_ORDER))
    if unknown:
        raise ValueError(f"unsupported devices: {', '.join(unknown)}")
    return tuple(device for device in DEVICE_ORDER if device in mapping)


def _normalized_constraint(
    supplied: Mapping[str, int] | None,
    devices: Sequence[str],
    defaults: Mapping[str, int],
    name: str,
) -> dict[str, int]:
    supplied = supplied or {}
    unknown = set(supplied) - set(devices)
    if unknown:
        raise ValueError(f"{name} supplied for inactive devices: {', '.join(sorted(unknown))}")
    result = {device: int(supplied.get(device, defaults[device])) for device in devices}
    if name == "alignment" and any(value <= 0 for value in result.values()):
        raise ValueError("alignments must be positive")
    return result


def _discover_scheduler_files(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    result: set[Path] = set()
    for value in values:
        path = Path(value)
        if path.is_dir():
            data_directory = path / "device-output" if (path / "device-output").is_dir() else path
            candidate = data_directory / "scheduler_trace.csv"
        else:
            candidate = path
        if candidate.name != "scheduler_trace.csv":
            raise ValueError(f"expected a run directory or scheduler_trace.csv, got {path}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        result.add(candidate.resolve())
    return sorted(result)


def _load_json_mapping(value: str | Path | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    with Path(value).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise ValueError(f"JSON document must be an object: {value}")
    return document


def _policy_from_run_directory(run_directory: Path) -> Mapping[str, object] | None:
    for directory in (run_directory, run_directory.parent):
        snapshot = directory / "input-policy.json"
        if snapshot.is_file():
            return _load_json_mapping(snapshot)
    config = run_directory / "config.txt"
    if not config.is_file():
        return None
    policy_path: Path | None = None
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() in {
            "POLICY_FILE",
            "POLICY",
            "BACKEND_POLICY_CONFIG",
            "BACKEND_POLICY_DEFAULT_CONFIG",
        } and value.strip():
            policy_path = Path(value.strip())
            break
    if policy_path is None:
        return None
    if policy_path.is_absolute():
        return _load_json_mapping(policy_path) if policy_path.is_file() else None
    for base in (run_directory, *run_directory.parents):
        candidate = (base / policy_path).resolve()
        if candidate.is_file():
            return _load_json_mapping(candidate)
    return None


def _policy_allocations(document: Mapping[str, object]) -> tuple[dict[str, dict[str, int]], dict[str, int] | None]:
    profiles: dict[str, dict[str, int]] = {}
    raw_profiles = document.get("profiles", {})
    if isinstance(raw_profiles, Mapping):
        for name, profile in raw_profiles.items():
            if not isinstance(profile, Mapping):
                continue
            ffn = profile.get("ffn_parallel", {})
            if isinstance(ffn, Mapping):
                split = _split_sizes(ffn.get("split_sizes"))
                if split:
                    profiles[str(name)] = split
    root_ffn = document.get("ffn_parallel", {})
    root = _split_sizes(root_ffn.get("split_sizes")) if isinstance(root_ffn, Mapping) else None
    return profiles, root


def _split_sizes(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, int] = {}
    for device in DEVICE_ORDER:
        try:
            result[device] = int(value[device])
        except (KeyError, TypeError, ValueError):
            continue
    return result or None


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ff", type=int, required=True)
    parser.add_argument(
        "--samples-json",
        type=Path,
        required=True,
        help='JSON object such as {"cpu":[{"size":1600,"latency_us":7000}], ...}',
    )
    parser.add_argument("--allow-zero", action="store_true", help="allow the optimizer to drop a device")
    parser.add_argument("--fit-intercept", action="store_true")
    args = parser.parse_args()
    samples = _load_json_mapping(args.samples_json)
    minimums = {device: 0 for device in samples} if args.allow_zero else None
    result = optimize_ffn_shards(
        args.n_ff,
        samples,  # type: ignore[arg-type]
        minimums=minimums,
        fit_intercept=args.fit_intercept,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
