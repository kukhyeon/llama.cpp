#!/usr/bin/env python3
"""Parse TrePrefill traces into clock states and FFN branch samples.

The module deliberately has no dependency on the experiment runner.  It can be
imported by an autotuner, or invoked directly to inspect a pair of CSV files.

``hardware_stats.csv`` and ``scheduler_trace.csv`` use different units for the
CPU clocks in existing logs.  Scheduler clocks are kHz, while the hardware log
usually records values such as ``4473.6`` (MHz).  The normalization helpers in
this file convert both forms to a canonical ``ClockPoint``:

* Prime and Performance/Gold CPU clocks: kHz
* GPU clock: Hz

``gpu_max_clock`` in older hardware logs is a KGSL constraint rather than a
kernel-level utilization-weighted frequency sample.  It is still useful for
identifying the reachable DVFS state, but callers should not interpret it as a
per-kernel effective clock.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


_PROFILE_RE = re.compile(r"(p\d+-g\d+-gpu\d+)", re.IGNORECASE)


@dataclass(frozen=True, order=True)
class ClockPoint:
    """Canonical clock tuple used by routing and tuning."""

    prime_khz: int
    gold_khz: int
    gpu_hz: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.prime_khz, self.gold_khz, self.gpu_hz)


@dataclass(frozen=True)
class SchedulerClockSample:
    profile: str | None
    point: ClockPoint
    query_id: int | None
    graph_id: int | None
    layer: int | None
    n_tokens: int | None
    source: str


@dataclass(frozen=True)
class HardwareClockSample:
    point: ClockPoint
    time_ms: float | None
    prime_column: str
    gold_column: str
    gpu_column: str
    source: str


@dataclass(frozen=True)
class ReachableClockState:
    """A clock state observed in at least one supplied trace."""

    profile: str
    point: ClockPoint
    aliases: tuple[str, ...]
    scheduler_samples: int
    hardware_samples: int
    first_hardware_time_ms: float | None
    last_hardware_time_ms: float | None
    distinct_queries: int
    input_tokens: tuple[int, ...]
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class ClockStateCatalog:
    """Serializable result returned by :func:`discover_clock_states`."""

    states: tuple[ReachableClockState, ...]
    scheduler_files: tuple[str, ...]
    hardware_files: tuple[str, ...]
    token_override: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "scheduler_files": list(self.scheduler_files),
            "hardware_files": list(self.hardware_files),
            "token_override": self.token_override,
            "states": [asdict(state) for state in self.states],
            "states_by_token": {
                str(token): [state.profile for state in states]
                for token, states in self.by_token().items()
            },
        }

    def by_token(self) -> dict[int, tuple[ReachableClockState, ...]]:
        grouped: dict[int, list[ReachableClockState]] = defaultdict(list)
        for state in self.states:
            for token in state.input_tokens:
                grouped[token].append(state)
        return {token: tuple(states) for token, states in sorted(grouped.items())}


@dataclass(frozen=True)
class FfnBranchSample:
    """One synchronized FFN device branch from scheduler_trace.csv."""

    clock_profile: str | None
    plan_profile: str | None
    device: str
    backend: str
    latency_us: float
    query_id: int | None
    graph_id: int | None
    group_id: int | None
    layer: int | None
    n_tokens: int | None
    source: str


@dataclass(frozen=True)
class FfnLatencySummary:
    clock_profile: str
    device: str
    samples: int
    mean_us: float
    median_us: float
    p95_us: float
    min_us: float
    max_us: float


def normalize_cpu_khz(value: str | int | float) -> int:
    """Normalize a CPU frequency expressed as MHz, kHz, or Hz to kHz.

    Existing hardware CSVs store values around 4473.6 (MHz), scheduler traces
    store 4473600 (kHz), and sysfs may expose 4473600000 (Hz).  The ranges do
    not overlap for mobile CPU clocks, so magnitude is a reliable discriminator.
    """

    number = _positive_number(value, "CPU frequency")
    if number < 10_000:  # MHz, often with one decimal place
        return int(round(number * 1_000))
    if number < 10_000_000:  # kHz
        return int(round(number))
    return int(round(number / 1_000))  # Hz


def normalize_gpu_hz(value: str | int | float) -> int:
    """Normalize a GPU frequency expressed as MHz, kHz, or Hz to Hz."""

    number = _positive_number(value, "GPU frequency")
    if number < 10_000:  # MHz
        return int(round(number * 1_000_000))
    if number < 10_000_000:  # kHz
        return int(round(number * 1_000))
    return int(round(number))  # Hz


def read_scheduler_clock_samples(
    path: str | Path,
    *,
    prefill_only: bool = True,
) -> list[SchedulerClockSample]:
    """Read rows that contain a valid measured scheduler clock tuple.

    If a row has a non-empty ``phase`` field, the default accepts only
    ``prefill``.  Older traces without that column (or with it empty) retain
    their previous behavior.
    """

    result: list[SchedulerClockSample] = []
    source = str(Path(path).resolve())
    for row in _csv_rows(path):
        if prefill_only and not _prefill_compatible(row):
            continue
        try:
            point = ClockPoint(
                normalize_cpu_khz(row.get("actual_prime_khz", "")),
                normalize_cpu_khz(row.get("actual_gold_khz", "")),
                normalize_gpu_hz(row.get("actual_gpu_hz", "")),
            )
        except ValueError:
            # Warm-up/build rows use -1 or empty strings for unavailable clocks.
            continue

        profile = _clean(row.get("actual_clock_profile"))
        result.append(
            SchedulerClockSample(
                profile=profile,
                point=point,
                query_id=_optional_int(row.get("query_id")),
                graph_id=_optional_int(row.get("graph_id")),
                layer=_optional_int(row.get("layer")),
                n_tokens=_optional_int(row.get("n_tokens")),
                source=source,
            )
        )
    return result


def read_hardware_clock_samples(
    path: str | Path,
    *,
    prime_column: str | None = None,
    gold_column: str | None = None,
    gpu_column: str | None = None,
) -> list[HardwareClockSample]:
    """Read clock tuples from hardware_stats.csv.

    Column names can be overridden for a different device.  For Galaxy S25
    traces the default resolver uses CPU6 for the Prime cluster and CPU0 for the
    Performance/Gold cluster.
    """

    source = str(Path(path).resolve())
    rows = _csv_rows(path)
    try:
        first = next(rows)
    except StopIteration:
        return []

    fields = tuple(first.keys())
    prime = _choose_column(
        fields,
        prime_column,
        ("actual_prime_khz", "prime_cur_freq", "cpu6_cur_freq", "cpu6_max_freq"),
        "Prime CPU clock",
    )
    gold = _choose_column(
        fields,
        gold_column,
        ("actual_gold_khz", "gold_cur_freq", "cpu0_cur_freq", "cpu0_max_freq"),
        "Performance/Gold CPU clock",
    )
    gpu = _choose_column(
        fields,
        gpu_column,
        ("actual_gpu_hz", "gpu_cur_clock", "gpu_cur_freq", "gpu_max_clock", "gpu_min_clock"),
        "GPU clock",
    )

    result: list[HardwareClockSample] = []
    for row in _prepend(first, rows):
        try:
            point = ClockPoint(
                normalize_cpu_khz(row.get(prime, "")),
                normalize_cpu_khz(row.get(gold, "")),
                normalize_gpu_hz(row.get(gpu, "")),
            )
        except ValueError:
            continue
        result.append(
            HardwareClockSample(
                point=point,
                time_ms=_optional_float(row.get("Time") or row.get("time_ms") or row.get("timestamp_ms")),
                prime_column=prime,
                gold_column=gold,
                gpu_column=gpu,
                source=source,
            )
        )
    return result


def collect_reachable_clock_states(
    scheduler_samples: Iterable[SchedulerClockSample] = (),
    hardware_samples: Iterable[HardwareClockSample] = (),
    *,
    cpu_tolerance_khz: int = 5_000,
    gpu_tolerance_hz: int = 5_000_000,
    include_hardware_only: bool = True,
    token_override: int | None = None,
) -> list[ReachableClockState]:
    """Merge observations into unique ``(input token, clock)`` states.

    Hardware samples are snapped to a scheduler-labelled point only when all
    three clocks fall within the supplied tolerances.  An unmatched hardware
    tuple remains an explicit ``observed-*`` state instead of being silently
    discarded when ``include_hardware_only`` is enabled.  Scheduler rows from
    different input lengths are never pooled for sample/query eligibility.
    """

    if token_override is not None and token_override <= 0:
        raise ValueError("token_override must be positive")
    scheduler = list(scheduler_samples)
    hardware = list(hardware_samples)
    scheduler_points = sorted({sample.point for sample in scheduler})

    StateKey = tuple[int | None, ClockPoint]
    sched_by_state: dict[StateKey, list[SchedulerClockSample]] = defaultdict(list)
    hardware_by_state: dict[StateKey, list[HardwareClockSample]] = defaultdict(list)
    tokens_by_run: dict[Path, set[int]] = defaultdict(set)
    for sample in scheduler:
        # Negative query IDs are graph-build/warm-up rows and must not inflate
        # the evidence for a real input-token state.
        if sample.query_id is None or sample.query_id < 0:
            continue
        token = token_override or sample.n_tokens
        if token is not None and token <= 0:
            token = None
        sched_by_state[(token, sample.point)].append(sample)
        if token is not None:
            tokens_by_run[Path(sample.source).parent].add(token)
    for sample in hardware:
        point = _nearest_clock_point(
            sample.point,
            scheduler_points,
            cpu_tolerance_khz=cpu_tolerance_khz,
            gpu_tolerance_hz=gpu_tolerance_hz,
        )
        run_tokens = tokens_by_run.get(Path(sample.source).parent)
        # A normal trial contains one token length.  If a custom trace contains
        # several, attach its clock evidence to each token bucket; scheduler
        # sample/query thresholds remain independent and authoritative.
        tokens: Iterable[int | None] = sorted(run_tokens) if run_tokens else (token_override,)
        for token in tokens:
            hardware_by_state[(token, point or sample.point)].append(sample)

    result: list[ReachableClockState] = []
    states = set(sched_by_state)
    if include_hardware_only:
        states |= set(hardware_by_state)
    for token, point in sorted(
        states,
        key=lambda item: (-1 if item[0] is None else item[0], item[1]),
        reverse=True,
    ):
        sched_rows = sched_by_state.get((token, point), [])
        hardware_rows = hardware_by_state.get((token, point), [])
        labels = Counter(sample.profile for sample in sched_rows if sample.profile)
        aliases = tuple(label for label, _ in sorted(labels.items(), key=lambda item: (-item[1], item[0])))
        profile = aliases[0] if aliases else _observed_profile_name(point)
        times = [sample.time_ms for sample in hardware_rows if sample.time_ms is not None]
        queries = {
            (sample.source, sample.query_id)
            for sample in sched_rows
            if sample.query_id is not None and sample.query_id >= 0
        }
        sources = sorted(
            {sample.source for sample in sched_rows} | {sample.source for sample in hardware_rows}
        )
        result.append(
            ReachableClockState(
                profile=profile,
                point=point,
                aliases=aliases,
                scheduler_samples=len(sched_rows),
                hardware_samples=len(hardware_rows),
                first_hardware_time_ms=min(times) if times else None,
                last_hardware_time_ms=max(times) if times else None,
                distinct_queries=len(queries),
                input_tokens=(token,) if token is not None else (),
                source_files=tuple(sources),
            )
        )
    return result


def parse_reachable_clock_states(
    scheduler_trace: str | Path | None,
    hardware_stats: str | Path | None,
    include_hardware_only: bool = True,
    **hardware_columns: str,
) -> list[ReachableClockState]:
    """Convenience wrapper for reading and merging two optional CSV files."""

    scheduler = read_scheduler_clock_samples(scheduler_trace) if scheduler_trace else []
    hardware = read_hardware_clock_samples(hardware_stats, **hardware_columns) if hardware_stats else []
    return collect_reachable_clock_states(
        scheduler,
        hardware,
        include_hardware_only=include_hardware_only,
    )


def discover_clock_states(
    paths: str | Path | Iterable[str | Path],
    *,
    token_override: int | None = None,
    min_samples: int = 1,
    min_queries: int = 1,
    allowed_gpu_indices: set[int] | frozenset[int] | None = None,
    allowed_profiles: set[str] | frozenset[str] | None = None,
    include_hardware_only: bool = False,
) -> ClockStateCatalog:
    """Discover runtime-reachable states from one or more run paths.

    Scheduler observations are authoritative by default.  Hardware-only tuples
    are retained only with ``include_hardware_only=True``.  ``min_samples`` is
    the minimum number of scheduler rows and ``min_queries`` counts distinct
    ``(source file, query_id)`` pairs, so repeated query IDs across runs do not
    collapse into one query.
    """

    if token_override is not None and token_override <= 0:
        raise ValueError("token_override must be positive")
    if min_samples < 0 or min_queries < 0:
        raise ValueError("minimum sample/query counts must be non-negative")

    scheduler_files, hardware_files = _discover_trace_files(paths)
    scheduler_samples = [
        sample
        for path in scheduler_files
        for sample in read_scheduler_clock_samples(path)
    ]
    hardware_samples = [
        sample
        for path in hardware_files
        for sample in read_hardware_clock_samples(path)
    ]
    states = collect_reachable_clock_states(
        scheduler_samples,
        hardware_samples,
        include_hardware_only=include_hardware_only,
        token_override=token_override,
    )

    filtered: list[ReachableClockState] = []
    normalized_profiles = {profile.lower() for profile in allowed_profiles} if allowed_profiles else None
    for state in states:
        # An explicit hardware-only request should not also require scheduler
        # evidence. Scheduler-backed states still have to satisfy both
        # thresholds, so sparse transient states cannot enter a policy catalog.
        if state.scheduler_samples > 0 and (
            state.scheduler_samples < min_samples or state.distinct_queries < min_queries
        ):
            continue
        if state.scheduler_samples == 0 and not include_hardware_only:
            continue
        if normalized_profiles is not None and not ({name.lower() for name in state.aliases} | {state.profile.lower()}) & normalized_profiles:
            continue
        gpu_index = _profile_gpu_index(state.profile)
        if allowed_gpu_indices is not None and gpu_index not in allowed_gpu_indices:
            continue
        filtered.append(state)

    return ClockStateCatalog(
        states=tuple(filtered),
        scheduler_files=tuple(str(path) for path in scheduler_files),
        hardware_files=tuple(str(path) for path in hardware_files),
        token_override=token_override,
    )


def read_ffn_branch_samples(
    path: str | Path,
    *,
    include_warmup: bool = False,
    prefill_only: bool = True,
) -> list[FfnBranchSample]:
    """Extract synchronized CPU/GPU/NPU FFN branch latencies.

    Rows with missing or negative ``compute_wall_us`` are asynchronous submits,
    not device service times, and are intentionally excluded.
    """

    result: list[FfnBranchSample] = []
    source = str(Path(path).resolve())
    for row in _csv_rows(path):
        if prefill_only and not _prefill_compatible(row):
            continue
        kind = (_clean(row.get("parallel_group_kind")) or "").lower()
        is_ffn = kind == "ffn" or _truthy(row.get("is_ffn_group"))
        if not is_ffn:
            continue

        latency = _optional_float(row.get("compute_wall_us"))
        if latency is None or latency < 0:
            continue
        query_id = _optional_int(row.get("query_id"))
        if not include_warmup and (query_id is None or query_id < 0):
            continue

        branch = _clean(row.get("parallel_branch")) or _clean(row.get("ffn_branch"))
        backend = _clean(row.get("backend")) or ""
        device = _branch_device(branch, backend)
        if device is None:
            continue
        plan_match = _PROFILE_RE.search(branch or "")
        result.append(
            FfnBranchSample(
                clock_profile=_clean(row.get("actual_clock_profile")),
                plan_profile=plan_match.group(1).lower() if plan_match else None,
                device=device,
                backend=backend,
                latency_us=latency,
                query_id=query_id,
                graph_id=_optional_int(row.get("graph_id")),
                group_id=_optional_int(row.get("group_id")),
                layer=_optional_int(row.get("layer")),
                n_tokens=_optional_int(row.get("n_tokens")),
                source=source,
            )
        )
    return result


def summarize_ffn_latencies(samples: Iterable[FfnBranchSample]) -> list[FfnLatencySummary]:
    """Summarize FFN latency by measured clock profile and device."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sample in samples:
        profile = sample.clock_profile or sample.plan_profile
        if profile:
            grouped[(profile, sample.device)].append(sample.latency_us)

    result: list[FfnLatencySummary] = []
    for (profile, device), values in sorted(grouped.items()):
        ordered = sorted(values)
        result.append(
            FfnLatencySummary(
                clock_profile=profile,
                device=device,
                samples=len(ordered),
                mean_us=statistics.fmean(ordered),
                median_us=statistics.median(ordered),
                p95_us=_percentile(ordered, 0.95),
                min_us=ordered[0],
                max_us=ordered[-1],
            )
        )
    return result


def _csv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    handle = Path(path).open("r", encoding="utf-8-sig", newline="")
    try:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        for row in reader:
            # Some hardware logs end their header with a comma.  DictReader then
            # creates a None/empty field which is irrelevant to this parser.
            yield {str(key).strip(): (value or "").strip() for key, value in row.items() if key}
    finally:
        handle.close()


def _prepend(first: Mapping[str, str], rest: Iterable[Mapping[str, str]]) -> Iterator[Mapping[str, str]]:
    yield first
    yield from rest


def _positive_number(value: str | int | float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"invalid {name}: {value!r}")
    return number


def _clean(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prefill_compatible(row: Mapping[str, str]) -> bool:
    phase = _clean(row.get("phase"))
    return phase is None or phase.lower() == "prefill"


def _choose_column(
    fields: Sequence[str],
    explicit: str | None,
    candidates: Sequence[str],
    description: str,
) -> str:
    if explicit:
        if explicit not in fields:
            raise ValueError(f"{description} column {explicit!r} not found")
        return explicit
    for candidate in candidates:
        if candidate in fields:
            return candidate
    raise ValueError(f"no {description} column found; tried {', '.join(candidates)}")


def _discover_trace_files(
    paths: str | Path | Iterable[str | Path],
) -> tuple[list[Path], list[Path]]:
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    scheduler: set[Path] = set()
    hardware: set[Path] = set()
    for value in values:
        path = Path(value)
        if path.is_dir():
            data_directory = path / "device-output" if (path / "device-output").is_dir() else path
            scheduler_path = data_directory / "scheduler_trace.csv"
            hardware_path = data_directory / "hardware_stats.csv"
            if scheduler_path.is_file():
                scheduler.add(scheduler_path.resolve())
            if hardware_path.is_file():
                hardware.add(hardware_path.resolve())
            if not scheduler_path.is_file() and not hardware_path.is_file():
                raise FileNotFoundError(
                    f"no scheduler_trace.csv or hardware_stats.csv in {path} or its device-output directory"
                )
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name == "scheduler_trace.csv":
            scheduler.add(path.resolve())
            sibling = path.with_name("hardware_stats.csv")
            if sibling.is_file():
                hardware.add(sibling.resolve())
        elif path.name == "hardware_stats.csv":
            hardware.add(path.resolve())
            sibling = path.with_name("scheduler_trace.csv")
            if sibling.is_file():
                scheduler.add(sibling.resolve())
        else:
            raise ValueError(f"unsupported trace filename: {path}")
    return sorted(scheduler), sorted(hardware)


def _nearest_clock_point(
    point: ClockPoint,
    candidates: Sequence[ClockPoint],
    *,
    cpu_tolerance_khz: int,
    gpu_tolerance_hz: int,
) -> ClockPoint | None:
    eligible = [
        candidate
        for candidate in candidates
        if abs(point.prime_khz - candidate.prime_khz) <= cpu_tolerance_khz
        and abs(point.gold_khz - candidate.gold_khz) <= cpu_tolerance_khz
        and abs(point.gpu_hz - candidate.gpu_hz) <= gpu_tolerance_hz
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (
            abs(point.prime_khz - candidate.prime_khz) / max(cpu_tolerance_khz, 1)
            + abs(point.gold_khz - candidate.gold_khz) / max(cpu_tolerance_khz, 1)
            + abs(point.gpu_hz - candidate.gpu_hz) / max(gpu_tolerance_hz, 1),
            candidate,
        ),
    )


def _observed_profile_name(point: ClockPoint) -> str:
    return f"observed-p{point.prime_khz}-g{point.gold_khz}-gpu{point.gpu_hz}"


def _profile_gpu_index(profile: str) -> int | None:
    match = _PROFILE_RE.search(profile)
    if not match:
        return None
    gpu = re.search(r"gpu(\d+)", match.group(1), re.IGNORECASE)
    return int(gpu.group(1)) if gpu else None


def _branch_device(branch: str | None, backend: str) -> str | None:
    branch_lower = (branch or "").lower()
    for device in ("cpu", "gpu", "npu"):
        if branch_lower == device or branch_lower.endswith(f".{device}"):
            return device
    backend_lower = backend.lower()
    if backend_lower.startswith("cpu"):
        return "cpu"
    if "opencl" in backend_lower or backend_lower.startswith("gpu"):
        return "gpu"
    if "htp" in backend_lower or "hexagon" in backend_lower or backend_lower.startswith("npu"):
        return "npu"
    return None


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-trace", type=Path)
    parser.add_argument("--hardware-stats", type=Path)
    parser.add_argument("--include-ffn", action="store_true", help="also print FFN latency summaries")
    args = parser.parse_args()
    if not args.scheduler_trace and not args.hardware_stats:
        parser.error("at least one trace path is required")

    states = parse_reachable_clock_states(args.scheduler_trace, args.hardware_stats)
    output: dict[str, object] = {"reachable_clock_states": [asdict(state) for state in states]}
    if args.include_ffn:
        if not args.scheduler_trace:
            parser.error("--include-ffn requires --scheduler-trace")
        output["ffn_latency"] = [
            asdict(summary)
            for summary in summarize_ffn_latencies(read_ffn_branch_samples(args.scheduler_trace))
        ]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
