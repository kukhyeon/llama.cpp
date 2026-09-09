#!/usr/bin/env python3
"""End-to-end host CLI for the TrePrefill policy-autotuning building blocks.

The command intentionally keeps device execution separate (``trial_plan.py``)
from offline processing.  Its five subcommands form a reproducible pipeline:

``inspect-model`` -> ``discover`` -> ``recommend`` -> ``generate`` -> ``validate``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .ffn_optimizer import SizeLatencySample, optimize_ffn_shards, parse_ffn_observations
    from .model_descriptor import ModelDescriptor, ModelDescriptorError, load_model_descriptor
    from .policy_generator import (
        ClockPoint as PolicyClockPoint,
        PolicyGenerationError,
        generate_policy,
        load_plans,
        load_policy_template,
        validate_generated_policy,
        write_policy,
    )
    from .profile_pruner import (
        ClockProfileCandidate,
        ObservedProfileState,
        prune_clock_profiles,
    )
    from .trace_parser import ClockPoint as TraceClockPoint, discover_clock_states
except ImportError:  # Direct execution from this directory.
    from ffn_optimizer import SizeLatencySample, optimize_ffn_shards, parse_ffn_observations
    from model_descriptor import ModelDescriptor, ModelDescriptorError, load_model_descriptor
    from policy_generator import (
        ClockPoint as PolicyClockPoint,
        PolicyGenerationError,
        generate_policy,
        load_plans,
        load_policy_template,
        validate_generated_policy,
        write_policy,
    )
    from profile_pruner import ClockProfileCandidate, ObservedProfileState, prune_clock_profiles
    from trace_parser import ClockPoint as TraceClockPoint, discover_clock_states


_PROFILE_RE = re.compile(r"(?:^|[^A-Za-z0-9])p(\d+)-g(\d+)-gpu(\d+)(?:$|[^A-Za-z0-9])")
_CANONICAL_DEVICES = ("cpu", "gpu", "npu")


class AutotuneError(ValueError):
    """An input set cannot be converted into a valid policy recommendation."""


@dataclass(frozen=True)
class _TemplateClock:
    profile: str
    prime_index: int
    gold_index: int
    gpu_index: int
    prime_khz: int
    gold_khz: int
    gpu_hz: int

    @property
    def frequencies(self) -> tuple[int, int, int]:
        return (self.prime_khz, self.gold_khz, self.gpu_hz)


@dataclass(frozen=True)
class _CatalogClock:
    profile: str
    aliases: tuple[str, ...]
    prime_khz: int
    gold_khz: int
    gpu_hz: int
    scheduler_samples: int
    distinct_queries: int
    input_tokens: tuple[int, ...]

    @property
    def frequencies(self) -> tuple[int, int, int]:
        return (self.prime_khz, self.gold_khz, self.gpu_hz)


@dataclass(frozen=True)
class _ResolvedClock:
    profile: str
    prime_index: int
    gold_index: int
    gpu_index: int
    prime_khz: int
    gold_khz: int
    gpu_hz: int
    source: str
    scheduler_samples: int

    @property
    def frequencies(self) -> tuple[int, int, int]:
        return (self.prime_khz, self.gold_khz, self.gpu_hz)

    def to_plan_dict(self) -> dict[str, object]:
        return {
            "prime": {"index": self.prime_index, "khz": self.prime_khz},
            "gold": {"index": self.gold_index, "khz": self.gold_khz},
            "gpu": {"index": self.gpu_index, "hz": self.gpu_hz},
        }


@dataclass(frozen=True)
class _FfnLayout:
    split_id_by_device: Mapping[str, str]
    alignments: Mapping[str, int]

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(device for device in _CANONICAL_DEVICES if device in self.split_id_by_device)


def _json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AutotuneError(f"failed to read JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise AutotuneError(f"JSON document must contain an object: {source}")
    return value


def _write_json(data: Mapping[str, object], path: Path | None, *, force: bool) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if path is None:
        print(text, end="")
        return
    destination = path.expanduser().resolve()
    if destination.exists() and not force:
        raise AutotuneError(f"output already exists (use --force to replace it): {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


def _nonnegative_int(value: object, source: str) -> int:
    if isinstance(value, bool):
        raise AutotuneError(f"{source} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AutotuneError(f"{source} must be a non-negative integer") from exc
    if result < 0:
        raise AutotuneError(f"{source} must be a non-negative integer")
    return result


def _positive_int(value: object, source: str) -> int:
    result = _nonnegative_int(value, source)
    if result == 0:
        raise AutotuneError(f"{source} must be positive")
    return result


def _profile_indices(*names: str) -> tuple[int, int, int] | None:
    for name in names:
        match = _PROFILE_RE.search(name)
        if not match:
            continue
        indices = tuple(int(part) for part in match.groups())
        # Prevent an ``observed-p4473600-g...`` frequency label from being
        # mistaken for a DVFS-index label.
        if all(index <= 255 for index in indices):
            return indices  # type: ignore[return-value]
    return None


def _template_clocks(template: Mapping[str, Any]) -> dict[str, _TemplateClock]:
    raw_profiles = template.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise AutotuneError("template.profiles must be an object")
    result: dict[str, _TemplateClock] = {}
    for raw_name, raw_profile in raw_profiles.items():
        name = str(raw_name)
        if not isinstance(raw_profile, Mapping) or not isinstance(
            raw_profile.get("clock_point"), Mapping
        ):
            continue
        try:
            point = PolicyClockPoint.from_dict(
                raw_profile["clock_point"], f"template.profiles.{name}.clock_point"
            )
        except PolicyGenerationError as exc:
            raise AutotuneError(str(exc)) from exc
        result[name] = _TemplateClock(
            profile=name,
            prime_index=point.prime.index,
            gold_index=point.gold.index,
            gpu_index=point.gpu.index,
            prime_khz=point.prime.frequency,
            gold_khz=point.gold.frequency,
            gpu_hz=point.gpu.frequency,
        )
    if not result:
        raise AutotuneError("template has no profile with a usable clock_point")
    return result


def _catalog_clocks(catalog: Mapping[str, Any]) -> list[_CatalogClock]:
    raw_states = catalog.get("states", catalog.get("reachable_clock_states"))
    if not isinstance(raw_states, list):
        raise AutotuneError("clock catalog must contain a states array")
    result: list[_CatalogClock] = []
    for index, raw_state in enumerate(raw_states):
        source = f"catalog.states[{index}]"
        if not isinstance(raw_state, Mapping):
            raise AutotuneError(f"{source} must be an object")
        profile = raw_state.get("profile")
        point = raw_state.get("point")
        if not isinstance(profile, str) or not profile or not isinstance(point, Mapping):
            raise AutotuneError(f"{source} requires profile and point")
        aliases_value = raw_state.get("aliases", [])
        tokens_value = raw_state.get("input_tokens", [])
        if not isinstance(aliases_value, (list, tuple)) or not all(
            isinstance(item, str) for item in aliases_value
        ):
            raise AutotuneError(f"{source}.aliases must be a string array")
        if not isinstance(tokens_value, (list, tuple)):
            raise AutotuneError(f"{source}.input_tokens must be an integer array")
        result.append(
            _CatalogClock(
                profile=profile,
                aliases=tuple(aliases_value),
                prime_khz=_positive_int(point.get("prime_khz"), f"{source}.point.prime_khz"),
                gold_khz=_positive_int(point.get("gold_khz"), f"{source}.point.gold_khz"),
                gpu_hz=_positive_int(point.get("gpu_hz"), f"{source}.point.gpu_hz"),
                scheduler_samples=_nonnegative_int(
                    raw_state.get("scheduler_samples", 0), f"{source}.scheduler_samples"
                ),
                distinct_queries=_nonnegative_int(
                    raw_state.get("distinct_queries", 0), f"{source}.distinct_queries"
                ),
                input_tokens=tuple(
                    _positive_int(item, f"{source}.input_tokens") for item in tokens_value
                ),
            )
        )
    return result


def _resolve_clock(
    state_name: str,
    input_tokens: int,
    catalog: Sequence[_CatalogClock],
    template: Mapping[str, _TemplateClock],
) -> _ResolvedClock:
    matching_catalog = [
        state
        for state in catalog
        if state_name == state.profile or state_name in state.aliases
    ]
    matching_catalog.sort(
        key=lambda state: (
            input_tokens in state.input_tokens,
            state_name == state.profile,
            state.scheduler_samples,
            state.distinct_queries,
        ),
        reverse=True,
    )
    catalog_state = matching_catalog[0] if matching_catalog else None
    exact_template = template.get(state_name)

    frequencies = (
        catalog_state.frequencies
        if catalog_state is not None
        else exact_template.frequencies
        if exact_template is not None
        else None
    )
    if frequencies is None:
        raise AutotuneError(
            f"clock state {state_name!r} ({input_tokens} tokens) is absent from both "
            "the catalog and template"
        )

    frequency_template = next(
        (entry for entry in template.values() if entry.frequencies == frequencies), None
    )
    label_names = [state_name]
    if catalog_state is not None:
        label_names.extend([catalog_state.profile, *catalog_state.aliases])
    label_indices = _profile_indices(*label_names)
    index_source = exact_template or frequency_template
    if label_indices is None and index_source is None:
        raise AutotuneError(
            f"cannot determine DVFS indices for clock state {state_name!r}; use a "
            "p<prime>-g<gold>-gpu<gpu> label or add its clock point to the template"
        )
    indices = label_indices or (
        index_source.prime_index,
        index_source.gold_index,
        index_source.gpu_index,
    )
    if exact_template is not None and label_indices is not None:
        template_indices = (
            exact_template.prime_index,
            exact_template.gold_index,
            exact_template.gpu_index,
        )
        if label_indices != template_indices:
            raise AutotuneError(
                f"profile name {state_name!r} encodes indices {label_indices}, but the "
                f"template declares {template_indices}"
            )

    output_profile = state_name
    if label_indices is None and index_source is not None:
        output_profile = index_source.profile
    return _ResolvedClock(
        profile=output_profile,
        prime_index=indices[0],
        gold_index=indices[1],
        gpu_index=indices[2],
        prime_khz=frequencies[0],
        gold_khz=frequencies[1],
        gpu_hz=frequencies[2],
        source=("catalog" if catalog_state is not None else "template"),
        scheduler_samples=catalog_state.scheduler_samples if catalog_state else 0,
    )


def _catalog_represents(
    state_name: str,
    input_tokens: int,
    catalog: Sequence[_CatalogClock],
) -> bool:
    return any(
        (state_name == state.profile or state_name in state.aliases)
        and (not state.input_tokens or input_tokens in state.input_tokens)
        for state in catalog
    )


def _backend_device(split_id: str, backend: str) -> str:
    normalized_id = split_id.lower()
    normalized_backend = backend.lower()
    if normalized_id in _CANONICAL_DEVICES:
        device = normalized_id
    elif "opencl" in normalized_backend or normalized_backend.startswith("gpu"):
        device = "gpu"
    elif "htp" in normalized_backend or "hexagon" in normalized_backend:
        device = "npu"
    elif normalized_backend.startswith("cpu"):
        device = "cpu"
    else:
        raise AutotuneError(
            f"cannot map FFN split {split_id!r} backend {backend!r} to cpu/gpu/npu"
        )
    if normalized_id != device:
        raise AutotuneError(
            "current trace parser records FFN sizes only under canonical split ids "
            f"cpu/gpu/npu; template uses {split_id!r} for {device}"
        )
    return device


def _ffn_layout(template: Mapping[str, Any]) -> _FfnLayout:
    section = template.get("ffn_parallel")
    if not isinstance(section, Mapping):
        raise AutotuneError("template.ffn_parallel must be an object")
    raw_layout = section.get("split_layout")
    if not isinstance(raw_layout, list) or not raw_layout:
        raise AutotuneError("template.ffn_parallel.split_layout must be a non-empty array")
    base_alignment = _positive_int(section.get("align", 256), "template.ffn_parallel.align")

    ordered_devices: list[str] = []
    split_ids: dict[str, str] = {}
    own_alignments: dict[str, int] = {}
    for index, raw_lane in enumerate(raw_layout):
        if not isinstance(raw_lane, Mapping):
            raise AutotuneError(f"template.ffn_parallel.split_layout[{index}] must be an object")
        split_id = raw_lane.get("id")
        backend = raw_lane.get("backend")
        if not isinstance(split_id, str) or not split_id:
            raise AutotuneError(f"template FFN lane {index} has no id")
        if not isinstance(backend, str) or not backend:
            raise AutotuneError(f"template FFN lane {split_id!r} has no backend")
        device = _backend_device(split_id, backend)
        if device in split_ids:
            raise AutotuneError(f"template contains more than one FFN lane for {device}")
        alignment = math.lcm(
            base_alignment,
            _positive_int(raw_lane.get("align", 1), f"FFN lane {split_id}.align"),
            256 if device == "npu" else 1,
        )
        ordered_devices.append(device)
        split_ids[device] = split_id
        own_alignments[device] = alignment

    # A later lane's start must satisfy that lane's alignment. Requiring each
    # preceding lane size to be divisible by later alignments is conservative
    # but guarantees every optimizer candidate passes policy validation.
    safe_alignments: dict[str, int] = {}
    for index, device in enumerate(ordered_devices):
        alignment = own_alignments[device]
        for later in ordered_devices[index + 1 :]:
            alignment = math.lcm(alignment, own_alignments[later])
        safe_alignments[device] = alignment
    return _FfnLayout(split_id_by_device=split_ids, alignments=safe_alignments)


def _load_or_discover_catalog(
    traces: Sequence[str | Path],
    catalog_path: str | Path | None,
    *,
    token_override: int | None,
    min_samples: int,
    min_queries: int,
) -> dict[str, Any]:
    if catalog_path is not None:
        return _json_object(catalog_path)
    return discover_clock_states(
        traces,
        token_override=token_override,
        min_samples=min_samples,
        min_queries=min_queries,
    ).to_dict()


def recommend_ffn_plans(
    model: ModelDescriptor,
    template: Mapping[str, Any],
    traces: Sequence[str | Path],
    *,
    measured_policy: str | Path | Mapping[str, object] | None = None,
    clock_catalog: Mapping[str, Any] | None = None,
    token_override: int | None = None,
    include_warmup: bool = False,
    include_edge_layers: bool = False,
    allowed_drops: Iterable[str] = (),
    fit_intercept: bool = False,
    min_device_samples: int = 1,
    max_alternatives: int = 4,
    max_relative_slowdown: float = 0.05,
    max_profile_regret: float | None = None,
    preserve_profiles: Iterable[str] = (),
) -> dict[str, object]:
    """Produce FFN plans from measured branch times.

    Groups using different active plans at the same measured clock point are
    pooled, giving the fitter multiple shard sizes when such data exists. With
    a single measured size, the optimizer's default through-origin fit remains
    valid and is explicitly recorded in the output.
    """

    model.validate()
    if min_device_samples <= 0:
        raise AutotuneError("min_device_samples must be positive")
    if max_alternatives < 0:
        raise AutotuneError("max_alternatives must be non-negative")
    if max_relative_slowdown < 0:
        raise AutotuneError("max_relative_slowdown must be non-negative")
    if max_profile_regret is not None and max_profile_regret < 0:
        raise AutotuneError("max_profile_regret must be non-negative")
    layout = _ffn_layout(template)
    drops = {device.lower() for device in allowed_drops}
    if "all" in drops:
        drops = set(layout.devices)
    unknown_drops = drops.difference(layout.devices)
    if unknown_drops:
        raise AutotuneError(
            "--allow-drop names device(s) absent from the template: "
            + ", ".join(sorted(unknown_drops))
        )

    observations = parse_ffn_observations(
        traces,
        policy=measured_policy,
        template=template,
        include_warmup=include_warmup,
        token_override=token_override,
    )
    if not observations.groups:
        raise AutotuneError(
            "no usable FFN observations were found; verify scheduler trace branch rows "
            "and the measured policy's split_sizes"
        )

    catalog_data = clock_catalog or discover_clock_states(
        traces,
        token_override=token_override,
        min_samples=1,
        min_queries=1,
    ).to_dict()
    catalog = _catalog_clocks(catalog_data)
    if not catalog:
        raise AutotuneError("clock catalog contains no states after filtering")
    template_clocks = _template_clocks(template)

    # First resolve labels into physical clock tuples. This merges aliases and
    # avoids generating duplicate clock points, which the engine rejects.
    grouped_samples: dict[
        tuple[int, tuple[int, int, int]], dict[str, list[SizeLatencySample]]
    ] = defaultdict(lambda: defaultdict(list))
    grouped_metadata: dict[
        tuple[int, tuple[int, int, int]], dict[str, Any]
    ] = {}
    skipped_catalog_groups = 0
    skipped_edge_observations = 0
    for group in observations.groups:
        if not _catalog_represents(group.actual_state, group.input_tokens, catalog):
            skipped_catalog_groups += 1
            continue
        resolved = _resolve_clock(
            group.actual_state, group.input_tokens, catalog, template_clocks
        )
        eligible_observations = []
        for observation in group.observations:
            is_interior = (
                observation.layer is not None
                and 1 <= observation.layer <= model.n_layer - 2
            )
            if include_edge_layers or is_interior:
                eligible_observations.append(observation)
            else:
                skipped_edge_observations += 1
        if not eligible_observations:
            continue
        key = (group.input_tokens, resolved.frequencies)
        metadata = grouped_metadata.setdefault(
            key,
            {
                "resolutions": [],
                "actual_states": set(),
                "active_plans": set(),
                "raw_observations": 0,
                "unknown_layer_observations": 0,
            },
        )
        metadata["resolutions"].append((len(eligible_observations), resolved))
        metadata["actual_states"].add(group.actual_state)
        metadata["active_plans"].add(group.active_plan)
        metadata["raw_observations"] += len(eligible_observations)
        metadata.setdefault("layers", set()).update(
            observation.layer
            for observation in eligible_observations
            if observation.layer is not None
        )
        metadata["unknown_layer_observations"] += sum(
            observation.layer is None for observation in eligible_observations
        )
        for observation in eligible_observations:
            if observation.device in layout.devices:
                grouped_samples[key][observation.device].append(
                    SizeLatencySample(observation.shard_size, observation.latency_us)
                )

    plans: list[dict[str, object]] = []
    pruning_inputs: dict[int, list[tuple[dict[str, object], Any, _ResolvedClock, float]]] = (
        defaultdict(list)
    )
    for (tokens, frequencies), device_samples in grouped_samples.items():
        metadata = grouped_metadata[(tokens, frequencies)]
        resolutions: list[tuple[int, _ResolvedClock]] = metadata["resolutions"]
        _, clock = max(
            resolutions,
            key=lambda item: (
                item[0],
                item[1].scheduler_samples,
                item[1].profile in template_clocks,
                item[1].profile,
            ),
        )

        insufficient = {
            device: len(device_samples.get(device, []))
            for device in layout.devices
            if len(device_samples.get(device, [])) < min_device_samples
        }
        required_missing = sorted(set(insufficient).difference(drops))
        if required_missing:
            details = ", ".join(
                f"{device}={insufficient[device]}" for device in required_missing
            )
            raise AutotuneError(
                f"{tokens}-token state {clock.profile} lacks required FFN samples "
                f"({details}); collect them or pass --allow-drop for those devices"
            )

        modeled_samples = {
            device: samples
            for device, samples in device_samples.items()
            if len(samples) >= min_device_samples
        }
        if not modeled_samples:
            raise AutotuneError(f"{tokens}-token state {clock.profile} has no modeled device")
        minimums = {
            device: 0 if device in drops else layout.alignments[device]
            for device in modeled_samples
        }
        result = optimize_ffn_shards(
            model.n_ff,
            modeled_samples,
            alignments={device: layout.alignments[device] for device in modeled_samples},
            minimums=minimums,
            fit_intercept=fit_intercept,
            max_alternatives=max_alternatives,
            max_relative_slowdown=max_relative_slowdown,
        )
        canonical_sizes = {
            device: int(result.best.split_sizes.get(device, 0)) for device in layout.devices
        }
        output_sizes = {
            layout.split_id_by_device[device]: canonical_sizes[device]
            for device in layout.devices
        }
        plans.append(
            {
                "profile": clock.profile,
                "input_tokens": tokens,
                "clock_point": clock.to_plan_dict(),
                "ffn_split_sizes": output_sizes,
                "measurement": {
                    "clock_source": clock.source,
                    "actual_states": sorted(metadata["actual_states"]),
                    "active_plans": sorted(metadata["active_plans"]),
                    "raw_observations": metadata["raw_observations"],
                    "layers": sorted(metadata["layers"]),
                    "unknown_layer_observations": metadata[
                        "unknown_layer_observations"
                    ],
                    "samples_per_device": {
                        device: len(device_samples.get(device, []))
                        for device in layout.devices
                    },
                    "distinct_sizes_per_device": {
                        device: len({sample.size for sample in device_samples.get(device, [])})
                        for device in layout.devices
                    },
                },
                "optimization": result.to_dict(),
            }
        )
        pruning_inputs[tokens].append(
            (
                plans[-1],
                result,
                clock,
                float(max(clock.scheduler_samples, metadata["raw_observations"], 1)),
            )
        )

    plans.sort(
        key=lambda plan: (
            int(plan["input_tokens"]),
            -int(plan["clock_point"]["prime"]["khz"]),  # type: ignore[index]
            -int(plan["clock_point"]["gold"]["khz"]),  # type: ignore[index]
            -int(plan["clock_point"]["gpu"]["hz"]),  # type: ignore[index]
            str(plan["profile"]),
        )
    )
    if not plans:
        edge_hint = " (try --include-edge-layers)" if skipped_edge_observations else ""
        raise AutotuneError(
            "no plans were produced from catalog-represented interior-layer observations"
            + edge_hint
        )

    pruning_output: dict[str, object] = {}
    if max_profile_regret is not None:
        requested_anchors = set(preserve_profiles)
        available_profiles = {str(plan["profile"]) for plan in plans}
        unknown_anchors = requested_anchors.difference(available_profiles)
        if unknown_anchors:
            raise AutotuneError(
                "--preserve-profile names unknown recommendation(s): "
                + ", ".join(sorted(unknown_anchors))
            )
        retained_by_token: dict[int, set[str]] = {}
        for tokens, items in sorted(pruning_inputs.items()):
            names = [str(item[0]["profile"]) for item in items]
            if len(set(names)) != len(names):
                raise AutotuneError(
                    f"{tokens}-token recommendations contain duplicate profile names"
                )
            observed_states = [
                ObservedProfileState(
                    name=str(plan["profile"]),
                    point=TraceClockPoint(
                        clock.prime_khz, clock.gold_khz, clock.gpu_hz
                    ),
                    weight=weight,
                    latency_models=result.models,
                    optimal_split=result.best.split_sizes,
                    optimal_makespan_us=result.best.predicted_makespan_us,
                )
                for plan, result, clock, weight in items
            ]
            candidates = [
                ClockProfileCandidate(
                    name=str(plan["profile"]),
                    point=TraceClockPoint(
                        clock.prime_khz, clock.gold_khz, clock.gpu_hz
                    ),
                    split_sizes=result.best.split_sizes,
                )
                for plan, result, clock, _ in items
            ]
            anchors = sorted(requested_anchors.intersection(names))
            pruning = prune_clock_profiles(
                observed_states,
                candidates,
                max_predicted_regret=max_profile_regret,
                preserve_anchors=anchors,
            )
            retained_by_token[tokens] = set(pruning.retained_profiles)
            pruning_output[str(tokens)] = pruning.to_dict()
        plans = [
            plan
            for plan in plans
            if str(plan["profile"]) in retained_by_token[int(plan["input_tokens"])]
        ]
    return {
        "schema_version": 1,
        "model": model.to_dict(),
        "fit": {
            "mode": "intercept-when-multiple-sizes" if fit_intercept else "through-origin",
            "single_size_supported": True,
            "allowed_device_drops": sorted(drops),
            "min_device_samples": min_device_samples,
            "layers": "all" if include_edge_layers else "interior-only",
        },
        "sources": {
            "scheduler_files": list(observations.scheduler_files),
            "skipped_without_shard_size": observations.skipped_without_shard_size,
            "skipped_catalog_groups": skipped_catalog_groups,
            "skipped_edge_observations": skipped_edge_observations,
        },
        "pruning": pruning_output,
        "plans": plans,
    }


def _command_inspect_model(args: argparse.Namespace) -> int:
    model = load_model_descriptor(args.model, compute_sha256=args.sha256)
    _write_json(model.to_dict(), args.output, force=args.force)
    return 0


def _command_discover(args: argparse.Namespace) -> int:
    allowed_gpu = set(args.gpu_index) if args.gpu_index else None
    allowed_profiles = set(args.profile) if args.profile else None
    catalog = discover_clock_states(
        args.traces,
        token_override=args.token_override,
        min_samples=args.min_samples,
        min_queries=args.min_queries,
        allowed_gpu_indices=allowed_gpu,
        allowed_profiles=allowed_profiles,
        include_hardware_only=args.include_hardware_only,
    )
    _write_json(catalog.to_dict(), args.output, force=args.force)
    return 0


def _command_recommend(args: argparse.Namespace) -> int:
    model = load_model_descriptor(args.model)
    template = load_policy_template(args.template)
    catalog = _load_or_discover_catalog(
        args.traces,
        args.catalog,
        token_override=args.token_override,
        min_samples=args.clock_min_samples,
        min_queries=args.clock_min_queries,
    )
    result = recommend_ffn_plans(
        model,
        template,
        args.traces,
        measured_policy=args.policy,
        clock_catalog=catalog,
        token_override=args.token_override,
        include_warmup=args.include_warmup,
        include_edge_layers=args.include_edge_layers,
        allowed_drops=args.allow_drop,
        fit_intercept=args.fit_intercept,
        min_device_samples=args.min_device_samples,
        max_alternatives=args.max_alternatives,
        max_relative_slowdown=args.max_relative_slowdown,
        max_profile_regret=args.max_profile_regret,
        preserve_profiles=args.preserve_profile,
    )
    _write_json(result, args.output, force=args.force)
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    model = load_model_descriptor(args.model)
    template = load_policy_template(args.template)
    plans = [
        plan
        for plan in load_plans(args.plans)
        if plan.input_tokens is None or plan.input_tokens == args.token
    ]
    if not plans:
        raise AutotuneError(f"plans file has no entry for {args.token} input tokens")
    policy = generate_policy(template, model, args.token, plans)
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.force:
        raise AutotuneError(f"output already exists (use --force to replace it): {destination}")
    write_policy(policy, destination)
    print(f"wrote {destination}")
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    model = load_model_descriptor(args.model)
    policy = _json_object(args.policy)
    validate_generated_policy(policy, model, args.token)
    profiles = policy.get("profiles", {})
    result = {
        "valid": True,
        "model_id": model.model_id,
        "input_tokens": args.token,
        "profile_count": len(profiles) if isinstance(profiles, Mapping) else 0,
        "policy": str(args.policy.expanduser().resolve()),
    }
    _write_json(result, args.output, force=args.force)
    return 0


def _add_output_options(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--output", type=Path, required=required)
    parser.add_argument("--force", action="store_true", help="replace an existing output file")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subcommands.add_parser(
        "inspect-model", help="extract a compact descriptor from GGUF or normalize descriptor JSON"
    )
    inspect_parser.add_argument("model", type=Path)
    inspect_parser.add_argument(
        "--sha256", action="store_true", help="hash the full GGUF file (can take time)"
    )
    _add_output_options(inspect_parser)
    inspect_parser.set_defaults(handler=_command_inspect_model)

    discover_parser = subcommands.add_parser(
        "discover", help="build a reachable clock-state catalog from collected traces"
    )
    discover_parser.add_argument("traces", nargs="+", type=Path)
    discover_parser.add_argument("--token-override", type=int)
    discover_parser.add_argument("--min-samples", type=int, default=1)
    discover_parser.add_argument("--min-queries", type=int, default=1)
    discover_parser.add_argument("--gpu-index", type=int, action="append")
    discover_parser.add_argument("--profile", action="append")
    discover_parser.add_argument("--include-hardware-only", action="store_true")
    _add_output_options(discover_parser)
    discover_parser.set_defaults(handler=_command_discover)

    recommend_parser = subcommands.add_parser(
        "recommend", help="fit FFN device latency and recommend aligned shards per state"
    )
    recommend_parser.add_argument("traces", nargs="+", type=Path)
    recommend_parser.add_argument("--model", type=Path, required=True)
    recommend_parser.add_argument("--template", type=Path, required=True)
    recommend_parser.add_argument(
        "--policy",
        type=Path,
        help="policy used to collect all traces; omit to use each run snapshot/config",
    )
    recommend_parser.add_argument("--catalog", type=Path)
    recommend_parser.add_argument("--token-override", type=int)
    recommend_parser.add_argument("--clock-min-samples", type=int, default=1)
    recommend_parser.add_argument("--clock-min-queries", type=int, default=1)
    recommend_parser.add_argument("--include-warmup", action="store_true")
    recommend_parser.add_argument(
        "--include-edge-layers",
        action="store_true",
        help="include layer 0 and the last layer (default: fit only interior layers)",
    )
    recommend_parser.add_argument(
        "--allow-drop",
        action="append",
        choices=(*_CANONICAL_DEVICES, "all"),
        default=[],
        help="allow a device to receive a zero FFN shard; repeat or use 'all'",
    )
    recommend_parser.add_argument(
        "--fit-intercept",
        action="store_true",
        help="fit launch overhead when multiple shard sizes exist; one-size data still uses origin",
    )
    recommend_parser.add_argument("--min-device-samples", type=int, default=1)
    recommend_parser.add_argument("--max-alternatives", type=int, default=4)
    recommend_parser.add_argument("--max-relative-slowdown", type=float, default=0.05)
    recommend_parser.add_argument(
        "--max-profile-regret",
        type=float,
        help="greedily prune profiles while predicted worst-case regret stays below this ratio",
    )
    recommend_parser.add_argument(
        "--preserve-profile",
        action="append",
        default=[],
        help="profile anchor that pruning must retain; repeat as needed",
    )
    _add_output_options(recommend_parser)
    recommend_parser.set_defaults(handler=_command_recommend)

    generate_parser = subcommands.add_parser(
        "generate", help="materialize one token-specific policy from a template and plans"
    )
    generate_parser.add_argument("--model", type=Path, required=True)
    generate_parser.add_argument("--template", type=Path, required=True)
    generate_parser.add_argument("--plans", type=Path, required=True)
    generate_parser.add_argument("--token", type=int, required=True)
    _add_output_options(generate_parser, required=True)
    generate_parser.set_defaults(handler=_command_generate)

    validate_parser = subcommands.add_parser(
        "validate", help="statically validate a generated policy against model dimensions"
    )
    validate_parser.add_argument("--model", type=Path, required=True)
    validate_parser.add_argument("--policy", type=Path, required=True)
    validate_parser.add_argument("--token", type=int, required=True)
    _add_output_options(validate_parser)
    validate_parser.set_defaults(handler=_command_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        AutotuneError,
        FileNotFoundError,
        ModelDescriptorError,
        PolicyGenerationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
