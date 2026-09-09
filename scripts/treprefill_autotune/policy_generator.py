#!/usr/bin/env python3

"""Generate a model/token-specific TrePrefill policy from a known-good template.

This module intentionally does not invent weight residency or ordinary op
placement.  Those correctness-sensitive sections are cloned from a template;
only model-wide dimensions, token applicability, clock profiles, and measured
partition plans are changed.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Support both package imports and direct execution in focused tests.
    from .model_descriptor import ModelDescriptor, ModelDescriptorError
except ImportError:  # pragma: no cover - exercised when run as a loose script
    from model_descriptor import ModelDescriptor, ModelDescriptorError


class PolicyGenerationError(ValueError):
    """Raised when a template or candidate plan cannot form a valid policy."""


def _integer(value: object, source: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise PolicyGenerationError(f"{source} must be a {qualifier} integer")
    return value


def _mapping(value: object, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyGenerationError(f"{source} must be an object")
    return value


@dataclass(frozen=True)
class ClockAxis:
    index: int
    frequency: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], unit_key: str, source: str) -> "ClockAxis":
        return cls(
            index=_integer(data.get("index"), f"{source}.index"),
            frequency=_integer(data.get(unit_key), f"{source}.{unit_key}", minimum=1),
        )


@dataclass(frozen=True)
class ClockPoint:
    prime: ClockAxis
    gold: ClockAxis
    gpu: ClockAxis

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: str = "clock_point") -> "ClockPoint":
        return cls(
            prime=ClockAxis.from_dict(
                _mapping(data.get("prime"), f"{source}.prime"), "khz", f"{source}.prime"
            ),
            gold=ClockAxis.from_dict(
                _mapping(data.get("gold"), f"{source}.gold"), "khz", f"{source}.gold"
            ),
            gpu=ClockAxis.from_dict(
                _mapping(data.get("gpu"), f"{source}.gpu"), "hz", f"{source}.gpu"
            ),
        )

    def to_policy_dict(self) -> dict[str, object]:
        return {
            "prime": {"index": self.prime.index, "khz": self.prime.frequency},
            "gold": {"index": self.gold.index, "khz": self.gold.frequency},
            "gpu": {"index": self.gpu.index, "hz": self.gpu.frequency},
        }


@dataclass(frozen=True)
class Plan:
    """One clock profile and its measured/optimized partition sizes."""

    profile: str
    clock_point: ClockPoint
    ffn_split_sizes: Mapping[str, int]
    input_tokens: int | None = None
    qkv_split_sizes: Mapping[str, Mapping[str, int]] | None = None
    attn_out_split_sizes: Mapping[str, int] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: str = "plan") -> "Plan":
        name = data.get("profile", data.get("name", data.get("id")))
        if not isinstance(name, str) or not name.strip():
            raise PolicyGenerationError(f"{source}.profile must be a non-empty string")

        raw_ffn = data.get("ffn_split_sizes", data.get("ffn_splits"))
        if raw_ffn is None and isinstance(data.get("ffn_parallel"), Mapping):
            raw_ffn = data["ffn_parallel"].get("split_sizes")
        ffn_mapping = _mapping(raw_ffn, f"{source}.ffn_split_sizes")
        ffn = {
            str(key): _integer(value, f"{source}.ffn_split_sizes.{key}")
            for key, value in ffn_mapping.items()
        }

        raw_qkv = data.get("qkv_split_sizes")
        if raw_qkv is None and isinstance(data.get("attn_qkv_shards"), Mapping):
            raw_qkv = data["attn_qkv_shards"].get("split_sizes")
        qkv = None
        if raw_qkv is not None:
            qkv_mapping = _mapping(raw_qkv, f"{source}.qkv_split_sizes")
            qkv = {}
            for projection, sizes in qkv_mapping.items():
                size_mapping = _mapping(sizes, f"{source}.qkv_split_sizes.{projection}")
                qkv[str(projection)] = {
                    str(key): _integer(
                        value, f"{source}.qkv_split_sizes.{projection}.{key}"
                    )
                    for key, value in size_mapping.items()
                }

        raw_attn_out = data.get("attn_out_split_sizes")
        if raw_attn_out is None and isinstance(data.get("attn_out_shards"), Mapping):
            raw_attn_out = data["attn_out_shards"].get("split_sizes")
        attn_out = None
        if raw_attn_out is not None:
            out_mapping = _mapping(raw_attn_out, f"{source}.attn_out_split_sizes")
            attn_out = {
                str(key): _integer(value, f"{source}.attn_out_split_sizes.{key}")
                for key, value in out_mapping.items()
            }

        token = data.get("input_tokens", data.get("token"))
        return cls(
            profile=name.strip(),
            clock_point=ClockPoint.from_dict(
                _mapping(data.get("clock_point"), f"{source}.clock_point"),
                f"{source}.clock_point",
            ),
            ffn_split_sizes=ffn,
            input_tokens=(
                None if token is None else _integer(token, f"{source}.input_tokens", minimum=1)
            ),
            qkv_split_sizes=qkv,
            attn_out_split_sizes=attn_out,
        )


def load_plans(path: str | Path) -> list[Plan]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyGenerationError(f"failed to read plan JSON {source}: {exc}") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("plans")
    if not isinstance(raw, list) or not raw:
        raise PolicyGenerationError("plan JSON must be a non-empty list or {\"plans\": [...]}")
    return [Plan.from_dict(_mapping(item, f"plans[{index}]"), f"plans[{index}]") for index, item in enumerate(raw)]


def load_policy_template(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyGenerationError(f"failed to read policy template {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolicyGenerationError("policy template must contain a JSON object")
    return raw


def _backend_needs_htp_alignment(backend: str) -> bool:
    normalized = backend.lower()
    return "htp" in normalized or "hexagon" in normalized or normalized.startswith("npu")


def _layout(section: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    raw = section.get("split_layout")
    if not isinstance(raw, list) or not raw:
        raise PolicyGenerationError(f"{source}.split_layout must be a non-empty array")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(raw):
        entry = _mapping(item, f"{source}.split_layout[{index}]")
        split_id = entry.get("id")
        backend = entry.get("backend")
        if not isinstance(split_id, str) or not split_id or split_id in ids:
            raise PolicyGenerationError(
                f"{source}.split_layout[{index}].id must be non-empty and unique"
            )
        if not isinstance(backend, str) or not backend:
            raise PolicyGenerationError(
                f"{source}.split_layout[{index}].backend must be non-empty"
            )
        ids.add(split_id)
        result.append(dict(entry))
    return result


def _lane_alignments(
    layout: Sequence[Mapping[str, Any]],
    *,
    base_alignment: int,
) -> list[int]:
    alignments: list[int] = []
    for index, entry in enumerate(layout):
        explicit = entry.get("align", 1)
        explicit = _integer(explicit, f"split_layout[{index}].align", minimum=1)
        alignment = math.lcm(base_alignment, explicit)
        if _backend_needs_htp_alignment(str(entry["backend"])):
            alignment = math.lcm(alignment, 256)
        alignments.append(alignment)
    return alignments


def _validate_partition(
    sizes: Mapping[str, Any],
    layout: Sequence[Mapping[str, Any]],
    total: int,
    alignments: Sequence[int],
    source: str,
) -> dict[str, int]:
    ids = [str(entry["id"]) for entry in layout]
    if set(sizes) != set(ids):
        raise PolicyGenerationError(
            f"{source} must contain exactly these split ids: {', '.join(ids)}"
        )
    normalized = {
        split_id: _integer(sizes[split_id], f"{source}.{split_id}") for split_id in ids
    }
    if sum(normalized.values()) != total:
        raise PolicyGenerationError(
            f"{source} sums to {sum(normalized.values())}, expected {total}"
        )
    start = 0
    active = 0
    for split_id, alignment in zip(ids, alignments):
        size = normalized[split_id]
        if size > 0:
            active += 1
            if start % alignment != 0 or size % alignment != 0:
                raise PolicyGenerationError(
                    f"{source}.{split_id} start={start}, size={size} must both be "
                    f"multiples of {alignment}"
                )
        start += size
    if active == 0:
        raise PolicyGenerationError(f"{source} must contain at least one active lane")
    return normalized


def _proportional_partition(
    original: Mapping[str, Any],
    layout: Sequence[Mapping[str, Any]],
    total: int,
    alignments: Sequence[int],
    source: str,
) -> dict[str, int]:
    """Find the closest aligned partition while preserving zero/active lanes."""

    ids = [str(entry["id"]) for entry in layout]
    if set(original) != set(ids):
        raise PolicyGenerationError(
            f"{source} must contain exactly these split ids: {', '.join(ids)}"
        )
    old = [_integer(original[split_id], f"{source}.{split_id}") for split_id in ids]
    old_total = sum(old)
    if old_total <= 0:
        raise PolicyGenerationError(f"{source} must have a positive total width")
    targets = [total * value / old_total for value in old]
    active = [value > 0 for value in old]

    best: tuple[float, tuple[int, ...]] | None = None

    def visit(index: int, start: int, chosen: list[int]) -> None:
        nonlocal best
        if index == len(ids) - 1:
            size = total - start
            if (not active[index] and size != 0) or (active[index] and size <= 0):
                return
            if size > 0 and (start % alignments[index] or size % alignments[index]):
                return
            candidate = tuple(chosen + [size])
            score = sum(
                ((actual - target) / max(target, float(alignment))) ** 2
                for actual, target, alignment in zip(candidate, targets, alignments)
            )
            ranked = (score, candidate)
            if best is None or ranked < best:
                best = ranked
            return

        if not active[index]:
            visit(index + 1, start, chosen + [0])
            return
        alignment = alignments[index]
        if start % alignment:
            return
        remaining_active = sum(active[index + 1 :])
        maximum = total - start - remaining_active
        for size in range(alignment, maximum + 1, alignment):
            visit(index + 1, start + size, chosen + [size])

    visit(0, 0, [])
    if best is None:
        raise PolicyGenerationError(
            f"{source} cannot be scaled to width {total} under lane alignments "
            f"{list(alignments)} while preserving active lanes"
        )
    return dict(zip(ids, best[1]))


def _template_last_layer(policy: Mapping[str, Any]) -> int:
    """Return the old model-wide end from the template's root FFN policy."""

    ffn = _mapping(policy.get("ffn_parallel"), "ffn_parallel")
    layer_range = ffn.get("layer_range")
    if (
        not isinstance(layer_range, list)
        or len(layer_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in layer_range)
        or layer_range[0] != 0
        or layer_range[1] < 0
    ):
        raise PolicyGenerationError(
            "ffn_parallel.layer_range must be a model-wide [0, last_layer] range"
        )
    return layer_range[1]


def _replace_full_layer_ranges(value: object, old_last_layer: int, new_last_layer: int) -> None:
    """Resize only ranges equal to the template's known model-wide range.

    A range such as ``[0, 13]`` may deliberately target only early layers.  It
    must not be expanded merely because it starts at zero.
    """

    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key == "layer_range"
                and isinstance(child, list)
                and len(child) == 2
                and all(isinstance(item, int) and not isinstance(item, bool) for item in child)
                and child == [0, old_last_layer]
            ):
                child[:] = [0, new_last_layer]
            else:
                _replace_full_layer_ranges(child, old_last_layer, new_last_layer)
    elif isinstance(value, list):
        for child in value:
            _replace_full_layer_ranges(child, old_last_layer, new_last_layer)


def _set_token_applicability(policy: dict[str, Any], input_tokens: int) -> None:
    defaults = policy.setdefault("profile_defaults", {})
    if not isinstance(defaults, dict):
        raise PolicyGenerationError("profile_defaults must be an object")
    applicability = defaults.setdefault("applicability", {})
    if not isinstance(applicability, dict):
        raise PolicyGenerationError("profile_defaults.applicability must be an object")
    applicability["input_tokens"] = [input_tokens, input_tokens]
    applicability["ubatch_tokens"] = [input_tokens, input_tokens]


def _adapt_qkv(policy: dict[str, Any], model: ModelDescriptor) -> None:
    root = policy.get("attn_qkv_shards")
    if root is None:
        return
    root = _mapping(root, "attn_qkv_shards")
    layout = _layout(root, "attn_qkv_shards")
    alignments = _lane_alignments(layout, base_alignment=model.head_dim)
    root_sizes = _mapping(root.get("split_sizes"), "attn_qkv_shards.split_sizes")
    targets = {"q": model.q_width, "k": model.k_width, "v": model.v_width}
    scaled_root: dict[str, dict[str, int]] = {}
    for projection, total in targets.items():
        scaled_root[projection] = _proportional_partition(
            _mapping(root_sizes.get(projection), f"attn_qkv_shards.split_sizes.{projection}"),
            layout,
            total,
            alignments,
            f"attn_qkv_shards.split_sizes.{projection}",
        )
    root["head_dim"] = model.head_dim
    root["split_sizes"] = scaled_root

    profiles = policy.get("profiles", {})
    if isinstance(profiles, Mapping):
        for name, entry in profiles.items():
            if not isinstance(entry, dict) or "attn_qkv_shards" not in entry:
                continue
            override = _mapping(
                entry["attn_qkv_shards"], f"profiles.{name}.attn_qkv_shards"
            )
            override_sizes = _mapping(
                override.get("split_sizes"),
                f"profiles.{name}.attn_qkv_shards.split_sizes",
            )
            entry["attn_qkv_shards"] = {
                "split_sizes": {
                    projection: _proportional_partition(
                        _mapping(
                            override_sizes.get(projection),
                            f"profiles.{name}.attn_qkv_shards.split_sizes.{projection}",
                        ),
                        layout,
                        total,
                        alignments,
                        f"profiles.{name}.attn_qkv_shards.split_sizes.{projection}",
                    )
                    for projection, total in targets.items()
                }
            }


def _adapt_attn_out(policy: dict[str, Any], model: ModelDescriptor) -> None:
    root = policy.get("attn_out_shards")
    if root is None:
        return
    root = _mapping(root, "attn_out_shards")
    layout = _layout(root, "attn_out_shards")
    alignments = _lane_alignments(layout, base_alignment=model.head_dim)
    root["head_dim"] = model.head_dim
    root["split_sizes"] = _proportional_partition(
        _mapping(root.get("split_sizes"), "attn_out_shards.split_sizes"),
        layout,
        model.attn_out_width,
        alignments,
        "attn_out_shards.split_sizes",
    )

    profiles = policy.get("profiles", {})
    if isinstance(profiles, Mapping):
        for name, entry in profiles.items():
            if not isinstance(entry, dict) or "attn_out_shards" not in entry:
                continue
            override = _mapping(
                entry["attn_out_shards"], f"profiles.{name}.attn_out_shards"
            )
            entry["attn_out_shards"] = {
                "split_sizes": _proportional_partition(
                    _mapping(
                        override.get("split_sizes"),
                        f"profiles.{name}.attn_out_shards.split_sizes",
                    ),
                    layout,
                    model.attn_out_width,
                    alignments,
                    f"profiles.{name}.attn_out_shards.split_sizes",
                )
            }


def _validate_ffn_sizes(
    sizes: Mapping[str, Any], section: Mapping[str, Any], model: ModelDescriptor, source: str
) -> dict[str, int]:
    layout = _layout(section, source.rsplit(".split_sizes", 1)[0])
    base_alignment = _integer(section.get("align", 256), f"{source}.align", minimum=1)
    return _validate_partition(
        sizes,
        layout,
        model.n_ff,
        _lane_alignments(layout, base_alignment=base_alignment),
        source,
    )


def _validate_qkv_sizes(
    sizes: Mapping[str, Any], section: Mapping[str, Any], model: ModelDescriptor, source: str
) -> dict[str, dict[str, int]]:
    layout = _layout(section, source.rsplit(".split_sizes", 1)[0])
    alignments = _lane_alignments(layout, base_alignment=model.head_dim)
    if set(sizes) != {"q", "k", "v"}:
        raise PolicyGenerationError(f"{source} must contain exactly q, k, and v")
    return {
        projection: _validate_partition(
            _mapping(sizes[projection], f"{source}.{projection}"),
            layout,
            total,
            alignments,
            f"{source}.{projection}",
        )
        for projection, total in (
            ("q", model.q_width),
            ("k", model.k_width),
            ("v", model.v_width),
        )
    }


def _validate_attn_out_sizes(
    sizes: Mapping[str, Any], section: Mapping[str, Any], model: ModelDescriptor, source: str
) -> dict[str, int]:
    layout = _layout(section, source.rsplit(".split_sizes", 1)[0])
    return _validate_partition(
        sizes,
        layout,
        model.attn_out_width,
        _lane_alignments(layout, base_alignment=model.head_dim),
        source,
    )


def _merge_patch(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_patch(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _nearest_template_profile(
    profiles: Mapping[str, Any],
    defaults: Mapping[str, Any],
    target: ClockPoint,
) -> Mapping[str, Any]:
    """Choose a donor using the engine's relative-frequency distance metric."""

    best: tuple[float, float, str, Mapping[str, Any]] | None = None
    for raw_name, raw_entry in profiles.items():
        if not isinstance(raw_name, str):
            raise PolicyGenerationError("template profile names must be strings")
        entry = _mapping(raw_entry, f"template profiles.{raw_name}")
        effective = _merge_patch(defaults, entry)
        clock_value = effective.get("clock_point")
        if clock_value is None:
            continue
        clock = ClockPoint.from_dict(
            _mapping(clock_value, f"template profiles.{raw_name}.clock_point"),
            f"template profiles.{raw_name}.clock_point",
        )
        distance = (
            abs(target.prime.frequency - clock.prime.frequency) / clock.prime.frequency
            + abs(target.gold.frequency - clock.gold.frequency) / clock.gold.frequency
            + abs(target.gpu.frequency - clock.gpu.frequency) / clock.gpu.frequency
        )
        total_khz = (
            clock.prime.frequency
            + clock.gold.frequency
            + clock.gpu.frequency / 1000.0
        )
        ranked = (distance, total_khz, raw_name, entry)
        if best is None or ranked[:3] < best[:3]:
            best = ranked
    if best is None:
        raise PolicyGenerationError(
            "a new plan profile requires at least one template profile with a clock_point"
        )
    return best[3]


def validate_generated_policy(policy: Mapping[str, Any], model: ModelDescriptor, input_tokens: int) -> None:
    """Validate constraints that the engine parser cannot infer from model shape."""

    model.validate()
    _integer(input_tokens, "input_tokens", minimum=1)
    profiles = _mapping(policy.get("profiles"), "profiles")
    if not profiles:
        raise PolicyGenerationError("profiles must not be empty")

    root_ffn = _mapping(policy.get("ffn_parallel"), "ffn_parallel")
    _validate_ffn_sizes(
        _mapping(root_ffn.get("split_sizes"), "ffn_parallel.split_sizes"),
        root_ffn,
        model,
        "ffn_parallel.split_sizes",
    )
    defaults = _mapping(policy.get("profile_defaults"), "profile_defaults")
    applicability = _mapping(defaults.get("applicability"), "profile_defaults.applicability")
    for key in ("input_tokens", "ubatch_tokens"):
        if applicability.get(key) != [input_tokens, input_tokens]:
            raise PolicyGenerationError(
                f"profile_defaults.applicability.{key} must equal "
                f"[{input_tokens}, {input_tokens}]"
            )

    root_qkv = policy.get("attn_qkv_shards")
    if root_qkv is not None:
        root_qkv = _mapping(root_qkv, "attn_qkv_shards")
        if root_qkv.get("head_dim") != model.head_dim:
            raise PolicyGenerationError("attn_qkv_shards.head_dim disagrees with model")
        _validate_qkv_sizes(
            _mapping(root_qkv.get("split_sizes"), "attn_qkv_shards.split_sizes"),
            root_qkv,
            model,
            "attn_qkv_shards.split_sizes",
        )

    root_out = policy.get("attn_out_shards")
    if root_out is not None:
        root_out = _mapping(root_out, "attn_out_shards")
        if root_out.get("head_dim") != model.head_dim:
            raise PolicyGenerationError("attn_out_shards.head_dim disagrees with model")
        _validate_attn_out_sizes(
            _mapping(root_out.get("split_sizes"), "attn_out_shards.split_sizes"),
            root_out,
            model,
            "attn_out_shards.split_sizes",
        )

    seen_clock_points: set[tuple[int, int, int]] = set()
    for name, raw_entry in profiles.items():
        entry = _mapping(raw_entry, f"profiles.{name}")
        effective = _merge_patch(defaults, entry)
        effective_ffn = _mapping(effective.get("ffn_parallel"), f"profiles.{name}.ffn_parallel")
        _validate_ffn_sizes(
            _mapping(effective_ffn.get("split_sizes"), f"profiles.{name}.ffn_parallel.split_sizes"),
            effective_ffn,
            model,
            f"profiles.{name}.ffn_parallel.split_sizes",
        )
        clock = ClockPoint.from_dict(
            _mapping(effective.get("clock_point"), f"profiles.{name}.clock_point"),
            f"profiles.{name}.clock_point",
        )
        frequencies = (
            clock.prime.frequency,
            clock.gold.frequency,
            clock.gpu.frequency,
        )
        if frequencies in seen_clock_points:
            raise PolicyGenerationError(
                f"profiles contains duplicate clock frequencies at profile {name!r}"
            )
        seen_clock_points.add(frequencies)

        if "attn_qkv_shards" in entry:
            if root_qkv is None:
                raise PolicyGenerationError(
                    f"profiles.{name}.attn_qkv_shards requires root attn_qkv_shards"
                )
            override = _mapping(entry["attn_qkv_shards"], f"profiles.{name}.attn_qkv_shards")
            if set(override) != {"split_sizes"}:
                raise PolicyGenerationError(
                    f"profiles.{name}.attn_qkv_shards may override only split_sizes"
                )
            _validate_qkv_sizes(
                _mapping(override["split_sizes"], f"profiles.{name}.attn_qkv_shards.split_sizes"),
                root_qkv,
                model,
                f"profiles.{name}.attn_qkv_shards.split_sizes",
            )
        if "attn_out_shards" in entry:
            if root_out is None:
                raise PolicyGenerationError(
                    f"profiles.{name}.attn_out_shards requires root attn_out_shards"
                )
            override = _mapping(entry["attn_out_shards"], f"profiles.{name}.attn_out_shards")
            if set(override) != {"split_sizes"}:
                raise PolicyGenerationError(
                    f"profiles.{name}.attn_out_shards may override only split_sizes"
                )
            _validate_attn_out_sizes(
                _mapping(override["split_sizes"], f"profiles.{name}.attn_out_shards.split_sizes"),
                root_out,
                model,
                f"profiles.{name}.attn_out_shards.split_sizes",
            )

    routes = _mapping(policy.get("runtime_routes"), "runtime_routes")
    route_names = routes.get("profiles")
    if not isinstance(route_names, list) or route_names != list(profiles):
        raise PolicyGenerationError("runtime_routes.profiles must match profiles in order")
    if routes.get("initial_profile") != next(iter(profiles)):
        raise PolicyGenerationError("runtime_routes.initial_profile must be the first profile")
    if routes.get("mode") != "clock" or routes.get("phase") != "prefill":
        raise PolicyGenerationError("generated runtime route must use clock/prefill mode")

    # Validate bounds without treating every range that begins at zero as a
    # model-wide range.  Intentional prefixes such as [0, 13] are valid.
    pending: list[object] = [policy]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "layer_range":
                    if (
                        not isinstance(child, list)
                        or len(child) != 2
                        or any(
                            isinstance(item, bool) or not isinstance(item, int)
                            for item in child
                        )
                        or child[0] < 0
                        or child[0] > child[1]
                        or child[1] >= model.n_layer
                    ):
                        raise PolicyGenerationError(
                            f"layer_range must stay within model layers 0.."
                            f"{model.n_layer - 1}, got {child}"
                        )
                else:
                    pending.append(child)
        elif isinstance(value, list):
            pending.extend(value)


def generate_policy(
    template: Mapping[str, Any],
    model: ModelDescriptor,
    input_tokens: int,
    plans: Sequence[Plan | Mapping[str, Any]],
) -> dict[str, Any]:
    """Clone ``template`` and materialize one token-specific clock policy."""

    try:
        model.validate()
    except ModelDescriptorError as exc:
        raise PolicyGenerationError(str(exc)) from exc
    if model.architecture.lower() != "llama":
        raise PolicyGenerationError(
            "current QKV/attention partition graph supports only LLaMA architecture"
        )
    if model.quantization not in {"UNKNOWN", "Q8_0"}:
        raise PolicyGenerationError(
            f"current HTP repack autotuning expects Q8_0, got {model.quantization}"
        )
    input_tokens = _integer(input_tokens, "input_tokens", minimum=1)
    normalized_plans = [
        plan if isinstance(plan, Plan) else Plan.from_dict(plan, f"plans[{index}]")
        for index, plan in enumerate(plans)
    ]
    if not normalized_plans:
        raise PolicyGenerationError("at least one plan is required")
    names = [plan.profile for plan in normalized_plans]
    if len(set(names)) != len(names):
        raise PolicyGenerationError("plan profile names must be unique")
    for plan in normalized_plans:
        if plan.input_tokens is not None and plan.input_tokens != input_tokens:
            raise PolicyGenerationError(
                f"plan {plan.profile!r} targets {plan.input_tokens} tokens, "
                f"not requested {input_tokens}"
            )

    policy = copy.deepcopy(dict(template))
    old_last_layer = _template_last_layer(policy)
    policy["version"] = 1
    policy["enabled"] = True
    _replace_full_layer_ranges(policy, old_last_layer, model.n_layer - 1)
    _set_token_applicability(policy, input_tokens)
    _adapt_qkv(policy, model)
    _adapt_attn_out(policy, model)

    root_ffn = _mapping(policy.get("ffn_parallel"), "ffn_parallel")
    ffn_layout = _layout(root_ffn, "ffn_parallel")
    ffn_align = _integer(root_ffn.get("align", 256), "ffn_parallel.align", minimum=1)
    ffn_alignments = _lane_alignments(ffn_layout, base_alignment=ffn_align)
    validated_ffn: dict[str, dict[str, int]] = {}
    for plan in normalized_plans:
        validated_ffn[plan.profile] = _validate_partition(
            plan.ffn_split_sizes,
            ffn_layout,
            model.n_ff,
            ffn_alignments,
            f"plans.{plan.profile}.ffn_split_sizes",
        )

    root_qkv = policy.get("attn_qkv_shards")
    root_out = policy.get("attn_out_shards")
    existing_profiles = policy.get("profiles", {})
    if not isinstance(existing_profiles, Mapping):
        raise PolicyGenerationError("template profiles must be an object")
    profile_defaults = _mapping(policy.get("profile_defaults"), "profile_defaults")
    generated_profiles: dict[str, Any] = {}
    any_qkv_override = False
    any_out_override = False
    for plan in normalized_plans:
        existing = (
            existing_profiles[plan.profile]
            if plan.profile in existing_profiles
            else _nearest_template_profile(
                existing_profiles, profile_defaults, plan.clock_point
            )
        )
        if not isinstance(existing, Mapping):
            raise PolicyGenerationError(f"template profile {plan.profile!r} must be an object")
        entry = copy.deepcopy(dict(existing))
        entry["clock_point"] = plan.clock_point.to_policy_dict()
        entry["ffn_parallel"] = {"split_sizes": validated_ffn[plan.profile]}

        if "applicability" in entry:
            entry["applicability"] = {
                "input_tokens": [input_tokens, input_tokens],
                "ubatch_tokens": [input_tokens, input_tokens],
            }
        if plan.qkv_split_sizes is not None:
            if root_qkv is None:
                raise PolicyGenerationError(
                    f"plan {plan.profile!r} supplies QKV sizes but template has no attn_qkv_shards"
                )
            qkv_section = _mapping(root_qkv, "attn_qkv_shards")
            entry["attn_qkv_shards"] = {
                "split_sizes": _validate_qkv_sizes(
                    plan.qkv_split_sizes,
                    qkv_section,
                    model,
                    f"plans.{plan.profile}.qkv_split_sizes",
                )
            }
        if plan.attn_out_split_sizes is not None:
            if root_out is None:
                raise PolicyGenerationError(
                    f"plan {plan.profile!r} supplies attention-output sizes but template "
                    "has no attn_out_shards"
                )
            out_section = _mapping(root_out, "attn_out_shards")
            entry["attn_out_shards"] = {
                "split_sizes": _validate_attn_out_sizes(
                    plan.attn_out_split_sizes,
                    out_section,
                    model,
                    f"plans.{plan.profile}.attn_out_split_sizes",
                )
            }
        any_qkv_override = any_qkv_override or "attn_qkv_shards" in entry
        any_out_override = any_out_override or "attn_out_shards" in entry
        generated_profiles[plan.profile] = entry

    policy["profiles"] = generated_profiles
    root_ffn["split_sizes"] = copy.deepcopy(validated_ffn[normalized_plans[0].profile])
    if normalized_plans[0].qkv_split_sizes is not None:
        _mapping(root_qkv, "attn_qkv_shards")["split_sizes"] = copy.deepcopy(
            generated_profiles[normalized_plans[0].profile]["attn_qkv_shards"]["split_sizes"]
        )
    if normalized_plans[0].attn_out_split_sizes is not None:
        _mapping(root_out, "attn_out_shards")["split_sizes"] = copy.deepcopy(
            generated_profiles[normalized_plans[0].profile]["attn_out_shards"]["split_sizes"]
        )

    if isinstance(policy.get("ffn_clock_switch"), dict):
        policy["ffn_clock_switch"]["enabled"] = False
    routes = policy.setdefault("runtime_routes", {})
    if not isinstance(routes, dict):
        raise PolicyGenerationError("runtime_routes must be an object")
    routes.update(
        {
            "enabled": True,
            "mode": "clock",
            "phase": "prefill",
            "initial_profile": normalized_plans[0].profile,
            "profiles": names,
            "transitions": "complete",
            "output_mode": "canonical",
        }
    )
    routes.setdefault(
        "boundary", {"node": "l_out", "backend": "auto", "granularity": "layer"}
    )
    routes["boundary"] = {
        "node": "l_out",
        "backend": routes["boundary"].get("backend", "auto")
        if isinstance(routes["boundary"], Mapping)
        else "auto",
        "granularity": "layer",
    }
    candidate_kinds = routes.get("candidate_kinds", ["ffn_block"])
    if not isinstance(candidate_kinds, list):
        raise PolicyGenerationError("runtime_routes.candidate_kinds must be an array")
    candidate_kinds = [str(item) for item in candidate_kinds]
    if "ffn_block" not in candidate_kinds:
        candidate_kinds.append("ffn_block")
    if any_qkv_override and "attn_qkv_block" not in candidate_kinds:
        candidate_kinds.append("attn_qkv_block")
    if any_out_override and "attn_out_block" not in candidate_kinds:
        candidate_kinds.append("attn_out_block")
    routes["candidate_kinds"] = candidate_kinds

    validate_generated_policy(policy, model, input_tokens)
    return policy


def write_policy(policy: Mapping[str, Any], path: str | Path) -> None:
    """Write a generated policy atomically without touching the template."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
