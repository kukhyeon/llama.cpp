#!/usr/bin/env python3
"""Expand and execute a reproducible TrePrefill thermal-discovery trial plan.

The JSON plan describes a Cartesian product of input-token lengths, two or more
GPU DVFS indices, query periods, and repetitions.  Every point becomes an
independent :class:`TrialSpec` consumed by ``adb_runner.py``.  Runs are always
sequential because parallel device experiments would invalidate thermal data.

Example configuration::

    {
      "schema_version": 1,
      "name": "llama32-thermal",
      "tokens": [64, 128, 256, 512],
      "gpu_indices": [14, 4],
      "query_periods_ms": [1000, 2000, 4000, 6000, 8000],
      "repeats": 3,
      "token_json_paths": {
        "64": "data/llama32_prefill_64.json",
        "128": "data/llama32_prefill_128.json",
        "256": "data/llama32_prefill_256.json",
        "512": "data/llama32_prefill_512.json"
      },
      "token_policies": {
        "64": "policies/llama32-i64.json",
        "128": "policies/llama32-i128.json",
        "256": "policies/llama32-i256.json",
        "512": "policies/llama32-i512.json"
      },
      "common_env": {
        "MODEL": "/data/local/tmp/gguf/Llama-3.2-3B-Instruct-Q8_0.gguf",
        "MODEL_TAG": "llama32-3b-q8-0",
        "MAX_QUERY_NUMBER": "100",
        "CPU_PRIME_P": "15",
        "CPU_GOLD_P": "15",
        "SCHED_TRACE": "on",
        "HARDWARE_STATS": "on"
      },
      "use_su": true
    }

Relative host paths (``results_root``, ``policy``, and ``token_policies``) are
resolved from the plan file directory.  Values in ``token_json_paths`` and
``common_env.MODEL`` are paths on the device.  ``policy`` is retained for a
single common policy; use the mutually exclusive ``token_policies`` mapping
when exact-token policies differ.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

try:
    # Works both as ``python -m scripts...trial_plan`` and as a direct script.
    from .adb_runner import (
        AdbRunner,
        RemoteStateUncertainError,
        RunnerError,
        TrialSpec,
        validate_trial_id,
    )
except ImportError:  # pragma: no cover - exercised by command-line smoke tests
    from adb_runner import (
        AdbRunner,
        RemoteStateUncertainError,
        RunnerError,
        TrialSpec,
        validate_trial_id,
    )


_ALLOWED_KEYS = {
    "schema_version",
    "name",
    "tokens",
    "gpu_indices",
    "query_periods_ms",
    "repeats",
    "token_json_paths",
    "common_env",
    "results_root",
    "remote_root",
    "remote_work_root",
    "script",
    "script_args",
    "policy",
    "token_policies",
    "required_artifacts",
    "timeout_seconds",
    "use_su",
    "adb",
    "serial",
    "shuffle_seed",
}
_GENERATED_ENV = {
    "INPUT_LENGTH",
    "JSON_PATH",
    "GPU_P",
    "GPU_D",
    "QUERY_PERIOD_MS",
    "RUN_REPLICATE",
}
_RETRY_RE_TEMPLATE = r"^{base}-retry([1-9][0-9]*)$"


class PlanError(ValueError):
    """The plan is malformed or conflicts with existing local results."""


@dataclass(frozen=True)
class PlanConfig:
    source: Path
    name: str
    tokens: tuple[int, ...]
    gpu_indices: tuple[int, ...]
    query_periods_ms: tuple[int, ...]
    repeats: int
    token_json_paths: dict[int, str]
    common_env: dict[str, str]
    results_root: Path
    remote_root: str
    remote_work_root: str
    script: str
    script_args: tuple[str, ...]
    policy: Path | None
    token_policies: dict[int, Path]
    required_artifacts: tuple[str, ...]
    timeout_seconds: float | None
    use_su: bool
    adb: str
    serial: str | None
    shuffle_seed: int | None


@dataclass(frozen=True)
class TrialDecision:
    logical_trial_id: str
    action: str
    reason: str
    spec: TrialSpec | None
    previous_attempts: tuple[tuple[str, str], ...]


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{name} must be a JSON object")
    return value


def _positive_unique_ints(value: Any, name: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PlanError(f"{name} must be a non-empty JSON array")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise PlanError(f"{name} must contain integers, got {item!r}")
        minimum = 0 if allow_zero else 1
        if item < minimum:
            qualifier = "non-negative" if allow_zero else "positive"
            raise PlanError(f"{name} must contain {qualifier} integers, got {item}")
        result.append(item)
    if len(set(result)) != len(result):
        raise PlanError(f"{name} contains duplicate entries")
    return tuple(result)


def _env_value(value: Any, name: str) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (str, int, float)) and not isinstance(value, complex):
        return str(value)
    raise PlanError(f"common_env.{name} must be a string, number, or boolean")


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PlanError(f"{key} must be a non-empty string when set")
    return value


def _resolve_host_path(config_dir: Path, value: str | None, default: Path) -> Path:
    path = Path(value) if value is not None else default
    if not path.is_absolute():
        path = config_dir / path
    return path.expanduser().resolve()


def load_plan(path: Path) -> PlanConfig:
    source = path.expanduser().resolve()
    try:
        raw_value = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanError(f"cannot read plan {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PlanError(f"invalid JSON in {source}: {exc}") from exc
    raw = _require_object(raw_value, "plan")

    unknown = sorted(set(raw).difference(_ALLOWED_KEYS))
    if unknown:
        raise PlanError(f"unknown plan key(s): {', '.join(unknown)}")
    if raw.get("schema_version", 1) != 1:
        raise PlanError("only schema_version 1 is supported")

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise PlanError("name must be a non-empty string")
    try:
        validate_trial_id(name)
    except argparse.ArgumentTypeError as exc:
        raise PlanError(f"invalid plan name: {exc}") from exc

    tokens = _positive_unique_ints(raw.get("tokens"), "tokens")
    gpu_indices = _positive_unique_ints(raw.get("gpu_indices"), "gpu_indices", allow_zero=True)
    periods = _positive_unique_ints(
        raw.get("query_periods_ms"), "query_periods_ms", allow_zero=True
    )
    repeats = raw.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise PlanError("repeats must be a positive integer")

    raw_token_paths = _require_object(raw.get("token_json_paths"), "token_json_paths")
    token_paths: dict[int, str] = {}
    for raw_token, remote_path in raw_token_paths.items():
        try:
            token = int(raw_token)
        except (TypeError, ValueError) as exc:
            raise PlanError(f"invalid token_json_paths key: {raw_token!r}") from exc
        if str(token) != str(raw_token):
            raise PlanError(f"token_json_paths key must be a canonical integer string: {raw_token!r}")
        if not isinstance(remote_path, str) or not remote_path:
            raise PlanError(f"token_json_paths.{raw_token} must be a non-empty remote path")
        token_paths[token] = remote_path
    missing_tokens = [str(token) for token in tokens if token not in token_paths]
    extra_tokens = [str(token) for token in token_paths if token not in tokens]
    if missing_tokens:
        raise PlanError(f"token_json_paths is missing token(s): {', '.join(missing_tokens)}")
    if extra_tokens:
        raise PlanError(f"token_json_paths contains unused token(s): {', '.join(extra_tokens)}")

    raw_env = _require_object(raw.get("common_env", {}), "common_env")
    common_env = {str(key): _env_value(value, str(key)) for key, value in raw_env.items()}
    invalid_env_names = [
        key for key in common_env if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
    ]
    if invalid_env_names:
        raise PlanError(f"invalid common_env key(s): {', '.join(sorted(invalid_env_names))}")
    conflicts = sorted(set(common_env).intersection(_GENERATED_ENV))
    if conflicts:
        raise PlanError(
            "common_env cannot override plan-generated variable(s): " + ", ".join(conflicts)
        )

    config_dir = source.parent
    results_value = _optional_string(raw, "results_root")
    results_root = _resolve_host_path(
        config_dir, results_value, Path("autotune-runs") / name
    )
    policy_value = _optional_string(raw, "policy")
    raw_token_policies_value = raw.get("token_policies")
    if policy_value is not None and raw_token_policies_value is not None:
        raise PlanError("policy and token_policies are mutually exclusive")
    policy = _resolve_host_path(config_dir, policy_value, Path()) if policy_value else None

    token_policies: dict[int, Path] = {}
    if raw_token_policies_value is not None:
        raw_token_policies = _require_object(raw_token_policies_value, "token_policies")
        for raw_token, host_path in raw_token_policies.items():
            try:
                token = int(raw_token)
            except (TypeError, ValueError) as exc:
                raise PlanError(f"invalid token_policies key: {raw_token!r}") from exc
            if str(token) != str(raw_token):
                raise PlanError(
                    f"token_policies key must be a canonical integer string: {raw_token!r}"
                )
            if not isinstance(host_path, str) or not host_path:
                raise PlanError(f"token_policies.{raw_token} must be a non-empty host path")
            token_policies[token] = _resolve_host_path(config_dir, host_path, Path())
        missing_policy_tokens = [str(token) for token in tokens if token not in token_policies]
        extra_policy_tokens = [str(token) for token in token_policies if token not in tokens]
        if missing_policy_tokens:
            raise PlanError(
                "token_policies is missing token(s): " + ", ".join(missing_policy_tokens)
            )
        if extra_policy_tokens:
            raise PlanError(
                "token_policies contains unused token(s): " + ", ".join(extra_policy_tokens)
            )

    if policy is not None and not policy.is_file():
        raise PlanError(f"policy file does not exist: {policy}")
    missing_policy_files = [
        f"{token}: {token_policies[token]}"
        for token in tokens
        if token in token_policies and not token_policies[token].is_file()
    ]
    if missing_policy_files:
        raise PlanError(
            "token_policies file(s) do not exist: " + "; ".join(missing_policy_files)
        )

    script_args_value = raw.get("script_args", [])
    if not isinstance(script_args_value, list) or not all(
        isinstance(item, (str, int, float)) and not isinstance(item, bool)
        for item in script_args_value
    ):
        raise PlanError("script_args must be an array of strings or numbers")

    artifacts_value = raw.get(
        "required_artifacts", ["scheduler_trace.csv", "hardware_stats.csv"]
    )
    if not isinstance(artifacts_value, list) or not all(
        isinstance(item, str) and item for item in artifacts_value
    ):
        raise PlanError("required_artifacts must be an array of non-empty strings")

    timeout = raw.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
    ):
        raise PlanError("timeout_seconds must be a positive number when set")
    use_su = raw.get("use_su", False)
    if not isinstance(use_su, bool):
        raise PlanError("use_su must be true or false")
    shuffle_seed = raw.get("shuffle_seed")
    if shuffle_seed is not None and (
        isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int)
    ):
        raise PlanError("shuffle_seed must be an integer when set")

    return PlanConfig(
        source=source,
        name=name,
        tokens=tokens,
        gpu_indices=gpu_indices,
        query_periods_ms=periods,
        repeats=repeats,
        token_json_paths=token_paths,
        common_env=common_env,
        results_root=results_root,
        remote_root=_optional_string(raw, "remote_root") or "/data/local/tmp/llama.cpp",
        remote_work_root=(
            _optional_string(raw, "remote_work_root")
            or "/data/local/tmp/treprefill-autotune"
        ),
        script=_optional_string(raw, "script") or "llama32_ffn_switch_hybrid.sh",
        script_args=tuple(str(item) for item in script_args_value),
        policy=policy,
        token_policies=token_policies,
        required_artifacts=tuple(artifacts_value),
        timeout_seconds=float(timeout) if timeout is not None else None,
        use_su=use_su,
        adb=_optional_string(raw, "adb") or "adb",
        serial=_optional_string(raw, "serial"),
        shuffle_seed=shuffle_seed,
    )


def expand_trials(plan: PlanConfig) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for token in plan.tokens:
        for gpu_index in plan.gpu_indices:
            for period_ms in plan.query_periods_ms:
                for repeat_index in range(1, plan.repeats + 1):
                    trial_id = (
                        f"{plan.name}-i{token}-gpu{gpu_index}-qp{period_ms}ms-r{repeat_index}"
                    )
                    try:
                        validate_trial_id(trial_id)
                    except argparse.ArgumentTypeError as exc:
                        raise PlanError(f"expanded trial id {trial_id!r} is invalid: {exc}") from exc
                    environment = dict(plan.common_env)
                    environment.update(
                        {
                            "INPUT_LENGTH": str(token),
                            "JSON_PATH": plan.token_json_paths[token],
                            "GPU_P": str(gpu_index),
                            "GPU_D": str(gpu_index),
                            "QUERY_PERIOD_MS": str(period_ms),
                            "RUN_REPLICATE": str(repeat_index),
                        }
                    )
                    specs.append(
                        TrialSpec(
                            trial_id=trial_id,
                            local_results_root=plan.results_root,
                            remote_root=plan.remote_root,
                            remote_work_root=plan.remote_work_root,
                            script=plan.script,
                            script_args=plan.script_args,
                            environment=environment,
                            policy=(
                                plan.token_policies[token]
                                if plan.token_policies
                                else plan.policy
                            ),
                            required_artifacts=plan.required_artifacts,
                            timeout_seconds=plan.timeout_seconds,
                            use_su=plan.use_su,
                        )
                    )
    if plan.shuffle_seed is not None:
        random.Random(plan.shuffle_seed).shuffle(specs)
    return specs


def _manifest_status(directory: Path) -> str:
    manifest_path = directory / "trial.json"
    if not manifest_path.is_file():
        return "incomplete"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid-manifest"
    status = manifest.get("status")
    return status if isinstance(status, str) and status else "unknown"


def _attempt_history(spec: TrialSpec) -> list[tuple[int, str, str]]:
    root = spec.local_results_root
    history: list[tuple[int, str, str]] = []
    base_dir = root / spec.trial_id
    if base_dir.exists():
        history.append((0, spec.trial_id, _manifest_status(base_dir)))
    if root.is_dir():
        retry_re = re.compile(_RETRY_RE_TEMPLATE.format(base=re.escape(spec.trial_id)))
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = retry_re.fullmatch(child.name)
            if match:
                history.append((int(match.group(1)), child.name, _manifest_status(child)))
    history.sort(key=lambda item: item[0])
    return history


def decide_trial(spec: TrialSpec, *, resume: bool) -> TrialDecision:
    history = _attempt_history(spec)
    summarized = tuple((trial_id, status) for _, trial_id, status in history)
    completed = [entry for entry in history if entry[2] == "complete"]
    if completed:
        return TrialDecision(
            logical_trial_id=spec.trial_id,
            action="skip",
            reason=f"completed as {completed[-1][1]}",
            spec=None,
            previous_attempts=summarized,
        )
    if not history:
        return TrialDecision(
            logical_trial_id=spec.trial_id,
            action="run",
            reason="no previous attempt",
            spec=spec,
            previous_attempts=(),
        )
    if not resume:
        return TrialDecision(
            logical_trial_id=spec.trial_id,
            action="conflict",
            reason="an incomplete or failed attempt exists; use --resume",
            spec=None,
            previous_attempts=summarized,
        )

    next_attempt = max(entry[0] for entry in history) + 1
    retry_spec = replace(spec, trial_id=f"{spec.trial_id}-retry{next_attempt}")
    try:
        validate_trial_id(retry_spec.trial_id)
    except argparse.ArgumentTypeError as exc:
        raise PlanError(f"retry trial id {retry_spec.trial_id!r} is invalid: {exc}") from exc
    return TrialDecision(
        logical_trial_id=spec.trial_id,
        action="retry",
        reason=f"preserving {len(history)} previous unsuccessful attempt(s)",
        spec=retry_spec,
        previous_attempts=summarized,
    )


def plan_decisions(plan: PlanConfig, *, resume: bool) -> list[TrialDecision]:
    return [decide_trial(spec, resume=resume) for spec in expand_trials(plan)]


def _print_list(decisions: Sequence[TrialDecision]) -> None:
    print("action\tlogical_trial_id\tnext_trial_id\treason")
    for decision in decisions:
        next_id = decision.spec.trial_id if decision.spec is not None else "-"
        print(f"{decision.action}\t{decision.logical_trial_id}\t{next_id}\t{decision.reason}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand and sequentially execute a TrePrefill thermal-discovery plan."
    )
    parser.add_argument("plan", type=Path, help="JSON trial-plan file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="list expanded trials (default)")
    mode.add_argument(
        "--dry-run", action="store_true", help="show resolved ADB commands without running them"
    )
    mode.add_argument("--run", action="store_true", help="execute trials sequentially")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip completed trials and retry failed/incomplete trials under a new suffix",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue with later trials after a failed device run",
    )
    parser.add_argument("--serial", help="override the ADB serial in the JSON plan")
    parser.add_argument("--quiet", action="store_true", help="do not mirror device logs to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.continue_on_error and not args.run:
        parser.error("--continue-on-error requires --run")

    try:
        plan = load_plan(args.plan)
        decisions = plan_decisions(plan, resume=args.resume)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list or (not args.dry_run and not args.run):
        _print_list(decisions)
        return 0

    runner = AdbRunner(plan.adb, args.serial or plan.serial, verbose=not args.quiet)
    conflicts = [decision for decision in decisions if decision.action == "conflict"]
    if conflicts:
        _print_list(conflicts)
        print("error: existing attempts found; rerun with --resume", file=sys.stderr)
        return 2

    if args.dry_run:
        descriptions = []
        try:
            for decision in decisions:
                if decision.spec is None:
                    descriptions.append(
                        {
                            "logical_trial_id": decision.logical_trial_id,
                            "action": decision.action,
                            "reason": decision.reason,
                        }
                    )
                else:
                    descriptions.append(
                        {
                            "logical_trial_id": decision.logical_trial_id,
                            "action": decision.action,
                            "reason": decision.reason,
                            "trial": runner.describe(decision.spec),
                        }
                    )
        except RunnerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(descriptions, indent=2, sort_keys=True))
        return 0

    attempted = 0
    completed = 0
    failed = 0
    skipped = sum(1 for decision in decisions if decision.action == "skip")
    for decision in decisions:
        if decision.spec is None:
            continue
        attempted += 1
        print(
            f"[{attempted}/{len(decisions) - skipped}] "
            f"{decision.action}: {decision.spec.trial_id}",
            flush=True,
        )
        try:
            runner.run(decision.spec)
            completed += 1
        except RemoteStateUncertainError as exc:
            failed += 1
            print(f"error: {decision.spec.trial_id}: {exc}", file=sys.stderr)
            print(
                "fatal: remote execution state is uncertain; aborting the plan even "
                "though --continue-on-error was requested",
                file=sys.stderr,
            )
            break
        except (OSError, RunnerError, subprocess.SubprocessError) as exc:
            failed += 1
            print(f"error: {decision.spec.trial_id}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    print(
        f"plan summary: completed={completed}, failed={failed}, "
        f"skipped={skipped}, total={len(decisions)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
