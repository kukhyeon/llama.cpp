#!/usr/bin/env python3
"""Run one TrePrefill profiling trial on an Android device through ADB.

The runner deliberately does not know how to optimize a policy.  It provides the
reproducible device-I/O boundary needed by an autotuner:

1. reserve a unique remote trial directory;
2. optionally push a generated policy;
3. execute one of the existing on-device shell scripts with explicit
   environment overrides;
4. pull the complete output directory; and
5. record a host-side JSON manifest and console log.

The Android package and model are assumed to be installed already.  No existing
on-device result directory is deleted or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import IO, Sequence


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRIAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_REMOTE_RC_PREFIX = "__TREPREFILL_REMOTE_RC__:"


class RunnerError(RuntimeError):
    """A recoverable trial setup, execution, or collection failure."""


class RemoteStateUncertainError(RunnerError):
    """The host lost control while the on-device experiment may still be running.

    Callers must not automatically start another thermal trial after this
    error.  Killing the local ``adb`` process does not prove that its remote
    shell child or the experiment script has exited.
    """


@dataclass(frozen=True)
class TrialSpec:
    """Description of one remote profiling trial.

    ``script`` is interpreted relative to ``remote_root`` unless it is an
    absolute POSIX path.  ``environment`` is passed through ``env`` only for
    the selected script; it does not mutate the host process environment.
    """

    trial_id: str
    local_results_root: Path
    remote_root: str = "/data/local/tmp/llama.cpp"
    remote_work_root: str = "/data/local/tmp/treprefill-autotune"
    script: str = "llama32_ffn_switch_hybrid.sh"
    script_args: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    policy: Path | None = None
    required_artifacts: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    use_su: bool = False


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    local_trial_dir: Path
    local_device_output: Path | None
    remote_trial_dir: str
    remote_output_dir: str
    remote_return_code: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _ProcessResult:
    local_return_code: int
    remote_return_code: int | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_trial_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"trial-{timestamp}-{uuid.uuid4().hex[:8]}"


def validate_trial_id(value: str) -> str:
    if not _TRIAL_ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "trial id must be 1-96 characters using only letters, digits, '.', '-', or '_'"
        )
    return value


def parse_env_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    name, raw_value = value.split("=", 1)
    if not _ENV_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(f"invalid environment variable name: {name!r}")
    return name, raw_value


def _absolute_remote_path(value: str, *, option: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise RunnerError(f"{option} must be an absolute normalized POSIX path: {value!r}")
    return str(path)


def _remote_child(parent: str, *parts: str) -> str:
    base = PurePosixPath(_absolute_remote_path(parent, option="remote path"))
    return str(base.joinpath(*parts))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap_remote_command(command: str) -> str:
    # Some older adb clients report only the transport status.  The sentinel
    # preserves the actual Android shell return code in stdout.
    return (
        f"{command}; _treprefill_rc=$?; "
        f"printf '\\n{_REMOTE_RC_PREFIX}%s\\n' \"$_treprefill_rc\"; "
        "exit \"$_treprefill_rc\""
    )


def _parse_remote_rc(line: str) -> int | None:
    stripped = line.strip()
    if not stripped.startswith(_REMOTE_RC_PREFIX):
        return None
    try:
        return int(stripped[len(_REMOTE_RC_PREFIX) :])
    except ValueError:
        return None


def build_trial_command(
    spec: TrialSpec,
    *,
    remote_output_dir: str,
    remote_policy_path: str | None,
) -> tuple[str, dict[str, str]]:
    """Return the remote shell command and the effective environment."""

    remote_root = _absolute_remote_path(spec.remote_root, option="--remote-root")
    effective_env = dict(spec.environment)
    generated_env = {
        "OUTPUT_DIR": remote_output_dir,
        "RESULT_ROOT": str(PurePosixPath(remote_output_dir).parent),
        "RUN_NAME": spec.trial_id,
        "RUN_ID": spec.trial_id,
        "CONFIG_PATH": str(PurePosixPath(remote_output_dir, "config.txt")),
        "SCHED_TRACE_PATH": str(PurePosixPath(remote_output_dir, "scheduler_trace.csv")),
        "NODE_WALL_TRACE_PATH": str(PurePosixPath(remote_output_dir, "node_wall_trace.csv")),
        "FFN_WORKER_TRACE_PATH": str(PurePosixPath(remote_output_dir, "ffn_worker_trace.csv")),
        "MODULE_BENCH_TRACE": str(PurePosixPath(remote_output_dir, "module_trace.csv")),
    }
    if remote_policy_path is not None:
        generated_env.update(
            {
                "BACKEND_POLICY": "on",
                "BACKEND_POLICY_CONFIG": remote_policy_path,
                "BACKEND_POLICY_DEFAULT_CONFIG": remote_policy_path,
            }
        )

    conflicts = sorted(set(effective_env).intersection(generated_env))
    if conflicts:
        joined = ", ".join(conflicts)
        raise RunnerError(f"runner-owned environment variable(s) cannot be overridden: {joined}")
    effective_env.update(generated_env)

    if PurePosixPath(spec.script).is_absolute():
        remote_script = str(PurePosixPath(spec.script))
    else:
        if ".." in PurePosixPath(spec.script).parts:
            raise RunnerError(f"--script cannot escape --remote-root: {spec.script!r}")
        remote_script = str(PurePosixPath(remote_root).joinpath(spec.script))

    env_words = [shlex.quote(f"{key}={value}") for key, value in sorted(effective_env.items())]
    script_words = ["sh", remote_script, *spec.script_args]
    inner = " ".join(["env", *env_words, shlex.join(script_words)])
    command = f"cd {shlex.quote(remote_root)} && {inner}"
    return command, effective_env


class AdbRunner:
    def __init__(self, adb: str = "adb", serial: str | None = None, *, verbose: bool = True):
        self.adb = adb
        self.serial = serial
        self.verbose = verbose

    def _adb_prefix(self) -> list[str]:
        command = [self.adb]
        if self.serial:
            command.extend(["-s", self.serial])
        return command

    def _shell_args(self, command: str, *, use_su: bool) -> list[str]:
        wrapped = _wrap_remote_command(command)
        if use_su:
            wrapped = f"su -c {shlex.quote(wrapped)}"
        return [*self._adb_prefix(), "shell", wrapped]

    def _run_capture(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        timeout: float | None = 30,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            list(args),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        if self.verbose and result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if check and result.returncode != 0:
            raise RunnerError(
                f"command failed with exit code {result.returncode}: {shlex.join(args)}\n"
                f"{result.stdout.rstrip()}"
            )
        return result

    def _run_remote_capture(
        self,
        command: str,
        *,
        use_su: bool = False,
        check: bool = True,
        timeout: float | None = 30,
    ) -> tuple[subprocess.CompletedProcess[str], int | None]:
        result = self._run_capture(
            self._shell_args(command, use_su=use_su), check=False, timeout=timeout
        )
        remote_rc = None
        for line in result.stdout.splitlines():
            parsed = _parse_remote_rc(line)
            if parsed is not None:
                remote_rc = parsed
        effective_rc = remote_rc if remote_rc is not None else result.returncode
        if check and effective_rc != 0:
            raise RunnerError(
                f"remote command failed with exit code {effective_rc}: {command}\n"
                f"{result.stdout.rstrip()}"
            )
        return result, remote_rc

    def _run_streamed(
        self,
        args: Sequence[str],
        log_path: Path,
        *,
        timeout: float | None,
    ) -> _ProcessResult:
        """Tee a long-running command to the terminal and a UTF-8 log."""

        process = subprocess.Popen(
            list(args),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        observed_remote_rc: list[int] = []

        def pump(source: IO[str], destination: IO[str]) -> None:
            for line in source:
                destination.write(line)
                destination.flush()
                if self.verbose:
                    print(line, end="", flush=True)
                parsed = _parse_remote_rc(line)
                if parsed is not None:
                    observed_remote_rc.append(parsed)

        with log_path.open("w", encoding="utf-8") as log_file:
            reader = threading.Thread(target=pump, args=(process.stdout, log_file), daemon=True)
            reader.start()
            try:
                local_rc = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                reader.join(timeout=5)
                interrupted = isinstance(exc, KeyboardInterrupt)
                raise RemoteStateUncertainError(
                    (
                        "ADB command was interrupted; "
                        if interrupted
                        else "ADB command timed out; "
                    )
                    + "the host adb process was stopped. "
                    "The on-device script may still be running; verify and stop it before "
                    "starting another thermal trial."
                ) from exc
            reader.join(timeout=5)

        if not observed_remote_rc:
            raise RemoteStateUncertainError(
                "ADB command ended without the on-device return-code sentinel. "
                "The transport may have failed while the script kept running; verify and "
                "stop it before starting another thermal trial."
            )
        return _ProcessResult(
            local_return_code=local_rc,
            remote_return_code=observed_remote_rc[-1],
        )

    def _write_manifest(self, path: Path, manifest: dict[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def describe(self, spec: TrialSpec) -> dict[str, object]:
        validate_trial_id(spec.trial_id)
        remote_trial_dir = _remote_child(spec.remote_work_root, spec.trial_id)
        remote_output_dir = _remote_child(remote_trial_dir, "output")
        remote_policy_path = _remote_child(remote_trial_dir, "policy.json") if spec.policy else None
        command, effective_env = build_trial_command(
            spec,
            remote_output_dir=remote_output_dir,
            remote_policy_path=remote_policy_path,
        )
        shell_command = _wrap_remote_command(command)
        if spec.use_su:
            shell_command = f"su -c {shlex.quote(shell_command)}"
        return {
            "trial_id": spec.trial_id,
            "local_trial_dir": str(spec.local_results_root / spec.trial_id),
            "remote_trial_dir": remote_trial_dir,
            "remote_output_dir": remote_output_dir,
            "remote_policy_path": remote_policy_path,
            "effective_environment": effective_env,
            "adb_shell_command": shlex.join([*self._adb_prefix(), "shell", shell_command]),
        }

    def run(self, spec: TrialSpec) -> TrialResult:
        """Execute and collect one trial, preserving diagnostics on failure."""

        plan = self.describe(spec)
        local_trial_dir = Path(str(plan["local_trial_dir"]))
        if local_trial_dir.exists():
            raise RunnerError(f"local trial directory already exists: {local_trial_dir}")

        policy_info: dict[str, object] | None = None
        resolved_policy: Path | None = None
        if spec.policy is not None:
            resolved_policy = spec.policy.expanduser().resolve()
            if not resolved_policy.is_file():
                raise RunnerError(f"policy file does not exist: {resolved_policy}")

        local_trial_dir.mkdir(parents=True)
        manifest_path = local_trial_dir / "trial.json"
        console_log = local_trial_dir / "adb-shell.log"

        if resolved_policy is not None:
            policy_snapshot = local_trial_dir / "input-policy.json"
            shutil.copy2(resolved_policy, policy_snapshot)
            policy_info = {
                "source": str(resolved_policy),
                "snapshot": policy_snapshot.name,
                "sha256": _sha256(policy_snapshot),
            }

        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "starting",
            "created_at": _utc_now(),
            "adb": {"executable": self.adb, "serial": self.serial, "use_su": spec.use_su},
            "spec": {
                **asdict(spec),
                "local_results_root": str(spec.local_results_root),
                "policy": str(spec.policy) if spec.policy else None,
            },
            "plan": plan,
            "policy": policy_info,
        }
        self._write_manifest(manifest_path, manifest)

        started = time.monotonic()
        remote_rc = -1
        local_device_output: Path | None = None
        remote_reserved = False
        pull_attempted = False
        try:
            self._run_capture([*self._adb_prefix(), "get-state"], timeout=15)

            remote_trial_dir = str(plan["remote_trial_dir"])
            remote_output_dir = str(plan["remote_output_dir"])
            reserve_command = (
                f"if [ -e {shlex.quote(remote_trial_dir)} ]; then "
                "echo 'remote trial directory already exists' >&2; false; "
                f"else mkdir -p {shlex.quote(remote_output_dir)}; fi"
            )
            # /data/local/tmp is normally writable by the adb shell user.  Do
            # this outside su so a subsequent `adb push` can create policy.json.
            self._run_remote_capture(reserve_command)
            remote_reserved = True

            remote_policy_path = plan["remote_policy_path"]
            if resolved_policy is not None and isinstance(remote_policy_path, str):
                self._run_capture(
                    [*self._adb_prefix(), "push", str(resolved_policy), remote_policy_path],
                    timeout=60,
                )

            command, _ = build_trial_command(
                spec,
                remote_output_dir=remote_output_dir,
                remote_policy_path=remote_policy_path if isinstance(remote_policy_path, str) else None,
            )
            manifest["status"] = "running"
            manifest["started_at"] = _utc_now()
            self._write_manifest(manifest_path, manifest)

            process_result = self._run_streamed(
                self._shell_args(command, use_su=spec.use_su),
                console_log,
                timeout=spec.timeout_seconds,
            )
            remote_rc = (
                process_result.remote_return_code
                if process_result.remote_return_code is not None
                else process_result.local_return_code
            )

            local_device_output = local_trial_dir / "device-output"
            pull_attempted = True
            pull_result = self._run_capture(
                [*self._adb_prefix(), "pull", remote_output_dir, str(local_device_output)],
                check=False,
                timeout=300,
            )
            if pull_result.returncode != 0:
                raise RunnerError(
                    f"adb pull failed with exit code {pull_result.returncode}: "
                    f"{pull_result.stdout.rstrip()}"
                )

            missing = [
                name
                for name in spec.required_artifacts
                if not any(path.is_file() for path in local_device_output.rglob(name))
            ]
            if missing:
                raise RunnerError(f"required artifact(s) were not collected: {', '.join(missing)}")
            if remote_rc != 0:
                raise RunnerError(f"on-device trial failed with exit code {remote_rc}")

            elapsed = time.monotonic() - started
            manifest.update(
                {
                    "status": "complete",
                    "finished_at": _utc_now(),
                    "elapsed_seconds": elapsed,
                    "remote_return_code": remote_rc,
                    "local_device_output": str(local_device_output),
                }
            )
            self._write_manifest(manifest_path, manifest)
            return TrialResult(
                trial_id=spec.trial_id,
                local_trial_dir=local_trial_dir,
                local_device_output=local_device_output,
                remote_trial_dir=remote_trial_dir,
                remote_output_dir=remote_output_dir,
                remote_return_code=remote_rc,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            partial_pull_error: str | None = None
            if remote_reserved and not pull_attempted:
                local_device_output = local_trial_dir / "device-output-partial"
                try:
                    partial_pull = self._run_capture(
                        [
                            *self._adb_prefix(),
                            "pull",
                            str(plan["remote_output_dir"]),
                            str(local_device_output),
                        ],
                        check=False,
                        timeout=60,
                    )
                    if partial_pull.returncode != 0:
                        partial_pull_error = partial_pull.stdout.rstrip()
                        local_device_output = None
                except Exception as pull_exc:  # retain the primary failure
                    partial_pull_error = str(pull_exc)
            manifest.update(
                {
                    "status": (
                        "remote-state-uncertain"
                        if isinstance(exc, RemoteStateUncertainError)
                        else "failed"
                    ),
                    "finished_at": _utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "remote_return_code": remote_rc,
                    "error": str(exc),
                    "local_device_output": (
                        str(local_device_output) if local_device_output is not None else None
                    ),
                    "partial_pull_error": partial_pull_error,
                }
            )
            self._write_manifest(manifest_path, manifest)
            raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one existing TrePrefill shell-script trial over ADB and collect its logs."
    )
    parser.add_argument("--adb", default="adb", help="adb executable (default: adb)")
    parser.add_argument("--serial", help="ADB device serial when more than one device is connected")
    parser.add_argument("--trial-id", type=validate_trial_id, default=default_trial_id())
    parser.add_argument("--results-root", type=Path, default=Path("autotune-runs"))
    parser.add_argument("--remote-root", default="/data/local/tmp/llama.cpp")
    parser.add_argument(
        "--remote-work-root", default="/data/local/tmp/treprefill-autotune"
    )
    parser.add_argument("--script", default="llama32_ffn_switch_hybrid.sh")
    parser.add_argument(
        "--script-arg",
        action="append",
        default=[],
        help="positional argument passed to the remote script; repeat as needed",
    )
    parser.add_argument("--policy", type=Path, help="generated policy JSON to push for this trial")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="remote script environment override; repeat as needed",
    )
    parser.add_argument(
        "--require-artifact",
        action="append",
        default=[],
        help="file name that must exist below the pulled output directory",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="host-side execution timeout in seconds (default: wait indefinitely)",
    )
    parser.add_argument(
        "--su", action="store_true", help="execute the experiment script through on-device su -c"
    )
    parser.add_argument("--quiet", action="store_true", help="do not mirror adb output to stdout")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the resolved trial plan without using adb"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    environment: dict[str, str] = {}
    for raw_assignment in args.env:
        try:
            name, value = parse_env_assignment(raw_assignment)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if name in environment:
            parser.error(f"duplicate --env variable: {name}")
        environment[name] = value

    spec = TrialSpec(
        trial_id=args.trial_id,
        local_results_root=args.results_root.expanduser().resolve(),
        remote_root=args.remote_root,
        remote_work_root=args.remote_work_root,
        script=args.script,
        script_args=tuple(args.script_arg),
        environment=environment,
        policy=args.policy.expanduser().resolve() if args.policy else None,
        required_artifacts=tuple(args.require_artifact),
        timeout_seconds=args.timeout,
        use_su=args.su,
    )
    runner = AdbRunner(args.adb, args.serial, verbose=not args.quiet)

    try:
        if args.dry_run:
            print(json.dumps(runner.describe(spec), indent=2, sort_keys=True))
            return 0
        result = runner.run(spec)
    except (OSError, RunnerError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"trial complete: {result.local_trial_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
