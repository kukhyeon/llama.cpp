from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.treprefill_autotune.adb_runner import (
    AdbRunner,
    RemoteStateUncertainError,
    RunnerError,
    TrialSpec,
    _parse_remote_rc,
    build_trial_command,
    parse_env_assignment,
    validate_trial_id,
)


class AdbRunnerUnitTest(unittest.TestCase):
    def test_parse_env_preserves_equals_in_value(self) -> None:
        self.assertEqual(parse_env_assignment("RUN_LABEL=a=b"), ("RUN_LABEL", "a=b"))

    def test_parse_env_rejects_invalid_name(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_env_assignment("NOT-AN-ENV=value")

    def test_trial_id_rejects_shell_metacharacters(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            validate_trial_id("trial;reboot")

    def test_remote_return_code_sentinel(self) -> None:
        self.assertEqual(_parse_remote_rc("__TREPREFILL_REMOTE_RC__:17\n"), 17)
        self.assertIsNone(_parse_remote_rc("normal output"))

    def test_build_command_quotes_environment_and_sets_owned_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = TrialSpec(
                trial_id="trial-1",
                local_results_root=Path(temporary),
                environment={"RUN_LABEL": "clock state; not a command", "GPU_P": "14"},
                script_args=("15", "9"),
            )
            command, environment = build_trial_command(
                spec,
                remote_output_dir="/data/local/tmp/treprefill-autotune/trial-1/output",
                remote_policy_path="/data/local/tmp/treprefill-autotune/trial-1/policy.json",
            )

        self.assertIn("'RUN_LABEL=clock state; not a command'", command)
        self.assertIn("sh /data/local/tmp/llama.cpp/llama32_ffn_switch_hybrid.sh 15 9", command)
        self.assertEqual(environment["BACKEND_POLICY"], "on")
        self.assertEqual(environment["RUN_NAME"], "trial-1")

    def test_runner_owned_environment_cannot_be_overridden(self) -> None:
        spec = TrialSpec(
            trial_id="trial-1",
            local_results_root=Path("results"),
            environment={"OUTPUT_DIR": "/tmp/not-the-trial-directory"},
        )
        with self.assertRaises(RunnerError):
            build_trial_command(
                spec,
                remote_output_dir="/data/local/tmp/treprefill-autotune/trial-1/output",
                remote_policy_path=None,
            )

    def test_stream_timeout_reports_uncertain_remote_state(self) -> None:
        process = MagicMock()
        process.stdout = io.StringIO("")
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="adb shell", timeout=0.01),
            0,
        ]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.treprefill_autotune.adb_runner.subprocess.Popen",
            return_value=process,
        ):
            runner = AdbRunner(verbose=False)
            with self.assertRaises(RemoteStateUncertainError):
                runner._run_streamed(
                    ["adb", "shell", "experiment"],
                    Path(temporary) / "adb-shell.log",
                    timeout=0.01,
                )

        process.terminate.assert_called_once_with()

    def test_keyboard_interrupt_reports_uncertain_remote_state(self) -> None:
        process = MagicMock()
        process.stdout = io.StringIO("")
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.treprefill_autotune.adb_runner.subprocess.Popen",
            return_value=process,
        ):
            runner = AdbRunner(verbose=False)
            with self.assertRaisesRegex(RemoteStateUncertainError, "interrupted"):
                runner._run_streamed(
                    ["adb", "shell", "experiment"],
                    Path(temporary) / "adb-shell.log",
                    timeout=None,
                )

        process.terminate.assert_called_once_with()

    def test_missing_remote_return_code_sentinel_is_uncertain(self) -> None:
        process = MagicMock()
        process.stdout = io.StringIO("adb transport ended without a sentinel\n")
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.treprefill_autotune.adb_runner.subprocess.Popen",
            return_value=process,
        ):
            runner = AdbRunner(verbose=False)
            with self.assertRaisesRegex(RemoteStateUncertainError, "without.*sentinel"):
                runner._run_streamed(
                    ["adb", "shell", "experiment"],
                    Path(temporary) / "adb-shell.log",
                    timeout=None,
                )

    def test_uncertain_remote_state_is_recorded_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = TrialSpec(trial_id="timeout-trial", local_results_root=root)
            runner = AdbRunner(verbose=False)
            capture_result = subprocess.CompletedProcess([], 0, stdout="", stderr=None)
            with patch.object(runner, "_run_capture", return_value=capture_result), patch.object(
                runner, "_run_remote_capture"
            ), patch.object(
                runner,
                "_run_streamed",
                side_effect=RemoteStateUncertainError("remote may still be running"),
            ):
                with self.assertRaises(RemoteStateUncertainError):
                    runner.run(spec)

            manifest = json.loads(
                (root / spec.trial_id / "trial.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["status"], "remote-state-uncertain")
        self.assertIn("remote may still be running", manifest["error"])


if __name__ == "__main__":
    unittest.main()
