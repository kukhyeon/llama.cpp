from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.treprefill_autotune.adb_runner import RemoteStateUncertainError
from scripts.treprefill_autotune.trial_plan import (
    PlanError,
    decide_trial,
    expand_trials,
    load_plan,
    main,
)


def _base_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "thermal-test",
        "tokens": [64, 128],
        "gpu_indices": [14, 4],
        "query_periods_ms": [1000, 2000],
        "repeats": 2,
        "token_json_paths": {
            "64": "data/llama32_prefill_64.json",
            "128": "data/llama32_prefill_128.json",
        },
        "common_env": {
            "MODEL": "/data/local/tmp/gguf/custom-model.gguf",
            "MODEL_TAG": "custom-model-q8-0",
            "MAX_QUERY_NUMBER": 100,
            "SCHED_TRACE": True,
        },
        "required_artifacts": [],
    }


class TrialPlanUnitTest(unittest.TestCase):
    def _write_plan(self, directory: Path, config: dict[str, object]) -> Path:
        path = directory / "plan.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_expands_full_product_with_token_specific_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = load_plan(self._write_plan(directory, _base_config()))
            specs = expand_trials(plan)

        self.assertEqual(len(specs), 2 * 2 * 2 * 2)
        first = specs[0]
        self.assertEqual(first.trial_id, "thermal-test-i64-gpu14-qp1000ms-r1")
        self.assertEqual(first.environment["JSON_PATH"], "data/llama32_prefill_64.json")
        self.assertEqual(first.environment["GPU_P"], "14")
        self.assertEqual(first.environment["GPU_D"], "14")
        self.assertEqual(first.environment["SCHED_TRACE"], "on")
        self.assertEqual(first.environment["MAX_QUERY_NUMBER"], "100")
        self.assertEqual(first.environment["MODEL"], "/data/local/tmp/gguf/custom-model.gguf")
        self.assertEqual(first.environment["MODEL_TAG"], "custom-model-q8-0")

    def test_token_specific_policies_are_resolved_and_selected(self) -> None:
        config = _base_config()
        config["token_policies"] = {
            "64": "policies/i64.json",
            "128": "policies/i128.json",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "policies").mkdir()
            (directory / "policies/i64.json").write_text("{}", encoding="utf-8")
            (directory / "policies/i128.json").write_text("{}", encoding="utf-8")
            plan = load_plan(self._write_plan(directory, config))
            specs = expand_trials(plan)

        policies_by_token = {
            int(spec.environment["INPUT_LENGTH"]): spec.policy for spec in specs
        }
        self.assertEqual(policies_by_token[64], (directory / "policies/i64.json").resolve())
        self.assertEqual(policies_by_token[128], (directory / "policies/i128.json").resolve())

    def test_single_policy_remains_shared_by_all_tokens(self) -> None:
        config = _base_config()
        config["policy"] = "policies/shared.json"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "policies").mkdir()
            (directory / "policies/shared.json").write_text("{}", encoding="utf-8")
            specs = expand_trials(load_plan(self._write_plan(directory, config)))

        expected = (directory / "policies/shared.json").resolve()
        self.assertTrue(specs)
        self.assertEqual({spec.policy for spec in specs}, {expected})

    def test_policy_and_token_policies_are_mutually_exclusive(self) -> None:
        config = _base_config()
        config["policy"] = "one-policy.json"
        config["token_policies"] = {
            "64": "policies/i64.json",
            "128": "policies/i128.json",
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PlanError, "mutually exclusive"):
                load_plan(self._write_plan(Path(temporary), config))

    def test_token_policies_must_cover_exact_token_set(self) -> None:
        config = _base_config()
        config["token_policies"] = {"64": "policies/i64.json"}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PlanError, "missing token"):
                load_plan(self._write_plan(Path(temporary), config))

    def test_all_token_policy_files_are_validated_before_any_trial(self) -> None:
        config = _base_config()
        config["token_policies"] = {
            "64": "policies/i64.json",
            "128": "policies/missing-i128.json",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "policies").mkdir()
            (directory / "policies/i64.json").write_text("{}", encoding="utf-8")
            plan_path = self._write_plan(directory, config)
            with patch(
                "scripts.treprefill_autotune.trial_plan.AdbRunner"
            ) as runner_class, redirect_stdout(io.StringIO()), patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr:
                return_code = main([str(plan_path), "--run", "--quiet"])

        self.assertEqual(return_code, 2)
        runner_class.assert_not_called()
        self.assertIn("128", stderr.getvalue())
        self.assertIn("missing-i128.json", stderr.getvalue())

    def test_single_policy_file_is_prevalidated(self) -> None:
        config = _base_config()
        config["policy"] = "policies/missing.json"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PlanError, "policy file does not exist"):
                load_plan(self._write_plan(Path(temporary), config))

    def test_missing_token_dataset_is_rejected(self) -> None:
        config = _base_config()
        config["token_json_paths"] = {"64": "data/llama32_prefill_64.json"}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PlanError, "missing token"):
                load_plan(self._write_plan(Path(temporary), config))

    def test_generated_environment_cannot_be_overridden(self) -> None:
        config = _base_config()
        config["common_env"] = {"GPU_P": "14"}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PlanError, "plan-generated"):
                load_plan(self._write_plan(Path(temporary), config))

    def test_resume_skips_complete_trial(self) -> None:
        config = _base_config()
        config.update(
            {
                "tokens": [64],
                "gpu_indices": [14],
                "query_periods_ms": [1000],
                "repeats": 1,
                "token_json_paths": {"64": "data/llama32_prefill_64.json"},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            spec = expand_trials(load_plan(self._write_plan(directory, config)))[0]
            trial_dir = spec.local_results_root / spec.trial_id
            trial_dir.mkdir(parents=True)
            (trial_dir / "trial.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            decision = decide_trial(spec, resume=True)

        self.assertEqual(decision.action, "skip")
        self.assertIsNone(decision.spec)

    def test_resume_preserves_failures_and_uses_next_retry_suffix(self) -> None:
        config = _base_config()
        config.update(
            {
                "tokens": [64],
                "gpu_indices": [14],
                "query_periods_ms": [1000],
                "repeats": 1,
                "token_json_paths": {"64": "data/llama32_prefill_64.json"},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            spec = expand_trials(load_plan(self._write_plan(directory, config)))[0]
            for suffix in ("", "-retry1"):
                trial_dir = spec.local_results_root / f"{spec.trial_id}{suffix}"
                trial_dir.mkdir(parents=True)
                (trial_dir / "trial.json").write_text(
                    json.dumps({"status": "failed"}), encoding="utf-8"
                )
            decision = decide_trial(spec, resume=True)

        self.assertEqual(decision.action, "retry")
        assert decision.spec is not None
        self.assertEqual(decision.spec.trial_id, f"{spec.trial_id}-retry2")

    def test_run_executes_expansion_sequentially(self) -> None:
        config = _base_config()
        config.update(
            {
                "tokens": [64],
                "gpu_indices": [14, 4],
                "query_periods_ms": [1000],
                "repeats": 2,
                "token_json_paths": {"64": "data/llama32_prefill_64.json"},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config["results_root"] = str(directory / "results")
            plan_path = self._write_plan(directory, config)
            with patch(
                "scripts.treprefill_autotune.trial_plan.AdbRunner"
            ) as runner_class, redirect_stdout(io.StringIO()):
                return_code = main([str(plan_path), "--run", "--quiet"])

        self.assertEqual(return_code, 0)
        runner = runner_class.return_value
        self.assertEqual(runner.run.call_count, 4)
        ids = [call.args[0].trial_id for call in runner.run.call_args_list]
        self.assertEqual(
            ids,
            [
                "thermal-test-i64-gpu14-qp1000ms-r1",
                "thermal-test-i64-gpu14-qp1000ms-r2",
                "thermal-test-i64-gpu4-qp1000ms-r1",
                "thermal-test-i64-gpu4-qp1000ms-r2",
            ],
        )

    def test_uncertain_remote_state_aborts_even_with_continue_on_error(self) -> None:
        config = _base_config()
        config.update(
            {
                "tokens": [64],
                "gpu_indices": [14, 4],
                "query_periods_ms": [1000],
                "repeats": 1,
                "token_json_paths": {"64": "data/llama32_prefill_64.json"},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config["results_root"] = str(directory / "results")
            plan_path = self._write_plan(directory, config)
            with patch(
                "scripts.treprefill_autotune.trial_plan.AdbRunner"
            ) as runner_class, redirect_stdout(io.StringIO()), patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr:
                runner_class.return_value.run.side_effect = RemoteStateUncertainError(
                    "remote may still be running"
                )
                return_code = main(
                    [
                        str(plan_path),
                        "--run",
                        "--continue-on-error",
                        "--quiet",
                    ]
                )

        self.assertEqual(return_code, 1)
        self.assertEqual(runner_class.return_value.run.call_count, 1)
        self.assertIn("aborting the plan", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
