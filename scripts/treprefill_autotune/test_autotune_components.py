#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffn_optimizer import (  # noqa: E402
    SizeLatencySample,
    fit_device_latency_model,
    optimize_ffn_shards,
    parse_ffn_observations,
)
from trace_parser import discover_clock_states, read_ffn_branch_samples  # noqa: E402


SCHEDULER_FIELDS = [
    "query_id",
    "phase",
    "graph_id",
    "n_tokens",
    "backend",
    "layer",
    "compute_wall_us",
    "is_ffn_group",
    "ffn_branch",
    "group_id",
    "parallel_group_kind",
    "parallel_branch",
    "actual_clock_profile",
    "actual_prime_khz",
    "actual_gold_khz",
    "actual_gpu_hz",
]


class ClockStateTests(unittest.TestCase):
    def test_include_hardware_only_does_not_require_scheduler_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            _write_csv(
                run / "hardware_stats.csv",
                ["Time", "cpu6_cur_freq", "cpu0_cur_freq", "gpu_max_clock"],
                [
                    {
                        "Time": 10,
                        "cpu6_cur_freq": 4473.6,
                        "cpu0_cur_freq": 3532.8,
                        "gpu_max_clock": 1_200_000_000,
                    }
                ],
            )

            catalog = discover_clock_states(run, include_hardware_only=True)

            self.assertEqual(len(catalog.states), 1)
            self.assertEqual(catalog.states[0].scheduler_samples, 0)
            self.assertEqual(catalog.states[0].hardware_samples, 1)

    def test_discovery_filters_and_normalizes_hardware_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            scheduler = run / "scheduler_trace.csv"
            rows = []
            for query in (1, 2):
                for _ in range(2):
                    rows.append(
                        _scheduler_row(
                            query=query,
                            profile="p15-g15-gpu14",
                            prime=4_473_600,
                            gold=3_532_800,
                            gpu=1_200_000_000,
                        )
                    )
            rows.append(
                _scheduler_row(
                    query=3,
                    profile="p15-g15-gpu12",
                    prime=4_473_600,
                    gold=3_532_800,
                    gpu=1_050_000_000,
                )
            )
            _write_csv(scheduler, SCHEDULER_FIELDS, rows)
            _write_csv(
                run / "hardware_stats.csv",
                ["Time", "cpu6_cur_freq", "cpu0_cur_freq", "gpu_max_clock"],
                [
                    {
                        "Time": 10,
                        "cpu6_cur_freq": 4473.6,
                        "cpu0_cur_freq": 3532.8,
                        "gpu_max_clock": 1_200_000_000,
                    },
                    {
                        "Time": 20,
                        "cpu6_cur_freq": 1000,
                        "cpu0_cur_freq": 1000,
                        "gpu_max_clock": 200_000_000,
                    },
                ],
            )

            catalog = discover_clock_states(
                run,
                token_override=128,
                min_samples=4,
                min_queries=2,
                allowed_gpu_indices={14},
            )

            self.assertEqual(len(catalog.states), 1)
            state = catalog.states[0]
            self.assertEqual(state.profile, "p15-g15-gpu14")
            self.assertEqual(state.point.prime_khz, 4_473_600)
            self.assertEqual(state.point.gold_khz, 3_532_800)
            self.assertEqual(state.point.gpu_hz, 1_200_000_000)
            self.assertEqual(state.hardware_samples, 1)
            self.assertEqual(state.distinct_queries, 2)
            self.assertEqual(state.input_tokens, (128,))
            self.assertEqual(catalog.by_token()[128], (state,))
            json.dumps(catalog.to_dict())

    def test_sample_and_query_thresholds_are_applied_per_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run64 = root / "i64"
            run256 = root / "i256"
            run64.mkdir()
            run256.mkdir()
            # The 64-token state has many rows but only one distinct query.  It
            # must not borrow the 256-token run's four query IDs.
            _write_csv(
                run64 / "scheduler_trace.csv",
                SCHEDULER_FIELDS,
                [
                    _scheduler_row(
                        query=1,
                        n_tokens=64,
                        profile="p15-g15-gpu14",
                        prime=4_473_600,
                        gold=3_532_800,
                        gpu=1_200_000_000,
                    )
                    for _ in range(20)
                ],
            )
            _write_csv(
                run256 / "scheduler_trace.csv",
                SCHEDULER_FIELDS,
                [
                    _scheduler_row(
                        query=query,
                        n_tokens=256,
                        profile="p15-g15-gpu14",
                        prime=4_473_600,
                        gold=3_532_800,
                        gpu=1_200_000_000,
                    )
                    for query in range(1, 5)
                    for _ in range(10)
                ],
            )

            catalog = discover_clock_states(
                [run64, run256],
                min_samples=10,
                min_queries=2,
            )

            self.assertEqual(set(catalog.by_token()), {256})
            self.assertEqual(len(catalog.states), 1)
            self.assertEqual(catalog.states[0].input_tokens, (256,))
            self.assertEqual(catalog.states[0].scheduler_samples, 40)
            self.assertEqual(catalog.states[0].distinct_queries, 4)

    def test_recommend_does_not_restore_a_sparse_filtered_token_state(self) -> None:
        from scripts.treprefill_autotune.autotune_policy import recommend_ffn_plans
        from scripts.treprefill_autotune.model_descriptor import ModelDescriptor

        split = {"cpu": 384, "gpu": 384, "npu": 256}
        layout = [
            {"id": "npu", "backend": "HTP0-REPACK"},
            {"id": "gpu", "backend": "OpenCL"},
            {"id": "cpu", "backend": "CPU_REPACK"},
        ]
        template = {
            "version": 1,
            "enabled": True,
            "ffn_parallel": {
                "enabled": True,
                "phase": "all",
                "layer_range": [0, 3],
                "align": 64,
                "split_layout": layout,
                "split_sizes": split,
            },
            "profile_defaults": {
                "applicability": {
                    "input_tokens": [256, 256],
                    "ubatch_tokens": [256, 256],
                },
                "ffn_parallel": {
                    "enabled": True,
                    "phase": "all",
                    "layer_range": [0, 3],
                    "align": 64,
                    "split_layout": layout,
                },
            },
            "profiles": {
                "p15-g15-gpu14": {
                    "clock_point": {
                        "prime": {"index": 15, "khz": 4_473_600},
                        "gold": {"index": 15, "khz": 3_532_800},
                        "gpu": {"index": 14, "hz": 1_200_000_000},
                    },
                    "ffn_parallel": {"split_sizes": split},
                }
            },
            "runtime_routes": {},
        }
        model = ModelDescriptor(
            model_id="tiny-q8",
            architecture="llama",
            quantization="Q8_0",
            n_layer=4,
            n_embd=128,
            n_ff=1024,
            n_head=1,
            n_head_kv=1,
            head_dim=128,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = []
            for tokens, queries in ((64, (1,)), (256, (1, 2, 3))):
                run = root / f"i{tokens}"
                run.mkdir()
                traces.append(run)
                rows = [
                    _scheduler_row(
                        query=query,
                        n_tokens=tokens,
                        profile="p15-g15-gpu14",
                        prime=4_473_600,
                        gold=3_532_800,
                        gpu=1_200_000_000,
                        backend=backend,
                        branch=f"p15-g15-gpu14.{device}",
                        latency=split[device],
                        layer=1,
                    )
                    for query in queries
                    for device, backend in (
                        ("cpu", "CPU"),
                        ("gpu", "OpenCL"),
                        ("npu", "HTP0"),
                    )
                ]
                _write_csv(run / "scheduler_trace.csv", SCHEDULER_FIELDS, rows)

            catalog = discover_clock_states(traces, min_samples=3, min_queries=2)
            result = recommend_ffn_plans(
                model,
                template,
                traces,
                measured_policy=template,
                clock_catalog=catalog.to_dict(),
            )

            self.assertEqual(set(catalog.by_token()), {256})
            self.assertEqual([plan["input_tokens"] for plan in result["plans"]], [256])

    def test_non_prefill_rows_do_not_create_clock_or_ffn_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            rows = [
                _scheduler_row(
                    query=1,
                    phase="prefill",
                    n_tokens=256,
                    profile="p15-g15-gpu14",
                    prime=4_473_600,
                    gold=3_532_800,
                    gpu=1_200_000_000,
                    backend="CPU",
                    branch="p15-g15-gpu14.cpu",
                ),
                _scheduler_row(
                    query=2,
                    phase="decode",
                    n_tokens=1,
                    profile="p4-g8-gpu14",
                    prime=1_958_400,
                    gold=1_996_800,
                    gpu=1_200_000_000,
                    backend="CPU",
                    branch="p4-g8-gpu14.cpu",
                ),
            ]
            scheduler = run / "scheduler_trace.csv"
            _write_csv(scheduler, SCHEDULER_FIELDS, rows)

            catalog = discover_clock_states(run)
            ffn = read_ffn_branch_samples(scheduler)

            self.assertEqual(set(catalog.by_token()), {256})
            self.assertEqual([state.profile for state in catalog.states], ["p15-g15-gpu14"])
            self.assertEqual(len(ffn), 1)
            self.assertEqual(ffn[0].clock_profile, "p15-g15-gpu14")


class ObservationTests(unittest.TestCase):
    def test_ffn_rows_join_actual_state_to_active_plan_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            rows = [
                _scheduler_row(
                    query=1,
                    profile="p4-g8-gpu14",
                    prime=1_958_400,
                    gold=1_996_800,
                    gpu=1_200_000_000,
                    backend=backend,
                    branch=f"p15-g15-gpu14.{device}",
                    latency=latency,
                )
                for device, backend, latency in (
                    ("cpu", "CPU", 8000),
                    ("gpu", "OpenCL", 9000),
                    ("npu", "HTP0", 8500),
                )
            ]
            _write_csv(run / "scheduler_trace.csv", SCHEDULER_FIELDS, rows)
            policy = {
                "profiles": {
                    "p15-g15-gpu14": {
                        "ffn_parallel": {
                            "split_sizes": {"cpu": 1600, "gpu": 3008, "npu": 3584}
                        }
                    }
                }
            }

            catalog = parse_ffn_observations(run, policy=policy)

            self.assertEqual(catalog.skipped_without_shard_size, 0)
            self.assertEqual(len(catalog.groups), 1)
            group = catalog.groups[0]
            self.assertEqual(group.input_tokens, 256)
            self.assertEqual(group.actual_state, "p4-g8-gpu14")
            self.assertEqual(group.active_plan, "p15-g15-gpu14")
            self.assertEqual(
                {row.device: row.shard_size for row in group.observations},
                {"cpu": 1600, "gpu": 3008, "npu": 3584},
            )
            json.dumps(catalog.to_dict())


class OptimizerTests(unittest.TestCase):
    def test_median_fit_ignores_a_single_repeat_outlier(self) -> None:
        model = fit_device_latency_model(
            "cpu",
            [
                SizeLatencySample(64, 64),
                SizeLatencySample(64, 64),
                SizeLatencySample(64, 640),
            ],
        )
        self.assertAlmostEqual(model.slope_us_per_element, 1.0)

    def test_optimizer_covers_width_and_respects_backend_alignment(self) -> None:
        samples = {
            "cpu": [(64, 64)],
            "gpu": [(64, 64)],
            "npu": [(256, 256)],
        }
        result = optimize_ffn_shards(1024, samples, max_alternatives=4)
        split = result.best.split_sizes

        self.assertEqual(sum(split.values()), 1024)
        self.assertEqual(split["cpu"] % 64, 0)
        self.assertEqual(split["gpu"] % 64, 0)
        self.assertEqual(split["npu"] % 256, 0)
        self.assertEqual(split, {"cpu": 384, "gpu": 384, "npu": 256})
        self.assertAlmostEqual(result.best.predicted_makespan_us, 384.0)
        json.dumps(result.to_dict())


def _scheduler_row(
    *,
    query: int,
    phase: str = "",
    profile: str,
    prime: int,
    gold: int,
    gpu: int,
    n_tokens: int = 256,
    backend: str = "CPU",
    branch: str = "",
    latency: int = 100,
    layer: int = 0,
) -> dict[str, object]:
    return {
        "query_id": query,
        "phase": phase,
        "graph_id": query,
        "n_tokens": n_tokens,
        "backend": backend,
        "layer": layer,
        "compute_wall_us": latency,
        "is_ffn_group": 1 if branch else 0,
        "ffn_branch": branch,
        "group_id": query,
        "parallel_group_kind": "ffn" if branch else "",
        "parallel_branch": branch,
        "actual_clock_profile": profile,
        "actual_prime_khz": prime,
        "actual_gold_khz": gold,
        "actual_gpu_hz": gpu,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
