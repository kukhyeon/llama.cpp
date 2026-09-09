from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.treprefill_autotune.autotune_policy import (
    AutotuneError,
    main,
    recommend_ffn_plans,
)
from scripts.treprefill_autotune.model_descriptor import ModelDescriptor


SCHEDULER_FIELDS = [
    "query_id",
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


def _model() -> ModelDescriptor:
    return ModelDescriptor(
        model_id="Tiny Llama Q8",
        architecture="llama",
        quantization="Q8_0",
        n_layer=4,
        n_embd=128,
        n_ff=1024,
        n_head=1,
        n_head_kv=1,
        head_dim=128,
    )


def _template(split: dict[str, int] | None = None) -> dict[str, object]:
    split = split or {"npu": 256, "gpu": 384, "cpu": 384}
    layout = [
        {"id": "npu", "backend": "HTP0-REPACK"},
        {"id": "gpu", "backend": "OpenCL"},
        {"id": "cpu", "backend": "CPU_REPACK"},
    ]
    return {
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
            "applicability": {"input_tokens": [256, 256], "ubatch_tokens": [256, 256]},
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


def _catalog() -> dict[str, object]:
    return {
        "states": [
            {
                "profile": "p15-g15-gpu14",
                "aliases": ["p15-g15-gpu14"],
                "point": {
                    "prime_khz": 4_473_600,
                    "gold_khz": 3_532_800,
                    "gpu_hz": 1_200_000_000,
                },
                "scheduler_samples": 6,
                "distinct_queries": 2,
                "input_tokens": [256],
            }
        ]
    }


def _write_scheduler(
    path: Path,
    split: dict[str, int],
    devices: tuple[str, ...],
    *,
    profile: str = "p15-g15-gpu14",
    prime_khz: int = 4_473_600,
    gold_khz: int = 3_532_800,
    gpu_hz: int = 1_200_000_000,
    layer: int = 1,
) -> None:
    backend = {"cpu": "CPU", "gpu": "OpenCL", "npu": "HTP0"}
    rows: list[dict[str, object]] = []
    for query in (1, 2):
        for device in devices:
            rows.append(
                {
                    "query_id": query,
                    "graph_id": query,
                    "n_tokens": 256,
                    "backend": backend[device],
                    "layer": layer,
                    "compute_wall_us": split[device],
                    "is_ffn_group": 1,
                    "ffn_branch": f"{profile}.{device}",
                    "group_id": query,
                    "parallel_group_kind": "ffn",
                    "parallel_branch": f"{profile}.{device}",
                    "actual_clock_profile": profile,
                    "actual_prime_khz": prime_khz,
                    "actual_gold_khz": gold_khz,
                    "actual_gpu_hz": gpu_hz,
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class AutotunePolicyCliTest(unittest.TestCase):
    def test_inspect_and_discover_subcommands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            descriptor_input = directory / "descriptor-input.json"
            descriptor_output = directory / "descriptor-output.json"
            catalog_output = directory / "catalog.json"
            scheduler = directory / "scheduler_trace.csv"
            descriptor_input.write_text(json.dumps(_model().to_dict()), encoding="utf-8")
            _write_scheduler(
                scheduler,
                {"npu": 256, "gpu": 384, "cpu": 384},
                ("cpu", "gpu", "npu"),
            )
            with redirect_stdout(io.StringIO()):
                inspect_rc = main(
                    [
                        "inspect-model",
                        str(descriptor_input),
                        "--output",
                        str(descriptor_output),
                    ]
                )
                discover_rc = main(
                    [
                        "discover",
                        str(scheduler),
                        "--min-samples",
                        "3",
                        "--min-queries",
                        "2",
                        "--gpu-index",
                        "14",
                        "--output",
                        str(catalog_output),
                    ]
                )
            descriptor = json.loads(descriptor_output.read_text(encoding="utf-8"))
            catalog = json.loads(catalog_output.read_text(encoding="utf-8"))

        self.assertEqual(inspect_rc, 0)
        self.assertEqual(discover_rc, 0)
        self.assertEqual(descriptor["n_ff"], 1024)
        self.assertEqual([state["profile"] for state in catalog["states"]], ["p15-g15-gpu14"])

    def test_two_backend_gpu_npu_layout_is_optimized_without_cpu_lane(self) -> None:
        template = _template()
        two_lane_layout = [
            {"id": "npu", "backend": "HTP0-REPACK"},
            {"id": "gpu", "backend": "OpenCL"},
        ]
        two_lane_split = {"npu": 512, "gpu": 512}
        template["ffn_parallel"]["split_layout"] = two_lane_layout
        template["ffn_parallel"]["split_sizes"] = two_lane_split
        template["profile_defaults"]["ffn_parallel"]["split_layout"] = two_lane_layout
        template["profiles"]["p15-g15-gpu14"]["ffn_parallel"][
            "split_sizes"
        ] = two_lane_split

        with tempfile.TemporaryDirectory() as temporary:
            scheduler = Path(temporary) / "scheduler_trace.csv"
            _write_scheduler(scheduler, two_lane_split, ("gpu", "npu"))
            result = recommend_ffn_plans(
                _model(),
                template,
                [scheduler],
                measured_policy=template,
                clock_catalog=_catalog(),
            )

        sizes = result["plans"][0]["ffn_split_sizes"]
        self.assertEqual(set(sizes), {"gpu", "npu"})
        self.assertEqual(sum(sizes.values()), 1024)
        self.assertEqual(sizes["npu"] % 256, 0)

    def test_recommend_accepts_one_measured_size_per_device(self) -> None:
        template = _template()
        with tempfile.TemporaryDirectory() as temporary:
            scheduler = Path(temporary) / "scheduler_trace.csv"
            _write_scheduler(
                scheduler,
                {"npu": 256, "gpu": 384, "cpu": 384},
                ("cpu", "gpu", "npu"),
            )
            result = recommend_ffn_plans(
                _model(),
                template,
                [scheduler],
                measured_policy=template,
                clock_catalog=_catalog(),
            )

        self.assertEqual(result["fit"]["mode"], "through-origin")
        self.assertEqual(len(result["plans"]), 1)
        plan = result["plans"][0]
        self.assertEqual(
            plan["ffn_split_sizes"], {"npu": 256, "gpu": 384, "cpu": 384}
        )
        self.assertEqual(plan["clock_point"]["prime"]["index"], 15)
        self.assertEqual(
            plan["measurement"]["distinct_sizes_per_device"],
            {"cpu": 1, "gpu": 1, "npu": 1},
        )

    def test_missing_device_requires_explicit_drop_permission(self) -> None:
        split = {"npu": 512, "gpu": 512, "cpu": 0}
        template = _template(split)
        with tempfile.TemporaryDirectory() as temporary:
            scheduler = Path(temporary) / "scheduler_trace.csv"
            _write_scheduler(scheduler, split, ("gpu", "npu"))
            with self.assertRaisesRegex(AutotuneError, "allow-drop"):
                recommend_ffn_plans(
                    _model(),
                    template,
                    [scheduler],
                    measured_policy=template,
                    clock_catalog=_catalog(),
                )
            result = recommend_ffn_plans(
                _model(),
                template,
                [scheduler],
                measured_policy=template,
                clock_catalog=_catalog(),
                allowed_drops={"cpu"},
            )

        self.assertEqual(result["plans"][0]["ffn_split_sizes"]["cpu"], 0)
        self.assertEqual(sum(result["plans"][0]["ffn_split_sizes"].values()), 1024)

    def test_catalog_filter_and_interior_layer_filter_are_enforced(self) -> None:
        template = _template()
        with tempfile.TemporaryDirectory() as temporary:
            scheduler = Path(temporary) / "scheduler_trace.csv"
            _write_scheduler(
                scheduler,
                {"npu": 256, "gpu": 384, "cpu": 384},
                ("cpu", "gpu", "npu"),
                layer=0,
            )
            with self.assertRaisesRegex(AutotuneError, "include-edge-layers"):
                recommend_ffn_plans(
                    _model(),
                    template,
                    [scheduler],
                    measured_policy=template,
                    clock_catalog=_catalog(),
                )
            edge_result = recommend_ffn_plans(
                _model(),
                template,
                [scheduler],
                measured_policy=template,
                clock_catalog=_catalog(),
                include_edge_layers=True,
            )
            unrelated_catalog = _catalog()
            unrelated_catalog["states"][0]["profile"] = "p14-g14-gpu14"
            unrelated_catalog["states"][0]["aliases"] = ["p14-g14-gpu14"]
            with self.assertRaisesRegex(AutotuneError, "catalog-represented"):
                recommend_ffn_plans(
                    _model(),
                    template,
                    [scheduler],
                    measured_policy=template,
                    clock_catalog=unrelated_catalog,
                    include_edge_layers=True,
                )

        self.assertEqual(len(edge_result["plans"]), 1)
        self.assertEqual(edge_result["plans"][0]["measurement"]["layers"], [0])
        self.assertEqual(edge_result["fit"]["layers"], "all")

    def test_optional_regret_pruning_preserves_requested_anchor(self) -> None:
        template = _template()
        template["profiles"]["p14-g14-gpu14"] = {
            "clock_point": {
                "prime": {"index": 14, "khz": 4_224_000},
                "gold": {"index": 14, "khz": 3_321_600},
                "gpu": {"index": 14, "hz": 1_200_000_000},
            },
            "ffn_parallel": {
                "split_sizes": {"npu": 256, "gpu": 384, "cpu": 384}
            },
        }
        catalog = _catalog()
        catalog["states"].append(
            {
                "profile": "p14-g14-gpu14",
                "aliases": ["p14-g14-gpu14"],
                "point": {
                    "prime_khz": 4_224_000,
                    "gold_khz": 3_321_600,
                    "gpu_hz": 1_200_000_000,
                },
                "scheduler_samples": 6,
                "distinct_queries": 2,
                "input_tokens": [256],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            traces = []
            for profile, prime, gold in (
                ("p15-g15-gpu14", 4_473_600, 3_532_800),
                ("p14-g14-gpu14", 4_224_000, 3_321_600),
            ):
                scheduler = directory / profile / "scheduler_trace.csv"
                scheduler.parent.mkdir()
                _write_scheduler(
                    scheduler,
                    {"npu": 256, "gpu": 384, "cpu": 384},
                    ("cpu", "gpu", "npu"),
                    profile=profile,
                    prime_khz=prime,
                    gold_khz=gold,
                )
                traces.append(scheduler)
            result = recommend_ffn_plans(
                _model(),
                template,
                traces,
                measured_policy=template,
                clock_catalog=catalog,
                max_profile_regret=0.0,
                preserve_profiles={"p15-g15-gpu14"},
            )

        self.assertEqual(
            [plan["profile"] for plan in result["plans"]], ["p15-g15-gpu14"]
        )
        self.assertEqual(
            result["pruning"]["256"]["removed_profiles"], ["p14-g14-gpu14"]
        )

    def test_generate_then_validate_subcommands(self) -> None:
        model = _model()
        template = _template()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            descriptor_path = directory / "model.json"
            template_path = directory / "template.json"
            scheduler = directory / "scheduler_trace.csv"
            plans_path = directory / "plans.json"
            policy_path = directory / "policy.json"
            descriptor_path.write_text(json.dumps(model.to_dict()), encoding="utf-8")
            template_path.write_text(json.dumps(template), encoding="utf-8")
            _write_scheduler(
                scheduler,
                {"npu": 256, "gpu": 384, "cpu": 384},
                ("cpu", "gpu", "npu"),
            )
            plans = recommend_ffn_plans(
                model,
                template,
                [scheduler],
                measured_policy=template,
                clock_catalog=_catalog(),
            )
            plans_path.write_text(json.dumps(plans), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                generate_rc = main(
                    [
                        "generate",
                        "--model",
                        str(descriptor_path),
                        "--template",
                        str(template_path),
                        "--plans",
                        str(plans_path),
                        "--token",
                        "256",
                        "--output",
                        str(policy_path),
                    ]
                )
                validate_rc = main(
                    [
                        "validate",
                        "--model",
                        str(descriptor_path),
                        "--policy",
                        str(policy_path),
                        "--token",
                        "256",
                    ]
                )

        self.assertEqual(generate_rc, 0)
        self.assertEqual(validate_rc, 0)


if __name__ == "__main__":
    unittest.main()
