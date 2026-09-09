from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from model_descriptor import ModelDescriptor
from policy_generator import (
    Plan,
    PolicyGenerationError,
    generate_policy,
    load_plans,
    validate_generated_policy,
    write_policy,
)


def template_policy() -> dict:
    return {
        "version": 1,
        "enabled": True,
        "devices": {"fallback_priority": ["OpenCL", "HTP0", "CPU"]},
        "ffn_clock_switch": {"enabled": False},
        "ffn_parallel": {
            "enabled": True,
            "phase": "all",
            "layer_range": [0, 27],
            "align": 64,
            "reduce_backend": "CPU",
            "split_layout": [
                {"id": "npu", "backend": "HTP0-REPACK"},
                {"id": "gpu", "backend": "OpenCL"},
                {"id": "cpu", "backend": "CPU_REPACK"},
            ],
            "split_sizes": {"npu": 4096, "gpu": 2560, "cpu": 1536},
        },
        "attn_qkv_shards": {
            "enabled": True,
            "phase": "prefill",
            "lane_activity": "per_projection",
            "layer_range": [0, 27],
            "head_dim": 128,
            "assemble_backend": "OpenCL",
            "split_layout": [
                {"id": "cpu", "backend": "CPU", "align": 128},
                {"id": "gpu", "backend": "OpenCL", "align": 128},
                {"id": "npu", "backend": "HTP0-REPACK", "align": 256},
            ],
            "split_sizes": {
                "q": {"cpu": 0, "gpu": 0, "npu": 3072},
                "k": {"cpu": 0, "gpu": 1024, "npu": 0},
                "v": {"cpu": 0, "gpu": 1024, "npu": 0},
            },
        },
        "attn_out_shards": {
            "enabled": True,
            "phase": "prefill",
            "partition_axis": "output",
            "layer_range": [0, 27],
            "head_dim": 128,
            "reduce_backend": "CPU",
            "split_layout": [
                {"id": "cpu", "backend": "CPU", "align": 128},
                {"id": "gpu", "backend": "OpenCL", "align": 128},
                {"id": "npu", "backend": "HTP0-REPACK", "align": 256},
            ],
            "split_sizes": {"cpu": 0, "gpu": 1024, "npu": 2048},
        },
        "profile_defaults": {
            "applicability": {
                "input_tokens": [256, 256],
                "ubatch_tokens": [256, 256],
            },
            "ffn_parallel": {
                "enabled": True,
                "phase": "all",
                "layer_range": [0, 27],
                "align": 64,
                "reduce_backend": "CPU",
                "split_layout": [
                    {"id": "npu", "backend": "HTP0-REPACK"},
                    {"id": "gpu", "backend": "OpenCL"},
                    {"id": "cpu", "backend": "CPU_REPACK"},
                ],
            },
            "ops": {
                "enabled": True,
                "rules": [
                    {
                        "name": "ffn_norm",
                        "op": "MUL",
                        "phase": "prefill",
                        "layer_range": [0, 27],
                        "backend": "HTP0-REPACK",
                    }
                ],
            },
        },
        "profiles": {
            "p15-g15-gpu14": {
                "clock_point": {
                    "prime": {"index": 15, "khz": 4473600},
                    "gold": {"index": 15, "khz": 3532800},
                    "gpu": {"index": 14, "hz": 1200000000},
                },
                "ffn_parallel": {
                    "split_sizes": {"npu": 4096, "gpu": 2560, "cpu": 1536}
                },
            },
            "p2-g5-gpu4": {
                "clock_point": {
                    "prime": {"index": 2, "khz": 1401600},
                    "gold": {"index": 5, "khz": 1363200},
                    "gpu": {"index": 4, "hz": 443000000},
                },
                "ffn_parallel": {
                    "split_sizes": {"npu": 5376, "gpu": 1792, "cpu": 1024}
                },
                "attn_qkv_shards": {
                    "split_sizes": {
                        "q": {"cpu": 0, "gpu": 0, "npu": 3072},
                        "k": {"cpu": 0, "gpu": 0, "npu": 1024},
                        "v": {"cpu": 0, "gpu": 1024, "npu": 0},
                    }
                },
                "attn_out_shards": {
                    "split_sizes": {"cpu": 0, "gpu": 2048, "npu": 1024}
                },
                "ops": {
                    "enabled": True,
                    "rules": [
                        {
                            "name": "low_clock_norm",
                            "op": "RMS_NORM",
                            "phase": "prefill",
                            "layer_range": [0, 27],
                            "backend": "CPU",
                        }
                    ],
                },
            },
        },
        "runtime_routes": {
            "enabled": True,
            "mode": "clock",
            "phase": "prefill",
            "initial_profile": "p15-g15-gpu14",
            "profiles": ["p15-g15-gpu14", "p2-g5-gpu4"],
            "transitions": "complete",
            "candidate_kinds": ["ffn_block"],
            "boundary": {"node": "l_out", "backend": "auto", "granularity": "layer"},
            "output_mode": "canonical",
        },
        "weights": {
            "enabled": True,
            "default": "CPU_REPACK",
            "rules": [{"pattern": "^blk\\.[0-9]+\\.ffn_", "backend": "CPU_REPACK"}],
        },
        "residency": {"enabled": True, "rules": []},
        "ops": {
            "enabled": True,
            "rules": [
                {
                    "name": "norm",
                    "op": "RMS_NORM",
                    "phase": "prefill",
                    "layer_range": [0, 27],
                    "backend": "OpenCL",
                },
                {
                    "name": "early_layers_only",
                    "op": "ADD",
                    "phase": "prefill",
                    "layer_range": [0, 13],
                    "backend": "CPU",
                },
            ],
        },
    }


def target_model() -> ModelDescriptor:
    return ModelDescriptor.from_dict(
        {
            "model_id": "example-4b",
            "architecture": "llama",
            "quantization": "Q8_0",
            "n_layer": 32,
            "n_embd": 4096,
            "n_ff": 11008,
            "n_head": 32,
            "n_head_kv": 8,
            "head_dim": 128,
        }
    )


def plan_dicts() -> list[dict]:
    return [
        {
            "profile": "p15-g15-gpu14",
            "input_tokens": 512,
            "clock_point": {
                "prime": {"index": 15, "khz": 4473600},
                "gold": {"index": 15, "khz": 3532800},
                "gpu": {"index": 14, "hz": 1200000000},
            },
            "ffn_split_sizes": {"npu": 5120, "gpu": 3328, "cpu": 2560},
        },
        {
            "profile": "p2-g5-gpu4",
            "input_tokens": 512,
            "clock_point": {
                "prime": {"index": 2, "khz": 1401600},
                "gold": {"index": 5, "khz": 1363200},
                "gpu": {"index": 4, "hz": 443000000},
            },
            "ffn_parallel": {
                "split_sizes": {"npu": 5376, "gpu": 3072, "cpu": 2560}
            },
            "attn_qkv_shards": {
                "split_sizes": {
                    "q": {"cpu": 0, "gpu": 1024, "npu": 3072},
                    "k": {"cpu": 0, "gpu": 1024, "npu": 0},
                    "v": {"cpu": 0, "gpu": 1024, "npu": 0},
                }
            },
            "attn_out_split_sizes": {"cpu": 0, "gpu": 1024, "npu": 3072},
        },
    ]


class PolicyGeneratorTests(unittest.TestCase):
    def test_generates_model_and_token_specific_policy_without_mutating_template(self) -> None:
        template = template_policy()
        original = copy.deepcopy(template)
        generated = generate_policy(template, target_model(), 512, plan_dicts())

        self.assertEqual(template, original)
        self.assertEqual(generated["ffn_parallel"]["layer_range"], [0, 31])
        self.assertEqual(generated["ops"]["rules"][0]["layer_range"], [0, 31])
        self.assertEqual(generated["ops"]["rules"][1]["layer_range"], [0, 13])
        self.assertEqual(
            generated["profile_defaults"]["applicability"]["input_tokens"],
            [512, 512],
        )
        self.assertEqual(
            generated["ffn_parallel"]["split_sizes"],
            {"npu": 5120, "gpu": 3328, "cpu": 2560},
        )
        self.assertEqual(generated["attn_qkv_shards"]["head_dim"], 128)
        self.assertEqual(
            sum(generated["attn_qkv_shards"]["split_sizes"]["q"].values()),
            4096,
        )
        self.assertEqual(
            sum(generated["attn_out_shards"]["split_sizes"].values()), 4096
        )
        self.assertEqual(
            generated["runtime_routes"]["profiles"],
            ["p15-g15-gpu14", "p2-g5-gpu4"],
        )
        self.assertIn("attn_qkv_block", generated["runtime_routes"]["candidate_kinds"])
        self.assertIn("attn_out_block", generated["runtime_routes"]["candidate_kinds"])
        self.assertEqual(generated["weights"], original["weights"])
        self.assertEqual(generated["residency"], original["residency"])
        validate_generated_policy(generated, target_model(), 512)

    def test_proportional_scaling_keeps_htp_start_and_size_aligned(self) -> None:
        generated = generate_policy(template_policy(), target_model(), 512, plan_dicts())
        for section_name in ("attn_qkv_shards", "attn_out_shards"):
            section = generated[section_name]
            projections = (
                section["split_sizes"].values()
                if section_name == "attn_qkv_shards"
                else [section["split_sizes"]]
            )
            ids = [entry["id"] for entry in section["split_layout"]]
            for sizes in projections:
                start = 0
                for split_id in ids:
                    size = sizes[split_id]
                    if split_id == "npu" and size:
                        self.assertEqual(start % 256, 0)
                        self.assertEqual(size % 256, 0)
                    start += size

    def test_new_clock_profile_clones_nearest_template_overrides(self) -> None:
        new_plan = {
            "profile": "p3-g6-gpu4",
            "input_tokens": 512,
            "clock_point": {
                "prime": {"index": 3, "khz": 1500000},
                "gold": {"index": 6, "khz": 1500000},
                "gpu": {"index": 4, "hz": 443000000},
            },
            "ffn_split_sizes": {"npu": 5376, "gpu": 3072, "cpu": 2560},
        }
        generated = generate_policy(template_policy(), target_model(), 512, [new_plan])
        entry = generated["profiles"]["p3-g6-gpu4"]

        # The nearby low-clock template sends K to NPU and uses a GPU-heavy
        # attention-output split.  Root inheritance would produce the opposite.
        self.assertEqual(
            entry["attn_qkv_shards"]["split_sizes"]["k"],
            {"cpu": 0, "gpu": 0, "npu": 1024},
        )
        out = entry["attn_out_shards"]["split_sizes"]
        self.assertGreater(out["gpu"], out["npu"])
        self.assertEqual(sum(out.values()), 4096)
        self.assertEqual(out["npu"] % 256, 0)
        self.assertEqual((out["cpu"] + out["gpu"]) % 256, 0)
        self.assertEqual(entry["ops"]["rules"][0]["name"], "low_clock_norm")
        self.assertEqual(entry["ops"]["rules"][0]["layer_range"], [0, 31])

    def test_exact_profile_name_wins_even_if_new_clock_is_closer_to_another(self) -> None:
        exact_plan = {
            "profile": "p2-g5-gpu4",
            "input_tokens": 512,
            # Deliberately use the high-clock point: exact-name inheritance must
            # still preserve the named low-clock profile's non-FFN overrides.
            "clock_point": {
                "prime": {"index": 15, "khz": 4473600},
                "gold": {"index": 15, "khz": 3532800},
                "gpu": {"index": 14, "hz": 1200000000},
            },
            "ffn_split_sizes": {"npu": 5376, "gpu": 3072, "cpu": 2560},
        }
        generated = generate_policy(template_policy(), target_model(), 512, [exact_plan])
        entry = generated["profiles"]["p2-g5-gpu4"]
        self.assertEqual(
            entry["attn_qkv_shards"]["split_sizes"]["k"],
            {"cpu": 0, "gpu": 0, "npu": 1024},
        )
        self.assertEqual(entry["ops"]["rules"][0]["name"], "low_clock_norm")

    def test_rejects_ffn_plan_with_unaligned_htp_size(self) -> None:
        plans = plan_dicts()[:1]
        plans[0]["ffn_split_sizes"] = {"npu": 5056, "gpu": 3392, "cpu": 2560}
        with self.assertRaisesRegex(PolicyGenerationError, "multiples of 256"):
            generate_policy(template_policy(), target_model(), 512, plans)

    def test_loads_plan_json_and_writes_generated_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans_path = root / "plans.json"
            plans_path.write_text(json.dumps({"plans": plan_dicts()}), encoding="utf-8")
            plans = load_plans(plans_path)
            self.assertTrue(all(isinstance(plan, Plan) for plan in plans))

            policy = generate_policy(template_policy(), target_model(), 512, plans)
            output = root / "nested" / "policy.json"
            write_policy(policy, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), policy)


if __name__ == "__main__":
    unittest.main()
