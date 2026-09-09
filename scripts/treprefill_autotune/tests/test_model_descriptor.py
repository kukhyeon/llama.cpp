from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import model_descriptor as descriptor_module
from model_descriptor import ModelDescriptor, ModelDescriptorError, load_model_descriptor


class ModelDescriptorTests(unittest.TestCase):
    def test_json_round_trip_and_derived_widths(self) -> None:
        source = {
            "schema_version": 1,
            "model_id": "example-4b-q8",
            "architecture": "llama",
            "quantization": "MOSTLY_Q8_0",
            "n_layer": 32,
            "n_embd": 4096,
            "n_ff": 11008,
            "n_head": 32,
            "n_head_kv": 8,
        }
        descriptor = ModelDescriptor.from_dict(source)
        self.assertEqual(descriptor.head_dim, 128)
        self.assertEqual(descriptor.quantization, "Q8_0")
        self.assertEqual(descriptor.q_width, 4096)
        self.assertEqual(descriptor.k_width, 1024)
        self.assertEqual(descriptor.v_width, 1024)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            descriptor.write_json(path)
            self.assertEqual(load_model_descriptor(path), descriptor)
            serialized = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(serialized["attn_out_width"], 4096)

    def test_rejects_inconsistent_or_unaligned_dimensions(self) -> None:
        base = {
            "model_id": "bad",
            "architecture": "llama",
            "quantization": "Q8_0",
            "n_layer": 28,
            "n_embd": 3072,
            "n_ff": 8192,
            "n_head": 24,
            "n_head_kv": 8,
            "head_dim": 64,
        }
        with self.assertRaisesRegex(ModelDescriptorError, "does not match"):
            ModelDescriptor.from_dict(base)

        base["head_dim"] = 128
        base["n_ff"] = 8193
        with self.assertRaisesRegex(ModelDescriptorError, "not divisible"):
            ModelDescriptor.from_dict(base)

    def test_optional_gguf_reader_extracts_standard_llama_metadata(self) -> None:
        class FieldType:
            def __init__(self, name: str):
                self.name = name

        class Field:
            def __init__(self, value: object, kind: str = "UINT32"):
                if kind == "STRING":
                    self.parts = [list(str(value).encode("utf-8"))]
                elif isinstance(value, list):
                    self.parts = [[item] for item in value]
                else:
                    self.parts = [[value]]
                self.data = list(range(len(self.parts)))
                self.types = [FieldType(kind)]

        fields = {
            "general.architecture": Field("llama", "STRING"),
            "general.name": Field("Synthetic Llama", "STRING"),
            "general.file_type": Field(7),
            "llama.block_count": Field(32),
            "llama.embedding_length": Field(4096),
            "llama.feed_forward_length": Field([11008, 11008]),
            "llama.attention.head_count": Field(32),
            "llama.attention.head_count_kv": Field(8),
            "llama.attention.key_length": Field(128),
            "llama.attention.value_length": Field(128),
        }

        class Reader:
            def get_field(self, key: str):
                return fields.get(key)

        class FileType(IntEnum):
            MOSTLY_Q8_0 = 7

        fake_gguf = SimpleNamespace(
            GGUFReader=lambda _path, mode="r": Reader(),
            LlamaFileType=FileType,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.gguf"
            path.write_bytes(b"not read by the fake reader")
            with mock.patch.object(
                descriptor_module, "_repo_gguf_module", return_value=fake_gguf
            ):
                descriptor = ModelDescriptor.from_gguf(path)

        self.assertEqual(descriptor.model_id, "Synthetic Llama")
        self.assertEqual(descriptor.n_layer, 32)
        self.assertEqual(descriptor.n_ff, 11008)
        self.assertEqual(descriptor.quantization, "Q8_0")

    def test_stdlib_gguf_metadata_fallback_needs_no_numpy_or_tensor_data(self) -> None:
        def packed_string(value: str) -> bytes:
            encoded = value.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded

        def packed_value(type_id: int, value: object) -> bytes:
            if type_id == 4:  # UINT32
                return struct.pack("<I", value)
            if type_id == 6:  # FLOAT32
                return struct.pack("<f", value)
            if type_id == 8:  # STRING
                return packed_string(value)
            if type_id == 9:  # ARRAY (subtype, length, elements)
                subtype, values = value
                return (
                    struct.pack("<IQ", subtype, len(values))
                    + b"".join(packed_value(subtype, item) for item in values)
                )
            raise AssertionError(f"unsupported synthetic type {type_id}")

        fields = [
            ("general.architecture", 8, "llama"),
            # Exercise skipping variable-size metadata that the descriptor does not use.
            ("tokenizer.synthetic", 9, (8, ["one", "two"])),
            ("general.name", 8, "Dependency-free Llama"),
            ("general.file_type", 4, 7),
            ("llama.block_count", 4, 32),
            ("unrelated.float", 6, 1.25),
            ("llama.embedding_length", 4, 4096),
            ("llama.feed_forward_length", 9, (4, [11008, 11008])),
            ("llama.attention.head_count", 4, 32),
            ("llama.attention.head_count_kv", 4, 8),
            ("llama.attention.key_length", 4, 128),
            ("llama.attention.value_length", 4, 128),
        ]
        metadata = b"".join(
            packed_string(key) + struct.pack("<I", type_id) + packed_value(type_id, value)
            for key, type_id, value in fields
        )
        # tensor_count=1 with no following tensor info proves the fallback stops
        # immediately after the metadata KV section.
        contents = b"GGUF" + struct.pack("<IQQ", 3, 1, len(fields)) + metadata

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata-only.gguf"
            path.write_bytes(contents)
            with mock.patch.object(
                descriptor_module,
                "_repo_gguf_module",
                side_effect=ModelDescriptorError("numpy unavailable"),
            ):
                descriptor = ModelDescriptor.from_gguf(path, compute_sha256=True)

        self.assertEqual(descriptor.model_id, "Dependency-free Llama")
        self.assertEqual(descriptor.architecture, "llama")
        self.assertEqual(descriptor.quantization, "Q8_0")
        self.assertEqual(descriptor.n_layer, 32)
        self.assertEqual(descriptor.n_ff, 11008)
        self.assertEqual(len(descriptor.sha256 or ""), 64)


if __name__ == "__main__":
    unittest.main()
