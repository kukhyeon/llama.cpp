#!/usr/bin/env python3

"""Model dimensions needed by the TrePrefill policy autotuner.

The descriptor deliberately contains only scheduling-relevant, stable model
metadata.  It can be checked into an experiment directory as JSON, so policy
generation does not require copying a multi-gigabyte GGUF file to the host.
When the GGUF is available locally, :func:`ModelDescriptor.from_gguf` uses the
repository's ``gguf-py`` reader on demand.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping


DESCRIPTOR_SCHEMA_VERSION = 1

# GGUF value type ids are part of the on-disk format.  Keeping this tiny table
# local lets ``inspect-model`` read the metadata header even when gguf-py's
# optional numpy dependency is not installed.
_GGUF_SCALAR_FORMATS = {
    0: "B",  # UINT8
    1: "b",  # INT8
    2: "H",  # UINT16
    3: "h",  # INT16
    4: "I",  # UINT32
    5: "i",  # INT32
    6: "f",  # FLOAT32
    7: "?",  # BOOL
    10: "Q",  # UINT64
    11: "q",  # INT64
    12: "d",  # FLOAT64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_SUPPORTED_VERSIONS = {2, 3}
_GGUF_MAX_KV_COUNT = 1_000_000
_GGUF_MAX_KEY_BYTES = 1 << 20
_GGUF_MAX_ARRAY_ITEMS = 10_000_000
_GGUF_MAX_RETAINED_ARRAY_ITEMS = 1_000_000
_GGUF_DESCRIPTOR_KEYS = {
    "general.architecture",
    "general.name",
    "general.file_type",
}
_GGUF_DESCRIPTOR_SUFFIXES = (
    ".block_count",
    ".embedding_length",
    ".feed_forward_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
)

# Values mirror llama_ftype/LlamaFileType.  Unknown future values remain
# explicit (FILE_TYPE_n) instead of being guessed.
_LLAMA_FILE_TYPE_NAMES = {
    0: "ALL_F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
}


class ModelDescriptorError(ValueError):
    """Raised when model metadata is missing or incompatible with the tuner."""


def _positive_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelDescriptorError(f"{key} must be a positive integer")
    return value


def _normalize_quantization(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("MOSTLY_"):
        text = text[len("MOSTLY_") :]
    return text or "UNKNOWN"


def _uniform_positive_int(value: object, key: str) -> int:
    """Accept a scalar or a per-layer array whose values are all identical."""

    if isinstance(value, (list, tuple)):
        if not value:
            raise ModelDescriptorError(f"GGUF field {key!r} is empty")
        converted = [_scalar_int(item, key) for item in value]
        if len(set(converted)) != 1:
            raise ModelDescriptorError(
                f"GGUF field {key!r} varies by layer; variable-width FFN models "
                "are not supported by this policy generator"
            )
        result = converted[0]
    else:
        result = _scalar_int(value, key)
    if result <= 0:
        raise ModelDescriptorError(f"GGUF field {key!r} must be positive")
    return result


def _scalar_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ModelDescriptorError(f"GGUF field {key!r} must be an integer")
    try:
        result = int(value)  # numpy integer scalars are intentionally accepted
    except (TypeError, ValueError) as exc:
        raise ModelDescriptorError(f"GGUF field {key!r} must be an integer") from exc
    return result


def _field_value(reader: object, key: str, *, required: bool = True) -> object | None:
    """Convert one gguf-py ReaderField without importing numpy here."""

    field = reader.get_field(key)  # type: ignore[attr-defined]
    if field is None:
        if required:
            raise ModelDescriptorError(f"GGUF metadata is missing {key!r}")
        return None

    type_names = [getattr(item, "name", str(item)).upper() for item in field.types]
    values: list[object] = []
    for part_index in field.data:
        raw = field.parts[part_index]
        if "STRING" in type_names:
            if isinstance(raw, str):
                values.append(raw)
            elif isinstance(raw, (bytes, bytearray, memoryview)):
                values.append(bytes(raw).decode("utf-8"))
            else:
                try:
                    values.append(bytes(int(item) for item in raw).decode("utf-8"))
                except (TypeError, ValueError, UnicodeDecodeError) as exc:
                    raise ModelDescriptorError(
                        f"cannot decode GGUF string field {key!r}"
                    ) from exc
            continue

        converted = raw.tolist() if hasattr(raw, "tolist") else raw
        if isinstance(converted, list):
            values.extend(converted)
        elif isinstance(converted, tuple):
            values.extend(converted)
        else:
            values.append(converted)

    if not values:
        raise ModelDescriptorError(f"GGUF metadata field {key!r} has no value")
    return values[0] if len(values) == 1 else values


def _repo_gguf_module() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    gguf_python = repo_root / "gguf-py"
    if not gguf_python.is_dir():
        raise ModelDescriptorError(
            f"repository gguf-py package was not found at {gguf_python}"
        )
    gguf_path = str(gguf_python)
    if gguf_path not in sys.path:
        sys.path.insert(0, gguf_path)
    try:
        return importlib.import_module("gguf")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModelDescriptorError(
            "reading a local GGUF requires the repository gguf-py dependencies "
            "(notably numpy)"
        ) from exc


class _GGUFMetadataParser:
    """Small, bounded GGUF header/KV reader with no third-party dependency."""

    def __init__(self, handle: BinaryIO, source: Path):
        self.handle = handle
        self.source = source
        self.endian = "<"
        self.file_size = source.stat().st_size

    def _read_exact(self, size: int, source: str) -> bytes:
        if size < 0 or self.handle.tell() + size > self.file_size:
            raise ModelDescriptorError(f"truncated GGUF while reading {source}")
        data = self.handle.read(size)
        if len(data) != size:
            raise ModelDescriptorError(f"truncated GGUF while reading {source}")
        return data

    def _unpack(self, fmt: str, source: str) -> object:
        size = struct.calcsize(fmt)
        return struct.unpack(self.endian + fmt, self._read_exact(size, source))[0]

    def _skip(self, size: int, source: str) -> None:
        if size < 0 or self.handle.tell() + size > self.file_size:
            raise ModelDescriptorError(f"truncated GGUF while skipping {source}")
        self.handle.seek(size, 1)

    def _string(self, source: str, *, retain: bool, max_bytes: int | None = None) -> str | None:
        size = int(self._unpack("Q", f"{source} length"))
        if max_bytes is not None and size > max_bytes:
            raise ModelDescriptorError(
                f"GGUF {source} length {size} exceeds safety limit {max_bytes}"
            )
        if not retain:
            self._skip(size, source)
            return None
        raw = self._read_exact(size, source)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelDescriptorError(f"GGUF {source} is not valid UTF-8") from exc

    def _value(self, type_id: int, source: str, *, retain: bool) -> object | None:
        scalar_fmt = _GGUF_SCALAR_FORMATS.get(type_id)
        if scalar_fmt is not None:
            if retain:
                return self._unpack(scalar_fmt, source)
            self._skip(struct.calcsize(scalar_fmt), source)
            return None
        if type_id == _GGUF_STRING:
            return self._string(source, retain=retain)
        if type_id != _GGUF_ARRAY:
            raise ModelDescriptorError(
                f"GGUF {source} has unsupported value type id {type_id}"
            )

        item_type = int(self._unpack("I", f"{source} element type"))
        count = int(self._unpack("Q", f"{source} element count"))
        if (
            item_type not in _GGUF_SCALAR_FORMATS
            and item_type not in {_GGUF_STRING, _GGUF_ARRAY}
        ):
            raise ModelDescriptorError(
                f"GGUF {source} has unsupported element type id {item_type}"
            )
        if count > _GGUF_MAX_ARRAY_ITEMS:
            raise ModelDescriptorError(
                f"GGUF {source} contains {count} items; safety limit is "
                f"{_GGUF_MAX_ARRAY_ITEMS}"
            )
        if retain and count > _GGUF_MAX_RETAINED_ARRAY_ITEMS:
            raise ModelDescriptorError(
                f"GGUF {source} contains {count} retained items; safety limit is "
                f"{_GGUF_MAX_RETAINED_ARRAY_ITEMS}"
            )

        scalar_item_fmt = _GGUF_SCALAR_FORMATS.get(item_type)
        if not retain and scalar_item_fmt is not None:
            self._skip(struct.calcsize(scalar_item_fmt) * count, source)
            return None

        values: list[object] | None = [] if retain else None
        for index in range(count):
            item = self._value(item_type, f"{source}[{index}]", retain=retain)
            if values is not None:
                values.append(item)
        return values

    def read(self) -> dict[str, object]:
        if self._read_exact(4, "magic") != b"GGUF":
            raise ModelDescriptorError("GGUF magic is invalid")

        raw_version = self._read_exact(4, "version")
        little_version = int.from_bytes(raw_version, "little")
        big_version = int.from_bytes(raw_version, "big")
        if little_version in _GGUF_SUPPORTED_VERSIONS:
            self.endian = "<"
        elif big_version in _GGUF_SUPPORTED_VERSIONS:
            self.endian = ">"
        else:
            raise ModelDescriptorError(
                f"unsupported GGUF version (little={little_version}, big={big_version})"
            )

        # Version 2 and 3 use uint64 tensor/KV counts.  Tensor metadata follows
        # the KV section, so this reader intentionally returns before touching it.
        _ = self._unpack("Q", "tensor count")
        kv_count = int(self._unpack("Q", "metadata count"))
        if kv_count > _GGUF_MAX_KV_COUNT:
            raise ModelDescriptorError(
                f"GGUF metadata count {kv_count} exceeds safety limit {_GGUF_MAX_KV_COUNT}"
            )

        result: dict[str, object] = {}
        for index in range(kv_count):
            key = self._string(
                f"metadata key {index}", retain=True, max_bytes=_GGUF_MAX_KEY_BYTES
            )
            assert key is not None
            value_type = int(self._unpack("I", f"metadata value type for {key!r}"))
            retain = key in _GGUF_DESCRIPTOR_KEYS or key.endswith(
                _GGUF_DESCRIPTOR_SUFFIXES
            )
            value = self._value(value_type, f"metadata {key!r}", retain=retain)
            if retain:
                if key in result:
                    raise ModelDescriptorError(f"duplicate GGUF metadata key {key!r}")
                result[key] = value
        return result


def _read_gguf_metadata_stdlib(source: Path) -> dict[str, object]:
    try:
        with source.open("rb") as handle:
            return _GGUFMetadataParser(handle, source).read()
    except OSError as exc:
        raise ModelDescriptorError(f"failed to read GGUF {source}: {exc}") from exc


def _sha256_file(source: Path) -> str:
    hasher = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ModelDescriptor:
    """Uniform dense-Llama dimensions used to construct partition policies."""

    model_id: str
    architecture: str
    quantization: str
    n_layer: int
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: int
    head_dim: int
    sha256: str | None = None

    @property
    def q_width(self) -> int:
        return self.n_head * self.head_dim

    @property
    def k_width(self) -> int:
        return self.n_head_kv * self.head_dim

    @property
    def v_width(self) -> int:
        return self.n_head_kv * self.head_dim

    @property
    def attn_out_width(self) -> int:
        return self.n_embd

    def validate(self, *, ffn_align: int = 64) -> None:
        if not self.model_id.strip():
            raise ModelDescriptorError("model_id must not be empty")
        if not self.architecture.strip():
            raise ModelDescriptorError("architecture must not be empty")
        for key in (
            "n_layer",
            "n_embd",
            "n_ff",
            "n_head",
            "n_head_kv",
            "head_dim",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelDescriptorError(f"{key} must be a positive integer")
        if self.n_head_kv > self.n_head or self.n_head % self.n_head_kv != 0:
            raise ModelDescriptorError(
                "n_head_kv must divide n_head for the supported GQA/MHA layout"
            )
        if self.n_embd % self.n_head != 0:
            raise ModelDescriptorError("n_embd must be divisible by n_head")
        expected_head_dim = self.n_embd // self.n_head
        if self.head_dim != expected_head_dim:
            raise ModelDescriptorError(
                f"head_dim={self.head_dim} does not match n_embd/n_head="
                f"{expected_head_dim}"
            )
        if isinstance(ffn_align, bool) or not isinstance(ffn_align, int) or ffn_align <= 0:
            raise ModelDescriptorError("ffn_align must be a positive integer")
        if self.n_ff % ffn_align != 0:
            raise ModelDescriptorError(
                f"n_ff={self.n_ff} is not divisible by FFN alignment {ffn_align}"
            )
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ModelDescriptorError("sha256 must contain exactly 64 hexadecimal characters")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": DESCRIPTOR_SCHEMA_VERSION,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "n_ff": self.n_ff,
            "n_head": self.n_head,
            "n_head_kv": self.n_head_kv,
            "head_dim": self.head_dim,
            "q_width": self.q_width,
            "k_width": self.k_width,
            "v_width": self.v_width,
            "attn_out_width": self.attn_out_width,
        }
        if self.sha256 is not None:
            result["sha256"] = self.sha256.lower()
        return result

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelDescriptor":
        if not isinstance(data, Mapping):
            raise ModelDescriptorError("model descriptor must be a JSON object")
        version = data.get("schema_version", DESCRIPTOR_SCHEMA_VERSION)
        if version != DESCRIPTOR_SCHEMA_VERSION:
            raise ModelDescriptorError(
                f"unsupported model descriptor schema_version {version!r}"
            )

        model_id = data.get("model_id", data.get("name"))
        architecture = data.get("architecture", data.get("arch"))
        if not isinstance(model_id, str) or not model_id.strip():
            raise ModelDescriptorError("model_id must be a non-empty string")
        if not isinstance(architecture, str) or not architecture.strip():
            raise ModelDescriptorError("architecture must be a non-empty string")

        n_embd = _positive_int(data, "n_embd")
        n_head = _positive_int(data, "n_head")
        supplied_head_dim = data.get("head_dim")
        if supplied_head_dim is None:
            if n_embd % n_head != 0:
                raise ModelDescriptorError(
                    "head_dim is missing and cannot be inferred because n_embd is "
                    "not divisible by n_head"
                )
            head_dim = n_embd // n_head
        else:
            head_dim = _positive_int(data, "head_dim")

        descriptor = cls(
            model_id=model_id.strip(),
            architecture=architecture.strip().lower(),
            quantization=_normalize_quantization(data.get("quantization", "unknown")),
            n_layer=_positive_int(data, "n_layer"),
            n_embd=n_embd,
            n_ff=_positive_int(data, "n_ff"),
            n_head=n_head,
            n_head_kv=_positive_int(data, "n_head_kv"),
            head_dim=head_dim,
            sha256=(str(data["sha256"]).strip().lower() if data.get("sha256") else None),
        )
        descriptor.validate()

        for key, expected in (
            ("q_width", descriptor.q_width),
            ("k_width", descriptor.k_width),
            ("v_width", descriptor.v_width),
            ("attn_out_width", descriptor.attn_out_width),
        ):
            if key in data and _positive_int(data, key) != expected:
                raise ModelDescriptorError(
                    f"{key}={data[key]} disagrees with dimensions-derived value {expected}"
                )
        return descriptor

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelDescriptor":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelDescriptorError(f"failed to read descriptor JSON {source}: {exc}") from exc
        if isinstance(raw, Mapping) and "model" in raw:
            raw = raw["model"]
        if not isinstance(raw, Mapping):
            raise ModelDescriptorError("descriptor JSON must contain an object")
        return cls.from_dict(raw)

    @classmethod
    def _from_gguf_metadata(
        cls,
        source: Path,
        metadata: Mapping[str, object],
        *,
        compute_sha256: bool = False,
        quantization_override: str | None = None,
    ) -> "ModelDescriptor":
        def field(key: str, *, required: bool = True) -> object | None:
            value = metadata.get(key)
            if value is None and required:
                raise ModelDescriptorError(f"GGUF metadata is missing {key!r}")
            return value

        architecture_raw = field("general.architecture")
        if not isinstance(architecture_raw, str) or not architecture_raw.strip():
            raise ModelDescriptorError(
                "GGUF metadata 'general.architecture' must be a non-empty string"
            )
        architecture = architecture_raw.strip().lower()
        name = field("general.name", required=False)
        if name is not None and not isinstance(name, str):
            raise ModelDescriptorError("GGUF metadata 'general.name' must be a string")
        prefix = architecture

        def required_int(suffix: str) -> int:
            key = f"{prefix}.{suffix}"
            return _uniform_positive_int(field(key), key)

        n_embd = required_int("embedding_length")
        n_head = required_int("attention.head_count")
        n_head_kv_raw = field(f"{prefix}.attention.head_count_kv", required=False)
        n_head_kv = (
            n_head
            if n_head_kv_raw is None
            else _uniform_positive_int(n_head_kv_raw, f"{prefix}.attention.head_count_kv")
        )
        key_length_raw = field(f"{prefix}.attention.key_length", required=False)
        value_length_raw = field(f"{prefix}.attention.value_length", required=False)
        if key_length_raw is None:
            if n_embd % n_head != 0:
                raise ModelDescriptorError(
                    "GGUF omits attention.key_length and n_embd/n_head is not integral"
                )
            head_dim = n_embd // n_head
        else:
            head_dim = _uniform_positive_int(
                key_length_raw, f"{prefix}.attention.key_length"
            )
        if value_length_raw is not None:
            value_head_dim = _uniform_positive_int(
                value_length_raw, f"{prefix}.attention.value_length"
            )
            if value_head_dim != head_dim:
                raise ModelDescriptorError(
                    "different attention key/value head dimensions are not supported"
                )

        file_type = field("general.file_type", required=False)
        quantization = quantization_override or "UNKNOWN"
        if quantization_override is None and file_type is not None:
            file_type_value = _scalar_int(file_type, "general.file_type")
            quantization = _LLAMA_FILE_TYPE_NAMES.get(
                file_type_value, f"FILE_TYPE_{file_type_value}"
            )

        digest = _sha256_file(source) if compute_sha256 else None

        descriptor = cls(
            model_id=name.strip() if name is not None else source.stem,
            architecture=architecture,
            quantization=quantization,
            n_layer=required_int("block_count"),
            n_embd=n_embd,
            n_ff=required_int("feed_forward_length"),
            n_head=n_head,
            n_head_kv=n_head_kv,
            head_dim=head_dim,
            sha256=digest,
        )
        descriptor.validate()
        return descriptor

    @classmethod
    def from_gguf(
        cls,
        path: str | Path,
        *,
        compute_sha256: bool = False,
    ) -> "ModelDescriptor":
        source = Path(path)
        if not source.is_file():
            raise ModelDescriptorError(f"GGUF file does not exist: {source}")

        try:
            gguf = _repo_gguf_module()
        except ModelDescriptorError:
            # gguf-py imports numpy eagerly.  Metadata inspection should still
            # work on a dependency-free host, so parse only the GGUF KV header.
            metadata = _read_gguf_metadata_stdlib(source)
            return cls._from_gguf_metadata(
                source, metadata, compute_sha256=compute_sha256
            )

        try:
            reader = gguf.GGUFReader(source, mode="r")
        except Exception as exc:  # gguf-py raises several format/numpy exceptions
            raise ModelDescriptorError(f"failed to read GGUF {source}: {exc}") from exc

        architecture = _field_value(reader, "general.architecture")
        if not isinstance(architecture, str) or not architecture.strip():
            raise ModelDescriptorError(
                "GGUF metadata 'general.architecture' must be a non-empty string"
            )
        prefix = architecture.strip().lower()
        keys = (
            "general.architecture",
            "general.name",
            "general.file_type",
            f"{prefix}.block_count",
            f"{prefix}.embedding_length",
            f"{prefix}.feed_forward_length",
            f"{prefix}.attention.head_count",
            f"{prefix}.attention.head_count_kv",
            f"{prefix}.attention.key_length",
            f"{prefix}.attention.value_length",
        )
        metadata = {
            key: value
            for key in keys
            if (value := _field_value(reader, key, required=False)) is not None
        }

        quantization_override = None
        file_type = metadata.get("general.file_type")
        if file_type is not None:
            file_type_value = _scalar_int(file_type, "general.file_type")
            enum_type = getattr(gguf, "LlamaFileType", None)
            if enum_type is not None:
                try:
                    quantization_override = _normalize_quantization(
                        enum_type(file_type_value).name
                    )
                except ValueError:
                    quantization_override = f"FILE_TYPE_{file_type_value}"

        return cls._from_gguf_metadata(
            source,
            metadata,
            compute_sha256=compute_sha256,
            quantization_override=quantization_override,
        )


def load_model_descriptor(path: str | Path, *, compute_sha256: bool = False) -> ModelDescriptor:
    """Load either a compact descriptor JSON or a local GGUF file."""

    source = Path(path)
    if source.suffix.lower() == ".gguf":
        return ModelDescriptor.from_gguf(source, compute_sha256=compute_sha256)
    return ModelDescriptor.from_json(source)
