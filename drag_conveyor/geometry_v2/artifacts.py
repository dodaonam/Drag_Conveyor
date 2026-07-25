from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_onnx_artifact_manifest(path: str | Path, *, class_names: dict[str, str] | None = None) -> dict[str, Any]:
    """Read immutable facts encoded in an ONNX artifact and derive its stable ID."""
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - exercised only in broken installs
        raise RuntimeError("onnx package is required to read a geometry model artifact manifest") from exc
    artifact_path = Path(path)
    model = onnx.load_model(str(artifact_path), load_external_data=False)
    manifest: dict[str, Any] = {
        "schema_version": "model_artifact/1.0",
        "sha256": _sha256_file(artifact_path),
        "format": "onnx",
        "onnx_ir_version": int(model.ir_version),
        "opset_imports": {item.domain: int(item.version) for item in model.opset_import},
        "input": _value_info(model.graph.input[0]) if model.graph.input else None,
        "outputs": [_value_info(item) for item in model.graph.output],
        "class_names": dict(sorted((class_names or {}).items())),
        "export_metadata_verbatim": dict(sorted((item.key, item.value) for item in model.metadata_props)),
    }
    manifest["artifact_manifest_id"] = canonical_record_hash(manifest, omit="artifact_manifest_id")
    return manifest


def canonical_record_hash(record: dict[str, Any], *, omit: str) -> str:
    payload = {key: value for key, value in record.items() if key != omit}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_info(value_info: Any) -> dict[str, Any]:
    tensor = value_info.type.tensor_type
    dtype = str(tensor.elem_type)
    try:
        import onnx

        dtype = onnx.TensorProto.DataType.Name(tensor.elem_type).lower()
    except (ImportError, ValueError):
        pass
    shape: list[int | str | None] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return {"name": value_info.name, "dtype": dtype, "shape": shape}
