import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "access_token",
    "bearer_token",
}


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def attach_stable_rollout_indices(
    non_tensor_batch: dict[str, Any], *, batch_size: int, rollout_n: int
) -> np.ndarray:
    if rollout_n <= 0 or batch_size % rollout_n != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by positive rollout_n={rollout_n}"
        )
    rollout_indices = np.tile(
        np.arange(rollout_n, dtype=np.int64), batch_size // rollout_n
    )
    non_tensor_batch["rollout_index"] = rollout_indices
    return rollout_indices


def build_actor_rpc_meta_info(*, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"metrics": dict(metrics)}


def _metadata_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def merge_rollout_dump_metadata(
    *,
    reward_extra_infos: Mapping[str, Any],
    non_tensor_batch: Mapping[str, Any],
    expected_size: int,
) -> dict[str, Any]:
    missing = {"uid", "rollout_index"} - set(non_tensor_batch)
    if missing:
        raise ValueError(
            f"rollout dump is missing stable join metadata: {sorted(missing)}"
        )
    result = dict(reward_extra_infos)
    joined_values = {}
    for key in ("uid", "rollout_index"):
        values = _metadata_list(non_tensor_batch[key])
        if len(values) != expected_size:
            raise ValueError(f"{key} must contain {expected_size} rollout values")
        values = [
            _json_scalar(value) if key == "uid" else int(value) for value in values
        ]
        if key in result and _metadata_list(result[key]) != values:
            raise ValueError(f"reward extra {key} conflicts with batch join metadata")
        result[key] = values
        joined_values[key] = values
    join_keys = list(
        zip(joined_values["uid"], joined_values["rollout_index"], strict=True)
    )
    if len(set(join_keys)) != expected_size:
        raise ValueError("(uid, rollout_index) must be unique within a rollout dump")
    if "request_id" in non_tensor_batch:
        request_ids = [
            _json_scalar(value)
            for value in _metadata_list(non_tensor_batch["request_id"])
        ]
        if len(request_ids) != expected_size:
            raise ValueError(f"request_id must contain {expected_size} rollout values")
        if (
            "request_id" in result
            and _metadata_list(result["request_id"]) != request_ids
        ):
            raise ValueError("reward extra request_id conflicts with batch metadata")
        result["request_id"] = request_ids
    return result


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path.name}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered.endswith("_env"):
        return False
    return lowered in _SENSITIVE_KEYS or any(
        (lowered.endswith(f"_{suffix}") for suffix in _SENSITIVE_KEYS)
    )


def sanitize_resolved_config(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): sanitize_resolved_config(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_resolved_config(item) for item in value]
    if isinstance(value, str) and (
        value.lower().startswith("bearer ") or value.startswith("sk-")
    ):
        return "<redacted>"
    return _json_scalar(value)


def write_resolved_config_snapshot(
    *, config: Mapping[str, Any], output_dir: str | os.PathLike[str]
) -> str:
    directory = Path(output_dir)
    config_path = directory / "resolved_config.json"
    hash_path = directory / "resolved_config.sha256"
    sanitized = sanitize_resolved_config(config)
    payload = (
        json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if config_path.exists() or hash_path.exists():
        if not config_path.exists() or not hash_path.exists():
            raise RuntimeError("resolved config snapshot is incomplete")
        existing_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        recorded_digest = hash_path.read_text(encoding="utf-8").strip()
        if existing_digest != digest or recorded_digest != digest:
            raise RuntimeError("resolved config snapshot does not match this run")
        return digest
    _atomic_write_new(config_path, payload)
    try:
        _atomic_write_new(hash_path, f"{digest}\n".encode("ascii"))
    except Exception:
        config_path.unlink(missing_ok=True)
        raise
    return digest
