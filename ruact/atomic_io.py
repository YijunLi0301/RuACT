from __future__ import annotations
import ctypes
import errno
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temp_bytes(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return temporary


def atomic_write_bytes_noreplace(target_value: str | Path, payload: bytes) -> None:
    target = Path(target_value)
    temporary = _write_temp_bytes(target, payload)
    try:
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite artifact: {target}") from None
        _fsync_directory(target.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_json_noreplace(target: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes_noreplace(
        target,
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )


def atomic_replace_bytes(target_value: str | Path, payload: bytes) -> None:
    target = Path(target_value)
    temporary = _write_temp_bytes(target, payload)
    try:
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_rename_dir_noreplace(
    source_value: str | Path, target_value: str | Path
) -> None:
    source = Path(source_value)
    target = Path(target_value)
    target.parent.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(
                    f"refusing to overwrite artifact directory: {target}"
                )
            raise OSError(error_number, os.strerror(error_number), str(target))
        _fsync_directory(target.parent)
        return
    import fcntl

    lock_path = target.parent / ".criterion-collision-rename.lock"
    lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 384)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if os.path.lexists(target):
            raise FileExistsError(f"refusing to overwrite artifact directory: {target}")
        os.rename(source, target)
        _fsync_directory(target.parent)
    finally:
        os.close(lock_descriptor)
