from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from uuid import uuid4


def is_zarr_store(path: Path) -> bool:
    """Recognize a Zarr v2/v3 group without opening arbitrary user data."""

    if (path / ".zgroup").is_file():
        return True
    metadata = path / "zarr.json"
    if not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("node_type") == "group" and value.get("zarr_format") == 3


def validate_publish_target(
    target: Path,
    *,
    overwrite: bool,
    operation: str,
    require_zarr_v3: bool = False,
) -> Path:
    """Validate an output target without deleting or renaming it."""

    target = target.expanduser()
    if target.is_symlink():
        raise ValueError(f"拒绝将{operation}输出写入符号链接：{target}")
    target = target.resolve(strict=False)
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"输出路径存在但不是目录：{target}")
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空：{target}")
        if not is_zarr_store(target):
            raise ValueError("拒绝覆盖普通非空目录；只能覆盖已识别的 Zarr 目录。")
        if require_zarr_v3 and not (target / "zarr.json").is_file():
            raise ValueError("当前操作只能覆盖 Zarr v3 目录。")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def make_staging_path(target: Path, operation: str) -> Path:
    return target.parent / f".{target.name}.{operation}-{uuid4().hex}.tmp"


def publish_staging(
    staging: Path,
    target: Path,
    operation: str,
    *,
    overwrite: bool = True,
    require_zarr_v3: bool = False,
) -> None:
    """Atomically publish a validated directory and restore on rename failure."""

    if not staging.is_dir():
        raise FileNotFoundError(f"待发布的临时目录不存在：{staging}")
    # Repeat target validation immediately before the rename.  Long-running
    # conversions must not overwrite an unrelated directory that appeared
    # after the initial preflight check.
    target = validate_publish_target(
        target,
        overwrite=overwrite,
        operation=operation,
        require_zarr_v3=require_zarr_v3,
    )
    backup: Path | None = None
    if target.exists():
        backup = target.parent / f".{target.name}.{operation}-backup-{uuid4().hex}"
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except Exception:
        if backup is not None and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
