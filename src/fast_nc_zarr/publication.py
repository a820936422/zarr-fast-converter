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

def preflight_writable(path: Path, operation: str) -> dict[str, object]:
    """Probe directory creation and a durable write before long-running work."""

    requested = Path(path).expanduser().resolve(strict=False)
    existing = requested
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if existing.exists() and not existing.is_dir():
        raise NotADirectoryError(f"{operation}路径不是目录：{existing}")
    requested.mkdir(parents=True, exist_ok=True)
    probe_parent = requested
    probe = probe_parent / f".fast-nc-zarr-{operation}-preflight-{uuid4().hex}.probe"
    try:
        with probe.open("wb") as handle:
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(probe_parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass
    return {
        "requested": str(requested),
        "probe_parent": str(probe_parent),
        "writable": True,
        "operation": str(operation),
    }


def make_staging_path(
    target: Path,
    operation: str,
    staging_root: Path | None = None,
) -> Path:
    """Return a UUID-scoped staging directory.

    By default the staging directory is created beside the output so the
    final publication can be an atomic same-filesystem rename.  When
    ``staging_root`` is provided (used for HDD read/write phase separation),
    staging is placed on that directory instead and ``publish_staging`` falls
    back to a copy-then-rename publish on the target filesystem.
    """
    if staging_root is not None:
        root = Path(staging_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root / f".{target.name}.{operation}-{uuid4().hex}.tmp"
    return target.parent / f".{target.name}.{operation}-{uuid4().hex}.tmp"


def _entry_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        stat_result = path.lstat()
    except FileNotFoundError:
        return None
    return (stat_result.st_dev, stat_result.st_ino, stat_result.st_mode)


def _fsync_tree(root: Path) -> None:
    """Durably flush every file and directory below ``root``."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            continue
    try:
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _prepare_publish_target(
    target: Path,
    *,
    overwrite: bool,
    operation: str,
    require_zarr_v3: bool,
) -> tuple[Path, tuple[int, int, int] | None, Path | None]:
    """Validate the target and move an existing Zarr store to a backup."""
    target = validate_publish_target(
        target,
        overwrite=overwrite,
        operation=operation,
        require_zarr_v3=require_zarr_v3,
    )
    target_identity = _entry_identity(target)
    if target_identity is not None and target_identity[2] & 0o170000 == 0o120000:
        raise ValueError(f"拒绝发布到符号链接：{target}")
    if _entry_identity(target) != target_identity:
        raise RuntimeError(f"发布目标在校验期间发生变化：{target}")
    backup: Path | None = None
    if target.exists():
        if _entry_identity(target) != target_identity:
            raise RuntimeError(f"发布目标在替换前发生变化：{target}")
        backup = target.parent / f".{target.name}.{operation}-backup-{uuid4().hex}"
        os.replace(target, backup)
    return target, target_identity, backup


def _restore_backup(target: Path, backup: Path | None) -> None:
    if backup is not None and not target.exists():
        os.replace(backup, target)


def _publish_same_device(
    staging: Path,
    target: Path,
    operation: str,
    *,
    overwrite: bool,
    require_zarr_v3: bool,
) -> None:
    """Atomically publish via same-filesystem rename."""
    target, _target_identity, backup = _prepare_publish_target(
        target,
        overwrite=overwrite,
        operation=operation,
        require_zarr_v3=require_zarr_v3,
    )
    try:
        os.replace(staging, target)
    except Exception:
        _restore_backup(target, backup)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _publish_cross_device(
    staging: Path,
    target: Path,
    operation: str,
    *,
    overwrite: bool,
    require_zarr_v3: bool,
) -> None:
    """Publish a staging tree located on a different filesystem.

    A cross-device rename would fail with ``EXDEV``, so the validated staging
    tree is copied into a sibling temporary directory on the target
    filesystem, made durable, and then atomically renamed into place.  The
    remote staging tree is removed afterwards.  This is the publish side of
    the HDD read/write phase separation feature: conversion can write to a
    scratch device while the final output stays on the target device.
    """
    target, _target_identity, backup = _prepare_publish_target(
        target,
        overwrite=overwrite,
        operation=operation,
        require_zarr_v3=require_zarr_v3,
    )
    import_temp = target.parent / f".{target.name}.{operation}-import-{uuid4().hex}.tmp"
    try:
        shutil.copytree(staging, import_temp, symlinks=False)
        _fsync_tree(import_temp)
        os.replace(import_temp, target)
    except Exception:
        _restore_backup(target, backup)
        raise
    finally:
        shutil.rmtree(import_temp, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def publish_staging(
    staging: Path,
    target: Path,
    operation: str,
    *,
    overwrite: bool = True,
    require_zarr_v3: bool = False,
    allow_cross_device: bool = True,
) -> None:
    """Atomically publish a validated directory and restore on rename failure.

    ``allow_cross_device`` enables the copy-then-rename fallback when staging
    lives on a different filesystem than the output (HDD read/write phase
    separation).  When disabled, cross-device staging is rejected exactly as
    before.
    """

    if not staging.is_dir() or staging.is_symlink():
        raise FileNotFoundError(f"待发布的临时目录不存在或不安全：{staging}")
    if staging.parent.stat().st_dev != target.parent.stat().st_dev:
        if not allow_cross_device:
            raise OSError("staging 与 output 必须位于同一文件系统")
        _publish_cross_device(
            staging,
            target,
            operation,
            overwrite=overwrite,
            require_zarr_v3=require_zarr_v3,
        )
        return
    _publish_same_device(
        staging,
        target,
        operation,
        overwrite=overwrite,
        require_zarr_v3=require_zarr_v3,
    )
