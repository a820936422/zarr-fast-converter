#!/usr/bin/env python3
"""Validate that the bundled desktop worker matches the current source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = ROOT / "VERSION"
SIDECAR_DIR = ROOT / "apps" / "desktop" / "src-tauri" / "binaries"
METADATA_SUFFIX = ".build.json"
METADATA_SCHEMA_VERSION = 1
PROTOCOL_VERSION = 1

# These are the inputs packaged by build_desktop_sidecar.sh.  Documentation and
# the Tauri shell are intentionally excluded: they do not change the worker.
INPUT_ROOTS = (
    Path("VERSION"),
    Path("pyproject.toml"),
    Path("pixi.toml"),
    Path("pixi.lock"),
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("rust-toolchain.toml"),
    Path("scripts/build_desktop_sidecar.sh"),
    Path("scripts/collect_sidecar_libs.py"),
    Path("src/fast_nc_zarr"),
    Path("rust/crates/fast-nc-zarr-model"),
    Path("rust/crates/fast-nc-zarr-python"),
    Path("rust/crates/fast-nc-zarr-zarr"),
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".pixi", "target", "build", "dist"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll"}


def _target_triple() -> str:
    value = os.environ.get("TAURI_TARGET") or os.environ.get("TARGET")
    if value:
        return value
    result = subprocess.run(
        ["rustc", "-vV"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ").strip()
    raise RuntimeError("无法从 rustc -vV 解析 host target")


def default_sidecar_path() -> Path:
    return SIDECAR_DIR / f"fast-nc-zarr-worker-{_target_triple()}"


def _iter_input_files() -> Iterable[Path]:
    for relative in INPUT_ROOTS:
        path = ROOT / relative
        if path.is_file():
            yield path
            continue
        if not path.is_dir():
            raise RuntimeError(f"sidecar 构建输入不存在：{relative}")
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            relative_parts = candidate.relative_to(ROOT).parts
            if any(part in EXCLUDED_PARTS for part in relative_parts):
                continue
            if candidate.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            yield candidate


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    seen: set[Path] = set()
    for path in sorted(_iter_input_files()):
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _metadata_path(sidecar: Path) -> Path:
    return Path(f"{sidecar}{METADATA_SUFFIX}")


def _write_metadata(sidecar: Path) -> Path:
    metadata_path = _metadata_path(sidecar)
    payload = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "sidecar": str(sidecar.relative_to(ROOT)),
        "target": _target_triple(),
        "project_version": VERSION_PATH.read_text(encoding="utf-8").strip(),
        "protocol_version": PROTOCOL_VERSION,
        "source_fingerprint": source_fingerprint(),
        "source_commit": _git_commit(),
    }
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata_path


def _read_metadata(sidecar: Path) -> dict[str, object]:
    path = _metadata_path(sidecar)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"缺少 sidecar 构建元数据：{path}；请先运行 pixi run desktop-sidecar。"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"sidecar 构建元数据损坏：{path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise RuntimeError(f"不支持的 sidecar 构建元数据版本：{path}")
    return payload


def _smoke(sidecar: Path) -> dict[str, object]:
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "sidecar-check",
        "command": "get_capabilities",
        "payload": {},
    }
    try:
        result = subprocess.run(
            [str(sidecar)],
            cwd=ROOT,
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("bundled worker capabilities smoke 超时") from exc
    if result.returncode != 0:
        diagnostics = result.stderr.strip()[-1000:]
        raise RuntimeError(
            f"bundled worker 退出码为 {result.returncode}：{diagnostics or '无 stderr'}"
        )

    events: list[dict[str, object]] = []
    for index, line in enumerate(result.stdout.splitlines(), start=1):
        if len(line.encode("utf-8")) > 1_048_576:
            raise RuntimeError(f"bundled worker 第 {index} 行超过 IPC 上限")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"bundled worker 输出非法 JSONL：第 {index} 行") from exc
        if not isinstance(event, dict):
            raise RuntimeError(f"bundled worker 第 {index} 行不是 JSON object")
        events.append(event)

    names = [str(event.get("event")) for event in events]
    if names[:2] != ["accepted", "started"] or names[-1:] != ["finished"]:
        raise RuntimeError(f"bundled worker capabilities 事件序列异常：{names}")
    if sum(name in {"finished", "failed", "cancelled"} for name in names) != 1:
        raise RuntimeError(f"bundled worker terminal event 数量异常：{names}")

    payload = events[-1].get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("bundled worker capabilities 缺少 payload")
    native = payload.get("native")
    if not isinstance(native, dict):
        raise RuntimeError("bundled worker capabilities 缺少 native capability")
    expected_version = VERSION_PATH.read_text(encoding="utf-8").strip()
    if native.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("bundled worker native protocol version 不匹配")
    if native.get("crate_version") != expected_version:
        raise RuntimeError(
            f"bundled worker native crate version {native.get('crate_version')!r} "
            f"不匹配项目版本 {expected_version!r}"
        )
    return {
        "events": names,
        "native_protocol_version": native.get("protocol_version"),
        "native_crate_version": native.get("crate_version"),
        "stderr": bool(result.stderr.strip()),
    }


def check(sidecar: Path, *, write_metadata: bool) -> None:
    sidecar = sidecar.expanduser().resolve()
    if not sidecar.is_file() or not os.access(sidecar, os.X_OK):
        raise RuntimeError(f"bundled worker 不存在或不可执行：{sidecar}")
    if write_metadata:
        metadata_path = _write_metadata(sidecar)
    else:
        metadata_path = _metadata_path(sidecar)
    metadata = _read_metadata(sidecar)
    expected_fingerprint = source_fingerprint()
    actual_fingerprint = metadata.get("source_fingerprint")
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "bundled worker 不是当前源码构建；请先运行 pixi run desktop-sidecar。"
        )
    expected_target = _target_triple()
    if metadata.get("target") != expected_target:
        raise RuntimeError(
            f"bundled worker target {metadata.get('target')!r} 不匹配当前 target {expected_target!r}"
        )
    smoke = _smoke(sidecar)
    print(f"sidecar check passed: {sidecar}")
    print(f"metadata: {metadata_path}")
    print(f"source fingerprint: {expected_fingerprint}")
    print(f"capabilities smoke: {' → '.join(smoke['events'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 bundled desktop worker 是否匹配当前源码。")
    parser.add_argument("--path", type=Path, help="指定 sidecar 路径；默认使用当前 Rust host target。")
    parser.add_argument(
        "--write",
        action="store_true",
        help="写入当前源码 fingerprint 后再执行完整检查；仅由 sidecar 构建脚本使用。",
    )
    args = parser.parse_args(argv)
    try:
        check(args.path or default_sidecar_path(), write_metadata=args.write)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"sidecar check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
