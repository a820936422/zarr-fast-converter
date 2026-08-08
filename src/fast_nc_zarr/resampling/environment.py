from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


class ResamplingEnvironmentError(RuntimeError):
    """Raised before a long task when xESMF/ESMF cannot be loaded safely."""


def _makefile_value(path: Path, key: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    prefix = key + "="
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            return value or None
    return None


def _environment_details() -> tuple[Path, Path, Path | None, Path]:
    prefix = Path(sys.prefix).resolve()
    makefile = Path(
        os.environ.get("ESMFMKFILE", prefix / "lib" / "esmf.mk")
    ).expanduser().resolve()
    configured_libdir_text = _makefile_value(makefile, "ESMF_LIBSDIR")
    configured_libdir = (
        Path(configured_libdir_text).expanduser()
        if configured_libdir_text is not None
        else None
    )
    actual_library = prefix / "lib" / "libesmf_fullylinked.so"
    return prefix, makefile, configured_libdir, actual_library


def validate_resampling_environment() -> str:
    """Import xESMF before any expensive conversion and return its version.

    Conda's ESMF makefile contains absolute installation paths. Copying a Pixi
    environment with the project can therefore leave a valid library in the
    new prefix while ``esmpy`` still loads it from a deleted old prefix. This
    check reports both paths before a pipeline creates intermediate data.
    """

    prefix, makefile, configured_libdir, actual_library = _environment_details()
    problems: list[str] = []
    if not makefile.is_file():
        problems.append(f"ESMFMKFILE 不存在：{makefile}")
    if configured_libdir is not None:
        configured_library = configured_libdir / "libesmf_fullylinked.so"
        if not configured_library.is_file():
            problems.append(f"ESMF 配置引用的动态库不存在：{configured_library}")
        try:
            configured_resolved = configured_libdir.resolve()
        except OSError:
            configured_resolved = configured_libdir
        expected_libdir = prefix / "lib"
        if configured_resolved != expected_libdir:
            problems.append(
                "ESMF 配置仍指向其他环境："
                f"{configured_libdir}；当前环境应为 {expected_libdir}"
            )
    if not actual_library.is_file():
        problems.append(f"当前环境缺少 ESMF 动态库：{actual_library}")

    try:
        module = importlib.import_module("xesmf")
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        detail = "\n".join(f"  - {item}" for item in problems)
        if detail:
            detail = "\n环境诊断：\n" + detail
        raise ResamplingEnvironmentError(
            "xESMF/ESMF 运行环境不可用，已在正式处理前停止。\n"
            f"Python 环境：{prefix}\n"
            f"ESMFMKFILE：{makefile}\n"
            f"导入错误：{reason}"
            f"{detail}\n"
            "请勿复制项目的 .pixi 目录；在当前项目目录执行 "
            "`pixi reinstall esmf esmpy --locked` 后重试。"
        ) from exc

    if problems:
        detail = "\n".join(f"  - {item}" for item in problems)
        raise ResamplingEnvironmentError(
            "xESMF 已导入，但 ESMF 安装路径检查失败：\n"
            f"{detail}\n"
            "请执行 `pixi reinstall esmf esmpy --locked` 修复当前环境。"
        )
    return str(getattr(module, "__version__", "unknown"))
