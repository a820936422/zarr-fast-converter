# 当前目录保留与清理清单

> 版本：1.7.2
>
> 目标：只保留当前 Tauri + React + TypeScript 桌面前端、Rust native runtime、现行兼容处理服务、CLI、协议、测试和当前文档。

## 当前唯一入口

| 场景 | 入口 |
|---|---|
| 桌面开发 | `pixi run gui` |
| Tauri 直接启动 | `npm --prefix apps/desktop run tauri:dev` |
| 浏览器预览 | `npm --prefix apps/desktop run dev` |
| 源数据 CLI | `pixi run convert` |
| 一条龙 CLI | `pixi run pipeline` |
| Zarr 重采样 CLI | `pixi run resample` |
| Zarr 重分块 CLI | `pixi run rechunk` |
| 原始数据校验 | `pixi run validate-raw` |

已移除 `gui.py`、`pixi run gui` 的 PySide6 实现和所有旧 GUI wrapper。桌面前端不再有 Python/PySide6 API 或文案；兼容数据处理服务仍由 Tauri Rust 在 capability 不足时内部调度。

## 保留内容

### 桌面应用

- `apps/desktop/src/`：React + TypeScript 工作台；
- `apps/desktop/src-tauri/src/`：Tauri commands、任务注册、native runtime、事件和兼容边界；
- `apps/desktop/src-tauri/tauri.conf.json`：唯一桌面打包配置；
- `apps/desktop/public/fonts/`：当前前端字体资源；
- `scripts/desktop_dev.sh`：唯一 Tauri 开发启动脚本；
- `scripts/build_desktop.sh`：桌面发布构建脚本；
- `scripts/build_desktop_sidecar.sh`：兼容处理服务 sidecar 构建脚本。

### 数据处理与运行时

- `src/fast_nc_zarr/` 中的 CLI、inspection、pipeline、resampling、rechunking、validation、publication 和 desktop worker；
- `rust/crates/` 中的 model、Zarr native 和 PyO3 bridge；
- `contracts/` 中的 request、event、error 和 capability schema/fixture；
- `tests/` 中的核心处理、pipeline、native、contract、worker 和服务测试。

Python 核心服务不是过期前端代码，仍负责 NetCDF/HDF/TIFF、复杂科学语义和未通过 native 正确性门的兼容路径，因此保留。

### 当前文档

- `README.md`：项目入口和运行说明；
- `docs/README.md`：模块文档索引；
- `docs/gui.md`：当前 Tauri 桌面应用说明；
- `docs/v1.7.2-development-plan.md`：当前 native-first 状态和路线；
- `docs/cleanup-plan.md`：当前目录边界；
- `docs/converter.md`、`pipeline.md`、`resampling.md`、`rechunking.md`、`raw-validation.md`：现行 CLI/数据处理说明；
- `contracts/README.md`：IPC 合同说明。

## 已清理内容

### 旧入口

```text
convert.py
pipeline.py
resample.py
rechunk.py
gui.py
```

这些文件只是旧的根目录转发 wrapper，当前 Pixi task 和模块入口已经覆盖其职责。

### 旧 GUI 实现

```text
src/fast_nc_zarr/gui/
tests/test_gui.py
tests/test_path_picker.py
```

PySide6 页面、Qt path picker、Qt worker、GUI 字体和 GUI 专属测试已删除。对应的核心服务行为已经迁移到 `tests/test_application_services.py` 或由现行 pipeline/native/worker 测试覆盖。

### 历史架构资料

```text
docs/architecture/
```

旧版 drawio、PNG、v1.6/v1.7 设计草案和过期 Rust backend 说明已删除。当前架构以 `docs/v1.7.2-development-plan.md`、`docs/gui.md` 和源码为准。

## 可以安全删除的再生目录

以下内容不是源码，不进入提交；磁盘紧张时可删除，随后按表重建：

| 路径 | 重建方式 |
|---|---|
| `.pixi/` | `pixi install` |
| `target/` | `pixi run rust-test` 或 Tauri 构建 |
| `build/`、`dist/` | `pixi run desktop-sidecar` |
| `apps/desktop/node_modules/` | `npm --prefix apps/desktop ci` |
| `apps/desktop/dist/` | `npm --prefix apps/desktop run build` |
| `apps/desktop/src-tauri/binaries/` | `pixi run desktop-sidecar` |
| `.pytest_cache/`、`__pycache__/`、`*.pyc` | 测试或下次运行自动重建 |

`release/` 只允许保留已确认发布或归档的当前版本包；旧版本包应在校验和外部归档后删除。

## 不应删除的内容

- `src/fast_nc_zarr/application/desktop_worker/`：Tauri 兼容处理边界；
- `src/fast_nc_zarr/pipeline/`、`resampling/`、`rechunking/`：当前 CLI 和兼容数据处理能力；
- `contracts/`：Rust、TypeScript 和兼容服务共享的 IPC 合同；
- `tests/test_application_services.py`、`test_pipeline.py`、`test_native_smoke.py`、`test_desktop_worker.py`：当前行为回归覆盖；
- `apps/desktop/src-tauri/src/worker.rs`：sidecar/开发环境兼容服务启动器；
- `apps/desktop/public/fonts/`：当前 TypeScript 前端运行时资源。
