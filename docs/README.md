# Fast NC Zarr v1.7.4 模块与运行指南

本文档合并项目各模块的入口、核心约束和验证方式。项目总览、环境安装和快速示例见根目录 [README](../README.md)。IPC 定义见 [contracts/README.md](../contracts/README.md)；历史基线审查见 [全面代码审查报告](comprehensive-code-audit-2026-08-15.md)。

## 共通约定

- 环境和命令由 `pixi.toml`、`pixi.lock` 管理。
- 最终产品固定为 Zarr v3，标准维度为 `time`、`lat`、`lon`。
- 时间轴统一为日精度；不支持小时、分钟或秒级时间轴。
- 输出先写任务专属 staging，完成结构、codec、样本和语义校验后再原子发布。
- 普通非空目录、符号链接目标和未知格式目录不会被 `--overwrite` 删除。
- `--dry-run` 只检查和规划，不写数据。

## 1. 源数据转换

入口：

```bash
pixi run convert -- --help
python -m fast_nc_zarr --help
```

支持 `complete`、`filename` 和 `auto` 三种输入模式。源文件可以是 NetCDF、HDF 或 TIFF；输出始终规范为 `time/lat/lon`。filename 模式从文件名的年、DOY 或年月日字段恢复时间轴，并检查重复、歧义和缺口。

常用命令：

```bash
pixi run convert -- \
  --input /data/source --output /data/result.zarr \
  --time '[2001-01-01,2022-12-31]' \
  --lat '[30,90]' --lon '[-180,180]' \
  --variables gpp quality

pixi run convert -- \
  --mode filename --input /data/daily-files \
  --output /data/result.zarr --template doy \
  --step-days 1 --continue-missing
```

核心服务位于 `fast_nc_zarr.application.services`：

- `inspect_source(SourceInspectionConfig(...))`
- `preview_conversion(inspection, ConversionConfig(...))`
- `run_conversion(inspection, ConversionConfig(...))`
- `save_inspection_snapshot(...)` / `load_inspection_snapshot(...)`

转换限制：日内时间、混合结构批次、额外维度和不支持的非数值变量会被拒绝或转入兼容路径。源变化、取消、校验失败和异常不会发布部分 staging。

## 2. 一条龙处理

入口：

```bash
pixi run pipeline -- --help
python -m fast_nc_zarr.pipeline --help
```

Pipeline 可组合源转换、空间重采样、重分块和重压缩。现有 Zarr 输入跳过转换，但至少需要请求一种后续操作。规划器会尽量把最终 chunks 和 codec 下推到前置阶段；只有物理布局不兼容时才执行独立最终化。

```bash
pixi run pipeline -- \
  --input /data/raw --input-kind raw \
  --output /data/product.zarr \
  --lat 30 90 --lon -180 180 \
  --resample --resolution 0.1 --method conservative \
  --rechunk --recompress --compression-codec blosc-zstd \
  --backend auto
```

每个任务保存 manifest、事件、资源预算、阶段状态和恢复检查点。恢复只接受经过校验的临时任务目录；失败和取消保留临时目录供排查，成功发布后才按策略清理中间结果。

## 3. 空间重采样

入口：

```bash
pixi run resample -- --help
python -m fast_nc_zarr.resampling --help
```

输入必须是规则经纬度 Zarr v3；`lat`、`lon` 必须是一维、单调、规则坐标，数据变量需要包含完整 `time/lat/lon`。支持 `bilinear`、`conservative`、`conservative_normed`、`patch`、`nearest_s2d` 和 `nearest_d2s`。

```bash
pixi run resample -- \
  --input /data/input.zarr --output /data/resampled.zarr \
  --resolution 0.25 --method bilinear --skipna
```

执行器按空间 tile 和时间 block 流式运行，统一受 `EffectiveResourceBudget`、owner buffer、worker 和物理 chunk ownership 约束。缺失值会被掩码；目标范围超出源覆盖时不外推，未覆盖格点保持缺测。替换规则通过 `--before-conditions/--before-results` 与 `--after-conditions/--after-results` 成对提供。

## 4. 重分块与重压缩

入口：

```bash
pixi run rechunk -- --help
python -m fast_nc_zarr.rechunking --help
```

输入必须是 Zarr v3，并包含完整 `time`、`lat`、`lon`。支持无损 Blosc Zstd/LZ4/LZ4HC/Zlib、原生 Zstd 和 Gzip。`time`、`space`、`custom` 分别适合时间序列、空间场和明确 chunks 的访问模式。

```bash
pixi run rechunk -- \
  --input /data/input.zarr --output /data/output.zarr \
  --strategy time --compression balanced
```

chunks 变化时使用有界两阶段流式写入；仅 codec 变化时逐源物理 chunk 转换；等价复制也先进入 staging。自动压缩会比较受控的无损候选并逐值验证。Rust 后端仅支持受 capability matrix 证明的单变量 float32 Zarr v3 场景；`auto` 不满足条件时回退 Python，`rust` 明确失败。

## 5. 真实源数据校验

入口：

```bash
pixi run validate-raw -- --help
python -m fast_nc_zarr.raw_validation --help
```

该模块处理输入根目录的直接子目录，每个子目录视为独立同构数据集，输出元数据、时间轴、经纬度、变量定义、分层源值样本和可选的小范围转换冒烟报告。

```bash
pixi run validate-raw -- \
  --input-root /data/RAW_DATA \
  --output /data/reports/raw-validation.json \
  --workers 4 --sample-files 9 \
  --smoke-output-root /fast-ssd/raw-smoke
```

抽样不能替代完整逐值科学验证；HDF-EOS、真实 GeoTIFF CRS/旋转、多 band 和 packed 数据仍需代表性 fixture 进一步验证。

## 6. Tauri 桌面应用

源码位置：

```text
apps/desktop/src/              React + TypeScript UI
apps/desktop/src-tauri/src/   Tauri/Rust runtime
```

启动和验证：

```bash
nvm use
npm --prefix apps/desktop ci
pixi run gui
npm --prefix apps/desktop run typecheck
npm --prefix apps/desktop run build
cargo test -p fast-nc-zarr-desktop
```

前端通过 Tauri commands 调用 Rust runtime；runtime 负责 worker、任务 registry、取消、资源快照、事件流、native capability 和恢复。主要 command 包括 `inspect_source`、`inspect_zarr`、`inspect_time_metadata`、`preview_pipeline`、`start_pipeline`、`resume_pipeline`、`start_native_task`、`list_tasks`、`get_task` 和 `cancel_task`。worker stdout 只承载 JSONL，诊断进入结构化事件或 stderr。

## v1.7.4 Linux 发布范围

v1.7.4 以 Linux `x86_64-unknown-linux-gnu` 为阻塞发布平台。发布前至少执行：

```bash
pixi run version-check
pixi run desktop-sidecar
pixi run desktop-typecheck
pixi run desktop-build
pixi run tauri-build
```

安装后还需确认 bundled worker、字体、图标、前端资源和至少一个真实 Tauri/worker IPC 流程可以从发布包路径加载。未完成 native parity 的操作必须记录实际 backend 和 fallback reason，不能伪装成 native 成功。
