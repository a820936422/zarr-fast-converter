# Fast NC Zarr 项目维护文档

> 本文档以当前项目状态为基准，记录项目信息、模块结构、开发/测试/发布命令和维护约定。
> **约定：每次版本更新时，必须同步更新本文档的“版本状态”与“版本历史”章节。**

- **最近更新**：2026-08-20
- **当前版本**：v1.8.1
- **当前分支**：`develop`
- **仓库**：`https://github.com/a820936422/zarr-fast-converter.git`

---

## 1. 项目基本信息

| 项 | 内容 |
|---|---|
| 项目名称 | Fast NC Zarr |
| 项目定位 | 批量 NetCDF / HDF / TIFF 数据到 Zarr v3 的转换工作台 |
| 当前版本 | 1.8.1 |
| 输出格式 | 固定为 Zarr v3，标准维度 `time/lat/lon` |
| 桌面端 | Tauri 2 + React 19 + TypeScript |
| 原生后端 | Rust workspace（model / zarr / python / desktop） |
| 兼容后端 | Python（xarray / dask / xESMF / netCDF4 / rasterio / zarr） |
| 环境管理 | Pixi（`pixi.toml` / `pixi.lock`），Node 版本见 `.nvmrc` |
| 支持平台 | Linux `x86_64-unknown-linux-gnu`（当前发布阻塞平台） |

---

## 2. 目录与模块地图

```text
.
├── apps/desktop/                 # Tauri 桌面应用
│   ├── src/                      # React + TypeScript UI
│   └── src-tauri/                # Tauri/Rust runtime 与 sidecar 配置
├── contracts/                    # IPC schema 与 fixture
├── docs/                         # 项目文档（含本维护文档）
├── rust/crates/
│   ├── fast-nc-zarr-model/       # 共享协议类型与 capability 定义
│   ├── fast-nc-zarr-python/      # PyO3 桥接层
│   └── fast-nc-zarr-zarr/        # Zarr v3 原生检查/写入/重分块/重采样/NetCDF 转换
├── scripts/                      # 构建、检查、发布脚本
├── src/fast_nc_zarr/             # Python 兼容服务与 CLI
│   ├── application/              # 应用编排与桌面 worker
│   ├── pipeline/                 # 端到端 pipeline
│   ├── resampling/               # 空间重采样
│   ├── rechunking/               # 重分块与重压缩
│   └── ...                       # 检查、规划、写入、发布、资源管理等
└── tests/                        # Python 测试与 fixture
```

### 2.1 关键模块职责

| 模块/目录 | 职责 |
|---|---|
| `src/fast_nc_zarr/application/services.py` | CLI 与桌面 worker 共享的应用编排入口 |
| `src/fast_nc_zarr/application/desktop_worker/` | Python worker JSONL 协议与任务执行 |
| `src/fast_nc_zarr/filename_mode.py` | 文件名时间轴推断与 filename 模式转换 |
| `src/fast_nc_zarr/pipeline/` | 转换 / 重采样 / 重分块 / 重压缩编排 |
| `src/fast_nc_zarr/resampling/` | xESMF 重采样与 native 快速路径 |
| `src/fast_nc_zarr/rechunking/` | 重分块、重压缩、自动压缩调参 |
| `src/fast_nc_zarr/system.py` | 资源预算、CPU/内存/cgroup、存储介质分类 |
| `src/fast_nc_zarr/publication.py` | staging 校验与原子发布 |
| `src/fast_nc_zarr/runtime.py` | 进程池生命周期与有界并行执行 |
| `rust/crates/fast-nc-zarr-zarr/` | Rust 原生 Zarr/NetCDF/resample/rechunk 能力 |
| `apps/desktop/src-tauri/` | Tauri command、任务 registry、worker 管理 |

---

## 3. 环境准备

```bash
# Pixi 环境
pixi install

# 前端 Node 环境
nvm use
npm --prefix apps/desktop ci
```

### 3.1 工具链要求

- Python：由 Pixi 管理，当前为 3.13.x
- Rust：`rust-toolchain.toml` 指定 1.97.1
- Node：`.nvmrc` 指定 24.19.0
- npm：`apps/desktop/package.json` 要求 npm >= 12

---

## 4. 常用命令

### 4.1 开发

```bash
# 启动 Tauri 桌面应用（唯一桌面入口）
pixi run gui

# 仅浏览器前端预览（不启动 Tauri runtime）
npm --prefix apps/desktop run dev
```

### 4.2 命令行处理

```bash
# 源数据转换
pixi run convert -- --input /path/to/nc --output /path/to/result.zarr

# 一条龙处理
pixi run pipeline -- --input /path/to/source --input-kind raw --output /path/to/result.zarr --rechunk --compression auto

# 重采样
pixi run resample -- --input /path/to/input.zarr --output /path/to/resampled.zarr --resolution 0.25

# 重分块/重压缩
pixi run rechunk -- --input /path/to/input.zarr --output /path/to/rechunked.zarr --strategy time --compression balanced

# 真实源数据校验
pixi run validate-raw -- --help
```

### 4.3 测试与静态检查

```bash
# Python 全量测试
pixi run test

# 跨后端/native 协议冒烟测试
pixi run cross-backend-test

# Python 静态检查（F/E9）
pixi run python-lint

# Rust 核心测试（不含桌面）
pixi run rust-test

# Rust 桌面测试
pixi run rust-desktop-test

# Rust 格式与 lint
pixi run rust-fmt-check
pixi run rust-clippy

# 前端类型检查与构建
pixi run desktop-typecheck
pixi run desktop-build
```

### 4.4 契约与版本检查

```bash
pixi run version-check
pixi run contract-check
```

### 4.5 原生能力检查

```bash
pixi run native-check
```

### 4.6 性能基准（P1 A/B 与 scaling）

```bash
# Filename chunk-owner writer 与 partial-region baseline A/B
pixi run benchmark-filename-ab -- <source_dir> --output-root <out_root>

# Conservative 与 conservative_normed 重采样 A/B（合成数据可重复）
pixi run benchmark-conservative-resample-ab -- --output-root <out_root>

# 转换写入 scaling 基准（合成或真实 NetCDF，workers 1/2/4/8/12）
pixi run benchmark-scaling -- --synthetic --output-root <out_root>
# 或真实源：
pixi run benchmark-scaling -- --input <source_dir> --output-root <out_root>
# 对比跨设备 staging 收益（HDD 读写分离；--staging-root 可指向另一设备）：
pixi run benchmark-scaling -- --synthetic --compare-staging --staging-root /fast-ssd/scratch --output-root <out_root>

# scaling smoke 门禁（发布前快速验证，非正吞吐即失败）
pixi run scaling-check
```

---

## 5. 发布与版本更新流程

### 5.1 版本更新需要同步修改的位置

版本一致性脚本 `scripts/check_version_consistency.py` 会校验以下位置必须保持一致：

- `VERSION`
- `pixi.toml` 的 `[workspace].version`
- `pyproject.toml` 的 `[project].version`
- `Cargo.toml` 的 `[workspace.package].version`
- `apps/desktop/package.json`
- `apps/desktop/package-lock.json`
- `apps/desktop/src-tauri/tauri.conf.json`
- `apps/desktop/src-tauri/Cargo.toml`
- `src/fast_nc_zarr/__init__.py`
- `apps/desktop/src-tauri/src/lib.rs`
- `contracts/README.md`
- `contracts/fixtures/capability-v1.json`
- `docs/README.md` 中的发布版本说明

### 5.2 版本更新 Checklist

1. **更新版本号**：按 `5.1` 列出的文件统一修改。
2. **更新文档**：
   - 更新 `docs/README.md` 中对应版本说明。
   - 更新本维护文档的“当前版本”“版本状态”“版本历史”。
   - 如架构、命令、模块或发布范围变化，同步更新本维护文档。
   - 如有重大审查/路线图，更新或新增 `docs/` 下对应文档。
   - 如有待办/路线图变更，同步更新 `docs/v1.8.1-development.md`。
3. **运行完整检查**：

```bash
pixi run version-check
pixi run contract-check
pixi run python-lint
pixi run cross-backend-test
pixi run desktop-typecheck
pixi run desktop-build
pixi run rust-fmt-check
pixi run rust-clippy
pixi run rust-test
pixi run rust-desktop-test
pixi run test
pixi run native-check
```

4. **构建并校验 sidecar**：

```bash
pixi run desktop-sidecar
pixi run sidecar-check
```

5. **构建桌面发布包**：

```bash
pixi run tauri-build
```

6. **收集发布候选**：

```bash
pixi run release-candidate
```

7. **提交并打 tag**：确保所有版本相关文件和本文档一起提交，再打对应 `vX.Y.Z` tag。

### 5.3 发布安全约束

- 输出必须先写 staging，结构/codec/样本/语义校验通过后才原子发布。
- `--overwrite` 不得删除普通非空目录、符号链接或未知格式目录。
- `backend=auto` 必须记录实际 resolved backend 与 fallback reason。
- 未通过 capability matrix 的 native 操作必须回退 Python，不能伪装成 Rust 成功。
- Python worker stdout 只允许 JSONL，诊断信息进入 log event 或 stderr。

---

## 6. 当前版本状态与已知问题

以下为当前工作区状态（v1.8.0 开发中），后续版本更新时需同步复核。

### 6.1 已验证状态

| 检查项 | 状态 |
|---|---|
| 版本一致性 | ✅ 通过 |
| 契约校验 | ✅ 通过 |
| 原生环境 | ✅ `ready_for_p0: true` |
| Rust fmt / clippy | ✅ 通过 |
| Rust 核心测试 | ✅ 23 个通过 |
| Rust 桌面测试 | ✅ 22 个通过 |
| 前端 typecheck / build | ✅ 通过 |
| Python 测试 | ✅ 290 passed，35 subtests passed，1 warning |
| `cross-backend-test` | ✅ 44 passed, 4 subtests passed |
| sidecar-check | ✅ 已重建并校验通过（v1.7.8） |
| Filename A/B smoke | ✅ 合成数据可重复执行 |
| Conservative A/B smoke | ✅ 合成数据可重复执行 |
| L0 真实数据受控样本 | ✅ 2/2 passed（FLUXSATv2 + GLASS-EVI） |
| L1 真实数据专项 | ✅ T1–T5 全部 passed，T6 `validate-raw` 全树通过 |
| `validate-raw` 全树 | ✅ 1240 文件 / 2 数据集 passed |
| 后端自适应优化 | ✅ 存储感知 worker/batch 初始值（非硬上限）+ 重采样全局线程预算已落地 |
| WorkerPool 全路径接入 | ✅ `direct_write`、文件名写入/检查、inspection、重分块两阶段与调参、重采样空间并行已接入；完整回归 290 passed |
| OnlineController 扩展 | ✅ 重分块阶段 1/2 与重采样空间并行动态 pending + `online_adjustments` 事件，pipeline manifest 顶层聚合 |
| HardwareProfile 调度集成 | ✅ 实测带宽初始 worker（fast HDD 提升到 8）+ P/E affinity spec 已落地并有单测 |
| scaling smoke 门禁 | ✅ `pixi run scaling-check` 通过（合成数据，workers 1/2） |
| v1.8.0 后端/调度优化 | ✅ PerformanceModel 默认启用与剪枝（`FAST_NC_ZARR_PERF_MODEL=0/off/false` 可关）、Storage-aware 初始 worker、P/E affinity spec（`test_performance_model.py` / `test_hardware.py`） |
| 跨设备/同设备 HDD I/O 分离 | ✅ `make_staging_path(staging_root=...)` + `publish_staging` 跨文件系统复制发布（`allow_cross_device` 默认开）；CLI `--staging-root`；filename `--phase-batch` 先读后写窗口（parity 测试通过） |
| filename 模式 | ✅ 28 passed：HDF-EOS Grid 低层坐标重建单测 + 端到端转换；HDF-EOS Swath `CoreMetadata.0`/`SwathStructMetadata` 解析、规则退化 2D GeoField → 1D 经纬度轴重建（微度归一化、不规则 Swath 拒绝、同维度退化回退）、低层扫描/端到端转换/xarray 归一化，GeoField 不进入输出变量 |
| native 未打包标准整数 | ✅ `test_native_smoke.py` 33 passed（int16 inspect/conversion roundtrip + 打包 int16 拒绝），capability 矩阵已同步 |
| native conservative 双后端 parity | ✅ native conservative/conservative_normed 已实现球面 cell-area overlap 与确定性边界触碰规则；Python/xESMF 路径按同一规则重算掩码与值，`test_resampling.py` 新增 skipna=False 边界触碰 parity 测试；v180 acceptance 9/9 通过（FLUXSAT conservative/conservative_normed 双后端值、NaN 掩码与 attrs 一致） |
| PipelineFusion 单遍流式 | ✅ 融合路径已实现：eligible 且 ≤2 GiB crop / 非 filename / 串行或 auto 重采样时，转换以惰性内存 `xr.Dataset` 直入重采样，不写磁盘中间 crop；`write_amplification=1.0`、`intermediate_validation_skipped=True`、`stages.conversion.status=fused_in_memory`；`--no-fusion` 关闭保留磁盘 checkpoint/恢复语义；`tests/test_pipeline.py` 43 passed（含中间 store 跳过 + 与磁盘路径逐值 parity） |
| 完整回归（当前基线） | ✅ `pixi run test` 290 passed、35 subtests、1 warning；`cross-backend-test` 44 passed、4 subtests；Rust 核心 23 passed |

### 6.2 已知问题

1. ~~CI workflow 引用未定义的 pixi 任务~~ 已修复：`pixi.toml` 已新增 `cross-backend-test`，本地运行通过。
2. ~~本地 sidecar 过期~~ 已修复：已执行 `pixi run desktop-sidecar` 并重新校验通过。
3. ~~前端浏览器预览 fallback 版本号为 1.7.5~~ 已修复：`App.tsx` 改为从 `package.json` 读取版本。
4. **文档/维护状态**：历史版本专项文档已整合清理；当前活跃文档为 `docs/MAINTENANCE.md`、`docs/README.md` 和 `docs/v1.8.1-development.md`。
5. **Python 可维护性**：`ruff`/`python-lint` 已落地并修复 F/E9 问题；大模块拆分仍作为后续 backlog。
6. ~~`validate-raw` 对 FLUXSATv2 float32 经纬度网格误报“不规则网格”~~ 已修复：`raw_validation._axis_report` 容差改为按坐标 dtype 自适应（float32 0.05° 网格通过）；新增单测 `test_axis_report_accepts_float32_regular_grid`。

> 后续待处理内容统一记录在 [v1.8.1 开发文档](v1.8.1-development.md)。

### 6.3 发布状态（开发阶段）

- **v1.7.7 修改已提交并同步远程**：当前 `develop` 分支已包含 v1.7.7 全部修改，远程已手动推送；本地 tag `v1.7.7` 已创建。
- **v1.7.8 真实数据测试已完成**：版本已提升至 1.7.8，L0/L1 与 `validate-raw` 全树验证通过，回归门禁保持绿色。
- **v1.7.8 后端优化分析已完成，P0 代码优化已落地**：存储感知 worker/batch **初始值**（仅作起始猜测，调优仍探索完整 worker 范围）、重采样全局线程预算已实现并通过测试。
- **v1.7.9 后端优化已实施**：已完成 HardwareProfile（含 NUMA/P/E）、PerformanceModel、重分块存储感知初始值、WorkerPool 工具类、文件亲和排序、OnlineController 可观测与保守调整、CPU affinity、PipelineFusion eligibility 标记；剩余项列入 [v1.8.1 开发文档](v1.8.1-development.md)。
- **v1.8.0 开发基线已提交并推送**：v1.8.0 完成项已收敛进版本历史（commit `18bf7bb`），包括 WorkerPool 全路径、HDD 读写阶段分离、PipelineFusion、OnlineController、native 整数、HDF-EOS Grid/Swath、native conservative parity、scaling 基础设施。
- **v1.8.1 待办推进（开发中）**：版本已提升至 1.8.1；剩余项转入 [v1.8.1 开发文档](v1.8.1-development.md)，重点为真实设备/真实数据验证、HDF-EOS/GeoTIFF fixture、剩余 native 能力扩展与发布门禁。
- **项目仍处于开发阶段**：**暂不打包安装包、暂不上传发布资产**（不创建 GitHub Release、不上传 `.deb`/`.rpm` 等安装包）。
- 本地 `release/` 下的候选安装包（含 `Fast NC Zarr_1.7.7_amd64.deb`）仅作本地留档，不用于分发；`release/*.deb` 已由 `.gitignore` 排除，不入库。
- 后续若进入正式发布阶段，再按本文件“5. 发布与版本更新流程”执行打包、收集与上传。

---

## 7. 维护约定

- 所有影响 IPC 的改动必须同步更新 `contracts/` 下的 schema 与 fixture，并运行 `pixi run contract-check`。
- 所有影响版本号的改动必须运行 `pixi run version-check`。
- 所有影响 Rust native capability 的改动必须同步 `rust/crates/fast-nc-zarr-model` 中的 capability 定义和 `contracts/fixtures/capability-v1.json`。
- 所有新增或修改的核心路径应补充 Python/Rust 测试。
- 每次版本更新必须同步更新本文档，并在“版本历史”中追加记录。

---

## 8. 版本历史

| 版本 | 日期 | 主要内容 |
|---|---|---|
| 1.8.1 | 2026-08-20 | 版本提升至 1.8.1；v1.8.0 完成项收敛进本文档；剩余项见 `docs/v1.8.1-development.md`，重点为真实设备/真实数据验证、HDF-EOS/GeoTIFF fixture、剩余 native 能力扩展与发布门禁 |
| 1.8.0 | 2026-08-20 | v1.8.0 开发基线提交并推送（commit `18bf7bb`）：WorkerPool 全路径、HDD 读写阶段分离、PipelineFusion、OnlineController、native 整数、HDF-EOS Grid/Swath、native conservative parity（v180 acceptance 9/9）、scaling 基础设施；开发阶段暂不打包上传 |
| 1.7.9 | 2026-08-18 | 版本提升至 1.7.9；开始实施后端优化方案：新增 HardwareProfile 存储微基准与缓存、PerformanceModel 候选排序、重分块 worker 初始值（全范围实测保留）；P1 其余项与 P2 继续推进；开发阶段暂不打包上传，剩余项见 `docs/v1.8.1-development.md` |
| 1.7.8 | 2026-08-18 | 版本提升至 1.7.8；完成 FLUXSATv2 + GLASS-EVI 真实数据测试（L0/L1、`validate-raw` 全树 1240 文件通过）；修复 `validate-raw` float32 坐标容差；落地后端自适应优化 P0（存储感知 worker/batch 初始值、重采样全局线程预算）；sidecar 已按 v1.7.8 重建并校验；开发阶段暂不打包上传 |
| 1.7.6 | 2026-08-18 | 当前基线：native-first 能力矩阵、Tauri 桌面、pipeline/resample/rechunk 能力；本文档首次建立 |
| 1.7.7 | 2026-08-18 | 版本提升至 1.7.7；P0、P1、P2 已完成；M6 发布门禁与发布准备完成（deb 候选包已收集）；修改已提交并同步远程（tag `v1.7.7`）；当前处于开发阶段，暂不打包安装包、暂不上传发布资产 |

> 后续每个版本发布时，在表格顶部插入新行，并更新“当前版本”与“当前版本状态”。
