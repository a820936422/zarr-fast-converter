# Fast NC Zarr 项目全面评估与审查报告

> **后续更新**：2026-08-18，v1.7.7 P0、P1、P2 已完成，M6 发布门禁与发布准备完成（版本已提升至 1.7.7，deb 候选包已收集）。`sidecar-check`、`cross-backend-test` 已修复；可靠性/CF/native parity/A-B 脚本与测试已落地；P2 评估、UI 诊断与 Python 静态检查已完成，详见 [v1.7.7 优化方案](v1.7.7-optimization-plan.md) 与 [P2 评估记录](p2-evaluation.md)。下文保留审查时的原始结论。

- **审查日期**：2026-08-18
- **项目版本**：v1.7.6
- **当前分支**：`develop`（`c1733d9 docs: rebuild v1.7.7 optimization roadmap`）
- **审查范围**：源码（Python / Rust / TypeScript / Tauri）、测试、CI、契约、脚本、文档；不审查 `target/`、`node_modules/`、`dist/`、`release/` 等构建产物细节。
- **审查方式**：静态代码阅读 + 本地命令验证（版本一致性、契约、类型检查、构建、Rust fmt/clippy、Rust 测试、Python 测试、原生环境、sidecar 检查）。

## 1. 项目概览

Fast NC Zarr 是一个面向批量 NetCDF / HDF / TIFF 数据到 Zarr v3 的转换工作台，采用：

- **桌面端**：Tauri 2 + React 19 + TypeScript；
- **原生运行时**：Rust workspace（`fast-nc-zarr-model`、`fast-nc-zarr-zarr`、`fast-nc-zarr-python`、`fast-nc-zarr-desktop`）；
- **兼容处理服务**：Python（xarray / dask / xESMF / netCDF4 / rasterio / zarr），由 capability matrix 决定是否回退。

核心设计约束包括：

- 输出固定为 Zarr v3，标准维度 `time/lat/lon`；
- 任务先写入专属 staging，结构、codec、样本和语义校验通过后原子发布；
- 取消、异常、磁盘不足、校验失败不得发布部分结果；
- `backend=auto` 必须记录 resolved backend 与 fallback reason；
- Python worker 使用 JSONL stdin/stdout，stdout 只承载结构化事件；
- manifest / events / capability 记录实际执行路径。

## 2. 代码规模

| 类别 | 规模（约） |
|---|---:|
| Python（`src/` + `tests/` + `scripts/`） | 34,453 行 |
| TypeScript / TSX（`apps/desktop/src`） | 1,968 行 |
| Rust（`rust/crates` + `apps/desktop/src-tauri/src`） | 6,811 行 |
| Python 测试收集 | 221 个 |
| Rust 核心测试 | 14 个 |
| Rust 桌面测试 | 22 个 |

主要 Python 模块：

- `application/services.py`、`application/desktop_worker/`：应用编排与桌面 worker 协议；
- `filename_mode.py`、`inspection.py`、`time_mapping.py`：源数据检查、文件名时间轴与时间映射；
- `pipeline/`：端到端转换 / 重采样 / 重分块 / 重压缩；
- `resampling/`：xESMF 空间重采样与 native 快速路径；
- `rechunking/`：重分块、重压缩、自动压缩调参；
- `system.py`、`runtime.py`、`publication.py`：资源预算、进程池、原子发布；
- `raw_validation.py`：真实源数据校验报告。

## 3. 验证结果

以下命令均在当前工作区实际执行。

| 检查项 | 结果 | 说明 |
|---|---|---|
| `pixi run version-check` | ✅ 通过 | 所有版本声明一致为 `1.7.6` |
| `pixi run contract-check` | ✅ 通过 | request/event/error/capability schema 校验通过 |
| `pixi run native-check` | ✅ 通过 | `ready_for_p0: true`，必需工具齐全 |
| `pixi run rust-fmt-check` | ✅ 通过 | `cargo fmt --all -- --check` |
| `pixi run rust-clippy` | ✅ 通过 | `-D warnings` 无告警 |
| `pixi run rust-test` | ✅ 通过 | 14 个 Rust 核心测试通过 |
| `pixi run rust-desktop-test` | ✅ 通过 | 22 个桌面 Rust 测试通过 |
| `pixi run desktop-typecheck` | ✅ 通过 | `tsc --noEmit` |
| `pixi run desktop-build` | ✅ 通过 | `tsc --noEmit && vite build` 成功 |
| `pixi run test` | ✅ 通过 | 221 passed, 33 subtests passed, 1 warning |
| `pixi run sidecar-check` | ❌ 失败 | bundled worker 不是当前源码构建，需先运行 `pixi run desktop-sidecar` |
| `pixi run cross-backend-test` | ❌ 失败 | pixi 任务不存在，CI workflow 引用未定义任务 |

### 3.1 Python 测试覆盖

测试覆盖了以下关键领域：

- 端到端转换、chunk/file/dask 策略、自动调参；
- 文件名模式、时间规则、非标准维度映射；
- 重采样（bilinear / conservative / nearest 等）、native typed-buffer 路径；
- 重分块、重压缩、自动压缩候选安全；
- pipeline 规划、执行、恢复、取消、失败发布保护；
- desktop worker 协议、事件序列、capability matrix；
- 原子发布、跨文件系统拒绝、符号链接 / 普通目录覆盖保护；
- 系统资源探测（cgroup、CPU affinity、存储介质分类）。

## 4. 主要优势

1. **能力驱动的 native-first 架构**：Rust 能力矩阵明确，未支持操作不会伪装成功，`auto` 回退路径有结构化原因。
2. **发布安全边界扎实**：staging + 校验 + 原子 rename，且对普通目录、符号链接、跨文件系统有明确保护。
3. **可观测性完整**：manifest、事件流、资源快照、恢复 checkpoint、terminal event 唯一性均有实现和测试。
4. **测试覆盖面较好**：257 个 Rust/Python 测试覆盖核心路径，Rust fmt/clippy 也通过。
5. **协议契约清晰**：IPC schema、JSONL 边界、大小限制、事件序列约束都有定义和 fixture 校验。

## 5. 发现的问题与风险

### 5.1 阻断 / 需立即处理

1. **CI workflow 引用未定义的 pixi 任务**
   - 文件：`.github/workflows/native-preparation.yml:47`
   - 命令：`pixi run cross-backend-test`
   - 现状：`pixi.toml` 中没有该任务，`pixi run cross-backend-test` 返回 `command not found`（exit 127）。
   - 影响：CI `native-preparation` 会在此步骤失败。
   - 建议：补充该任务定义（例如 native 协议回归测试别名），或从 workflow 中移除/替换为已有任务。

2. **本地 sidecar 与当前源码不同步**
   - 命令：`pixi run sidecar-check` 失败。
   - 原因：`apps/desktop/src-tauri/binaries/fast-nc-zarr-worker-*` 的 source fingerprint 与当前源码不匹配。
   - 影响：当前工作区不能直接进入发布验证流程；`pixi run desktop-sidecar` 后才会恢复。
   - 建议：发布前执行 `pixi run desktop-sidecar`，并让 CI 或 pre-release 脚本强制检查该步骤。

### 5.2 文档与 UI 一致性

3. **`docs/README.md` 引用已删除的路线图**
   - 发现时 `docs/README.md` 链接到已删除的 `docs/v1.7.7-optimization-roadmap.md`。
   - 影响：文档链接失效。
   - 处理：本轮已将链接更新为本报告 `project-review.md`，问题已修复。

4. **`README.md` 代码块缺开头的反引号**
   - 发现时 `README.md` 的“构建桌面发布包”代码块缺少起始 ` ``` `，格式不完整。
   - 处理：本轮已补上反引号，问题已修复。

5. **浏览器预览 fallback 版本号过旧**
   - `apps/desktop/src/App.tsx:738` 中浏览器预览版本写死为 `1.7.5`，而项目当前为 `1.7.6`。
   - 建议：改为从 `package.json` 或统一常量读取，避免版本漂移。

### 5.3 可维护性与工程化

6. **部分 Python 模块体量过大**
   - `resampling/engine.py` 3,186 行、`rechunking/engine.py` 2,807 行、`filename_mode.py` 2,193 行、`pipeline/engine.py` 1,806 行。
   - 建议：按阶段/策略拆分，降低单文件复杂度，便于审查和测试定位。

7. **Python 侧缺少 CI 静态检查**
   - 当前 CI 有 Rust fmt/clippy、TS typecheck，但没有 Python linter/type checker（如 ruff / mypy / pyright）。
   - 建议：至少加入 `ruff check`；如可行再引入 mypy/pyright 对核心模块做类型检查。

8. **存在较多裸 `except ...: pass`**
   - 部分用于诊断/清理场景可以接受，但需确认没有吞掉关键错误。
   - 建议：将关键路径的异常改为显式记录日志或限定异常类型，避免静默失败。

### 5.4 发布与交付

9. **发布包体积较大**
   - `dist/fast-nc-zarr-worker` 约 361 MB；`.deb` 约 360–377 MB。
   - 这是当前 sidecar 打包策略（PyInstaller + 大量科学计算库）的结果；不一定是缺陷，但值得在发布说明中记录。

10. **`release/` 目录保留多版本安装包**
    - 保留 1.7.3 / 1.7.4 / 1.7.6 的 `.deb` 与 SHA256SUMS，属于本地发布资产；需确保发布流程与 Git tag 一致。

## 6. 结论

项目整体处于**健康、可发布前检查基本完善**的状态：核心 Rust/Python 测试、类型检查、构建、格式与 lint 均通过，架构边界清晰，安全发布机制有测试支撑。

但当前存在两个需要优先处理的工程问题：

- CI workflow 引用了不存在的 `cross-backend-test` 任务，会导致 CI 失败；
- 本地 bundled sidecar 与源码不一致，`sidecar-check` 失败。

建议在进入 v1.7.7 或下一次发布前：

1. 修复 CI 任务定义；
2. 重新构建并校验 sidecar；
3. 文档链接与 README 格式问题已在本轮修复；继续统一前端 fallback 版本号；
4. 评估 Python 静态检查与大型模块拆分；
5. 根据已删除的 v1.7.7 优化路线图中的 P0 故障注入清单，继续补强失败/恢复/发布可靠性测试。
