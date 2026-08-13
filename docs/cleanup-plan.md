# 项目目录精简与清理清单

> 审计日期：2026-08-13
> 审计分支：`refactor/v1.7.1-tauri`
> 审计基线：`2571b854`（扫描时本地与远端分支一致）；清单提交为后续提交。

## 1. 结论先行

当前目录“乱”的主要来源不是源码数量，而是**构建树、运行环境、发布包和源码同时堆在工作树**：

- 工作树磁盘占用约 **18G**；其中 `.pixi` 约 3.9G、`target` 约 12G、`build` 约 280M、根 `dist` 约 245M、`release` 约 386M。
- `apps/desktop/node_modules` 约 100M，`apps/desktop/dist` 约 18M，Tauri sidecar 目录约 267M。
- Python GUI 和 React/Tauri 各自携带一份相同的 Noto Sans SC 字体，各约 17M；两份哈希相同，但当前分别被两个运行时直接引用，不能仅凭重复哈希删除一份。
- `release/` 中的 deb/rpm 是交付包，不是普通缓存；本次推送失败的直接诱因就是把约 303M deb 和约 82M rpm 作为 Git blob 推送。已将它们从未推送历史移除，并把 `release/*.deb`、`release/*.rpm` 加入 `.gitignore`，随后成功推送分支。
- 目录中存在多个版本的架构规划、可编辑 draw.io 源图和 PNG 导出；可以明显收敛，但历史设计文档和可编辑源图应先归档再删，不建议直接粗暴删除。

**推荐策略：先清理可再生产物，再收敛发布资产和文档；最后才动兼容入口、协议实现和字体。** 当前源码、测试、契约和桌面实现不应因为“看起来多”而直接删除。

---

## 2. 审计范围与现状证据

### 2.1 扫描范围

本次只读审计覆盖：

- 根目录全部条目、根级配置、入口脚本、README；
- `src/` 全部 Python 包及其子包；
- `rust/` 全部 Cargo crate；
- `apps/desktop/` 前端、Tauri Rust、资源、配置；
- `contracts/` schema、fixture、说明；
- `tests/` 全部测试文件；
- `docs/`、`Architecture/` 全部文档和图纸；
- `scripts/`、`.github/`、`release/`；
- ignored/untracked 的 `.pixi`、`target`、`build`、`dist`、`node_modules`、Python 缓存、Tauri 生成目录和发布包。

审计期间未删除文件，未运行格式化、lint 或项目级测试。

### 2.2 Git 与文件数量

| 项目 | 观测值 | 说明 |
|---|---:|---|
| Git 跟踪文件 | 179 | 源码、测试、配置、文档和资源 |
| `git status --short --ignored` 行数 | 20 | 主要是 ignored 构建/缓存目录 |
| `git clean -ndX` 候选目录/文件组 | 20 | 可再生物与本地缓存；不能盲删 release 包 |
| 推送分支 | `refactor/v1.7.1-tauri` | 与远端同名分支一致 |
| 当前 HEAD | `2571b854` | 清理历史大包后的提交 |
| Git pack 体积 | 约 13.14MiB | 已移除未推送历史中的 deb/rpm 大 blob |

根目录跟踪内容按职责分布如下：

| 路径 | 跟踪条目数 | 职责 |
|---|---:|---|
| `src/` | 65 | Python 产品代码、GUI、pipeline、resampling、rechunking、desktop worker |
| `apps/` | 43 | React/Vite/Tauri 桌面壳和跨平台资源 |
| `tests/` | 18 | Python、GUI、Rust/native、协议和 pipeline 行为测试 |
| `Architecture/` | 11 | 历史方案、可编辑架构图和渲染导出 |
| `docs/` | 7 | 模块使用文档和索引 |
| `contracts/` | 6 | IPC schema、fixture、协议说明 |
| `rust/` | 6 | 三个 Rust crate 的 manifest 与源码 |
| `scripts/` | 3 | native 环境检查和桌面 sidecar 构建 |
| `.github/` | 1 | native/desktop CI |
| `release/` | 1 | 跟踪的发布图标；deb/rpm 已改为 ignored |

### 2.3 当前主要磁盘占用

| 路径 | 约占用 | 状态 | 处理建议 |
|---|---:|---|---|
| `.pixi/` | 3.9G | ignored，本地 Pixi 环境 | 可重建；离线开发前不要删 |
| `target/` | 12G | ignored，Cargo/Tauri/maturin 构建树 | 第一批清理 |
| `build/` | 280M | ignored，PyInstaller 中间产物 | 第一批清理 |
| 根 `dist/` | 245M | ignored，PyInstaller 输出 | 第一批清理 |
| `release/` | 386M | deb/rpm ignored，图标 tracked | 包发布/校验后清理包 |
| `apps/desktop/node_modules/` | 100M | ignored，npm 依赖树 | `npm ci` 可重建 |
| `apps/desktop/dist/` | 18M | ignored，Vite 输出 | `npm run build` 可重建 |
| `apps/desktop/src-tauri/binaries/` | 267M | ignored，sidecar | 重建 sidecar 后可清理 |
| `apps/desktop/src-tauri/gen/schemas/` | 316K | ignored，Tauri 生成 schema | 先确认生成流程，暂留 |
| `src/fast_nc_zarr/gui/assets/` | 17M | tracked，PySide6 GUI 字体 | 当前必须保留 |
| `apps/desktop/public/fonts/` | 17M | tracked，React/Tauri 字体 | 当前必须保留 |
| `Architecture/` | 1.9M | tracked，设计文档和图纸 | 归档、去重复导出 |

---

## 3. A 类：可立即清理的内容

这里的“立即”指：不改变源码和提交历史；删除后由现有配置重新生成。第一次执行前仍应确认没有正在运行的构建、没有需要离线使用的产物。

### A1. 整体删除可再生构建树

| 路径 | 证据 | 风险 | 重建方式 |
|---|---|---|---|
| `target/` | 含 debug/release/deps/build/fingerprint/incremental，已观测到 worker 244.6M、desktop 215.1M | 下次 Cargo/Tauri/native 构建重新编译，耗时较长 | `pixi run rust-test`、`pixi run native-develop` 或 `pixi run tauri-build` |
| `build/` | PyInstaller `fast-nc-zarr-worker.pkg` 约 244.5M、`PYZ-00.pyz` 约 25M | sidecar 需重新打包 | `pixi run desktop-sidecar` |
| 根 `dist/` | PyInstaller one-file worker 输出约 245M | 当前已生成的 worker 暂时不可直接运行 | `pixi run desktop-sidecar` |
| `apps/desktop/dist/` | Vite 的 JS/CSS/字体输出，`tauri.conf.json` 的 `frontendDist` 指向它 | Tauri 预览/打包前端暂不可用 | `npm --prefix apps/desktop ci && npm --prefix apps/desktop run build` |
| `apps/desktop/node_modules/` | npm 依赖树和 `.vite` 缓存 | 离线前端开发不可用；重装耗时 | `npm --prefix apps/desktop ci` |
| `apps/desktop/src-tauri/binaries/` | Tauri sidecar 的 target triple 与无后缀副本，`worker.rs` 明确搜索它 | 已构建桌面程序不能直接找到 sidecar | `pixi run desktop-sidecar` |
| `.pixi/` | 完整 Python/conda/native 工具环境，嵌套条目超过 1800 | 首次 `pixi install` 可能耗时很久；无网络时不可恢复 | `pixi install` |

**推荐第一批命令（本清单只给方案，本次未执行）：**

```bash
rm -rf target build dist
rm -rf apps/desktop/node_modules apps/desktop/dist
rm -rf apps/desktop/.vite apps/desktop/.vite-temp
rm -rf apps/desktop/src-tauri/binaries
rm -rf .pixi
```

若当前只是想释放空间、又需要保留开发环境，可以先删前五项，暂时保留 `.pixi/`。

### A2. 删除 Python/pytest 缓存

| 路径 | 证据 | 结论 |
|---|---|---|
| `tests/__pycache__/` | 20 多个 Python 3.13/3.14 `.pyc` | 直接删除 |
| `src/**/__pycache__/` | 各 Python 子包均有字节码缓存 | 直接删除 |
| `.pytest_cache/` | pytest 节点缓存和本地 README | 直接删除 |
| PyInstaller `build/**/localpycs/` | 打包中间字节码 | 随 `build/` 删除 |

可用：

```bash
rm -rf .pytest_cache tests/__pycache__
find src -type d -name __pycache__ -prune -exec rm -rf {} +
```

这些目录已被 `.gitignore` 的 `__pycache__/`、`*.py[cod]` 和 `.pytest_cache/` 覆盖。

### A3. 删除本地发布包，但必须先完成交付确认

`release/` 当前包含：

- `Fast NC Zarr_1.7.1_amd64.deb`，约 303.4M；
- `Fast NC Zarr-1.7.1-1.x86_64.rpm`，约 81.7M；
- `fast-nc-zarr-icon.png`，约 32.7K，tracked。

deb/rpm 已被 `.gitignore` 忽略，并且不在当前推送分支历史中。它们可以在**已上传 GitHub Release/制品库、完成 SHA-256 记录并确认可重新生成**后删除：

```bash
sha256sum release/*.deb release/*.rpm
# 将 checksum 保存到外部发布记录后再执行
rm -f release/*.deb release/*.rpm
```

不要把 `release/` 当作普通缓存直接执行 `rm -rf release`，因为图标目前仍是 tracked 文件，而且 `pixi.toml` 的 `release-candidate` task 会把构建包复制到此处。

---

## 4. B 类：可以删除或合并，但需要先确认策略

### B1. 发布图标只保留一个源文件

证据：

```text
apps/desktop/src-tauri/icons/icon.png
release/fast-nc-zarr-icon.png
```

两者 SHA-256 均为：

```text
7caed0b5ba81bbba515650019e18c658bee3bdba26898a77730d8d70ed5391f2
```

`pixi.toml:62` 明确执行：

```text
cp apps/desktop/src-tauri/icons/icon.png release/fast-nc-zarr-icon.png
```

因此建议：

1. 把 `apps/desktop/src-tauri/icons/icon.png` 定为唯一源资源；
2. 从 Git 中删除 `release/fast-nc-zarr-icon.png`；
3. 将 `release-candidate` 改为只在发布阶段生成外部制品；
4. 发布验证通过后不在仓库工作树保留生成副本。

这是低风险的**源文件/生成文件合并**，但应与一次发布流程验证放在同一变更中，避免发布脚本失效。

### B2. Architecture 图纸去掉重复渲染导出

当前有两组“一个 `.drawio` 源图 + 两个 PNG 导出”：

| 源图 | PNG 1 | PNG 2 | 建议 |
|---|---|---|---|
| `docs/architecture/rust-backend.drawio` | `docs/architecture/rust-backend.png` 约 399.5K | 已删除 `Architecture/architecture-v1.7.0-rust-refactor.drawio.png` 约 401.7K | 保留 `.drawio` + 一个规范命名 PNG |
| `docs/architecture/archive/architecture-v1.6.9-optimization.drawio` | `docs/architecture/archive/architecture-v1.6.9-optimization.png` 约 654.6K | 已删除 `Architecture/architecture-v1.6.9-optimization.drawio.png` 约 303.3K | 保留 `.drawio` + 一个规范命名 PNG |

重复 `.drawio.png` 导出已删除；普通 PNG 和可编辑 `.drawio` 源图保留。需要 PNG 内嵌 XML 的场景可从 `.drawio` 源图重新导出，不再在仓库长期保存第二份导出。

此外，以下旧图目前没有 README/docs 引用：

- `docs/architecture/archive/gui-redesign-v1.6.8.drawio`；
- `docs/architecture/archive/architecture-v1.6.7.drawio`。

上述旧图已移入 `docs/architecture/archive/`，它们不是运行时依赖；后续若确认不再需要设计历史，可再删除。

### B3. 历史架构计划合并/归档

`v1.7.0-rust-refactor-plan.md` 已整理为当前入口 `docs/architecture/rust-backend.md`；`v1.7.0-rust-preparation.md` 已移入 `docs/architecture/archive/`。两者仍保留原始实施决策和历史基线，但不再在仓库根目录并列维护。

当前 Rust 架构入口保留：

- `docs/architecture/rust-backend.md`：方案、能力边界和 fallback 说明；
- `docs/architecture/rust-backend.drawio`：当前方案可编辑图；
- `docs/architecture/archive/v1.7.0-rust-preparation.md`：准备阶段历史记录。

`v1.6.9-optimization-plan.md` 及对应图纸已移入 `docs/architecture/archive/`，避免历史设计稿与当前用户模块文档混排。

### B4. 两份字体和两份 OFL 许可文件

证据：

```text
src/fast_nc_zarr/gui/assets/NotoSansSC-VF.ttf
apps/desktop/public/fonts/NotoSansSC-VF.ttf
```

两份 TTF SHA-256 均为：

```text
a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da
```

两份 `OFL.txt` SHA-256 均为：

```text
1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9
```

当前不能直接合并：

- Python GUI 的 `src/fast_nc_zarr/gui/fonts.py` 使用 `gui/assets/NotoSansSC-VF.ttf`；
- React 的 `apps/desktop/src/styles.css` 使用 `/fonts/NotoSansSC-VF.ttf`；
- Tauri/Vite 构建需要 `public/fonts` 在前端输出中存在；
- Python 包和桌面包的分发边界不同。

后续可选方案：

1. 构建阶段从一个仓库源复制到两个包的临时 staging 目录；
2. 或将字体放进单独的资源包，两个运行时从统一安装位置读取；
3. 运行 Python GUI、Vite build、Tauri bundle 和字体注册 smoke 后，再删除其中一份 tracked TTF/OFL。

在没有完成上述 packaging redesign 前，两份字体都应保留。这里是**高收益但不能现在直接做**的合并项，可一次减少约 17M 的 Git 工作树文件；不会减少两套最终安装包同时需要的内容。

---

## 5. C 类：后续代码优化后才能清理

### C1. 根入口 wrapper

当前根目录有：

- `convert.py` → `fast_nc_zarr.cli`；
- `pipeline.py` → `fast_nc_zarr.pipeline.cli`；
- `resample.py` → `fast_nc_zarr.resampling.cli`；
- `rechunk.py` → `fast_nc_zarr.rechunking.cli`；
- `gui.py` → `fast_nc_zarr.gui.app`。

每个文件只有约十几行，实际功能在 `src/`。`pixi.toml` 主要使用 `python -m ...`，根 README 也推荐 `pixi run`。从纯目录观感看它们是重复入口，但根脚本可能是外部自动化或用户习惯依赖。

建议：

1. 在 README 中明确只支持 `pixi run` / `python -m`；
2. 给 wrapper 增加一个版本周期的 deprecated 提示，或在发行说明中登记；
3. 搜索外部自动化后再批量移除五个 wrapper；
4. 不要把五个 wrapper 合并成一个“万能脚本”，那会增加参数分发和兼容复杂度。

### C2. `gui-legacy` task 与旧 GUI 页面

证据：

- `pixi.toml:61` 仍有 `gui-legacy = "python -m fast_nc_zarr.gui.app"`；
- `gui/main_window.py` 明确保留旧页面对象供 API/自动化兼容；
- `docs/gui.md` 说明 `path_picker.py`、`path_chooser.py` 和旧页面仍有兼容职责；
- `tests/test_path_picker.py` 覆盖 `pathPicker/v1` 到 `v2` 的迁移行为。

不要现在删除：

- `src/fast_nc_zarr/gui/path_chooser.py`；
- `src/fast_nc_zarr/gui/path_picker.py`；
- `gui/main_window.py` 中的 legacy page 对象；
- `pixi.toml` 的 `gui-legacy`。

后续若能确认没有外部调用：

1. 先把 `path_chooser` 和 `path_picker` 的共享数据/选择器逻辑收敛到一个模块；
2. 保留一次性 settings migration；
3. 更新 GUI 测试和文档；
4. 删除旧页面与 `gui-legacy` task；
5. 最后才删除根 `gui.py` wrapper。

### C3. `fast-nc-zarr-worker.spec`

该 PyInstaller spec 存在两个问题：

- `binaries` 写死了本机绝对路径：`/run/media/owen/HDD/zarr-fast-converter-v1/src/...so`；
- 当前 `scripts/build_desktop_sidecar.sh` 直接调用 `pyinstaller --onefile --paths src --add-binary ...`，没有引用该 spec。

因此它是高概率的遗留/重复打包入口，且不可移植。建议：

- 先在 CI 和发布说明中确认没有人通过 `pyinstaller fast-nc-zarr-worker.spec` 构建；
- 将必要参数迁移到 `pyproject.toml` 或构建脚本；
- 删除 `fast-nc-zarr-worker.spec`；
- 只保留一个可跨机器工作的 sidecar 构建入口。

这是适合后续清理的文件，不建议在没有构建验证前直接删除。

### C4. `scripts/build_native.sh`

它只有：

```bash
exec maturin develop --release "$@"
```

而 `pixi.toml` 已提供：

```text
native-develop = "CFLAGS=-std=gnu17 maturin develop --release"
```

当前脚本没有被 README、CI、其他脚本或 Pixi task 引用；并且缺少项目统一的 `CFLAGS`。建议将外部使用迁移到 `pixi run native-develop` 后删除该脚本。不要与 `check_native_environment.py` 合并，环境检查和构建是两个清晰职责。

### C5. Python/Rust/TypeScript 三份协议定义

IPC 协议目前同时存在于：

- `contracts/request-v1.schema.json`、`event-v1.schema.json`、`error-v1.schema.json`；
- `contracts/README.md` 和两个 JSONL/JSON fixture；
- Python `desktop_worker/protocol.py`；
- Tauri Rust `src-tauri/src/protocol.rs`、`error.rs`；
- TypeScript `apps/desktop/src/api.ts` 的 `TaskEvent` 和请求类型。

这是跨语言边界的必要重复，不是立即删除项。风险在于命令、事件、错误枚举变更时人工同步遗漏。

推荐后续方案：

1. 以 `contracts/*.schema.json` 作为唯一 wire contract；
2. 用稳定 codegen 生成 TypeScript 类型和 Rust/Python 常量/验证代码；
3. 保留 `test_contracts.py`、Rust protocol tests 和 desktop worker fixture smoke；
4. 生成文件放入 `contracts/generated/` 并保持 ignored，或明确决定将生成结果纳入 Git；
5. 删除手写的重复枚举前，必须验证三种语言的错误消息、终止事件和未知字段语义一致。

在没有 codegen 和跨语言 smoke 之前，不能删除任何一份协议实现。

### C6. 测试文件不要按“数量多”合并

当前测试按行为边界拆分：

- `test_pipeline.py`：复杂 pipeline 计划、执行、恢复和 CLI；
- `test_rechunking.py`、`test_resampling.py`：各自引擎边界；
- `test_native_smoke.py`：Python/Rust backend 能力、取消、数据一致性；
- `test_gui.py`、`test_path_picker.py`：Qt 页面、路径和兼容迁移；
- `test_contracts.py`、`test_desktop_worker.py`：schema/fixture 与独立 worker 行为；
- 其他文件覆盖时间映射、元数据、替换规则、系统资源、压缩调优等。

`test_contracts.py` 和 `test_desktop_worker.py` 都启动 worker，但前者验证 fixture/contract，后者验证错误和 terminal event；建议只提取测试 helper，不合并整个测试文件。合并测试文件会扩大冲突面、降低失败定位速度，收益很小。

---

## 6. 必须保留的内容

以下目录/文件当前都有实际消费者，不应作为目录瘦身直接删除：

- `src/fast_nc_zarr/`：核心 Python 产品逻辑和公共服务层；
- `rust/crates/`、根 `Cargo.toml`、`Cargo.lock`：native backend workspace；
- `apps/desktop/src/`、`apps/desktop/src-tauri/src/`：React/Tauri 产品实现；
- `apps/desktop/package.json`、`package-lock.json`：前端依赖和确定性安装；
- `contracts/` 与 fixture：Python worker、Tauri worker 和测试共同依赖；
- `tests/`：当前版本的行为和跨 backend 回归边界；
- `pixi.toml`、`pixi.lock`：项目声明的 Python/native 环境唯一来源；
- `pyproject.toml`：maturin/Python 包构建入口；
- `.github/workflows/native-preparation.yml`：native、Python 和桌面检查；
- `.cargo/config.toml`：GCC 16/C23 兼容的 `CFLAGS` 约束；
- `rust-toolchain.toml`、`.nvmrc`：Rust/Node 工具链约束；
- `VERSION`：当前版本文件，但应在后续版本治理中成为唯一来源；
- `docs/README.md` 及六个模块文档：入口和用户行为说明；
- `apps/desktop/src-tauri/icons/`：Tauri bundle 配置直接引用的图标集合；
- 两份 Noto Sans SC 字体：当前分别被 Python GUI 和前端直接引用；
- `Cargo.lock`、`pixi.lock`、`package-lock.json`：不要为了减少文件数删除锁文件。

### 版本治理风险

版本目前至少出现在 `VERSION`、`pixi.toml`、`pyproject.toml`、根 `Cargo.toml`、桌面 `package.json`、Tauri `tauri.conf.json` 和桌面 Cargo manifest。当前值均为 1.7.1，但这仍是维护负担。

后续可选择一个源（推荐 `VERSION` 或 `pyproject.toml`），由发布脚本生成其他 manifest；完成发布流水线和版本一致性测试后，才删除重复定义。不要现在删除任一版本字段。

---

## 7. 文档目录精简方案

当前 `docs/` 是用户文档，`Architecture/` 是设计历史，两者职责不同，不建议把所有 Markdown 粗暴合并成一个超大 README。

推荐目标结构：

```text
docs/
├── README.md                  # 用户文档索引
├── converter.md
├── pipeline.md
├── resampling.md
├── rechunking.md
├── gui.md
├── raw-validation.md
└── architecture/
    ├── rust-backend.md        # 当前 Rust 方案与能力边界
    ├── rust-backend.drawio    # 当前架构可编辑图
    └── archive/               # 历史方案与历史架构图
        ├── v1.7.0-rust-preparation.md
        ├── v1.6.9-optimization-plan.md
        ├── architecture-v1.6.9-optimization.drawio
        ├── architecture-v1.6.9-optimization.png
        ├── architecture-v1.6.7.drawio
        └── gui-redesign-v1.6.8.drawio
```

实施顺序：

1. 修正 `docs/README.md:3` 的过时版本描述；
2. 将当前架构入口和历史资料收敛到 `docs/architecture/`；
3. 更新根 README、模块索引和归档文档中的链接；
4. 删除重复 `.drawio.png` 导出，保留源图和普通预览 PNG；
5. 字体、协议实现和兼容入口仍按后续阶段处理。

这样可以把项目根部的历史设计噪音收进 `docs/architecture/`，又不会丢失对 Rust 重构决策有价值的上下文。

---

## 8. 推荐执行顺序

### Phase 0：已完成

- 提交当前 v1.7.1 Tauri 修改；
- 处理 GitHub 推送因 deb/rpm 大 blob 失败的问题；
- 从未推送历史移除 deb/rpm；
- 推送 `refactor/v1.7.1-tauri`；
- 清理本地 `refs/original` 和 reflog 残留，Git pack 从约 397M 降到约 13.14M；
- 本次审计不改源码、不删除工作树产物。

### Phase 1：低风险释放磁盘

1. 停止正在运行的 Cargo/Pixi/npm/PyInstaller 进程；
2. 保留需要交付的 `release/*.deb`、`release/*.rpm` 和 wheel，记录 checksum；
3. 删除 `target/`、`build/`、根 `dist/`、`apps/desktop/dist/`、`node_modules/`、Tauri binaries；
4. 删除 `.pytest_cache` 和所有 `__pycache__`；
5. 视是否需要离线环境决定是否删除 `.pixi/`；
6. 执行 `git status --short --ignored`，确认没有误删 tracked 文件。

### Phase 2：发布资产治理

1. 将 deb/rpm/wheel 上传到 Release 或制品库；
2. 记录 SHA-256、构建 commit、平台和工具链；
3. 让 CI 直接上传 `target/debug/bundle`，不再把包复制到长期存在的 `release/`；
4. 删除工作树发布包；
5. 删除 `release/fast-nc-zarr-icon.png`，保留 Tauri `icons/icon.png` 源文件。

### Phase 3：文档和资源收敛

1. 统一 Architecture 历史文档目录；
2. 每组架构图只保留 `.drawio` 和一种规范 PNG；
3. 合并 v1.7.0 两份规划文档；
4. 更新 `docs/README.md`、根 README 和链接；
5. 评估两个 TTF/OFL 是否由构建阶段共享；
6. 删除确认不再引用的旧图纸。

### Phase 4：兼容路径收缩

1. 统计根 wrapper 的外部使用；
2. 将 `gui-legacy`、旧 GUI page、path picker migration 标记弃用；
3. 收敛 `path_chooser.py`/`path_picker.py`；
4. 删除未使用的 `build_native.sh` 和硬编码路径的 `.spec`；
5. 删除 wrapper 与 legacy task；
6. 保留一轮完整兼容测试和发布说明。

### Phase 5：协议与版本治理

1. 建立 schema codegen；
2. 让 Python、Rust、TypeScript 共享生成的命令/事件/错误定义；
3. 集中版本源并添加一致性检查；
4. 确认 generated schema 的生成路径；
5. 只在跨语言 smoke 和桌面打包验证通过后删除手写重复定义。

---

## 9. 每阶段验收清单

### 清理缓存后

```bash
git status --short
git check-ignore -v target build dist .pixi apps/desktop/node_modules
# 需要确认的路径不存在时：
test ! -e target
test ! -e build
test ! -e dist
test ! -e apps/desktop/node_modules
```

### 重新建立 Python/native 后

```bash
pixi run native-check
pixi run cross-backend-test
pixi run test
```

### 重新建立桌面前端后

```bash
pixi run desktop-install
pixi run desktop-typecheck
pixi run desktop-build
```

### 重新建立 Tauri/sidecar 后

```bash
pixi run desktop-sidecar
pixi run tauri-build
```

同时确认：

- `apps/desktop/dist/` 被重新生成；
- `apps/desktop/src-tauri/binaries/` 有正确 target triple；
- `apps/desktop/src-tauri/gen/schemas/desktop-schema.json` 可被 capabilities 配置解析；
- worker fixture 仍输出 `accepted → started → finished`；
- deb/rpm bundle 可安装或至少通过文件存在、架构和 checksum 检查。

### 文档/资源收敛后

```bash
git grep -n "Architecture/\|gui-redesign\|v1.7.0-rust-preparation\|release/fast-nc-zarr-icon"
sha256sum src/fast_nc_zarr/gui/assets/NotoSansSC-VF.ttf \
  apps/desktop/public/fonts/NotoSansSC-VF.ttf
```

任何移动/删除操作都必须做到：无断链、无失效构建路径、无丢失的协议 fixture、无破坏的兼容迁移。

---

## 10. 最终建议清单（按优先级）

### 可清理

- [ ] `target/`
- [ ] `build/`
- [ ] 根 `dist/`
- [ ] `apps/desktop/dist/`
- [ ] `apps/desktop/node_modules/`
- [ ] `apps/desktop/.vite/`、`.vite-temp/`
- [ ] `apps/desktop/src-tauri/binaries/`（确认不需要现成 sidecar 后）
- [ ] `.pytest_cache/`
- [ ] 全部 `__pycache__/`、`.pyc`
- [ ] `.pixi/`（确认不需要离线环境后）
- [ ] `release/*.deb`、`release/*.rpm`（发布/校验/备份后）

### 可合并/重组

- [x] `release/fast-nc-zarr-icon.png` 合并回 Tauri icon 生成流程
- [x] 每组 Architecture 图只保留 `.drawio` + 一种 PNG
- [ ] `v1.7.0-rust-preparation.md` 与 `v1.7.0-rust-refactor-plan.md` 合并为一个历史/现状文档
- [x] `Architecture/` 迁入 `docs/architecture/`
- [ ] 两份字体改为一个源文件 + 两个构建 staging 输出
- [ ] 两份 `OFL.txt` 随字体构建复制，不在两个运行时目录各维护一份
- [ ] 协议 schema 作为唯一源，生成 Python/Rust/TypeScript 定义

### 后续优化后可清理

- [ ] `fast-nc-zarr-worker.spec`
- [ ] `scripts/build_native.sh`
- [ ] 根目录五个兼容 wrapper
- [ ] `pixi.toml` 的 `gui-legacy`
- [ ] `gui/main_window.py` 中旧页面对象
- [ ] `gui/path_chooser.py` 与 `gui/path_picker.py` 的重复/兼容层
- [ ] 未引用的 `architecture-v1.6.7.drawio`、`gui-redesign-v1.6.8.drawio`（已归档，待后续删除决策）
- [x] 重复的普通 PNG 或 `.drawio.png` 导出
- [ ] 分散的版本号定义
- [ ] 已确认可自动生成的 `apps/desktop/src-tauri/gen/schemas/`

### 必须保留

- [ ] `src/`、`rust/`、`apps/desktop/src/`、`apps/desktop/src-tauri/src/`
- [ ] `contracts/` 和 fixture
- [ ] `tests/`
- [ ] `Cargo.lock`、`pixi.lock`、`package-lock.json`
- [ ] `pyproject.toml`、`pixi.toml`、Cargo/Tauri/Node 配置
- [ ] 当前被两个运行时引用的字体
- [ ] 当前 Tauri 配置引用的 icon 资源
- [ ] 当前用户文档与模块文档

**最优先的实际动作是 Phase 1：清掉 12G `target`、280M `build`、245M `dist` 和前端缓存；这些动作不改变代码结构，却能立即把工作树从“构建机状态”恢复为“源码仓库状态”。**

---

## 11. 第一次实际执行记录（2026-08-13）

本次已按 A 类低风险项执行实际清理，并完成一项 B 类资源合并：

### 已完成

- [x] 删除 `.pixi/`（约 3.9G，可由 `pixi install` 重建）。
- [x] 删除 `target/`（约 12G，可由 Cargo/Tauri/maturin 重建）。
- [x] 删除 `build/` 和根 `dist/`（PyInstaller 中间/输出产物）。
- [x] 删除 `apps/desktop/node_modules/`、`apps/desktop/dist/`、`.vite` 缓存（可由 `npm ci` 和 Vite build 重建）。
- [x] 删除 `apps/desktop/src-tauri/binaries/`（sidecar 可由 `pixi run desktop-sidecar` 重建）。
- [x] 删除 `.pytest_cache/`、全部已发现的 `__pycache__/` 和 `.pyc` 缓存。
- [x] 删除 `src/fast_nc_zarr/_native*.so` 本地 native 构建产物。
- [x] 删除 tracked 的 `release/fast-nc-zarr-icon.png`；保留 `apps/desktop/src-tauri/icons/icon.png` 唯一源图标。
- [x] 修改 `pixi.toml:release-candidate`，不再把源图标复制为冗余 release 副本。
- [x] 保留 `release/*.deb`、`release/*.rpm`，因为本次没有外部发布、checksum 归档或交付确认。
- [x] 保留 `apps/desktop/src-tauri/gen/schemas/`，因为 `capabilities/default.json` 仍通过 `$schema` 引用其中的 `desktop-schema.json`。

### 本次未执行

- 未移动/删除 `Architecture/` 历史文档和 PNG 导出；需要先确认归档位置与预览/嵌入 XML 策略。
- 未合并两份 Noto Sans SC 字体；Python GUI 与 React/Tauri 当前分别直接读取它们。
- 未删除根入口 wrapper、`gui-legacy`、旧 GUI 页面、`fast-nc-zarr-worker.spec` 或 `scripts/build_native.sh`；这些仍涉及兼容调用或缺少外部使用确认。
- 未改动协议 schema、Python/Rust/TypeScript 协议实现或版本字段；这些需要 codegen/发布流程改造后再处理。

### 清理后观测

- `.pixi/`、`target/`、`build/`、根 `dist/`、桌面 `node_modules/dist/binaries`、`.pytest_cache/` 均已不存在。
- `git clean -ndX` 仅剩 Tauri 生成 schema 目录和未发布的 deb/rpm；没有残留 Python 缓存。
- `release/*.deb`、`release/*.rpm` 仍被 `.gitignore` 覆盖。
- 唯一保留的源图标 SHA-256 为 `7caed0b5ba81bbba515650019e18c658bee3bdba26898a77730d8d70ed5391f2`。
- 本次未运行完整测试套件：清理同时删除了本地 Pixi/编译依赖环境；后续重建环境后按第 9 节命令验证。

---

## 12. 第二次实际执行记录（2026-08-13）

- [x] 将当前 Rust 方案整理到 `docs/architecture/rust-backend.md`。
- [x] 将准备阶段、v1.6.9 优化方案和旧架构图移入 `docs/architecture/archive/`。
- [x] 将根 README 和 `docs/README.md` 的架构入口更新到新路径。
- [x] 为归档方案添加历史状态说明，并修正归档优化方案的图纸相对路径。
- [x] 删除两份重复 `.drawio.png` 导出；保留两个 `.drawio` 源图和普通 PNG 预览。
- [x] 保留两份字体、协议实现、兼容入口和 Tauri 生成 schema，未扩大清理范围。
- [x] `Architecture/` 已不再包含跟踪文件；历史设计内容集中在 `docs/architecture/`。

本阶段未运行完整测试套件；验证重点为 Git 路径、文档链接、图纸源文件和归档结构。
