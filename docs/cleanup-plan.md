# 项目目录精简与清理清单

> 状态：当前版本报告与后续执行清单
>
> 更新日期：2026-08-13
>
> 分支：`refactor/v1.7.2-native-first`
>
> 审计基线：`a8e231f`（`refactor/v1.7.1-tauri`）
>
> 当前版本：`1.7.2`

本文件只保留当前真实状态、已经完成的治理工作和仍然可执行的清理项。v1.7.2 Native-first
开发已完成 P0 capability contract 地基，但默认 Python backend、复杂格式 fallback 和
PySide6 legacy 路径仍保留；开发细节见 [`v1.7.2-development-plan.md`](v1.7.2-development-plan.md)。

## 1. 结论

项目源码目录已经完成第一轮结构收敛，当前“目录很大”的主要原因是**本地重建后的 ignored 构建环境和发布包**，不是 tracked 源码失控：

- Git 仅跟踪 178 个文件；`.git` pack 约 13.14 MiB。
- 根目录旧的 `Architecture/` 已移除，当前架构文档集中在 `docs/architecture/`。
- `target/`、`.pixi/`、PyInstaller、Vite、npm 和 Tauri sidecar 产物均已加入忽略规则，可按需删除并重建。
- `release/` 中的 deb/rpm 是本地发布缓存，不在 Git 跟踪范围，也不在当前分支历史中；它们仍需完成外部发布或备份后才能删除。
- Python GUI 和 React/Tauri 仍分别需要一份 Noto Sans SC 字体；当前不能仅凭内容相同就删除其中一份。
- 根入口 wrapper、`gui-legacy`、旧 GUI 兼容层以及三种语言的协议定义仍有消费者或兼容职责，不能作为目录瘦身直接删除。

**下一步优先级：**需要释放磁盘时删除可再生构建树；完成发布资产外部归档后删除本地包；之后再处理字体 staging、协议 codegen、兼容入口和任务别名。

## 2. 当前仓库状态

### 2.1 Git 和目录证据

| 项目 | 当前值 | 结论 |
|---|---:|---|
| 分支 | `refactor/v1.7.1-tauri` | 本地与远端同名分支一致 |
| HEAD | `c2d4b8b` | 最新治理收尾提交 |
| Git 跟踪文件 | 178 | 源码、测试、配置、文档、契约和资源 |
| Git pack | 约 13.14 MiB | deb/rpm 大 blob 已不在当前历史 |
| Git 跟踪的 `release/` 文件 | 0 | 发布包不是仓库内容 |
| 当前版本 | `1.7.1` | `pixi run version-check` 已通过 |
| 根 `Architecture/` | 不存在 | 历史架构内容已归档到 `docs/architecture/` |
| 根 wrapper | 5 个 | 仍需兼容性评估，不应直接删除 |

当前 tracked 文件按主要职责分布：

| 路径 | 文件数 | 职责 |
|---|---:|---|
| `src/` | 65 | Python 产品逻辑、GUI、处理引擎和 desktop worker |
| `apps/` | 43 | React/Vite/Tauri 桌面应用 |
| `tests/` | 18 | Python、GUI、native、协议和 worker 行为测试 |
| `docs/` | 17 | 用户文档、当前架构和历史归档 |
| `contracts/` | 6 | IPC schema、协议说明和 fixture |
| `rust/` | 6 | 核心 Rust crate |
| `scripts/` | 5 | 环境检查、版本检查、sidecar 和 Tauri 构建入口 |

### 2.2 当前工作树体积

以下是本次重建验证后工作树中的实际观测值。所有构建树均为 ignored；体积会随编译缓存和平台变化。

| 路径 | 约占用 | 当前状态 | 处理建议 |
|---|---:|---|---|
| `.pixi/` | 3.9G | ignored，完整 Python/conda 环境 | 离线开发需要时保留；否则可重建后删除 |
| `target/` | 7.6G | ignored，Cargo/Tauri/maturin 构建树 | 需要释放空间时第一批删除 |
| `build/` | 559M | ignored，PyInstaller 中间产物 | 可直接删除 |
| `dist/` | 245M | ignored，PyInstaller sidecar 输出 | 可直接删除 |
| `apps/desktop/node_modules/` | 92M | ignored，npm 依赖树 | 可由 `npm ci` 重建 |
| `apps/desktop/dist/` | 18M | ignored，Vite 输出 | 可由前端 build 重建 |
| `apps/desktop/src-tauri/binaries/` | 245M | ignored，target-specific sidecar | 可由 sidecar task 重建 |
| `apps/desktop/src-tauri/gen/schemas/` | 316K | ignored，Tauri 生成 schema | 与 Tauri 构建一起清理和重建 |
| `release/` | 386M | ignored，本地 deb/rpm 发布缓存 | 外部归档前保留，不能盲删 |
| `src/fast_nc_zarr/gui/assets/` | 17M | tracked，Python GUI 字体和许可 | 当前必须保留 |
| `apps/desktop/public/fonts/` | 17M | tracked，桌面前端字体和许可 | 当前必须保留 |
| `.git/` | 14M | Git 元数据 | 不处理 |

当前 `release/` 包的本地 SHA-256 记录如下；这不是外部发布凭证，删除前仍须确认已上传或备份：

```text
8c2c1f96b5592b131467cd24a3878dd2f81651a981fb17a053501128fd7289b0  release/Fast NC Zarr-1.7.1-1.x86_64.rpm
cebe0db00dd7d562bbc85ab60fb6dca0adb2eab0a6e27ff7f57759fe8d216333  release/Fast NC Zarr_1.7.1_amd64.deb
```

## 3. 已完成的清理与治理

### 3.1 目录和资源收敛

- [x] 将当前 Rust 架构入口整理为 `docs/architecture/rust-backend.md`。
- [x] 将历史方案和旧架构图移入 `docs/architecture/archive/`。
- [x] 删除根目录 `Architecture/` 的重复组织方式。
- [x] 删除重复的 `.drawio.png` 导出；每组当前图纸保留可编辑源图和一种规范预览。
- [x] 删除 `release/fast-nc-zarr-icon.png` 这个重复图标；`apps/desktop/src-tauri/icons/` 是唯一图标源。
- [x] 修改 `release-candidate`，不再把源图标复制为长期存在的 release 副本。
- [x] 将 deb、rpm、AppImage、压缩包、dmg、msi 和 wheel 加入 `release/` 忽略规则。
- [x] 清理历史提交中的大型发布 blob；当前分支可正常推送。

### 3.2 构建入口收敛

- [x] 删除含本机绝对路径的 `fast-nc-zarr-worker.spec`。
- [x] 删除旧的 `scripts/build_native.sh`。
- [x] 由 `scripts/build_desktop_sidecar.sh` 统一管理 PyInstaller 参数、workpath、specpath 和 distpath。
- [x] PyInstaller 临时 spec 放入 ignored 的 `build/fast-nc-zarr-worker/`，不会重新污染项目根目录。
- [x] 使用 `pixi run native-develop` 作为 native 开发构建入口，并统一 `--skip-install`。
- [x] 使用 `scripts/build_desktop.sh` 作为 Tauri 构建入口。
- [x] Linux Tauri 构建显式使用宿主 GTK/WebKit linker，避免误用 Pixi conda linker。
- [x] 在缺少 `rpmbuild` 时默认构建 deb；显式请求 RPM 时立即报告缺少工具，不再长时间挂起。
- [x] 增加 `scripts/check_version_consistency.py` 和 `pixi run version-check`。

### 3.3 测试与 CI 治理

- [x] `rust-test` 和 `rust-clippy` 排除 Tauri 桌面壳，只验证核心 Rust workspace。
- [x] CI 的 `native-preparation` job 不再调用桌面 Node/sidecar 步骤。
- [x] `desktop-release` job 的顺序固定为：Linux 系统库 → Pixi sidecar → Node/npm → 前端 → Tauri。
- [x] 桌面 CI 配置包含 Linux、Windows 和 macOS 检查矩阵。
- [x] 删除没有测试文件支撑的桌面 Vitest/`desktop-test` 入口和依赖。
- [x] 保留 Python、Rust、native、协议和桌面 worker 的行为测试，不按文件数量粗暴合并。

## 4. A 类：现在可以清理的内容

A 类只包含不改变 tracked 源码的 ignored 产物。删除前确认没有正在运行的构建、调试会话或需要离线使用的环境。

### A1. 构建树和缓存

可按需执行：

```bash
rm -rf target build dist
rm -rf apps/desktop/node_modules apps/desktop/dist
rm -rf apps/desktop/src-tauri/binaries
rm -rf .pytest_cache
find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
find src tests -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
rm -f src/fast_nc_zarr/_native*.so
```

这组删除的重建入口：

| 删除内容 | 重建命令 |
|---|---|
| `.pixi/` | `pixi install` |
| `target/` | `pixi run rust-test`、`pixi run native-develop` 或 `pixi run tauri-build` |
| `build/`、`dist/` | `pixi run desktop-sidecar` |
| `node_modules/` | `npm --prefix apps/desktop ci` |
| `apps/desktop/dist/` | `npm --prefix apps/desktop run build` |
| Tauri sidecar | `pixi run desktop-sidecar` |
| Python/native 缓存 | 对应测试或 native task |

`.pixi/` 是唯一需要单独决策的目录：它占用约 3.9G，删除后必须重新下载或恢复整个环境。需要离线开发时不要删除。

### A2. Tauri 生成 schema

`apps/desktop/src-tauri/gen/schemas/` 被 `.gitignore` 忽略，且 `capabilities/default.json` 的 `$schema` 指向其中的 `desktop-schema.json`。它是可生成文件，但不应和普通缓存无条件混删：

1. 删除前确认 Tauri CLI 可用；
2. 删除 `node_modules` 时可一并删除；
3. 运行 `npm --prefix apps/desktop ci` 和 `npm --prefix apps/desktop exec tauri info` 或 Tauri build；
4. 确认 `capabilities/default.json` 仍能解析后再结束清理。

### A3. 本地发布包

当前 `release/` 中只有未跟踪、被忽略的 deb/rpm。只有在以下条件全部满足后才能删除：

- 已上传 GitHub Release 或其他制品库；
- 已保存上面的 SHA-256、构建提交和目标平台信息；
- 已确认可由 CI 或本地命令重建；
- 不再需要本机直接安装测试。

仅删除包，不要删除图标源或整个目录：

```bash
rm -f release/*.deb release/*.rpm
```

## 5. B 类：可以合并，但需要设计和验证

### B1. 两份字体改为一个源文件

当前重复资源：

```text
src/fast_nc_zarr/gui/assets/NotoSansSC-VF.ttf
src/fast_nc_zarr/gui/assets/OFL.txt
apps/desktop/public/fonts/NotoSansSC-VF.ttf
apps/desktop/public/fonts/OFL.txt
```

两套资源内容相同，但消费者不同：

- Python GUI 的 `src/fast_nc_zarr/gui/fonts.py` 直接读取 `gui/assets`；
- React 的 `apps/desktop/src/styles.css` 通过 `/fonts/NotoSansSC-VF.ttf` 读取 `public/fonts`；
- Python 包和 Tauri 桌面包有不同的分发边界。

推荐方案：选一份 canonical source，在 Python wheel 和 Vite/Tauri build 阶段分别 staging 到运行时需要的位置。必须先验证 Python GUI 启动、Vite build、Tauri bundle 和字体许可文件，再删除第二份 tracked 资源。当前阶段不要直接删除任一字体目录。

### B2. 协议定义收敛为 schema + codegen

当前协议边界存在于：

- `contracts/request-v1.schema.json`、`event-v1.schema.json`、`error-v1.schema.json`；
- `contracts/README.md` 和 fixtures；
- Python `desktop_worker/protocol.py`；
- Tauri Rust 的 `protocol.rs`、`error.rs`；
- TypeScript API 类型。

这些重复不是无用文件，而是跨语言边界的手写同步风险。后续应：

1. 将 `contracts/*.schema.json` 定为 wire contract；
2. 选择稳定 codegen 工具，生成 TypeScript 类型以及 Rust/Python 常量或验证代码；
3. 保留现有 fixture 和跨 backend smoke；
4. 明确 `contracts/generated/` 是 ignored 生成物还是纳入 Git；
5. 在三种语言的未知字段、错误枚举、终止事件和版本语义一致后，才删除手写重复定义。

### B3. 构建 task 别名清理

当前 `pixi.toml` 中以下任务存在潜在重复或未使用风险：

- `desktop-sidecar` 与 `desktop-sidecar-target` 当前命令文本相同；
- `desktop-release-check` 当前没有 CI 调用；
- `gui` 与 `gui-legacy` 指向同一个 Python GUI 模块，但可能分别服务于外部调用者。

先用仓库外部调用记录和发布脚本确认使用者，再删除别名。不要为了减少三行配置而破坏已有自动化入口。

## 6. C 类：完成代码优化后才能清理

### C1. 根目录五个 wrapper

当前文件：

```text
convert.py
pipeline.py
resample.py
rechunk.py
gui.py
```

它们只是转发到 `src/fast_nc_zarr/` 的入口。建议顺序：

1. README 明确 `pixi run` 和 `python -m` 是正式入口；
2. 检查外部脚本、用户习惯和发行包是否调用 wrapper；
3. 如需弃用，先保留一个版本周期的提示；
4. 通过兼容测试后一次性删除。

不要把五个入口合并为一个万能脚本；这会增加参数分发和兼容复杂度。

### C2. 旧 GUI 兼容层

暂时保留：

- `pixi.toml` 的 `gui-legacy`；
- `src/fast_nc_zarr/gui/main_window.py` 中的 legacy page 对象；
- `path_chooser.py`、`path_picker.py`；
- `tests/test_path_picker.py` 覆盖的 settings/path picker migration。

后续顺序：收敛选择器共享逻辑 → 保留一次性 migration → 更新测试和文档 → 删除旧 page/task → 最后删除根 `gui.py` wrapper。

### C3. 历史架构资料

历史资料已归档到 `docs/architecture/archive/`。当前不再有根目录架构文件堆积。以下文件是否删除，取决于是否需要保留设计决策历史：

- `architecture-v1.6.7.drawio`；
- `gui-redesign-v1.6.8.drawio`；
- `v1.6.9-optimization-plan.md` 及对应图纸；
- `v1.7.0-rust-preparation.md`。

删除前检查文档链接、issue/PR 引用和设计追溯需求。归档不是删除；目前保留是低成本的历史记录策略。

### C4. 分散的版本字段

`VERSION` 已作为检查基准，`check_version_consistency.py` 已验证以下 manifest 与运行时版本一致：

- Pixi；
- Python project/runtime；
- 根 Cargo workspace；
- desktop npm/Tauri/Cargo；
- Tauri Rust runtime。

这解决了“版本漂移不可见”的问题，但没有消除各 manifest 中的重复字段。后续可让发布脚本从 `VERSION` 生成其他版本字段；在发布流水线完成前不要删除任何字段。

### C5. Tauri 生成目录和未使用 Rust 符号

- `apps/desktop/src-tauri/gen/schemas/` 当前是可再生文件，但被 capabilities schema 引用；确认 Tauri CLI 每次构建都会恢复后，才考虑完全不保留本地目录。
- Tauri release 编译仍有 5 个未使用 Rust enum/function 警告。它们不阻塞构建，但应先定位是否为预留协议或未接入命令，再删除无用符号。

## 7. 必须保留的内容

以下内容目前有明确消费者、构建职责或回归价值，不应为了减少文件数量删除：

- `src/fast_nc_zarr/`：核心 Python 产品代码；
- `rust/`、根 `Cargo.toml`、`Cargo.lock`：native backend workspace；
- `apps/desktop/src/`、`apps/desktop/src-tauri/src/`：React/Tauri 实现；
- `apps/desktop/package.json`、`package-lock.json`：确定性前端安装；
- `contracts/` 及 fixtures：IPC wire contract 和 worker 测试边界；
- `tests/`：当前行为、兼容和跨 backend 回归；
- `pixi.toml`、`pixi.lock`、`pyproject.toml`：环境和 Python/native 构建入口；
- `.github/workflows/native-preparation.yml`：核心检查和桌面发布检查；
- `.cargo/config.toml`、`rust-toolchain.toml`、`.nvmrc`：工具链和编译约束；
- `VERSION`：版本一致性检查的 canonical value；
- `apps/desktop/src-tauri/icons/`：Tauri bundle 直接引用的图标资源；
- 两份 Noto Sans SC 字体及两份 OFL：当前分别被 Python 和前端直接读取；
- `docs/` 用户文档和 `docs/architecture/` 当前/历史架构资料；
- `Cargo.lock`、`pixi.lock`、`package-lock.json`：锁文件不能为了“目录精简”删除。

## 8. 推荐执行顺序

### P0：需要释放本机磁盘时

1. 停止 Cargo、Pixi、npm 和 PyInstaller 进程；
2. 保留需要离线开发的 `.pixi/`，否则删除；
3. 删除 `target/`、`build/`、`dist/`、前端缓存、sidecar、Python cache 和 native `.so`；
4. 重新运行 `git status --short --ignored`，确认没有 tracked 文件被误删；
5. 只在发布包外部归档完成后删除 `release/*.deb` 和 `release/*.rpm`。

### P1：发布资产治理

1. 让 CI 直接上传 Tauri bundle，不把大包长期放在工作树；
2. 保存构建提交、目标平台、工具链和 SHA-256；
3. 将 `release-candidate` 限定为本地/发布阶段 staging；
4. 确认外部制品可下载后删除本地包；
5. 保持 Tauri icon 目录为唯一图标源。

### P2：源码和资源收敛

1. 设计字体 canonical source + 双端 staging；
2. 建立 schema codegen 和跨语言 fixture 验证；
3. 清理 `desktop-sidecar-target`、`desktop-release-check` 等确认无外部使用的 task；
4. 评估根 wrapper 和旧 GUI 兼容层的弃用周期；
5. 清理确认不再需要的历史架构图。

### P3：质量收尾

1. 处理 Tauri 未使用符号警告；
2. 在 Windows/macOS runner 上完成真实桌面构建或明确 sidecar 发布策略；
3. 将版本生成和发布上传流程接入 CI；
4. 删除已经由 codegen、发布脚本或 staging 完全替代的手写/副本文件。

## 9. 验收命令

### 9.1 清理后确认

```bash
git status --short
git check-ignore -v target build dist .pixi \
  apps/desktop/node_modules apps/desktop/dist \
  apps/desktop/src-tauri/binaries
```

需要完全删除时再确认：

```bash
test ! -e target
test ! -e build
test ! -e dist
test ! -e apps/desktop/node_modules
test ! -e apps/desktop/src-tauri/binaries
```

不要用 `git clean -fdX` 代替分项清理；它会同时删除 `.pixi/`、发布包和所有其他 ignored 产物。

### 9.2 Python/native/Rust

```bash
pixi install
pixi run version-check
pixi run native-check
pixi run cross-backend-test
pixi run test
pixi run rust-test
pixi run rust-clippy
```

### 9.3 前端/Tauri

```bash
npm --prefix apps/desktop ci
pixi run desktop-typecheck
pixi run desktop-build
pixi run desktop-sidecar
pixi run tauri-build
```

同时检查：

- `apps/desktop/dist/` 已生成；
- sidecar 文件名包含正确 target triple；
- `capabilities/default.json` 的 schema 可解析；
- worker fixture 仍输出 `accepted → started → finished`；
- Linux deb 存在且架构、版本和 SHA-256 正确；
- RPM 仅在安装 `rpmbuild` 的环境中验证。

### 9.4 文档和资源变更

```bash
git grep -n "Architecture/\|fast-nc-zarr-worker.spec\|scripts/build_native.sh"
git grep -n "release/fast-nc-zarr-icon"
sha256sum src/fast_nc_zarr/gui/assets/NotoSansSC-VF.ttf \
  apps/desktop/public/fonts/NotoSansSC-VF.ttf
```

任何移动、删除或合并必须同时满足：无断链、无失效构建路径、无丢失协议 fixture、无破坏兼容迁移、无遗漏许可文件。

## 10. 当前验证结果和边界

已验证：

- Python 全量回归：`188 passed`、`29 subtests passed`、`1 warning`；
- Rust 核心测试：`1 passed`；
- Rust 核心 clippy：通过；
- native 跨 backend smoke：`12 passed`；
- `pixi run version-check`：通过，版本 `1.7.1`；
- 桌面 TypeScript typecheck：通过；
- Vite production build：通过；
- PyInstaller sidecar：target-specific worker 和协议 fixture 通过；
- Linux Tauri deb 构建：通过；
- CI workflow 的 job 分离和步骤顺序：已检查。

边界：

- Windows/macOS 当前只在 CI 矩阵中配置检查，未在本机交叉构建；
- RPM 构建依赖 `rpmbuild`，本机最新 governed build 只验证 deb；
- 两份字体、协议重复定义、根 wrapper、GUI legacy 层和历史架构资料尚未删除；
- 当前工作树仍保留重建产物，这是本地开发缓存，不代表它们应进入 Git。

**当前最安全、收益最高的动作是：按需删除 ignored 构建树；不要继续删除 tracked 源码。**
