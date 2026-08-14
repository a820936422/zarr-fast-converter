# 项目目录清理清单

> 审计基线：`develop` 工作区，v1.7.3 Linux 收尾阶段。  
> 审计日期：2026-08-14。  
> 目标：减小仓库与工作区噪声，不删除仍承担运行时、兼容路径或发布职责的代码。

## 结论

当前目录不需要继续合并业务模块。主要噪声来自本地构建缓存、桌面打包输出和外部数据 smoke 的临时目录；这些内容应保留在本地生成、由 Git 忽略，不应进入仓库。源码、协议、测试和模块文档已经按职责分层，强行合并会增加耦合。

已执行的本地清理：

- `build/`、`dist/`、`target/`：Rust、PyInstaller、Tauri 构建输出；可随时重建。
- `apps/desktop/node_modules/`、`apps/desktop/dist/`、`apps/desktop/.vite/`：前端依赖与构建输出。
- `apps/desktop/src-tauri/binaries/*`：本机 sidecar；发布时由 CI 重新生成。
- `tests/external_data/inputs/`、`work/`、`logs/`、`manifest.local.json`：外部数据链接、临时 Zarr、日志和本机绝对路径配置。

保留的验证报告：

- `tests/external_data/results/external_sample_report.json`：5 个受控外部样本均通过；该目录继续被 `.gitignore` 忽略，报告只作为本机审计证据，不提交绝对路径或结果大文件。

## 清理决策矩阵

| 路径 | 当前用途 | 决策 | 时机与动作 |
|---|---|---|---|
| `src/fast_nc_zarr/` | Python 公共 API、兼容服务、CLI 和数据处理实现 | 保留 | native parity 完成前不能删除 compatibility worker |
| `rust/crates/` | Rust native model、Zarr、Python 扩展 | 保留 | 与 Python 结果完成 golden 对照后再收缩重复实现 |
| `apps/desktop/` | React/TypeScript 前端和 Tauri runtime | 保留 | 前端、Rust coordinator、sidecar 仍是完整产品链路 |
| `apps/desktop/src-tauri/binaries/` | Tauri 外部 sidecar 输入目录 | 保留目录，清理文件 | 只保留 `.gitkeep`（若需要）；每次发布由 `desktop-sidecar` 生成 |
| `contracts/` | IPC schema、fixture 和协议说明 | 保留 | schema 与 Rust/TypeScript 行为测试必须同步 |
| `tests/` | Python、native、desktop、协议和恢复测试 | 保留 | 仅删除缓存，不删除测试分层 |
| `tests/external_data/` | 受控真实数据 smoke runner 和清单模板 | 保留 runner/template/README | 每次运行后删除 `inputs/work/logs/manifest.local.json`，按需保留 ignored report |
| `scripts/` | 版本、环境、sidecar、桌面构建入口 | 保留 | `build_desktop.sh` 与 `build_desktop_sidecar.sh` 仍是不同发布阶段，不合并 |
| `.github/workflows/` | Linux blocking native/release gate | 保留 | Linux-only 发布闭环稳定后再减少重复步骤 |
| `docs/` | 模块边界、GUI、验证范围和开发计划 | 保留并维护索引 | 各文档有独立入口和边界，不合并成单一长文档 |
| `Cargo.toml`/`Cargo.lock` | Rust workspace 与锁定依赖 | 保留 | 发布构建依赖锁定版本 |
| `pyproject.toml`/`pixi.toml`/`pixi.lock` | Python/Pixi 环境和任务入口 | 保留 | `pixi.toml` 是 CI 与本地统一入口 |
| `apps/desktop/package.json`/`package-lock.json` | 前端依赖与 Tauri 命令 | 保留 | npm lock 必须与 package manifest 同步 |
| `.cargo/`、`rust-toolchain.toml`、`pytest.ini` | 编译器、C 编译选项和测试约束 | 保留 | 是 native 可复现构建门的一部分 |
| `.pixi/` | 本机 Pixi 环境 | 不入 Git，可选清理 | 磁盘紧张时用 `pixi clean` 或删除后重新 `pixi install`；正常开发保留 |
| `target/`、`build/`、`dist/` | 编译/打包缓存 | 不入 Git，已清理 | 发布或验证后可重复删除 |
| `release/` | 发布包暂存目录 | 不入 Git | `.deb`、`.rpm`、AppImage 等只作为 CI artifact 或 GitHub Release asset |
| `__pycache__/`、`.pytest_cache/` | Python 测试缓存 | 不入 Git，可清理 | 测试后删除；不影响源码 |

## 可以合并但不应现在合并

1. **环境检查脚本**：`scripts/check_version_consistency.py` 和 `scripts/check_native_environment.py` 都是轻量检查，可在后续统一成一个 `verify_environment.py`，但当前 CI task 名称清晰，立即合并会破坏调用入口；先保持独立。
2. **桌面构建脚本**：`build_desktop.sh` 负责 Tauri bundle，`build_desktop_sidecar.sh` 负责 Python/native sidecar。两者共享 linker 环境，但执行顺序、失败语义和输出不同；只抽取无状态公共 shell 函数，不合并为一个多模式脚本。
3. **模块文档**：`converter.md`、`pipeline.md`、`resampling.md`、`rechunking.md`、`raw-validation.md` 和 `gui.md` 分别对应用户入口；只在 `docs/README.md` 维护导航，避免重复复制内容。
4. **协议 schema 与 fixture**：`contracts/*.schema.json` 和 `contracts/fixtures/` 必须保持分离；schema 描述约束，fixture 描述可执行样例，不能用一个大 JSON 替代两者。

## 后续优化后才可清理

- `src/fast_nc_zarr/application/desktop_worker/`：只有当所有未支持 native capability 已迁移、真实数据 A/B、失败/取消/恢复证据完整后，才能删除 Python compatibility worker。
- Python 与 Rust 中看似重复的 NetCDF/Zarr/resampling 实现：只有对应 capability 从 fallback 切换到 native 并完成 golden fixture 对照后，才能删除旧实现；不能以“代码重复”为理由提前删除。
- `tests/external_data/run_samples.py`：外部数据挂载测试不应并入普通单元测试；只有 CI 拥有稳定、受控的数据源后才考虑单独的发布 smoke job。
- `apps/desktop/public/fonts/`：字体是桌面包自包含中文渲染的一部分；只有确认所有目标发行环境提供等价字体并完成截图回归后，才能移除。
- `apps/desktop/src-tauri/icons/`：Tauri bundle 使用多尺寸图标；只有确认目标打包器不再需要某一尺寸后，才能裁剪。

## 明确禁止提交的内容

- 外部数据原文件、复制出来的 NetCDF/HDF/TIFF/Zarr。
- `tests/external_data/manifest.local.json` 中的本机绝对路径。
- `target/`、PyInstaller `build/`/`dist/`、前端 `node_modules/` 和 Tauri sidecar 二进制。
- `.deb`、`.rpm`、AppImage、压缩包等发布生成物；这些文件可能超过 Git 托管单文件限制，应上传 CI artifact 或 Release asset。
- Python 缓存、pytest 缓存、临时日志和状态文件。

## 执行顺序

1. 每次 smoke/build 完成后删除生成目录，保留需要审计的 JSON 报告。
2. 提交前运行 `git status --short --ignored`，确认只剩源码、配置、文档和测试。
3. 运行 `pixi run version-check`、`pixi run native-check`、前端检查和 Rust/Python 测试。
4. Linux 发布只由 CI 生成 `.deb`/`.rpm`，在 CI 中检查包内 worker、字体、图标和前端资源，再上传 artifact。
5. 任何删除兼容代码的提交必须同时删除对应配置、文档、fixture 和测试；否则只执行本地生成物清理，不动源码目录。
