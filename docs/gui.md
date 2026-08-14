# Tauri 桌面应用

## 定位

桌面应用由 Tauri 2、React 19 和 TypeScript 构成。前端通过 `@tauri-apps/api` 调用 Rust commands；Rust 负责窗口生命周期、任务注册、取消、资源快照、事件流、native capability 和兼容处理路径编排。

源码位置：

```text
apps/desktop/src/              React + TypeScript UI
apps/desktop/src-tauri/src/   Tauri/Rust runtime
apps/desktop/src-tauri/       Tauri 配置、权限和图标
```

## 启动

从项目根目录执行：

```bash
nvm use
npm --prefix apps/desktop ci
pixi run gui
```

`pixi run gui` 调用 `scripts/desktop_dev.sh`，脚本会设置桌面 Rust linker 环境并启动 Tauri dev window。
在 Linux Wayland 会话中，如果存在可用的 X11 display，Tauri 会自动使用 X11 fallback，避免 WebKitGTK 因 Wayland protocol error 无法创建窗口。需要强制选择时可设置 `FAST_NC_ZARR_DISPLAY_BACKEND=x11` 或 `FAST_NC_ZARR_DISPLAY_BACKEND=wayland`；强制 `wayland` 会绕过自动回退，只有当前 compositor 支持所需协议时才应使用。

直接使用 npm：

```bash
npm --prefix apps/desktop run tauri:dev
```

只启动 Vite 浏览器预览：

```bash
npm --prefix apps/desktop run dev
```

浏览器预览不会提供 Tauri commands；界面会进入 preview runtime，文件检查和任务执行需要在 Tauri desktop window 中验证。

## 页面

### 总览

显示最近路径、native capability 数量、活动任务和当前执行策略，并提供数据检查与任务中心快捷入口。

### 数据检查

流程为：

1. 选择原始数据目录或现有 Zarr；
2. 对原始数据检查时间轴和时间规则；
3. 读取完整结构；
4. 查看变量、警告和 capability matrix；
5. 保存检查快照或进入处理流程。

### 处理流程

Pipeline Builder 组织以下阶段：

- 输入检查；
- 空间重采样；
- 重分块；
- 重压缩；
- staging 校验和输出发布。

后端路由只有两种前端可见策略：

- `auto`：按 capability 优先选择 native 操作；
- `rust`：要求选中的操作具备 native capability，否则明确失败。

兼容执行路径不会在界面中伪装成 native 成功，实际路由和原因由 manifest、事件和 capability 记录。

### 任务中心

任务中心显示：

- 运行中、已完成、失败和已取消任务；
- CPU 和可用内存快照；
- manifest 路径；
- 实时事件流；
- 取消操作；
- checkpoint 检查和恢复到新输出。

### 路径设置

路径设置只保存当前桌面用户的 localStorage 数据：

- 当前输入目录；
- 收藏路径；
- 最近目录。

## 前端 API 边界

`apps/desktop/src/api.ts` 只定义 TypeScript 到 Tauri command 的请求和响应类型。前端不直接访问兼容处理服务的进程、stdin/stdout 或内部模块；这些实现细节由 Rust runtime 隔离。

主要 command：

- `get_backend_info`；
- `native_capabilities`；
- `inspect_source`；
- `inspect_zarr`；
- `inspect_time_metadata`；
- `save_inspection_snapshot`；
- `preview_pipeline`；
- `start_pipeline`；
- `resume_pipeline`；
- `inspect_pipeline_recovery`；
- `start_native_task`；
- `list_tasks`；
- `get_task`；
- `cancel_task`。

## 验证

```bash
npm --prefix apps/desktop run typecheck
npm --prefix apps/desktop run build
```

Tauri Rust runtime：

```bash
cargo test -p fast-nc-zarr-desktop
```

## v1.7.3 Linux 发布范围

v1.7.3 当前只把 Linux `x86_64-unknown-linux-gnu` 作为阻塞发布平台。必须分别验证 X11、Wayland、sidecar 和安装包启动；Windows/macOS 构建与运行时验证延期，不影响本轮 Linux release gate。

Linux 发布前至少执行：

```bash
pixi run version-check
pixi run desktop-sidecar
pixi run desktop-typecheck
pixi run desktop-build
pixi run tauri-build
```

安装后的验证必须确认 bundled worker、字体、图标和前端资源均能从发布包路径加载。未完成 native parity 的操作仍通过 Python compatibility worker 执行，实际 backend 和 fallback reason 以 manifest 与 events 为准。
