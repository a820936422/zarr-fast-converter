# Fast NC Zarr

面向批量 NetCDF、HDF 和 TIFF 数据的 Zarr v3 转换工作台。项目采用 Tauri + React + TypeScript 桌面界面、Rust native-first runtime，以及由 capability 控制的兼容数据处理服务。

## 当前架构

```text
React + TypeScript
        ↓ Tauri commands
Rust desktop runtime
        ├── 已声明能力的 Zarr 操作：Rust native
        └── 复杂输入或未通过正确性门的操作：兼容处理服务
```

前端只通过 Tauri commands 访问文件选择、数据检查、处理计划、任务事件和恢复功能。Rust 负责任务注册、取消、资源快照、事件转发和 native capability；兼容服务负责当前尚未完成 native parity 的 NetCDF/HDF/TIFF、CF 元数据、复杂重采样和科学结果校验。

当前 native 能力包括：

- Zarr v3 结构检查；
- Float32/Float64 chunk、region 和数组写入；
- 单变量 Float32/Float64 重分块；
- 有界并行、取消、进度、staging 校验和原子发布；
- capability matrix、manifest 和事件证据。

复杂输入、多变量发布、整数 dtype、fill/scale/CF 属性、复杂 codec、标准 NetCDF native conversion 和规则网格 native resampling 仍由兼容路径负责，并在 capability 中明确记录。

## 安装环境

项目使用 Pixi 管理数据处理和 Rust/Python 构建环境：

```bash
pixi install
```

桌面前端使用 Node 版本由 `.nvmrc` 指定：

```bash
nvm use
npm --prefix apps/desktop ci
```

## 启动桌面应用

唯一的桌面开发入口是 Tauri：

```bash
pixi run gui
```

等价命令：

```bash
bash scripts/desktop_dev.sh
```

也可以直接启动 Tauri CLI：

```bash
npm --prefix apps/desktop run tauri:dev
```

仅查看浏览器前端预览，不启动 Tauri runtime：

```bash
npm --prefix apps/desktop run dev
```

`gui.py`、PySide6 GUI 和旧根目录 GUI wrapper 已移除，不再是项目入口。

## 命令行处理

源数据转换：

```bash
pixi run convert -- \
  --input /path/to/nc \
  --output /path/to/result.zarr \
  --time '[2003, 2010]' \
  --lat '[30, 90]' \
  --lon '[-100, 100]' \
  --variables a1 a2
```

一条龙处理：

```bash
pixi run pipeline -- \
  --input /path/to/source \
  --input-kind raw \
  --output /path/to/result.zarr \
  --rechunk \
  --compression auto
```

现有 Zarr 重采样：

```bash
pixi run resample -- \
  --input /path/to/input.zarr \
  --output /path/to/resampled.zarr \
  --resolution 0.25
```

现有 Zarr 重分块或重压缩：

```bash
pixi run rechunk -- \
  --input /path/to/input.zarr \
  --output /path/to/rechunked.zarr \
  --strategy time \
  --compression balanced
```

真实源数据校验：

```bash
pixi run validate-raw -- --help
```

## 构建与验证

前端类型检查和生产构建：

```bash
pixi run desktop-typecheck
pixi run desktop-build
```

Rust workspace 测试：

```bash
pixi run rust-test
```

完整 Python/兼容路径测试：

```bash
pixi run test
```

构建桌面发布包：

```bash
pixi run desktop-sidecar
pixi run tauri-build
```

## 文档

- [模块文档](docs/README.md)
- [Tauri 桌面应用](docs/gui.md)
- [v1.7.3 开发方案](docs/v1.7.3-development-plan.md)
- [IPC contracts](contracts/README.md)

## 结果安全边界

- 输出固定为 Zarr v3；
- 任务先写入专属 staging 目录；
- 结构、codec、样本和语义校验通过后才原子发布；
- 取消、异常、权限错误和校验失败不能产生伪成功输出；
- manifest、events 和 capability 记录实际执行路径；
- 未通过 native 正确性门的操作不会伪装成 Rust 成功。
