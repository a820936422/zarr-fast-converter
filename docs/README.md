# 模块文档

本文档目录按可执行模块组织，描述当前 v1.7.3 功能、使用方式和边界，以及 v1.7.3 的正式开发方案。项目总览、环境安装和快速示例见根目录 [README](../README.md)。

| 模块 | 作用 | 入口 |
|---|---|---|
| [源数据转换](converter.md) | 检查批量 NC/HDF/TIFF，恢复日级时间轴并转换为 Zarr v3 | `pixi run convert` |
| [一条龙处理](pipeline.md) | 组合转换、重采样、重分块和重压缩，只发布一个最终 Zarr | `pixi run pipeline` |
| [空间重采样](resampling.md) | 使用 xESMF 对规则经纬网格 Zarr 进行流式重采样 | `pixi run resample` |
| [重分块与重压缩](rechunking.md) | 调整 Zarr v3 chunks 和无损 codec | `pixi run rechunk` |
| [Tauri 桌面应用](gui.md) | React + TypeScript 工作台、Tauri commands、任务中心和恢复 | `pixi run gui` |
| [真实源数据校验](raw-validation.md) | 批量检查数据集目录，抽样源值并可执行小范围转换 | `pixi run validate-raw` |

## 共通约定

- 环境和命令统一由 `pixi.toml`、`pixi.lock` 管理。
- 最终产品固定为 Zarr v3；标准维度名为 `time`、`lat`、`lon`。
- 时间坐标统一为日精度日期，当前不支持小时、分钟或秒级时间轴。
- 写入模块先生成任务专属临时 store，校验通过后再发布；普通非空目录不会被当作 Zarr 删除。
- `--dry-run` 只检查和规划，不写数据；生产处理建议保留默认校验。
- 桌面应用内置 Noto Sans SC 字体，不依赖宿主环境的中文字体。

## 建议阅读顺序

首次处理源文件：先读 [源数据转换](converter.md)，再读 [一条龙处理](pipeline.md)。只处理现有 Zarr 时，直接阅读 [空间重采样](resampling.md) 和 [重分块与重压缩](rechunking.md)。批量接入真实数据前，使用 [真实源数据校验](raw-validation.md) 建立兼容性报告。

## 当前架构

- [v1.7.3 开发方案](v1.7.3-development-plan.md)
- [IPC contracts](../contracts/README.md)
