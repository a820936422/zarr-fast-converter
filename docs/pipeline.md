# 一条龙处理模块

## 定位

`fast_nc_zarr.pipeline` 把原始数据转换、空间重采样、重分块和重压缩组合成一个写入最少的计划，并只发布一份最终 Zarr。输入既可以是 NC/HDF/TIFF 源目录，也可以是现有 Zarr v3 或失败任务的临时检查点。

入口：

```bash
pixi run pipeline -- --help
python -m fast_nc_zarr.pipeline --help
```

## 操作语义

- 原始数据输入必定执行转换。
- 现有 Zarr 输入跳过转换，并要求至少选择 `--resample`、`--rechunk` 或 `--recompress` 之一。
- `--resample`、`--rechunk`、`--recompress` 表达最终产品意图，不保证各自形成独立物理阶段。
- 最终 chunks 和明确 codec 会优先融合到转换器或重采样器；只有布局、变量、物理 chunk ownership 不兼容，或请求真实样本自动压缩时才执行独立最终化。
- 源网格与目标网格完全一致且没有替换规则时，重采样请求可作为 no-op 满足；float32 坐标使用受像元比例硬上限约束的 ULP-aware 比较，不把量化误差误判为新网格。

## 规划与执行流程

1. 复用已完成的源数据或 Zarr 检查结果。
2. 根据用户范围和分辨率建立目标网格。
3. 重采样时按方法计算带 halo 的连续源读取窗口，避免边界像元误差。
4. 统一计算转换 chunks、最终 chunks、codec 和每项操作的 disposition。
5. 在直接下推最终布局前验证任务边界不会切穿物理 Zarr chunk；不安全时自动保留最终化阶段。
6. `workers=auto` 时，最终化阶段 1 用互不重叠的真实源 chunk、阶段 2 用最终物理 chunk 对齐 region，在各自真实文件系统上独立实测安全候选。
7. 自动压缩在代表性 begin/middle/end 数据上比较受控无损候选，校验逐值一致性并按 speed/balanced/compact 目标从 Pareto 前沿选择。
8. 创建任务目录和原子写入的 `manifest.json`。
9. 执行转换；需要时执行 xESMF 重采样。
10. 对重采样结果做有界局部数学抽样验证。
11. 仅在必要时执行兼容性最终化。
12. 执行语义抽样检查并发布最终 Zarr。

## 常用命令

原始数据转换、重采样和存储优化：

```bash
pixi run pipeline -- \
  --input /data/raw \
  --input-kind raw \
  --output /data/product.zarr \
  --lat 30 90 --lon -180 180 \
  --resample --resolution 0.1 \
  --method conservative --skipna \
  --rechunk --strategy time \
  --recompress --compression-codec blosc-zstd \
  --compression-level 4 \
  --backend auto \
  --temporary-dir /fast-ssd/pipeline-temp \
  --inspection-cache /data/cache/inspection.json
```

`--backend python` 强制现有 Python 路径；`--backend auto` 仅在最终化阶段满足
Rust 能力矩阵时尝试 Rust，否则回退 Python；`--backend rust` 在请求的阶段不支持
时明确失败。Rust 当前不替换源转换、xESMF 重采样或 `--compression auto` 调优。

处理现有 Zarr：

```bash
pixi run pipeline -- \
  --input /data/input.zarr --input-kind zarr \
  --output /data/output.zarr \
  --resample --resolution 0.25 \
  --rechunk --recompress --backend auto
```

只生成计划：

```bash
pixi run pipeline -- \
  --input /data/raw --input-kind raw \
  --output /data/product.zarr --resample --resolution 0.1 --dry-run
```

值替换规则通过 `--before-conditions/--before-results` 和 `--after-conditions/--after-results` 成对提供；统计表达式由 `--statistics-policy auto|sample|exact` 控制。

## Python 入口

配置模型位于 `fast_nc_zarr.pipeline.models`，核心入口为：

- `build_pipeline_plan(inspection, config)`
- `preview_pipeline(inspection, config)`
- `run_pipeline(inspection, config)`

桌面应用通过 `fast_nc_zarr.application.services.preview_pipeline` 和 `run_pipeline` 调用。

## Manifest 与恢复

每个任务目录包含 schema version 6 的 `manifest.json`，记录：

- 请求操作、操作决策和实际物理阶段；
- 统一 `EffectiveResourceBudget`、CPU/内存/cgroup/WSL 与源、临时、输出存储证据及置信度；
- 临时目录和输出目录写入预检结果、源读取窗口、目标 shape、最终布局和运行时选定压缩配置；
- conversion/resampling 检查点、stage1/stage2 worker 候选、吞吐、RSS、失败与选择原因；
- 压缩候选的写入、durable、冷热读取、体积、Pareto 与无损验证结果；
- 每阶段的 candidate trials、selection、resolved plan、per-worker/aggregate 内存语义、状态、耗时、错误、恢复历史、临时写入和写放大。

恢复模块要求 schema version 6；旧版 v5 清单必须显式迁移，不能被静默解释为新版。

失败或取消时保留临时目录。恢复模块验证 manifest、Zarr v3、维度、变量和检查点状态后，从最近的有效阶段继续；成功发布后才按清理策略删除上游临时 store。

## 安全与限制

- 用户最终输出在校验成功前不会被部分结果替换。
- `--cleanup-intermediate` 只清理已有有效下游副本的中间 store。
- 超出源覆盖范围不会外推：重采样目标格点保持缺测；未重采样时超出部分不出现在输出中。
- 目前只接受可标准化为一维规则经纬网格的三维数值变量。
- 不提供多来源拼接、曲线/非规则网格、投影重投影、时间聚合或分类变量专用重采样。
