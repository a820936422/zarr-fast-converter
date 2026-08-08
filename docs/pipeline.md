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
- 最终 chunks 和 codec 会优先融合到转换器或重采样器；只有布局或变量不兼容时才执行独立最终化。
- 源网格与目标网格完全一致且没有替换规则时，重采样请求可作为 no-op 满足。

## 规划与执行流程

1. 复用已完成的源数据或 Zarr 检查结果。
2. 根据用户范围和分辨率建立目标网格。
3. 重采样时按方法计算带 halo 的连续源读取窗口，避免边界像元误差。
4. 统一计算转换 chunks、最终 chunks、codec 和每项操作的 disposition。
5. 创建任务目录和原子写入的 `manifest.json`。
6. 执行转换；需要时执行 xESMF 重采样。
7. 对重采样结果做有界局部数学抽样验证。
8. 仅在必要时执行兼容性最终化。
9. 执行语义抽样检查并发布最终 Zarr。

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
  --temporary-dir /fast-ssd/pipeline-temp \
  --inspection-cache /data/cache/inspection.json
```

处理现有 Zarr：

```bash
pixi run pipeline -- \
  --input /data/input.zarr --input-kind zarr \
  --output /data/output.zarr \
  --resample --resolution 0.25 \
  --rechunk --recompress
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

GUI 通常通过 `fast_nc_zarr.application.services.preview_pipeline` 和 `run_pipeline` 调用。

## Manifest 与恢复

每个任务目录包含 schema version 5 的 `manifest.json`，记录：

- 请求操作、操作决策和实际物理阶段；
- 源读取窗口、目标 shape、最终布局和压缩配置；
- conversion/resampling 检查点；
- 阶段状态、耗时、错误和恢复历史；
- 最终逻辑字节、临时写入、写放大及避免的最终化 I/O。

失败或取消时保留临时目录。恢复模块验证 manifest、Zarr v3、维度、变量和检查点状态后，从最近的有效阶段继续；成功发布后才按清理策略删除上游临时 store。

## 安全与限制

- 用户最终输出在校验成功前不会被部分结果替换。
- `--cleanup-intermediate` 只清理已有有效下游副本的中间 store。
- 超出源覆盖范围不会外推：重采样目标格点保持缺测；未重采样时超出部分不出现在输出中。
- 目前只接受可标准化为一维规则经纬网格的三维数值变量。
- 不提供多来源拼接、曲线/非规则网格、投影重投影、时间聚合或分类变量专用重采样。
