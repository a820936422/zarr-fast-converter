# 重分块与重压缩模块

## 定位

`fast_nc_zarr.rechunking` 调整现有 Zarr v3 的 chunks，并可同时应用无损压缩 codec。模块根据目标访问模式、变量 dtype、可用内存和磁盘关系规划有界的两阶段流式写入。

入口：

```bash
pixi run rechunk -- --help
python -m fast_nc_zarr.rechunking --help
```

## 输入约束

- 输入必须是 Zarr v3，并包含完整 `time`、`lat`、`lon` 维度。
- 非标量数据变量必须包含这三个维度；变量维度顺序可以不同。
- 支持无损 codec：Blosc Zstd/LZ4/LZ4HC/Zlib、原生 Zstd 和 Gzip。
- 最终输出使用普通 Zarr v3 chunks；大规模中间层可能使用 sharding 减少文件数。

## 分块策略

- `time`：提高长时间序列读取连续性。
- `space`：提高单时间切片空间场读取连续性。
- `custom`：使用 `--chunks '[time,lat,lon]'` 明确指定。

自动策略以 `--target-chunk-mib` 为目标，默认 128 MiB，并根据实际 shape、dtype、worker 内存预算和安全上下限调整。

## 常用命令

时间连续型重分块并压缩：

```bash
pixi run rechunk -- \
  --input /data/input.zarr \
  --output /data/time-optimized.zarr \
  --strategy time \
  --compression balanced
```

显式 codec：

```bash
pixi run rechunk -- \
  --input /data/input.zarr \
  --output /data/output.zarr \
  --strategy space \
  --compression-codec blosc-zstd \
  --compression-level 4 \
  --compression-shuffle bitshuffle
```

自定义 chunks：

```bash
pixi run rechunk -- \
  --input /data/input.zarr \
  --output /data/custom.zarr \
  --strategy custom --chunks '[16,256,256]' \
  --compression none
```

检查和 dry-run：

```bash
pixi run rechunk -- --input /data/input.zarr --inspect-only
pixi run rechunk -- --input /data/input.zarr --output /data/output.zarr --strategy time --dry-run
```

## 工作流

1. 读取 Zarr v3 根元数据、变量 shape、chunks、dtype、codec 和属性。
2. 生成 `ChunkPlan` 与 `CompressionPlan`。
3. 根据源/目标 chunks 判断直接路径或两阶段路径。
4. 阶段一按源 chunks 读取一次，写入与目标合并方向对齐的中间布局。
5. 阶段二按最终 chunk 合并有界区域并一次性编码写出。
6. 抽样逐值校验后原子发布。

所有进程池使用 `spawn`。worker 数受 CPU、未压缩块内存和源/目标磁盘关系限制；同一机械硬盘会主动降低并发，避免顺序 I/O 退化为随机寻道。

## Python 入口

- `inspect_store(path)`
- `plan_chunks(info, strategy, ...)`
- `make_compression_plan(...)`
- 核心执行器 `rechunking.engine.run_rechunk(...)`，或服务门面 `application.services.run_rechunk(RechunkConfig(...))`

GUI 通常通过 `application.services.preview_rechunk` 和 `run_rechunk` 使用该模块。

## 输出与安全

- 中间 store 可放在单独 SSD；独立 CLI 的具体临时路径由执行计划管理。
- 最终 store 在校验前不会发布。
- 覆盖已有输出需要 `--overwrite`，且目标必须能识别为 Zarr。
- `--no-validate` 会关闭输出抽样逐值校验，不建议用于正式产品。

## 当前限制

- 不接受 Zarr v2。
- 不负责空间重采样、坐标重投影或时间聚合。
- 压缩转换保持无损；数据 dtype 和数值语义不会为了压缩而缩窄。
