# 重分块与重压缩模块

## 定位

`fast_nc_zarr.rechunking` 调整现有 Zarr v3 的 chunks，并可同时应用无损压缩 codec。模块先识别等价复制和仅换 codec 的单阶段路径；只有 chunks 真正变化时才根据目标访问模式、变量 dtype、可用内存和磁盘关系执行有界的两阶段流式写入。

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

## Rust 后端（v1.7.0 实验性）

Rust 后端通过 `fast_nc_zarr._native` 提供可选的 Zarr v3 重分块执行器：

```bash
pixi run rechunk -- \
  --input /data/input.zarr \
  --output /data/output.zarr \
  --strategy custom \
  --chunks '[1,256,256]' \
  --compression-codec zstd \
  --compression-level 1 \
  --backend rust \
  --workers 4
```

当前 Rust 适用范围：一个三维 `float32` 数据变量、维度严格为
`(time, lat, lon)`、Zarr v3 目录型 store；可以改变目标 chunks，并在显式 codec
配置下执行无损 Zstd、Blosc 或 Gzip 重压缩。`--compression auto` 仍由 Python
调优器执行，能力不足时 `--backend auto` 回退 Python，`--backend rust` 明确失败。

Rust 执行按互不重叠的目标 chunk 建立有界线程池，worker ceiling、目标 chunk 数、
内存预算和 codec 内部并发共同限制实际并发。输出先写入 staging，随后由 Python
完成元数据、codec、抽样数值校验和原子发布；取消、异常或校验失败不会发布最终输出。
Rust 后端目前不支持多变量、非 `float32`、Zarr v2、NetCDF/HDF/TIFF 转换、重采样
或完整 pipeline 执行。

## 分块策略

- `time`：提高长时间序列读取连续性。
- `space`：提高单时间切片空间场读取连续性。
- `custom`：使用 `--chunks '[time,lat,lon]'` 明确指定。

自动策略以 `--target-chunk-mib` 为目标，默认 128 MiB，并根据实际 shape、dtype、当前进程 CPU/内存边界和安全上下限调整。`workers=auto` 会在真实临时/输出文件系统上分别实测两阶段候选；显式整数仍作为硬上限。

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
2. 生成 `ChunkPlan`；自动压缩时以最终 chunk 形状抽取有界 begin/middle/end 真实数据，比较无损 Zstd/LZ4 候选的写入、fsync、冷热读取和体积。
3. chunks、codec 和 metadata 等价时，将独立文件复制到 staging 后校验发布，不创建 hardlink。
4. 数据物理 chunks 相同但 codec 不同时，逐源物理 chunk 单阶段解码并写最终 codec。
5. chunks 不同时，阶段一按真实源 chunks 实测并发后写入对齐中间布局；阶段二按最终物理 region 重新实测并发，再有界合并。
6. 所有自动压缩候选逐值验证；从 Pareto 前沿按 speed、balanced 或 compact 目标选择。
7. 抽样逐值校验后原子发布；worker 和压缩完整报告随 metrics 返回。

所有进程池使用 `spawn`。worker 先受当前进程 affinity/cgroup CPU、有效可用内存和物理 chunk ownership 约束，再由真实样本吞吐选择；网络/9p/低置信存储只收敛候选，不替代实测。显式 worker 仍是硬上限。

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
