# 源数据转换模块

## 定位

`fast_nc_zarr` 核心转换模块将同构的 NetCDF、HDF 或 TIFF 文件目录转换为 Zarr v3。它负责文件发现、时间解释、空间/变量选择、自适应执行计划、写入校验和安全发布。

入口：

```bash
pixi run convert
python -m fast_nc_zarr
```

无参数启动时进入交互模式；自动化任务应显式传参。

## 输入模式与约束

- `complete`：源文件本身具有时间维度。
- `filename`：单文件通常代表一个时间切片，从文件名的年 + DOY 或年 + 月 + 日字段建立时间轴。
- `auto`：检查首个文件结构后选择上述模式，歧义时要求人工确认。
- `.tif/.tiff` 使用 rasterio/rioxarray；HDF 优先使用 netCDF4；其他 NetCDF 默认使用 h5netcdf。
- 源维度可以通过 `--time-dim`、`--lat-dim`、`--lon-dim` 映射；输出始终规范为 `time/lat/lon`。
- 文件名模式会检查重复时间、字段歧义和缺口；只有指定 `--continue-missing` 才为缺失日期建立空值切片。
- 当前时间精度最低为日，不支持日内时间轴。

## 工作流

1. 枚举文件并并行读取元数据。
2. 校验变量定义、维度、经纬度网格和时间唯一性。
3. 构建 `Inventory`，再按时间、空间和变量生成 `Selection`。
4. 根据文件大小分布、原生 chunks、CPU、内存和源/目标磁盘关系选择 file、chunk 或 Dask 路径。
5. 可在目标文件系统上使用真实样本调优吞吐和压缩率。
6. 写入 staging Zarr，抽样比对源值、坐标和维度。
7. 校验成功后原子发布。

## 常用命令

完整时间维度数据：

```bash
pixi run convert -- \
  --input /data/source \
  --output /data/result.zarr \
  --time '[2001-01-01,2022-12-31]' \
  --lat '[30,90]' \
  --lon '[-180,180]' \
  --variables gpp quality
```

文件名 DOY 模式：

```bash
pixi run convert -- \
  --mode filename \
  --input /data/daily-files \
  --output /data/result.zarr \
  --template doy --year 2001 --doy 001 \
  --step-days 1 --continue-missing
```

非标准维度：

```bash
pixi run convert -- \
  --input /data/source --output /data/result.zarr \
  --time-dim time --lat-dim latitude --lon-dim longitude
```

只检查或只规划：

```bash
pixi run convert -- --input /data/source --inspect-only
pixi run convert -- --input /data/source --output /data/result.zarr --dry-run
```

关闭实测调优可用 `--no-tune`；限制物理核心数可用 `--max-workers`；通过 `--reserve-memory` 为系统保留内存。

当一条龙规划器已固定转换 chunks 或最终 `OutputLayout` 时，自动调优仍会保持该布局不变，并只实测安全的 worker 数和任务批量。chunk 写入按变量和连续 time slab 分组；每个物理输出 chunk 只有一个 owner，源文件句柄缓存同时受时间块、worker、`RLIMIT_NOFILE` 和硬上限约束。

## Python 入口

主要服务入口位于 `fast_nc_zarr.application.services`：

- `inspect_source(SourceInspectionConfig(...))`
- `preview_conversion(inspection, ConversionConfig(...))`
- `run_conversion(inspection, ConversionConfig(...))`
- `save_inspection_snapshot(...)` / `load_inspection_snapshot(...)`

GUI 和一条龙模块复用这些服务，不需要调用子进程。

## 输出与安全

- 输出为 Zarr v3，默认使用 Blosc Zstd 无损压缩。
- 写入前会估算目标磁盘空间；自动调优时使用实测压缩率并留安全余量。
- `--overwrite` 只允许替换可识别的 Zarr，拒绝删除普通非空目录或符号链接目标。
- 取消或异常不会把部分 staging 当作成功结果。
- 检查快照记录文件路径、大小和修改时间；源文件变化后旧快照会被拒绝。

## 当前限制

- 日内时间暂不支持。
- 同一批次应为同构数据，混合后缀或不兼容变量结构会被拒绝。
- 非数值变量、额外维度等无法直接写入时可能回退到 xarray/Dask；一条龙模块的约束更严格。
- 机械硬盘上的海量小文件仍受随机寻道限制，增加 worker 不能消除物理 I/O 瓶颈。
