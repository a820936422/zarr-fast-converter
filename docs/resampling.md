# 空间重采样模块

## 定位

`fast_nc_zarr.resampling` 使用 xESMF/ESMF 对现有 Zarr v3 的规则经纬度网格进行空间重采样。执行器按空间 tile 和时间 block 流式处理，避免构建覆盖完整数据集的计算图。

入口：

```bash
pixi run resample -- --help
python -m fast_nc_zarr.resampling --help
```

## 输入约束

- 输入必须是 Zarr v3。
- `lat`、`lon` 必须是一维、规则、单调坐标；数据变量需要包含完整 `time/lat/lon`。
- 支持 `bilinear`、`conservative`、`conservative_normed`、`patch`、`nearest_s2d` 和 `nearest_d2s`。
- conservative 系列方法会派生网格边界。
- 浮点数据可用 `--compute-dtype float32` 降低内存；默认 `source` 保持原浮点 dtype。

## 工作流

1. 检查 Zarr 元数据、变量、坐标规则性、分辨率、chunks 和 codec。
2. 根据 `source` 或 `global` 范围建立目标网格。
3. 自动规划目标 tile、时间 block、源窗口和 worker；也可手动指定。
4. 将缺失值和 `_FillValue` 掩码后执行 xESMF。
5. 可在采样前后应用值替换规则。
6. 写入与最终 chunks 对齐的空间块，避免进程并发改写同一 chunk。
7. 校验输出结构；作为一条龙阶段时还会执行局部数学抽样比对。

## 常用命令

```bash
pixi run resample -- \
  --input /data/input.zarr \
  --output /data/resampled.zarr \
  --resolution 0.25 \
  --method bilinear \
  --skipna
```

全局网格与 float32 计算：

```bash
pixi run resample -- \
  --input /data/input.zarr \
  --output /data/global.zarr \
  --resolution 0.1 --extent global \
  --method conservative_normed \
  --compute-dtype float32 \
  --temporary-dir /fast-ssd/resample-temp
```

资源手动控制：

```bash
pixi run resample -- \
  --input /data/input.zarr --output /data/output.zarr \
  --resolution 0.25 --method nearest_s2d \
  --tile-size 256 --time-block 8 \
  --compute-workers 2 --space-workers 4
```

只检查或规划：

```bash
pixi run resample -- --input /data/input.zarr --inspect-only
pixi run resample -- --input /data/input.zarr --output /data/output.zarr --resolution 0.25 --dry-run
```

## 替换与统计

`--before-conditions` 和 `--before-results` 在采样前执行；`--after-conditions` 和 `--after-results` 在采样后执行。规则按声明顺序匹配。统计表达式支持 `auto`、`sample` 和 `exact` 策略；`exact` 可能扫描完整变量，应根据数据规模选择。

## 中转层与输出

当计算时间 block 小于最终 time chunk，且多个批次会更新同一最终 chunk 时，模块建立时间中转 Zarr，全部完成后受控合并一次。空间 tile 则直接对齐最终 chunk 边界。中转 store 和权重文件写入 `--temporary-dir`；成功发布后清理，失败时保留。

输入属性、时间坐标和可兼容 codec 会被保留；显式最终布局由一条龙规划器下推时，重采样器直接写目标 chunks/codec。

## Python 入口

- `inspect_resample_input(path)`
- `plan_resample(ResampleConfig(...), inspection)`
- `run_resample(ResampleConfig(...), inspection=...)`

相应模型位于 `fast_nc_zarr.resampling.models`。

## 当前限制

- 不支持非规则网格、曲线坐标或投影坐标重投影。
- 不会自动为分类变量选择专用算法，方法需由用户确认。
- 不执行时间聚合。
- 目标范围超出源覆盖时不进行外推，未覆盖格点保持缺测值。
