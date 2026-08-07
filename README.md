# 快速 Zarr 转换器

面向成批 NetCDF/HDF/TIFF 数据，自适应转换为 Zarr v3；源文件可自带时间维度，
也可由文件名重建时间轴。
源维度默认名为 `time`、`lat`、`lon`，其他名称可通过映射指定。程序会先并行检查全部文件，再根据文件形态、源 chunk、物理核心、
可用内存和源/目标磁盘关系生成候选方案，并在输出磁盘上用真实数据小样本测速。

当前版本最低支持 1 日尺度的数据转换；小时、分钟、秒等日内尺度数据暂不支持。
无论源数据使用源时间、年 + DOY 还是年 + 月 + 日，输出 Zarr 的 `time` 坐标统一
按 `YYYY-MM-DD` 日期格式表示，例如 `2001001` 转换为 `2001-01-01`。

## 运行

当前 Python 环境和项目命令统一由 Pixi 管理。首次运行会根据 `pixi.lock` 创建当前项目
专属环境，之后直接使用：

```bash
pixi run convert
```

无参数时依次提示输入目录、输出目录、时间范围、经度范围、纬度范围和变量编号。
范围采用列表格式；直接回车表示全部。

启动后程序会根据文件后缀和首个文件的维度自动判断完整维度模式或文件名时间模式。
文件名模式支持年 + DOY（如 `2001001`）和年 + 月 + 日（如 `20010101`），会扫描
全部文件检查时间字段唯一性、命名结构和时间缺口，并在确认后为缺口建立空值切片；
HDF 和 TIFF 分别自动选择 `netcdf4` 和 `rasterio` 后端。自动识别存在歧义时，
程序才要求用户手动确认时间字段或模板。

GUI 模式会先单独执行“文件时间维度信息检查”：比较排序后的文件名字段、读取首文件的
`time` 元数据，并允许用户确认完整时间来源或组合规则。支持完整时间来自文件名、完整时间
来自 `time`，以及文件名年份 + `time` DOY 等混合情况；确认时间规则之前不会解锁后续操作。
HDF-EOS 网格（例如 `YDim:*`/`XDim:*`）会优先从 `StructMetadata.0` 恢复像元中心
经纬度，再统一重命名为 `lat/lon`。
文件名时间模式在使用 `netcdf4` 引擎时，会优先通过 `netCDF4.Dataset` 直接读取维度、
变量定义、关键属性和 HDF-EOS 网格元数据，避免为每个文件构造完整的 xarray Dataset；
若底层接口不适用，则自动回退到 xarray。该优化减少软件层开销，但海量文件在机械硬盘
上的逐文件寻道仍受物理 I/O 限制，源文件放在 SSD 上通常会明显加快扫描。

也可完全使用命令行：

```bash
pixi run convert \
  --input /path/to/nc \
  --output /path/to/result.zarr \
  --time '[2003, 2010]' \
  --lat '[30, 90]' \
  --lon '[-100, 100]' \
  --variables a1 a2
```

时间范围中的日期可以直接写成 `--time '[2001-01-01, 2022-12-31]'`，不需要为日期加引号；年份范围仍可写成 `--time '[2003, 2010]'`。

## GUI（当前 MVP）

当前已提供 PySide6 图形界面，包含数据检查、转换、Zarr 优化、Zarr 重采样和任务日志页面。
GUI 与命令行共用同一套核心检查和写入引擎。

使用项目 Pixi 环境启动：

```bash
pixi run gui
```

## 可组合一条龙模块（v1.3.0）

一条龙模块会先执行完整的数据检查和时间规则确认。原始数据转换是必经步骤；
重采样、重分块和重压缩可以独立选择，所有组合始终只发布一份最终 Zarr。规划器按统一
`OutputLayout` 融合用户意图：选择的 chunks 和 codec 优先由转换器或重采样器直接写出，
而不是强制再写一遍完整数据。需要重采样时，转换阶段的源窗口会根据目标边界、源/目标
分辨率和重采样方法自动增加边界 stencil，避免直接裁到用户填写的边界导致海岸线或边界
像元结果偏差。最终 Zarr 统一使用 `time/lat/lon`，纬度从北到南、经度从西到东。
网格完全一致时转换器直接写最终 chunks 和 codec；需要重采样时，重采样器直接写最终布局。
只有变量类型或布局不兼容时才回退到独立最终重分块，因此常规三维数值数据不再进行一次
完整 Zarr 的重复读取和写入。任务 manifest 会记录逻辑写放大和避免的最终化 I/O。

命令行入口：

```bash
pixi run pipeline \
  --input /path/to/gosif-tif \
  --output /path/to/gosif.zarr \
  --lat 30 90 --lon -180 180 \
  --resample --resolution 0.1 \
  --rechunk --recompress \
  --method conservative --skipna \
  --strategy time --compression balanced \
  --inspection-cache /path/to/cache/gosif-inspection.json \
  --temporary-dir /path/to/ssd/pipeline-temporary
```

`--dry-run` 只执行检查和计划，不写数据；
`--resample`、`--rechunk`、`--recompress` 分别开启对应可选操作；三者都不提供时
执行仅转换流程。`--resolution` 只在同时选择 `--resample` 时有效。
`--cleanup-intermediate` 会在下游验证通过后删除已不再需要的上游临时 Zarr。完整设计与
阶段边界见 [docs/ONE_STOP_PIPELINE_V1.md](docs/ONE_STOP_PIPELINE_V1.md)。
对于数千个源文件，建议固定使用同一个 `--inspection-cache` 路径。快照记录文件路径、
大小、纳秒修改时间和读取后端；再次运行时仍会枚举当前目录，但只打开新增或指纹变化的
文件读取元数据。直接导入快照用于转换时会严格拒绝已变化的源文件，避免使用过期索引。

## Zarr 重采样（第一版）

项目现在提供基于 xESMF 的 Zarr v3 空间重采样模块。它会先执行与 Zarr 优化模块相同的
输入检查，然后显示当前经纬度分辨率、目标网格大小、输入 chunks 和 codec。第一版要求
`lat`、`lon` 是一维规则经纬度坐标，默认覆盖输入空间范围；边界不足一个目标单元时会向外取整，目标范围也可以选择全球范围。

命令行示例：

```bash
pixi run resample \
  --input /path/to/input.zarr \
  --output /path/to/resampled.zarr \
  --resolution 0.25 \
  --method bilinear \
  --skipna
```

支持的 xESMF 方法包括 `bilinear`、`conservative`、`conservative_normed`、`patch`、
`nearest_s2d` 和 `nearest_d2s`。可以使用 `--no-skipna` 关闭缺失值跳过。输出数据变量的
chunks 会按输入变量逐维保持，若目标维度变短则自动截断；输入 Zarr v3 codec、时间坐标、
变量属性和全局属性会被保留。执行时采用有界的空间/时间流式块，避免把完整 Dask 图和
完整数据同时载入内存；`--tile-size`、`--time-block`、`--compute-workers` 和
`--space-workers` 可分别调节空间块、时间批次、块内线程和空间进程并行度。空间块和时间
批次默认支持 `auto`：空间 auto 会根据源/目标 chunks、分辨率比例、数据 dtype、xESMF 方法、
周期经度行为、源窗口实际跨越的 chunk 数和当前可用内存选择目标块；时间 auto 会尽量
贴合源时间 chunk，同时限制单个批次的内存。空间块会对齐到所有输出变量的 chunk 边界，
避免多个进程并发读改写同一个 Zarr chunk；周期网格只读取当前空间块所需的局部经度窗口。
计划会显示自动估算依据，资源不足时会给出先重分块或降低并行度的警告；也可以传入正整数
手动指定。

浮点数据还可以使用 `--compute-dtype float32` 在重采样前转换为 float32，降低源窗口、
中间结果和最终输出的内存/存储压力；默认 `source` 保持原始浮点 dtype。该选项只作用于
浮点数据变量，整数变量仍走安全的浮点输出路径，不支持把 int64 直接转换为 int16。
当自动时间块小于最终输出的 time chunk 时，程序会在 `--temporary-dir` 指定的目录生成
临时中转 Zarr，以较小时间 chunk 写入；全部重采样完成后按最终 chunk 一次性合并，避免
反复读改写同一个大输出 chunk。中转目录中的任务专属文件和目录会在成功发布最终结果后
自动删除；失败时会保留中间目录用于排查。未指定 `--temporary-dir` 时使用输出目录旁的
隐藏临时目录。

例如：

```bash
pixi run resample \
  --input /path/to/input.zarr \
  --output /path/to/resampled.zarr \
  --resolution 0.25 \
  --method bilinear \
  --compute-dtype float32 \
  --temporary-dir /path/to/ssd/resample-temporary
```

GUI 中新增“Zarr 重采样”页面。项目 Pixi 环境已经提供 xESMF/ESMF 运行时。

GUI 的推荐操作顺序是先在“数据检查”页面完成源目录检查并确认时间规则，再进入“转换”；“Zarr 优化”页面接收已经存在的 Zarr v3 目录。
“Zarr 优化”与原始数据检查相互独立，即使尚未检查源数据，也可以直接输入已有的 Zarr v3 目录执行优化。
在“Zarr 优化”页面可以分别勾选重分块和重压缩；两项同时勾选时，程序在一次任务中生成一份同时完成重分块和重压缩的输出 Zarr，不会产生两份中间结果。
同时勾选两项时还可以指定“中间处理目录”（例如 SSD 上的目录）。阶段间反复读取的中间 Zarr 会写入该目录，最终已校验的数据直接写入输出目录所在磁盘；任务成功后自动删除中间目录内容，失败时保留临时目录以便排查。
数据检查成功后可以保存完整检查快照，之后通过“导入检查快照”恢复逐文件时间索引和变量结构，直接进入转换；缺少逐文件索引或修改时间指纹的旧版快照会要求重新检查。
转换页面的变量表支持输出变量重命名，以及按原始值配置多个填充值、缩放因子和输出填充值；不填写输出变量名时保留原名。
“任务与日志”页面提供紧凑的 CPU/RSS 实时曲线、读取/写入速率、自动识别的磁盘空间使用表、请求取消和当前会话任务历史。取消会在当前 I/O 块或阶段边界安全停止，已生成的部分输出不会被视为成功结果。

文件名模式示例：

```bash
pixi run convert --mode filename \
  --input /path/to/files --output /path/to/result.zarr \
  --template doy --year 2001 --doy 001 --step-days 1 \
  --continue-missing
```

程序默认要求源文件使用 `time`、`lat`、`lon` 作为时间、纬度和经度维度名。
如果源数据使用其他名称，需要显式提供映射；例如源维度为
`time`、`latitude`、`longitude` 时：

```bash
pixi run convert \
  --input /path/to/nc \
  --output /path/to/result.zarr \
  --time-dim time \
  --lat-dim latitude \
  --lon-dim longitude
```

交互模式会在检查到非标准维度名后逐项询问实际名称。无论源文件使用什么名称，
输出 Zarr 的维度和坐标名始终统一为 `time`、`lat`、`lon`。

只检查结构或只查看计划：

```bash
pixi run convert --input /path/to/nc --inspect-only
pixi run convert --input /path/to/nc --output /path/to/result.zarr --dry-run
```

## 自适应策略

程序不会套用固定参数：

- 文件数量极多且单文件较小（典型为数千个以上）：使用文件优先策略。一个 worker 打开一个
  文件后连续处理其中全部所选变量和空间块，`time chunk` 通常对齐单文件时间。
- 单文件包含数月或全年数据：使用 chunk 优先策略。同一大文件内部的独立输出
  chunk 可由多个进程并发读取、压缩和写入。
- 文件数量和单文件大小均处于中间范围：生成平衡计划，并同时实测文件优先和
  chunk 优先候选；最终分类还会参考文件大小分位数、每文件时间点数量以及存储设备。
- 非数值变量、额外维度等不满足直接写入条件时，自动回退到 xarray/Dask，
  回退路径仍受内存预算约束。

自动调参同时测试 worker、时间 chunk、空间 chunk、小文件任务批量大小以及
Zstd/LZ4 的 shuffle 组合。大文件、海量小文件和均衡型数据使用不同的候选范围；
同一机械硬盘上的源/目标会限制随机 I/O 并发。每项
测试写到输出目录同一文件系统中的隐藏临时目录，并对测试文件逐个 `fsync` 后
计算耐久吞吐。调参样本从时间轴的开头、中间和结尾分层抽取，并限制单次样本
逻辑数据量，避免大文件候选在调参阶段生成数 GiB 临时数据；候选按轮次交替测试，
避免第一个候选耗尽整个预算。Linux 下每个候选开始前还会用 `POSIX_FADV_DONTNEED`
丢弃本次样本源文件的页缓存，避免后测候选因缓存命中获得虚假的优势。临时目录无论
成功或失败都会清理。默认预算 60 秒，可用
`--tune-budget` 调整；`--no-tune` 可仅使用启发式计划。

调参同时记录三种速度：逻辑吞吐（未压缩数据处理速度）、物理吞吐（实际压缩后
写入速度）和耐久吞吐（包含 `fsync`）。正式写入不在每个任务后执行 `fsync`，因此
默认按逻辑吞吐选择方案，同时展示耐久吞吐；压缩率则使用分层样本的汇总结果估算
输出体积，并额外预留 25% 的压缩率波动安全裕量。

上述自动调参和多进程正式写入同时适用于源文件自带时间维度模式和文件名时间模式。
正式写入时会显示实时进度、吞吐率以及可用时的 CPU/RSS 监测信息。

实测还会得到当前变量和压缩器的压缩率，据此估算正式输出体积。程序只在预计
可容纳的候选中按逻辑吞吐选择方案；全部方案都会占用目标磁盘 95% 以上可用
空间时，程序在正式写入前停止。

## 输出与安全

- 输出固定为 Zarr v3，默认 Blosc Zstd level 1，保留变量属性和 `_FillValue`。
- 检查阶段会展示变量的 `_FillValue`、`missing_value`、`scale_factor` 和
  `add_offset`；交互模式可为每个变量输入任意长度填充值列表和单个缩放因子。
  填充值按原始值、缩放之前识别；空回车表示不执行对应操作。
- 已存在的空输出目录可直接使用。
- 非空 Zarr 输出目录不会自动覆盖；交互模式要求确认，命令行要求
  `--overwrite`。即使指定覆盖，程序也拒绝删除不含 Zarr 根标记的普通非空目录。
- 正式写完后校验维度、经纬度和多个源文件抽样值。
- 中断后的 store 可能不完整，重试时应明确使用 `--overwrite`。

Zarr v3 当前建议这样打开：

```python
import xarray as xr

ds = xr.open_zarr("/path/to/result.zarr", consolidated=False)
```

## 开发测试

第二个模块的设计流程见 [docs/RECHUNK_WORKFLOW.md](docs/RECHUNK_WORKFLOW.md)。
当前已经提供独立的 Zarr 重分块与无损重压缩入口：

~~~bash
pixi run rechunk --input /path/to/input.zarr --output /path/to/rechunked.zarr --strategy time --compression balanced
~~~

第一版重分块器只接受 Zarr v3，并要求输入同时包含完整的 `time`、`lat`、`lon`
维度。策略包括时间连续型、空间连续型和三元素自定义 chunks。默认目标 chunk 为
128 MiB，程序会根据实际维度、dtype、可用内存和 worker 数自动计算实际 chunks。
压缩方案包括 `fast`、`balanced` 和 `maximum` 三种无损 Zstd 配置；不指定
`--compression` 时命令行默认保留输入 codec，交互模式会询问用户。
对于源 chunks 与目标 chunks 差异较大的大型数据，重分块采用源 chunk 对齐的两阶段
流式流程：第一阶段每个源 chunk 只读取一次并写入与最终空间块对齐的中间 Zarr，
第二阶段沿目标连续维度合并；阶段 1 将一个源 chunk 作为单次批量写入交给 Zarr
codec pipeline，阶段 2 也按最终 chunk 单次写入，避免数百次独立的小块写操作。两阶段都限制局部任务图和并发量，避免完整变量任务图产生内存峰值。中间层在
启用重压缩时使用快速 codec，最终输出再使用用户选择的压缩方案；worker 数还会依据
源 chunk 大小、可用内存和源/目标存储设备类型自动收敛到安全范围。时间连续型的
阶段 1 按源 time chunk 隔离后使用多进程，阶段 2 按最终 chunk 隔离后使用多进程；
同一块机械硬盘默认最多使用两个进程，避免并发随机 I/O 造成寻道放大。数据写入采用
Zarr v3 Fused codec pipeline，以提高批量压缩和写入的 CPU 利用率。阶段 2 会将多个
相邻的中间逻辑 chunk 合并为一个有界读取区域，再一次性写出对应的最终 chunks，
减少 Python/Dask 任务和 codec 调用。预计中间逻辑 chunk 数达到约 1,000,000 以上
时，阶段 1 才会额外启用 Zarr v3 sharding，把多个逻辑小 chunk 打包为较少的物理
文件；逻辑 chunk 形状不变，time 方向保持与源 time chunk 对齐，shard 的未压缩大小
受安全上限约束。较小的数据默认使用普通中间 chunk，避免阶段 2 的 sharded 解码
开销。最终输出不使用 sharding，仍是普通 Zarr v3 chunk，便于后续工具读取。

转换、重采样和重分块都在目标同一文件系统的 UUID 临时目录中写入，通过校验后再
原子发布。覆盖时旧 Zarr 会在发布期间保留为可恢复备份，不会在计算开始前删除。

所有进程池显式使用 `spawn`，避免 Python 3.13 中多线程父进程 `fork` 的死锁风险。
每个 worker 默认限制为一个 OpenMP/BLAS 线程；确有需要时可设置
`FAST_NC_ZARR_THREADS_PER_WORKER`。

只检查输入 Zarr：

~~~bash
pixi run rechunk --input /path/to/input.zarr --inspect-only
~~~

```bash
pixi run test
```

测试覆盖小文件文件优先写入、跨 NetCDF 文件边界的大文件 chunk 写入、范围裁剪、
变量筛选、Zarr v3、填充值和逐值校验。测试数据统一位于 `/tmp/codex_test`。

对真实数据根目录中所有数据集执行全文件元数据检查、分层数值抽样和小范围转换：

```bash
pixi run validate-raw \
  --input-root /media/owen/机械硬盘/zarr处理/RAW_DATA \
  --output /media/owen/机械硬盘/codex_test_hdd/p0_hardening/raw-validation.json \
  --time-field GLASS-PAR=3 \
  --smoke-output-root /media/owen/机械硬盘/codex_test_hdd/p0_hardening/smoke
```

项目根目录的 `pixi.toml` 和 `pixi.lock` 是唯一的环境与依赖来源，锁定 Python、GDAL、
ESMF、xESMF、Zarr 及 GUI 运行时。`pixi.toml` 的激活配置会把 `src` 加入
`PYTHONPATH`，不依赖外部共享环境或本地包安装。
