# 快速 Zarr 转换器

面向成批 NetCDF/HDF/TIFF 数据，自适应转换为 Zarr v3；源文件可自带时间维度，
也可由文件名重建时间轴。
源维度默认名为 `time`、`lat`、`lon`，其他名称可通过映射指定。程序会先并行检查全部文件，再根据文件形态、源 chunk、物理核心、
可用内存和源/目标磁盘关系生成候选方案，并在输出磁盘上用真实数据小样本测速。

当前版本最低支持 1 日尺度的数据转换；小时、分钟、秒等日内尺度数据暂不支持。
无论源数据使用源时间、年 + DOY 还是年 + 月 + 日，输出 Zarr 的 `time` 坐标统一
按 `YYYY-MM-DD` 日期格式表示，例如 `2001001` 转换为 `2001-01-01`。

模块功能与使用手册见 [docs/README.md](docs/README.md)。

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

## GUI（v1.6.9）

PySide6 图形界面将主导航收敛为“数据检查、处理流程、任务中心”。转换、重采样、重分块
和重压缩不再作为相互割裂的主页面出现，而是在处理流程中按需组合；未勾选的操作不会进入
物理执行计划。旧页面对象和独立命令行入口暂时保留，供已有自动化脚本兼容使用。
界面使用 Fusion + QSS 主题，在 1280x820 和较小窗口下保持紧凑的参数区与计划摘要区。
v1.6.8 将路径字段简化为“路径 + 可访问状态 + 浏览”；收藏、最近浏览和收藏管理集中在“浏览”打开的嵌入式路径选择界面左侧面板中，可直接搜索、进入收藏目录、选择当前目录、收藏当前目录和管理收藏。收藏设置从 `pathPicker/v1` 自动迁移到 `pathPicker/v2`；任务中心增加 CPU、RSS、读写和磁盘指标卡，数据检查和处理流程使用步骤条与状态徽标。
v1.6.9 将存储介质从静态 worker 限速改为有效资源约束与真实候选选择：统一 CPU/内存/文件描述符资源契约，conversion、resampling、rechunk 分别记录 worker/tile 候选和选择证据；临时与输出目录在处理前执行可写性预检；pipeline manifest 升级为 schema 6，并写入 resource budget、preflight、候选选择、内存语义和 `events.jsonl`。

## v1.7.0 Rust 重构状态

`refactor/v1.7.0-rust` 分支已完成 Rust Zarr v3 核心、单变量 `float32` 重分块和
有界目标 chunk 并行后端。Rust 后端仍为实验性 opt-in，不改变 v1.6.9 的 Python
默认路径；完整说明见 [`docs/rechunking.md`](docs/rechunking.md) 和
[`Architecture/v1.7.0-rust-refactor-plan.md`](Architecture/v1.7.0-rust-refactor-plan.md)。

当前 Rust 后端支持：一个 `(time, lat, lon)` 三维 `float32` 数据变量、Zarr v3、
不改变 codec 的重分块。输出通过 staging、结构校验、抽样逐值校验和原子发布完成。
多变量、其他 dtype、重压缩、NetCDF/HDF/TIFF 转换、重采样和 pipeline 融合仍使用
Python 路径或回退 Python。

v1.6.7 在每次任务前记录当前进程真正可用的 CPU、内存、WSL/cgroup 限制以及源、临时和输出文件系统证据；WSL 虚拟块设备不再因不可信的 `rotational=1` 被直接判为机械硬盘。兼容性最终化默认分别用真实源 chunk 和最终 region 小样本实测阶段 1/2 worker。自动压缩在真实输出文件系统上比较受控的无损 Zstd/LZ4 候选，综合 durable 写入、冷热读取和体积从 Pareto 前沿选择。GUI 的主要路径选择器支持持久收藏、最近目录和失效路径保留。

v1.6.6 针对真实生产任务的 CPU 欠利用与中间写放大进行通用优化：规则网格等价判断考虑坐标 dtype/ULP，物理 Zarr chunk 强制单一写者；流水线固定布局仍会实测 worker 并发；转换按时间片提高源句柄局部性；重采样优先以最终 chunk owner 有界聚合，重分块对等价布局和仅换 codec 使用快速路径。任务日志与资源事件会持久化到用户缓存目录，便于性能回归分析。

v1.6.5 增加真实源数据校验模块的自动化覆盖；任务中心会从转换、重采样、重分块和流水线日志中提取实际百分比或完成量；时间检查与计划预览均响应安全取消。修改输入类型、读取引擎、递归选项、worker、维度映射或时间规则后，旧检查结果会立即失效并锁定下游处理，避免使用过期计划。

v1.6.4 将现有 Zarr 和失败任务续跑的最终 chunks/codec 直接下推到重采样阶段，避免重采样成功后再次完整重分块；仅时间批次会部分更新同一最终 chunk 时才建立中转 Zarr，空间 chunk 差异改用边界对齐直接写入。中转 chunk 使用受控并行合并，并分别记录时间批次、合并耗时、内部写入和避免的最终化 I/O。ESMF 自动 worker 使用生产任务校准后的内存基线，GUI 同时显示实际占用核心数与整机 CPU 百分比。

输出发布前会清理指向未输出变量的 CF 引用属性；成功续跑会把旧错误转入 `error_history`。抽样语义检查会根据显式约束、`valid_min/valid_max` 以及标准误差/不确定度元数据报告越界或负值，但不会静默修改科研数据。

v1.6.3 新增失败流水线续跑：在“数据检查”的“输入类型”中选择“临时处理产物”，输入
上次选择的临时目录，程序会从任务目录内的相对 `manifest.json` 和检查点路径识别最新的
可恢复任务，验证 Zarr v3、维度、变量和阶段状态，并恢复原处理参数。“处理流程”会显示
“继续执行”按钮，从最近一个已验证检查点执行剩余阶段；成功后才按原清理策略删除旧中间
产物。新任务清单以相对路径导出 conversion/resampling 检查点，不依赖原计算机的项目路径。
项目源码和启动脚本只通过包导入或相对 `src` 路径定位代码；`.pixi` 环境被 Git 忽略，
NetCDF/Zarr 输入输出等用户数据路径仍按运行时选择记录，不属于代码库依赖路径。

使用项目 Pixi 环境启动：

```bash
pixi run gui
```

## 可组合处理流程（v1.6.8）

一条龙模块会先执行完整的数据检查和时间规则确认。原始数据转换是必经步骤；
重采样、重分块和重压缩可以独立选择，所有组合始终只发布一份最终 Zarr。规划器按统一
`OutputLayout` 融合用户意图：选择的 chunks 和 codec 优先由转换器或重采样器直接写出，
而不是强制再写一遍完整数据。需要重采样时，转换阶段的源窗口会根据目标边界、源/目标
分辨率和重采样方法自动增加边界 stencil，避免直接裁到用户填写的边界导致海岸线或边界
像元结果偏差。最终 Zarr 统一使用 `time/lat/lon`，纬度从北到南、经度从西到东。
网格完全一致且存储方案已明确时，转换器可直接写最终 chunks 和 codec；需要重采样时，重采样器优先直接写最终布局。选择“自动压缩”时程序会保留最终化阶段，在已验证的真实代表数据和最终输出文件系统上完成 codec 实测后再写最终产品。其他情况下只有变量类型、布局或物理 chunk ownership 不兼容时才回退到独立最终化。任务 manifest 会记录资源快照、worker/压缩候选、选择原因、逻辑写放大和避免的最终化 I/O。

命令行入口：

```bash
pixi run pipeline \
  --input /path/to/gosif-tif \
  --input-kind raw \
  --output /path/to/gosif.zarr \
  --lat 30 90 --lon -180 180 \
  --resample --resolution 0.1 \
  --rechunk --recompress \
  --method conservative --skipna \
  --strategy time \
  --compression-codec blosc-zstd --compression-level 4 \
  --inspection-cache /path/to/cache/gosif-inspection.json \
  --temporary-dir /path/to/ssd/pipeline-temporary
```

`--dry-run` 只执行检查和计划，不写数据；
`--input-kind auto|raw|zarr` 用于声明输入类别；`auto` 会根据 Zarr v3 根元数据自动区分
现有 Zarr 与原始 NC/HDF/TIFF 目录。现有 Zarr 会跳过转换，并要求至少选择重采样、
重分块或重压缩中的一项。仅选择一项时直接发布最终结果；重采样与存储优化组合时，
规划器生成“重采样 → 最终化”两阶段任务，仍然只发布一份最终 Zarr；
`--resample`、`--rechunk`、`--recompress` 分别开启对应可选操作；三者都不提供时
执行仅转换流程。`--resolution` 只在同时选择 `--resample` 时有效。
替换规则通过 `--before-conditions`、`--before-results`、`--after-conditions` 和
`--after-results` 提供；`--statistics-policy auto|sample|exact` 控制统计表达式。
重压缩既可以继续使用旧 `--compression` profile，也可以通过 `--compression-codec`、
`--compression-level` 和 `--compression-shuffle` 明确指定最终 Zarr v3 codec。
`--cleanup-intermediate` 会在下游验证通过后删除已不再需要的上游临时 Zarr。完整工作流与
阶段边界见 [一条龙处理模块文档](docs/pipeline.md)。
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
当自动时间块小于最终输出的 time chunk 时，每个空间任务会独占完整的最终物理 chunk，
将各时间批次填入受预算约束的内存缓冲或任务专属 memmap，填满后只编码写入一次。
这避免了完整时间中转 Zarr 的写入和回读；临时 memmap 位于 `--temporary-dir`，成功、取消和失败路径都会清理。

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

重分块的工作流和完整参数见 [重分块与重压缩模块文档](docs/rechunking.md)。
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
启用重压缩时使用快速 codec，最终输出再使用用户选择的压缩方案；worker 数依据有效 CPU、内存、文件描述符和物理 chunk ownership 生成候选，并在源/临时/目标文件系统上分别真实测速选择。源/目标存储 profile 只作为 benchmark 上下文，不再静态截断并发。时间连续型的
阶段 1 按源 time chunk 隔离后使用多进程，阶段 2 按最终 chunk 隔离后使用多进程；数据写入采用
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
