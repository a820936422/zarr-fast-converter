# Fast NC Zarr v1.7.5 代码审查与性能优化发布状态报告

- **审查日期**：2026-08-15
- **审查对象**：`/run/media/owen/HDD/zarr-fast-converter-v1`
- **项目定位**：面向个人用户的 NetCDF/HDF/TIFF 等遥感数据到 Zarr v3 的转换、重采样、重分块、重压缩和 Tauri 桌面处理平台
- **审查方式**：4 个只读审查子代理覆盖 Rust 核心、Python 数据处理链、Tauri/桌面安全、质量与发布运维；随后对 P0、非 P0 和性能优化进行源码复核、定向行为复现与测试验证。严重度统计保留原始 v1.7.3 基线，以下状态以当前 v1.7.5 工作树为准。

## 1. 结论摘要

当前版本已完成上一轮 P0 修复，并完成一轮非 P0 优先级收敛，同时落地首轮性能优化：native resampling 改为 typed buffer bridge，Rust lookup 预计算并移除每个目标格点的轴复制/线性扫描；native NetCDF staging/语义校验、取消终态、能力矩阵、Dask CF 引用清理、Tauri CSP、worker stderr、sidecar stale artifact、release target 路径、wheel 依赖声明和协议 schema 校验均已补强。

**仍不建议将当前版本宣称为“数据正确性保证”的稳定发布版。** 剩余主要风险集中在 native NetCDF 的大变量内存与 packed-data 完整 parity、非标准 calendar、恢复 manifest 信任边界、同用户 TOCTOU/symlink 竞态、sidecar 签名与安装包签名、以及安装后桌面 IPC smoke。

已完成的非 P0 修复不等于全部风险关闭；本报告把“已修复”“部分缓解”“仍开放”分开记录，避免把历史基线发现误当作当前代码状态。

### 严重度统计

| 严重度 | 含义 | 本报告主要问题数 |
|---|---|---:|
| Critical | 不可信输入可触发崩溃、任务悬挂或直接造成严重数据错误 | 1 |
| High | 可能产生错误科学结果、伪成功/部分输出、关键功能不可用或发布信任边界失效 | 17 |
| Medium | 资源耗尽、竞态、契约/发布可靠性或防御纵深问题 | 17 |
| Low | 隐私、文档、覆盖率和长期维护问题 | 2 |

### 当前 v1.7.5 修复状态

| 范围 | 状态 | 证据 |
|---|---|---|
| P0 数据发布、语义校验、取消终态、native resample 越界 | 已完成 | 当前 Rust/Python 定向回归与 native smoke 已通过；仍需持续补充真实格式 parity fixture |
| native resampling typed buffer 与 Rust lookup | 已完成 | `resample_f32_buffer`、预计算 axis brackets、`tests/test_native_smoke.py` 与 `benchmark-native-resample` |
| Python wheel 依赖 | 已完成 | `pyproject.toml` `[project].dependencies` 与 `resampling` extra |
| worker stderr、CSP、sidecar stale artifact、release target 路径 | 已完成/部分缓解 | `worker.rs`、`tauri.conf.json`、sidecar/release scripts；签名与 digest enforcement 仍开放 |
| 协议 schema、fixture、版本一致性 | 已完成基础门禁 | `scripts/check_contracts.py`、`contract-check`、`check_version_consistency.py` |
| 安装后桌面启动/IPC、非标准 calendar、恢复边界、TOCTOU | 仍开放 | 见 H-07、M-05、M-06、M-08、M-16 |

## 2. 审查范围与架构梳理

### 2.1 主要组件

- **Python 数据处理链**：`src/fast_nc_zarr/`，负责 CLI、inspection、filename mode、转换、pipeline、resampling、rechunking、publication、校验和 desktop worker。
- **Rust workspace**：
  - `rust/crates/fast-nc-zarr-model`：能力矩阵、请求/计划/事件模型。
  - `rust/crates/fast-nc-zarr-zarr`：Zarr v3 访问、NetCDF native conversion、native resampling、rechunk、write。
  - `rust/crates/fast-nc-zarr-python`：PyO3 扩展。
- **Tauri 桌面端**：`apps/desktop/src-tauri/src/`，负责命令、任务 registry、native task、sidecar worker、事件和恢复；React 前端位于 `apps/desktop/src/`。
- **跨进程协议**：Rust 与 Python worker 使用 JSONL；契约和 fixture 位于 `contracts/`。
- **发布模型**：Python 和 native NetCDF 主路径均使用 staging + validation + atomic publish；仍需继续审查 native task 与恢复路径的统一 commit 边界。

### 2.2 关键边界

当前最重要的边界是：前端只调用 Tauri commands；Tauri 将复杂处理转给 Python worker，同时对部分操作走 Rust native；能力矩阵与桌面 dispatcher 已有 fixture/schema 门禁，但仍维护两套清单，尚未收敛为单一事实源。

## 3. 关键问题清单

### 3.1 Critical：输入错误可触发 native 线程 panic，任务可能悬挂

**C-01 — 非数值标准坐标可能触发 `unreachable!()`。**

- **证据**：`rust/crates/fast-nc-zarr-zarr/src/netcdf_native.rs:127-158` 只按变量名和一维维度判断 `time`/`lat`/`lon` 为标准坐标，没有要求 dtype 为 numeric；后续 conversion 在 `:444-553` 按 dtype 分派，未支持类型会落到 `_ => unreachable!()`。任务线程在 `apps/desktop/src-tauri/src/native.rs:79-93` detached spawn，未见统一 panic 边界。
- **触发条件**：输入存在精确命名的非数值一维标准坐标，同时还有满足形状的三维变量。
- **影响**：不可信输入可导致 native 线程 panic；任务可能停留在 `Running`，并可能已经创建部分输出。
- **建议**：inspection 阶段要求标准坐标为数值型；所有 dtype fallback 返回结构化错误，禁止 `unreachable!()` 处理外部输入；native task 入口增加 panic 转 terminal failure 和 cleanup 边界。

### 3.2 High：科学数据正确性和发布一致性

**H-01 — native NetCDF conversion 直接写最终输出目录，失败会留下部分 Zarr。**

- **证据**：`rust/crates/fast-nc-zarr-zarr/src/netcdf_native.rs:396-431` 只检查 `output.exists()`，随后直接 `create_dir_all(output)` 和写 group metadata/data；没有 staging、回滚或失败清理。
- **影响**：磁盘、权限、读取或写 chunk 失败后，最终路径仍可能存在半成品；自动化或用户可能把它当成有效结果。这与模块总览 `docs/README.md` 和根 README 的 staging 承诺冲突。
- **建议**：所有 native output 先写唯一 sibling staging，执行结构/语义校验，再原子 rename；失败、取消和 panic 路径都清理 staging，必要时对 metadata 和父目录执行持久化同步。

**H-02 — `_FillValue`、`scale_factor`、`add_offset` 没有保持为统一的 Zarr 语义。**

- **证据**：float builder 在 `netcdf_native.rs:192-197,245-250` 固定使用 `NaN` fill，integer builder 在 `:344-362` 固定使用零；源属性仅复制到 attrs，未用 `_FillValue` 配置 Zarr fill，也未应用 packed data 的 scale/offset。
- **影响**：缺测值可能被当作普通值；原始 packed 值和物理量值可能混淆；下游 CF/xarray 读取结果会改变。能力模型又在 `rust/crates/fast-nc-zarr-model/src/lib.rs:157-164` 声明 fill 保持，形成错误承诺。
- **建议**：明确 conversion 输出是 raw storage 还是 decoded physical values；据此一致处理数据、Zarr fill、`_FillValue`、`missing_value`、scale/offset 和 attrs；增加非 NaN fill、packed float/integer 的端到端测试。

**H-03 — native nearest 对越界目标坐标返回边缘值，而不是 NaN。**

- **证据**：`rust/crates/fast-nc-zarr-zarr/src/resample_native.rs:73-96` 对每个目标坐标直接选择最近源索引；只有 bilinear 的 `bracket` 在 `:98-105` 做越界判断。能力声明和 README 要求越界为 NaN。
- **复现**：源坐标 `[0,1]`、目标 `(2,2)` 的 native nearest 结果为 `4.0`，预期应为 `NaN`。
- **影响**：源范围外的地理区域被填入边缘观测值，产生伪数据。
- **建议**：nearest 查找前显式比较两个轴的 min/max，任何轴越界直接写 NaN；补充升序、降序、边界内外测试。

**H-04 — native regular resample 破坏文档宣称的流式/有界内存模型。**

- **证据**：`src/fast_nc_zarr/resampling/engine.py:1989-2024` 使用 `.values` 读取整个变量，转为 Python nested list，JSON 序列化后传给 Rust；多个变量的输出保存在 `variables` 中再统一写出。Rust 端 `resample_native.rs:69-72` 又 clone 输入并分配完整输出。
- **影响**：源数组、Python object/list、JSON、Rust buffer、解码结果和输出数组同时占用内存；大规模个人数据可触发 OOM 或桌面进程退出。
- **建议**：改为 chunk/region native API 或 typed buffer 零拷贝/低拷贝传输；按变量和 chunk 增量写出，并把 `ResamplePlan` 的内存预算落实为硬限制。

**H-05 — native resample 输出没有完整保留源 encoding/fill 语义。**

- **证据**：native 路径在 `engine.py:2017-2018` 只复制变量 attrs；普通路径在 `:214-224,299-337` 另行构造 `_FillValue`、`missing_value`、chunks、codec 等 encoding。`_validate_output` 的 `:1822-1875` 未比较这些语义字段。
- **影响**：输出可通过结构校验，但缺测处理、压缩布局、scale/offset 或 CF 解码行为已改变。
- **建议**：显式复制兼容 encoding，或明确规范化并在 manifest 中记录；校验 fill、scale/offset、CF refs、coordinate attrs 和 codec。

**H-06 — TIFF filename mode CRS/投影一致性校验。**【已修复】

- **修复**：`src/fast_nc_zarr/filename_mode.py:1000-1026` 将 rasterio fallback 的规范化 CRS/WKT 与 transform 纳入跨文件 signature；不一致时沿用结构不一致错误拒绝。
- **验证**：`tests/test_filename_mode.py` 新增 CRS 与 transform 差异回归；现有 rasterio low-level 路径继续保留原生 CRS/transform signature。

**H-17 — Python package wheel 未声明运行时依赖。**【已修复】

- **修复**：`pyproject.toml` 已声明 dask、h5netcdf、netCDF4、numpy、rasterio、rioxarray、xarray、zarr 等核心依赖，并将 xESMF 放入 `resampling` optional extra；Pixi 环境同步锁定。
- **验证**：`tests/test_protocol.py` 检查 project metadata；`pixi run version-check` 通过。尚未执行独立干净环境 wheel install smoke。

**H-07 — 非标准 calendar 可能被强制转换为 Gregorian `datetime64`。**

- **证据**：`src/fast_nc_zarr/inspection.py:366-381` 可得到 cftime 对象，但 `_normalize_daily_times()` 在 `:97-129` 强制通过 `np.datetime64` 并保存 Gregorian 日期；ISO fallback 也只能解析 Gregorian 日期。
- **影响**：`360_day`、`365_day`、`all_leap` 等 calendar 的日期身份可能丢失或无法表示，时间筛选和输出坐标会错误。
- **建议**：保留 calendar-aware 类型并写入明确 calendar metadata，或显式路由到 compatibility backend；拒绝时返回确定性的 capability/error，而不是隐式强转。

**H-08 — inspection 宣称整数 NetCDF 可支持，但 conversion 明确拒绝。**

- **证据**：inspection 在 `netcdf_native.rs:78-96,150-159` 对 numeric integer data 仍可给出 `supported_subset=true`；conversion 在 `:414-421` 只接受 float32/float64 data。
- **复现**：构造含 int16 data、非 NaN fill、scale/offset 的 NetCDF，native inspection 返回 `supported_subset=True`，conversion 返回 `RuntimeError: native conversion supports float32/float64...`。
- **建议**：让 inspection predicate 与 conversion 完全一致，或真正实现整数 conversion；加入 signed/unsigned、fill、packed integer fixtures。

### 3.3 High：任务状态、协议和桌面可靠性

**H-09 — 取消机制可能报告错误终态，甚至“已完成但输出已被删除”。**【已修复：commit 状态已补强】

- **证据**：`apps/desktop/src-tauri/src/native.rs:157-200` 对 conversion/resample/write 没有传递取消状态，只在返回后根据 cancellation file 选择 terminal event；`native_rechunk` 在 `:366-392` 先 join，若检测到取消则删除 target，但随后仍可将成功 `result` 转成 `Ok`。Python pipeline worker 在 `src/fast_nc_zarr/application/desktop_worker/worker.py:378-383` 也在 `run_pipeline()` 返回后才判断 cancel，发布可能已发生。
- **影响**：任务可能显示 `cancelled` 但已有输出，或显示 `finished` 但输出刚被删除；重试、恢复和用户判断均不可靠。
- **建议**：引入显式 commit 状态：发布前取消可中止，发布成功后取消只记录“late cancel”而不改终态；所有 output-producing native 操作均使用 staging、周期性 poll、发布前最后一次检查；terminal registry 更新成功后再发 terminal event。

**H-10 — Python worker 的 JSONL stdout 可能被人类可读 inspection 输出污染。**【已修复】

- **证据**：`worker.py:354-379` 在 inspection/planning 阶段没有像 execution 一样包裹 stdout/stderr；`inspection.py:647-651` 和 `filename_mode.py:1050-1058` 存在直接 print；`worker.py:396-409` 将每个 stdout line 当作 JSON 请求/事件协议处理。
- **影响**：Rust/Tauri 端可能遇到非 JSON 行，导致 JSON decode failure 或事件顺序异常。
- **建议**：worker stdout 只允许协议事件；所有进度和诊断走注入的 event sink 或 stderr；为 `inspect_source`、`preview_pipeline`、snapshot 等命令补充 JSONL 集成测试。

**H-11 — `cancel_task` 无法取消同一 worker 正在执行的请求。**【协议边界已明确；文件取消路径已生效】

- **证据**：`worker.py:396-409` 同步逐行 dispatch；pipeline 在 `:360-392` 阻塞；`cancel_task` 在 `:310-312` 只设置当前 dispatch 调用收到的 event；每个请求在 `:407-408` 新建一个 `ThreadEvent`。
- **影响**：取消命令要等当前长任务结束后才会被读取，因此对原任务无效；只有独立 cancellation-file monitor 在特定路径上可能生效。
- **建议**：维护 `task_id -> cancel event` 注册表，让长任务在线程/进程中执行并由 cancel 命令路由到对应 event；或从 Python worker 协议删除该命令并明确只支持 cancellation file。

**H-12 — semantic constraints 只是 warning，且 direct-final 路径先发布后校验。**【已修复】

- **证据**：`src/fast_nc_zarr/validation.py:66-92` 将约束违规记录为 warning；`pipeline/engine.py:1349-1355,1385-1389` 仍可将 manifest 标为 succeeded。raw direct-final 在 `pipeline/engine.py:1078-1145` 调用 conversion，resampling direct-final 在 `resampling/engine.py:2271-2299` 发布，而语义校验位于后续 pipeline 路径。
- **影响**：配置了 `min/max/nonnegative` 等约束但数据不满足时，任务仍可能成功；即便未来改为 strict，直接输出路径也已替换用户目标。
- **建议**：约束存在时把 semantic validation 作为硬 gate；所有路径统一写私有 staging，完整校验通过后只进行一次 atomic publish；增加“语义失败时目标保持不变”的测试。

**H-13 — raw validation 对完整 time/lat/lon 数据集可能误拒绝。**

- **证据**：`src/fast_nc_zarr/raw_validation.py:52-85` 只抽样规范化维度恰好为 `{lat,lon}` 的变量；无此变量就抛错。`101-143` 的 smoke conversion 无条件调用 `convert_filename()`；`filename_mode.py:849-867` 对 size>1 的 time 维度拒绝。
- **影响**：完整的多时间 NetCDF/HDF 数据集无法生成文档所述 validation report 或 smoke output，模块实际只覆盖 filename-mode 2-D slice。
- **建议**：按 `Inventory.source_mode` 分支；完整数据走正常 source reader 和 `run_conversion()`，只有 filename-mode 使用 `convert_filename()`；补充多时间完整数据 fixture。

**H-14 — Dask fallback 重命名变量后未清理 CF references。**【已修复】

- **证据**：`src/fast_nc_zarr/engine.py:154-220` 的 Dask path 重命名后直接写出，没有调用 `sanitize_cf_references()`；direct writer 在 `writer.py:454-457` 会调用，metadata helper 在 `metadata.py:60-99` 处理 bounds、grid_mapping、coordinates、ancillary_variables、cell_measures、formula_terms。
- **影响**：输出仍引用旧变量名或已被 selection 删除的变量；CF 消费者可能静默丢失边界、质量标记、网格映射或公式项。
- **建议**：在 Dask rename 和 selection 后调用相同 sanitization，并增加包含 bounds/grid_mapping/ancillary/formula 的测试。

**H-15 — native single-variable rechunk 允许 source/target 嵌套路径。**

- **证据**：`apps/desktop/src-tauri/src/native.rs:338-352` 只检查 target 是否已存在；`fast-nc-zarr-zarr/src/lib.rs:672-718` 没有 source/target overlap guard；copy walk 在 `:153-208` 会递归遍历 source。
- **影响**：当 target 是 source 的新建子目录时，创建 target 后 copy walk 可能再次遍历 target，造成路径增长、磁盘耗尽或源/目标污染。
- **建议**：在创建任何目录前解析 existing ancestors、拒绝 source identity/nesting，并对 symlink/alias 做安全检查；single 和 multi entry point 使用同一 guard。

### 3.4 High：桌面安全与供应链边界

**H-16 — release worker 选择信任环境变量和可写开发路径。**

- **证据**：`apps/desktop/src-tauri/src/worker.rs:19-61` 依次信任 `FAST_NC_ZARR_WORKER`、可执行文件旁 worker、`FAST_NC_ZARR_PROJECT_ROOT` 下 binary，最后回退到 `PYTHON`；仅使用 `is_file()`，不校验签名、digest、owner、symlink 或 trusted install root，并继承 `PYTHONPATH`。
- **影响**：在被篡改的启动环境、可写安装目录或开发 checkout 中，应用可执行攻击者控制的 sidecar/Python，获得当前用户权限。
- **建议**：release 构建忽略这些 override，只从受信任资源目录加载 regular executable；校验 digest/signature，sidecar 缺失时 fail closed；环境 override 只保留在明确的 development build。

### 3.5 Medium：资源、竞态、契约和发布可靠性

**M-01 — native NetCDF conversion 整变量 materialize，资源快照没有形成硬预算。**

- **证据**：`netcdf_native.rs:444-552` 对每个变量调用 `get_values::<T,_>(..)` 后才分块；`native.rs:61-73,151-156` 只记录 resource snapshot，不将内存预算传给 conversion。
- **影响**：大变量或多变量输入可造成高 RSS 和进程终止。
- **建议**：使用 NetCDF hyperslab/chunk 增量读取，设置输入/输出估算上限，并将 buffer 内存纳入资源 gate。

**M-02 — 错误的 shuffle 字符串静默降级为 `NoShuffle`。**

- **证据**：`rust/crates/fast-nc-zarr-zarr/src/lib.rs:742-746,1114-1117` 对未知值使用默认 `NoShuffle`。
- **影响**：用户输入拼写错误时任务成功但 codec 与请求不同，压缩率和性能不可预测。
- **建议**：只接受显式枚举值，未知值返回 `InvalidRequest`；把 no-shuffle 作为明确字符串而非 fallback。

**M-03 — task lifecycle 错误可能留下永久 Running 或删除旧状态。**

- **证据**：`native.rs:63-94` 在 thread spawn 失败时已持久化 Running 但未 terminalize；`tasks.rs:268-301` 在 fallback rename 前移除旧状态；多个 registry/event 更新错误被忽略。
- **影响**：恢复 UI、取消句柄和历史状态不可靠，严重时任务记录消失。
- **建议**：spawn/persist/event 失败都进入可恢复 terminal failure；使用同目录 durable replacement，确认新文件安装后再删除旧文件；区分事件失败和任务状态正确性。

**M-04 — conversion/rechunk 没有源快照或并发修改检测。**

- **证据**：rechunk 在 `fast-nc-zarr-zarr/src/lib.rs:672-718,888-938,1145-1250` 读取 metadata、复制 auxiliary 文件后再读 chunks；NetCDF conversion 在 `netcdf_native.rs:406-454` 先 inspect 后读取变量；未验证 source identity/version。
- **影响**：源文件或 Zarr 在执行期间被修改时，输出可能混合不同版本的 metadata 和 data，且没有确定性错误。
- **建议**：要求输入只读/不可变，或记录并复核文件 size/mtime/inode/store metadata；发现变化就拒绝发布。

**M-05 — Python publication 存在同用户 TOCTOU 删除/替换窗口。**

- **证据**：`src/fast_nc_zarr/publication.py:89-121` 分开执行目标校验、`os.replace(target, backup)` 和最终 publish；代码注释也承认 race。
- **影响**：同一用户的竞争进程可在校验后替换 target，导致应用备份并递归删除非预期目录。
- **建议**：使用 directory fd、`openat`/`renameat`、`O_NOFOLLOW` 等绑定目录项的文件系统原语，或在 destructive cleanup 前验证 inode/identity；不递归删除未经证明是原目标的 backup。

**M-06 — inspection snapshot 的可预测 `.tmp` 可被 symlink 竞态利用。**

- **证据**：`src/fast_nc_zarr/application/services.py:904-913` 使用 `destination.name + ".tmp"`，调用 `write_text()` 后 `replace()`；没有 exclusive create/no-follow。
- **影响**：同用户进程可预先创建 symlink，使 snapshot 写入其他用户可写文件。
- **建议**：使用 `O_CREAT|O_EXCL|O_NOFOLLOW` 创建随机临时文件，写完并 fsync 后再原子替换。

**M-07 — task、worker、payload 和历史没有统一上限。**

- **证据**：`native.rs:32-94`、`pipeline.rs:12-68`、`tasks.rs:102-132` 为每个请求创建线程/worker；`worker.rs:14-16` 长期保留 sequences/terminal_tasks；`worker.rs:105-121`、Python protocol `:85-96` 和 native payload `native.rs:510-549` 未限制 JSON 行长度、深度、数组 cardinality 或估算输出。
- **影响**：恶意/故障前端可耗尽线程、进程、CPU、内存、临时磁盘和 task-state；大 JSON/数组可在反序列化前造成放大。
- **建议**：设置 active task、worker、payload bytes、JSON depth、array length、dimension、output estimate 和 history retention 上限；超限在分配前拒绝。

**M-08 — recovery manifest 被当作可信配置，输出路径边界不完整。**（部分为假设，需运行时确认）

- **证据**：`src/fast_nc_zarr/pipeline/recovery.py:82-188,253-316` 从用户选择的 manifest 恢复 output、overwrite、backend、worker 等；checkpoint 有 containment 校验，但恢复的 final output 不一定限制在 recovery job 目录。Tauri 在 `apps/desktop/src-tauri/src/pipeline.rs:90-116` 转发用户路径。
- **风险**：恶意 manifest 可能指定其他输出路径并启用 overwrite。
- **建议**：把 manifest 视为不可信导入；恢复前重新确认 resolved output，禁止越出用户选择的目标范围，忽略持久化 overwrite，并重新执行 source/output preflight。

**M-09 — `csp: null` 缺少 WebView 防御纵深。**【已修复】

- **修复**：`apps/desktop/src-tauri/tauri.conf.json:22-24` 已配置 restrictive CSP，限制脚本、连接和资源来源；当前 frontend 仍未发现 `innerHTML`、`eval` 或远程 origin sink。
- **验证**：Tauri 配置解析与前端构建门禁继续覆盖该配置；未来新增远程资源时仍需同步收紧 capability。

**M-10 — release package 签名与 runtime sidecar integrity verification。**【仍开放；部分缓解】

- **当前状态**：CI 现在要求 sidecar 生成、可执行且通过 `.sha256` 校验；发布收集脚本也输出 `SHA256SUMS`。
- **剩余风险**：CI 仍使用 `--no-sign`，Tauri runtime 未在启动时验证 sidecar digest，deb/rpm 仍无发布签名/provenance gate。
- **下一步**：引入发行签名和可信 provenance；把 sidecar digest 纳入构建时或安装后可验证的信任根。

**M-11 — worker stderr 被丢弃，启动和 import 失败缺少诊断。**【已修复】

- **修复**：`apps/desktop/src-tauri/src/worker.rs` 使用有界 stderr reader 保留最近 32 行，并在 worker EOF/JSON decode 错误中附带诊断尾部。
- **验证**：Rust desktop worker tests 覆盖 worker 生命周期与错误路径；stderr 不再使用 `Stdio::null()`。

**M-12 — sidecar 脚本可能在没有生成新 worker 时成功退出。**【已修复】

- **修复**：`scripts/build_desktop_sidecar.sh` 在构建前删除目标 sidecar/checksum；缺少 PyInstaller、生成物不可执行或复制失败均返回非零；成功后写出 checksum。
- **验证**：`bash -n` 通过；CI 在 Tauri build 前检查 sidecar 与 checksum。

**M-13 — release-candidate 的 Cargo 输出路径未带 target triple。**【已修复】

- **修复**：`pixi.toml` 改为调用 `scripts/collect_release_candidate.sh`；脚本按 `TAURI_TARGET`/`TARGET` 和 profile 选择 `target/<triple>/<profile>/bundle`，并生成 `SHA256SUMS`。
- **验证**：临时 target-triple deb/rpm bundle smoke 已通过。

### 3.6 Medium/Low：测试、契约、文档和隐私

**M-14 — contract 版本和 fixture 仍是旧版本，版本检查未覆盖它们。**【已修复】

- **修复**：`contracts/README.md`、`contracts/fixtures/capability-v1.json` 已同步 1.7.5；`scripts/check_version_consistency.py` 已将 contract 与 docs release marker 纳入检查。
- **验证**：`pixi run version-check` 通过，所有覆盖项均为 1.7.5。

**M-15 — contract 测试没有真正校验 JSON Schema 和 canonical fixture。**【已修复基础门禁】

- **修复**：新增 `scripts/check_contracts.py`，使用 Draft 2020-12 validator 校验 request/event/error/capability schemas 与 checked-in fixtures；新增 Pixi `contract-check` 和 CI step。
- **验证**：`contract schema validation passed: 2 event fixture(s)`；`tests/test_protocol.py` 与 fixture 行为回归通过。动态 Tauri native command 与 Python worker command 仍保持分层契约。

**M-16 — 桌面 CI 未测试安装后应用。**【部分修复】

- **已修复**：workflow 已执行 `cargo test -p fast-nc-zarr-desktop --lib`，构建 sidecar 后验证 executable/checksum，并继续执行 frontend/typecheck/Tauri package checks。
- **剩余风险**：CI 尚未启动已安装 deb/rpm 或 packaged binary，也未执行真实 Tauri command/worker IPC smoke；Wayland/X11、sidecar runtime libs 和安装后资源加载仍需单独门禁。

**M-17 — Python wheel 之外的真实格式覆盖仍不足。**【仍开放】

- **当前覆盖**：`tests/test_filename_mode.py` 已包含 rasterio GeoTIFF low-level、cleanup/conversion 和 HDF-EOS `StructMetadata.0` 网格 fixture；当前全套 Python tests 已通过。
- **剩余缺口**：仍缺多 band/旋转 GeoTIFF 的完整转换 smoke、packed integer/calendar parity、HDF group 变体和安装后真实数据 smoke。

**L-01 — 绝对路径会被持久化到浏览器存储、task state、manifest 和错误。**

- **证据**：`apps/desktop/src/App.tsx:199-219` 使用 localStorage 保存路径；`tasks.rs:39-55`、`pipeline/engine.py:662-715`、`error.rs:71-82` 保存或返回原始路径。
- **影响**：同一桌面 profile 的其他软件/用户可获知本地数据位置、目录结构和工作流；错误信息也可能泄漏路径。
- **建议**：UI/history 使用 opaque label，诊断文件限制权限，错误向用户返回脱敏路径，详细绝对路径只在受保护日志中保留。

**L-02 — 文档链接引用不存在的开发方案文件。**【已修复】

- **修复**：当前 `README.md` 与 `docs/README.md` 已移除不存在的 `v1.7.3-development-plan.md` 链接，改为实际存在的模块指南与当前审查报告。
- **验证**：仓库搜索未发现该失效链接。

## 4. 已验证的工程优势

以下内容在源码、定向测试或两者中得到验证，不应因上述问题而被忽略：

- `array_relative_path` 在 `fast-nc-zarr-zarr/src/lib.rs:94-109` 拒绝非 normal path component；copy store 还拒绝 symlink 和不支持的 filesystem entry（`:153-186`）。
- multi-variable rechunk 对重复 array path、dtype/dimension 不一致、source/target lexical overlap、fill/attrs 不一致有前置校验（`lib.rs:1214-1225,1334-1442`）。
- rechunk 使用不重叠 chunk ownership 和有界 Rayon；progress 文件采用原子替换（`lib.rs:127-150,1233-1284`）。
- Python publication/rechunk/resample 的主路径普遍使用 staging、结构验证和 atomic publish；`tests/test_runtime_publication.py` **8 passed**。
- filename mode 已覆盖 DOY inference、歧义、缺口、transform、axis reversal 和固定输出 chunks；相关 focused tests 已通过。
- native resampling 快速路径通过 typed buffer 避免 Python list/JSON 往返；Rust 核心预计算 target axis lookup，并直接按 flat offset 读取 source values。
- Python runtime 有 bounded process execution、取消检查和失败时终止 pending work 的机制（`runtime.py:58-132`）。
- Rust worker 有 protocol version、request ID、event type 和 sequence 检查（`protocol.rs:56-100`、`worker.rs:119-161`）。
- task registry 能持久化 terminal state、重启时标准化 active task 并清理 cancellation handle（`tasks.rs:183-248`）。
- Tauri capability 当前只声明 `core:default`/`dialog:default`，未发现远程 URL capability；审查范围内未发现 shell command injection、Python `eval/exec` 或前端 HTML injection sink。
- 版本一致性脚本对 runtime/build manifests、contract 和 release docs 检查通过，当前版本为 1.7.5。
- sidecar worker 现在有 stderr diagnostics、stale-output fail-closed 和 checksum artifact；安装包签名与 runtime digest trust root 仍开放。

## 5. 本次验证结果

### 5.1 定向测试与命令

| 检查 | 结果 |
|---|---|
| `pixi run contract-check` | 通过，2 个 event fixture 与 request/error/capability schema 校验通过 |
| `pixi run version-check` | 通过，所有覆盖项为 1.7.5 |
| `pixi run python -m pytest -q tests/test_protocol.py tests/test_filename_mode.py` | **26 passed** |
| `python -m py_compile`、`bash -n` | 通过，覆盖本次编辑的 Python 与 release scripts |
| target-triple release collector smoke | 通过，临时 deb/rpm bundle 被正确复制并生成 `SHA256SUMS` |
| `pixi run test` | **194 passed**, 1 warning, 29 subtests passed |
| `pixi run rust-test` | Rust workspace excluding desktop passed: model 1, Zarr 10, native crate 0; doc-tests passed |
| `cargo test -p fast-nc-zarr-desktop --lib` | **21 passed** |
| `pixi run desktop-typecheck` / `pixi run desktop-build` | both passed |
| `cargo fmt --all -- --check` / `pixi run native-check` | both passed |
| `benchmark-native-resample` smoke | 通过，2×64×64→2×32×32 样例 typed bridge 相对 JSON bridge 约 308×；该数值为单机 smoke，不作为固定 SLA |
| `pixi run python -m pytest -q tests/test_native_smoke.py tests/test_resampling.py` | **60 passed** |
| `pixi run rust-desktop-test` | **21 passed**，Pixi task 已使用 host linker environment |
| `pixi run rust-clippy` | 通过，workspace excluding desktop，`-D warnings` |

### 5.2 历史基线与当前边界

上一轮 v1.7.3 审查曾直接复现整数 NetCDF capability drift 和 native nearest 越界；当前代码已加入对应 P0 回归/contract 覆盖，native smoke 与 Rust tests 必须继续作为 release gate。该结果不等于 native 与 Python 在所有 packed dtype、calendar、CRS 和大数组上的科学 parity 已证明。

## 6. 修复优先级与发布门槛

### P0：当前工作树已完成，发布前继续保持回归门禁

1. native NetCDF staging + validation + atomic publish、失败清理和 panic-to-terminal handling。
2. `_FillValue`/packed scale/offset 语义、nearest 越界、inspection capability/dtype parity。
3. cancellation/commit state、semantic hard gate、Dask CF reference sanitization。

### P1：下一迭代按用户数据损害和可恢复性排序

1. 消除 JSON/list native resample 的全量 materialization，落实内存/输出配额。
2. 明确非标准 calendar 的 compatibility backend 或 deterministic rejection。
3. 将 recovery manifest、publication snapshot 和 destructive overwrite 绑定到可信目录项，修复 same-user TOCTOU/symlink 竞态。
4. 收敛 capability matrix 与 dispatcher 为单一事实源，并补齐整数/packed/TIFF/HDF-EOS 真实 parity fixtures。
5. 为 sidecar/安装包建立签名、provenance 和 runtime digest trust root。
6. CI 启动 packaged/installed desktop binary，执行至少一次真实 Tauri command/worker IPC smoke。

### P2：长期工程质量

1. 为 task、JSONL、数组、维度、输出估算和历史设置统一上限。
2. 脱敏绝对路径，恢复或更新缺失的开发文档链接。
3. 对 TIFF/HDF-EOS/packed/calendar/异常维度建立小型真实数据集并持续纳入 CI。

### 建议的 release gate

只有在以下条件同时满足后，才建议把版本标记为“数据正确性可接受”：

- 所有 P0 项有回归测试；
- native 与 Python 结果在 fill/scale/offset、越界、CRS、calendar 维度完成 parity 证明；
- 所有 output-producing path 都能证明“校验前不可见、失败无伪成功输出”；
- 能力矩阵与可执行 dispatcher 通过自动化一致性测试；
- CI 能启动安装包并执行至少一个真实 Tauri command/worker IPC；
- 至少一组真实 TIFF、HDF-EOS 和 packed NetCDF 数据进入自动化 smoke。

## 7. 审查限制

- 已执行完整 Python 回归套件（194 passed）、Rust core/desktop tests、frontend typecheck/build、contract/version/native checks 和 typed-buffer benchmark；仍未执行完整 Linux deb/rpm release build、安装后 packaged launch 或图形界面人工操作。
- 没有对 `zarrs`、filesystem store、Tauri/WebView 等第三方实现做源码级安全审计；因此 `array_path` traversal、manifest 恢复边界和 CSP 项目中的部分问题标记为“需运行时/依赖确认”。
- 抽样校验不等价于全部像元的科学等价证明；真实数据中的 HDF/TIFF/CRS/calendar 仍需 fixture 和 end-to-end 验证。

---

**总体评级：有工程基础，但当前仍处于“功能可运行、数据正确性和发布边界未闭合”的阶段。** 对个人用户产品，优先保证结果不被错误标记、部分输出不被误用、缺测/坐标语义不被静默改变，再扩展格式和性能能力。
