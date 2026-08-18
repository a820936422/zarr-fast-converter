# Fast NC Zarr 自动优化代码逻辑汇总

- **文档更新**：2026-08-18
- **当前版本**：v1.7.9
- **关联文档**：[v1.7.8 后端优化分析](v1.7.8-backend-optimization-analysis.md)、[v1.7.9 后端处理逻辑优化方案](v1.7.9-backend-optimization-plan.md)、[维护文档](MAINTENANCE.md)

本文档汇总当前程序中所有“自动优化 / 自动调参”相关代码逻辑，作为后续优化方案的基线。内容以当前工作区源码为准。

---

## 1. 目的与范围

本文档覆盖以下自动优化能力：

- 资源探测与预算（CPU / 内存 / 存储 / cgroup / FD）
- 转换（NetCDF/HDF/TIFF → Zarr v3）的 plan 生成与实测调优
- 重采样（xESMF / native）的 space workers、time block、tile size、buffer 预算自动选择
- 重分块 / 重压缩的 worker 与 codec 自动选择
- 一条龙 pipeline 的融合 / 最终化自动决策
- 进程调度与线程环境约束

---

## 2. 总体数据流

```text
硬件/资源探测 (system.py)
        │
        ▼
工作负载分类 (planner.workload_kind / pipeline 输入检查)
        │
        ▼
初始计划 (initial_plan / plan_resample / plan_chunks)
        │
        ▼
候选生成 (candidate_plans / compression_candidates / worker_candidates)
        │
        ▼
样本实测调优 (benchmark.tune / benchmark_worker_candidates / benchmark_compression_candidates)
        │
        ▼
选择最优 plan（speed / balanced / compact + 磁盘可行性）
        │
        ▼
执行（direct_write / dask / resample / rechunk / pipeline 阶段）
        │
        ▼
staging 校验 → 原子发布 → manifest/event 记录
```

---

## 3. 模块清单

| 模块 | 职责 | 关键函数 / 入口 | 自动优化内容 | 关键参数 / 默认值 |
|---|---|---|---|---|
| `src/fast_nc_zarr/system.py` | 资源探测与预算 | `runtime_resource_snapshot()`、`effective_resource_budget()`、`storage_profile()` | CPU 物理/逻辑/affinity/cgroup；内存；存储介质；同设备关系；FD 限制 | `worker_ceiling = min(physical, effective)`；`memory_per_worker_bytes=512MiB`；`FD_PER_WORKER=64` |
| `src/fast_nc_zarr/hardware.py` | 硬件画像（v1.7.9 新增） | `build_hardware_profile()`、`benchmark_storage_path()`、`load/save_cached_profile()`、`detect_numa_nodes()` | 存储顺序读/写带宽、随机 4K IOPS、NUMA 拓扑、缓存 | `DEFAULT_SAMPLE_MIB=64`；缓存目录 `~/.cache/fast-nc-zarr/hardware-profiles/` |
| `src/fast_nc_zarr/performance_model.py` | 性能模型（v1.7.9 新增） | `estimate_plan()`、`rank_candidates()`、`prune_candidates()` | 候选耗时估算与排序/剪枝 | `DEFAULT_READ_MIB_S=100`；`DEFAULT_WRITE_MIB_S=100`；`SPAWN_SECONDS_PER_WORKER=0.5` |
| `src/fast_nc_zarr/worker_pool.py` | 进程池复用（v1.7.9 新增） | `WorkerPool` | 可复用进程池与顺序保持 map | `max_workers`；`pending_limit = workers*2` |
| `src/fast_nc_zarr/online_controller.py` | 在线反馈控制器（v1.7.9 新增） | `OnlineController`、`AdjustmentEvent` | 运行期启发式（IO/CPU/内存压力） | `cpu_low=50`；`cpu_high=90`；`rss_high_ratio=0.9` |
| `src/fast_nc_zarr/runtime.py` | 进程调度 | `bounded_process_map()`、`configure_process_runtime()`、`parse_cpu_affinity()`、`apply_cpu_affinity()` | 有界 pending（支持动态 `pending_limit_fn`）、线程环境约束、CPU affinity | `FAST_NC_ZARR_THREADS_PER_WORKER=1`；`FAST_NC_ZARR_CPU_AFFINITY` |
| `src/fast_nc_zarr/planner.py` | 转换计划生成 | `workload_kind()`、`initial_plan()`、`candidate_plans()`、`fixed_layout_candidate_plans()`、`storage_aware_initial_workers()` | 策略选择（file/chunk/dask）、chunk 尺寸、worker 初始值、task_batch、codec 候选；`FAST_NC_ZARR_PERF_MODEL=1` 时按性能模型排序候选 | `target_mib`：large=64 / balanced=32 / many-small=4；HDD 初始 worker：large/balanced=4、many-small=8；HDD `task_batch=4` |
| `src/fast_nc_zarr/benchmark.py` | 转换实测调优 | `tune()`、`representative_selections()`、`drop_source_page_cache()` | 分层样本、候选 round-robin、预算控制、目标选择、磁盘可行性 | `budget_seconds=60`；`max_samples=3`；`near_best_threshold=0.95`；objective：`speed/balanced/compact` |
| `src/fast_nc_zarr/writer.py` | 直接写入执行 | `direct_write()`、`_task_batches()`、`source_cache_limit()` | 进程池并发、有界 pending、源文件 LRU、batch 局部性、CPU/RSS 监控 | `SOURCE_CACHE_HARD_LIMIT=64`；`TASK_BATCH_HARD_LIMIT=64`；`pending_limit=workers` |
| `src/fast_nc_zarr/engine.py` | 转换 Dask 回退 | `convert()`、`_dask_write()` | 不兼容变量回退到 Dask processes；磁盘估算 | `dask scheduler=processes`；`num_workers=plan.workers` |
| `src/fast_nc_zarr/resampling/autotune.py` | 重采样自动参数 | `resolve_auto_space_workers()`、`resolve_auto_time_block()`、`resolve_auto_tile_size()`、`resolve_owner_buffer_budget()` | 空间 worker、时间批、tile、owner buffer 的内存模型选择 | `ESMF_WORKER_BASELINE_BYTES=4GiB`；`MAX_AUTO_TIME_BLOCK=64`；`MIN/MAX_TILE_SIZE=16/1024`；全局线程预算 `space×compute ≤ effective_cpus` |
| `src/fast_nc_zarr/resampling/engine.py` | 重采样执行 | `plan_resample()`、`run_resample()`、`_resample_tile_variable()` | 自动 tile/worker 候选探测；native 快速路径 | `compute_workers=2`；`space_workers=auto`；`tune_budget=60`；native 仅 float32 规则网格 nearest/bilinear |
| `src/fast_nc_zarr/rechunking/autotune.py` | 重分块 worker 调优 | `worker_candidates()`、`benchmark_worker_candidates()`、`select_worker_trial()` | 全范围 worker 实测、预算控制、目标选择；支持 `initial_workers` 优先评估 | `worker_candidates = 1..safe_ceiling`（初始值可置首）；objective：`speed/balanced/compact` |
| `src/fast_nc_zarr/rechunking/engine.py` | 重分块执行 | `_tune_source_workers()`、`_tune_stage2_workers()`、`_parallel_workers()`、`_storage_initial_workers()` | 两阶段 worker 调优；存储感知初始值 + 全范围实测 | `stage_peak_bytes` 估算；`requested=auto` 时自动实测；HDD 初始值 4 |
| `src/fast_nc_zarr/rechunking/compression.py` | 压缩自动选择 | `generate_compression_candidates()`、`benchmark_compression_candidates()`、`select_compression_candidate()` | dtype 剪枝候选；真实写/耐久/读基准；Pareto + 对数评分 | profile：`fast/balanced/maximum/compact`；`max_candidates`；objective 权重 |
| `src/fast_nc_zarr/pipeline/planner.py` | 流水线自动决策 | `build_pipeline_plan()`、`build_zarr_pipeline_plan()`、`_final_layout()` | 同网格检测、转换 chunk 与目标布局融合、是否独立最终化、`streaming_fusion_eligible` 标记 | `finalization_required = compression_auto or !direct_layout or !chunk_ownership_safe` |
| `src/fast_nc_zarr/runtime.py` | 进程调度 | `bounded_process_map()`、`configure_process_runtime()` | 有界 pending、单 worker 串行、spawn 上下文、线程环境约束 | `FAST_NC_ZARR_THREADS_PER_WORKER=1`（OMP/BLAS/NUMEXPR 默认 1） |

---

## 4. 各调优流程详解

### 4.1 转换自动调优

1. **工作负载分类**：`workload_kind()` 根据文件大小分布、中位数文件大小、每文件时间点数分为 `many-small-files` / `large-files` / `balanced`。
2. **初始计划**：`initial_plan()` 选择 `file` / `chunk` / `dask` 策略，计算 `chunk_time`、`spatial_chunks()`，并用 `storage_aware_initial_workers()` 给出 worker 初始值（HDD 初始 4/8，network 2/4，SSD/unknown 保持 CPU/内存上限）。
3. **候选生成**：`candidate_plans()` 保留 `1..worker_ceiling` 全部 worker 候选，并追加 chunk 尺寸、时间块、codec（zstd/lz4/bitshuffle）、task_batch 候选；`fixed_layout_candidate_plans()` 在固定布局下只扫 worker/batch。
4. **样本调优**：`tune()` 取最多 3 个时间分层样本，round-robin 执行候选；每个候选先 `drop_source_page_cache()` 再写临时 Zarr，记录逻辑/耐久吞吐、压缩率、CPU、RSS。
5. **选择**：按 `speed/balanced/compact` 目标，在磁盘可行性约束下从最快候选的 95% 邻域中选优；`compact` 额外限制在最快 80% 邻域内降低 RSS/体积。
6. **执行**：`direct_write()` 用进程池 + 源 LRU + batch 写入；不兼容变量走 `_dask_write()`。

### 4.2 重采样自动调优

1. `resolve_auto_space_workers()`：按内存模型（ESMF 4GiB/worker 基线 + compute 线程增量）和全局 CPU 预算选择空间 worker 数。
2. `resolve_auto_time_block()`：按内存预算选择每个空间 tile 内一次向量化计算的时间批。
3. `resolve_auto_tile_size()`：在候选 tile 尺寸中选“能放入内存预算的最大 tile”；空间 worker 候选按 objective 从多到少或从少到多探测。
4. `resolve_owner_buffer_budget()`：为最终 chunk 缓冲分配内存预算，超出则 memmap spill。
5. `plan_resample()` 汇总上述参数；若满足 native 快速路径条件（nearest/bilinear、float32、规则网格、无替换规则），优先走 Rust native。

### 4.3 重分块 / 重压缩自动调优

1. `worker_candidates()` 暴露 `1..safe_ceiling` 全部 worker。
2. `benchmark_worker_candidates()` 在预算内逐个实测，`select_worker_trial()` 按目标选择。
3. 重分块分 `stage1`（source 读取）和 `stage2`（最终写入）两阶段分别调优。
4. 压缩：`generate_compression_candidates()` 按 dtype 剪枝生成 zstd/lz4/blosc 候选；`benchmark_compression_candidates()` 实测写/耐久/读（含热/冷读）；`select_compression_candidate()` 做 Pareto 过滤 + 对数评分选择。

### 4.4 Pipeline 自动决策

- 对原始数据一条龙：
  - 检查目标网格与源网格是否 exact align；若一致且无替换规则，则重采样为 noop。
  - 计算 `conversion_chunks`：若需要重采样，使用 `_resampling_conversion_chunks()` 让转换直接产出适合重采样的 chunk；否则用 `resolve_conversion_plan()`。
  - 计算最终 chunk/压缩；若 `compression_auto`、变量不 direct-compatible、或转换任务无法完整拥有最终物理 chunk，则设置独立 `finalization` 阶段；否则将重分块/重压缩融合进转换或重采样阶段。
- 对现有 Zarr 输入：只规划重采样/最终化，跳过转换。

---

## 5. 当前参数与默认值速查

| 参数 | 默认值 | 位置 |
|---|---|---|
| 转换 `tune_budget` | `60.0` 秒 | `benchmark.tune` / CLI |
| 转换 `memory_per_worker_bytes` | `512 MiB` | `system.worker_ceiling` |
| 转换 `target_mib` | large=64 / balanced=32 / many-small=4 | `planner.initial_plan` |
| 转换 HDD 初始 worker | large/balanced=4，many-small=8 | `planner.storage_aware_initial_workers` |
| 转换 HDD `task_batch` | 4 | `planner.initial_plan` |
| `SOURCE_CACHE_HARD_LIMIT` | 64 | `writer` |
| `TASK_BATCH_HARD_LIMIT` | 64 | `writer` |
| 重采样 `compute_workers` | 2 | `resampling.models.ResampleConfig` |
| 重采样 `space_workers` | auto | 同上 |
| 重采样 `tune_budget` | 60 秒 | 同上 |
| 重采样 `ESMF_WORKER_BASELINE_BYTES` | 4 GiB | `resampling.autotune` |
| 重采样 `MAX_AUTO_TIME_BLOCK` | 64 | 同上 |
| 重采样 tile 范围 | 16..1024 | 同上 |
| 线程环境 | `OMP_NUM_THREADS=1` 等 | `runtime.configure_process_runtime` |
| 调优目标 | `balanced`（支持 speed/compact） | 各调优点 |

---

## 6. 已知局限与后续优化切入点

1. **调优是离线样本式**：正式执行期没有运行期反馈控制，机器负载变化后不会动态调整。
2. **进程池生命周期短**：每次 `direct_write`/`resample`/`rechunk` 创建销毁 `ProcessPoolExecutor`，spawn 开销在短任务和多阶段 pipeline 中不可忽略。
3. **存储 profile 参与度不均衡**：转换初始计划已使用存储感知初始值；重分块 `_parallel_workers` 目前仅把存储作为上下文，不参与静态决策；HDD 之间差异需要实测而非固定值。
4. **内存模型部分固定**：转换按 512 MiB/worker、重采样按 4 GiB/worker 估算，未用实测 RSS 校准。
5. **无硬件微基准**：没有顺序/随机读写带宽、IOPS、NUMA/P-E 核等实测，无法为不同硬件建立性能模型。
6. **流水线融合是“布局融合”而非“单遍流式”**：`direct_finalization` 避免了独立最终化阶段，但转换/重采样/重分块仍可能多次读写中间 Zarr。
7. **无 NUMA/affinity 绑核**：worker 调度未感知大小核/多路拓扑。
8. **无在线可观测调整记录**：manifest 记录最终选择，但没有运行期调整事件。

---

## 7. 与 v1.7.8 已落地优化的关系

v1.7.8 已落地：
- `storage_aware_initial_workers()`：HDD/network 使用初始 worker 提示（非硬上限）；
- `candidate_plans` / `fixed_layout_candidate_plans` 仍保留完整 worker 扫描；
- `resolve_auto_space_workers()` 增加全局线程预算 `space×compute ≤ effective_cpus`；
- 新增 `tests/test_backend_adaptive.py`。

本文档是上述工作的系统化基线，后续 `docs/v1.7.9-backend-optimization-plan.md` 将在此基础上给出分阶段优化方案。
