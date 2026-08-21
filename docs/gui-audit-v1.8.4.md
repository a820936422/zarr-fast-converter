## 1. 命令层
| 集合 | 缺失于前端 | 缺失于 Rust 注册 | 缺失于 contracts 文档 |
|---|---|---|---|
| `cancel_task` |  |  |  |
| `clear_task_history` |  |  |  |
| `get_backend_info` |  |  |  |
| `get_capabilities` | worker-only（预期不注册不调用） | | |
| `get_task` |  |  |  |
| `inspect_pipeline_recovery` |  |  |  |
| `inspect_source` |  |  |  |
| `inspect_time_metadata` |  |  |  |
| `inspect_zarr` |  |  |  |
| `list_tasks` |  |  |  |
| `native_capabilities` |  |  |  |
| `preview_pipeline` |  |  |  |
| `resume_pipeline` |  |  |  |
| `run_pipeline` | worker-only（预期不注册不调用） | | |
| `save_inspection_snapshot` |  |  |  |
| `shutdown` | worker-only（预期不注册不调用） | | |
| `start_inspection` |  |  |  |
| `start_native_task` |  |  |  |
| `start_pipeline` |  |  |  |

## 2. 字段层（payload 键）
| 键 | 来源 |
|---|---|
| `after_conditions` | worker、App.buildPipelinePayload |
| `after_results` | worker、App.buildPipelinePayload |
| `auto_tune` | worker |
| `backend` | worker、App.buildPipelinePayload |
| `before_conditions` | worker、App.buildPipelinePayload |
| `before_results` | worker、App.buildPipelinePayload |
| `cache_path` | api.InspectionRequest |
| `cleanup_intermediate` | worker |
| `compression` | worker、App.buildPipelinePayload |
| `compression_codec` | worker、App.buildPipelinePayload |
| `compression_level` | worker、App.buildPipelinePayload |
| `compression_objective` | worker、App.buildPipelinePayload |
| `compression_shuffle` | worker、App.buildPipelinePayload |
| `compression_tune_budget` | worker、App.buildPipelinePayload |
| `compute_dtype` | worker、App.buildPipelinePayload |
| `compute_workers` | worker |
| `custom_chunks` | worker、App.buildPipelinePayload |
| `engine` | api.InspectionRequest、App.buildPipelinePayload |
| `field_values` | api.InspectionRequest |
| `input_dir` | api.InspectionRequest、App.buildPipelinePayload |
| `input_kind` | worker、App.buildPipelinePayload |
| `inspection_kind` | App.buildPipelinePayload |
| `inspection_snapshot_path` | App.buildPipelinePayload |
| `lat_max` | worker、App.buildPipelinePayload |
| `lat_min` | worker、App.buildPipelinePayload |
| `lon_max` | worker、App.buildPipelinePayload |
| `lon_min` | worker、App.buildPipelinePayload |
| `max_workers` | worker |
| `method` | worker、App.buildPipelinePayload |
| `mode` | api.InspectionRequest |
| `na_thres` | worker、App.buildPipelinePayload |
| `output` | worker、App.buildPipelinePayload |
| `output_storage` | worker |
| `overwrite` | worker |
| `rechunk` | worker、App.buildPipelinePayload |
| `rechunk_tune_budget` | worker、App.buildPipelinePayload |
| `recompress` | worker、App.buildPipelinePayload |
| `recursive` | api.InspectionRequest、App.buildPipelinePayload |
| `resample` | worker、App.buildPipelinePayload |
| `reserve_memory_gib` | worker |
| `resolution` | worker、App.buildPipelinePayload |
| `semantic_constraints` | worker |
| `skipna` | worker、App.buildPipelinePayload |
| `source_dimensions` | api.InspectionRequest |
| `source_storage` | worker |
| `space_workers` | worker |
| `statistics_policy` | worker |
| `strategy` | worker、App.buildPipelinePayload |
| `target_extent_enabled` | App.buildPipelinePayload |
| `target_mib` | worker、App.buildPipelinePayload |
| `template` | api.InspectionRequest |
| `temporary_dir` | worker、App.buildPipelinePayload |
| `temporary_storage` | worker |
| `tile_size` | worker |
| `time_block` | worker |
| `time_end` | worker、App.buildPipelinePayload |
| `time_rule` | api.InspectionRequest、App.buildPipelinePayload |
| `time_start` | worker、App.buildPipelinePayload |
| `tune_budget` | worker、App.buildPipelinePayload |
| `tuning_objective` | worker |
| `validate` | worker、App.buildPipelinePayload |
| `validate_snapshot` | App.buildPipelinePayload |
| `validation_mode` | api.InspectionRequest、App.buildPipelinePayload |
| `variable_names` | worker、App.buildPipelinePayload |
| `variable_resampling` | App.buildPipelinePayload |
| `variable_transforms` | App.buildPipelinePayload |
| `variables` | worker、App.buildPipelinePayload |
| `workers` | worker、api.InspectionRequest |

### 仅 worker 消费（前端未发送）
`auto_tune`、`cleanup_intermediate`、`compute_workers`、`max_workers`、`output_storage`、`overwrite`、`reserve_memory_gib`、`semantic_constraints`、`source_storage`、`space_workers`、`statistics_policy`、`temporary_storage`、`tile_size`、`time_block`、`tuning_objective`

### 仅前端发送（worker 未消费）
`cache_path`、`engine`、`field_values`、`input_dir`、`inspection_kind`、`inspection_snapshot_path`、`mode`、`recursive`、`source_dimensions`、`target_extent_enabled`、`template`、`time_rule`、`validate_snapshot`、`validation_mode`、`variable_resampling`、`variable_transforms`

## 3. 事件层
| 事件 | protocol | fixtures | 前端类型 | 前端缺失 |
|---|---|---|---|---|
| `accepted` | ✅ | ✅ | ✅ |  |
| `cancelled` | ✅ | ✅ | ✅ |  |
| `failed` | ✅ | ✅ | ✅ |  |
| `finished` | ✅ | ✅ | ✅ |  |
| `inspection_ready` | ✅ | ✅ | ✅ |  |
| `log` | ✅ | ✅ | ✅ |  |
| `plan_ready` | ✅ | ✅ | ✅ |  |
| `progress` | ✅ | ✅ | ✅ |  |
| `resource` | ✅ | ✅ | ✅ |  |
| `started` | ✅ | ✅ | ✅ |  |

## 4. capability 层
| 操作 | 前端引用 |
|---|---|
| `probe` | ❌（未在前端引用） |
| `raw.netcdf.convert` | ❌（未在前端引用） |
| `raw.netcdf.inspect` | ❌（未在前端引用） |
| `resample.bilinear` | ❌（未在前端引用） |
| `resample.conservative` | ❌（未在前端引用） |
| `resample.conservative_normed` | ❌（未在前端引用） |
| `resample.nearest` | ❌（未在前端引用） |
| `zarr.inspect` | ✅ |
| `zarr.read_chunk_f32` | ✅ |
| `zarr.read_chunk_f64` | ✅ |
| `zarr.read_region_f32` | ✅ |
| `zarr.read_region_f64` | ✅ |
| `zarr.rechunk_f32` | ✅ |
| `zarr.rechunk_f32_cancel` | ❌（未在前端引用） |
| `zarr.rechunk_f32_codec` | ❌（未在前端引用） |
| `zarr.rechunk_f64` | ✅ |
| `zarr.rechunk_f64_cancel` | ❌（未在前端引用） |
| `zarr.rechunk_multi` | ❌（未在前端引用） |
| `zarr.write_f32` | ✅ |
| `zarr.write_f64` | ✅ |

## 5. 审计结论（需人工确认）
- worker 消费但前端未发送的键：auto_tune、cleanup_intermediate、compute_workers、max_workers、output_storage、overwrite、reserve_memory_gib、semantic_constraints、source_storage、space_workers、statistics_policy、temporary_storage、tile_size、time_block、tuning_objective
- 前端发送但 worker 未消费的键：cache_path、engine、field_values、input_dir、inspection_kind、inspection_snapshot_path、mode、recursive、source_dimensions、target_extent_enabled、template、time_rule、validate_snapshot、validation_mode、variable_resampling、variable_transforms
- contracts 文档命令集与实现不一致
## 6. 人工判定与处置（审计结论）

### 6.1 命令层：全部对齐 ✅
- Tauri 注册 16 命令 ↔ `api.ts` 16 个 invoke 调用 ↔ `contracts/README.md` 命令清单完全一致（`get_capabilities`/`run_pipeline`/`shutdown` 为 Python worker JSONL 专属命令，预期不注册 Tauri）。

### 6.2 事件层：全部对齐 ✅
- protocol `EVENTS`、contract event fixtures、前端 `TaskEvent` 联合类型 10 个事件名完全一致。

### 6.3 字段层：两处真实缺口 + 一处记录
- **[缺口 G1] `fusion` 未暴露**：后端 `PipelineGeneralConfig.fusion`（v1.8.1 引入）在 `_pipeline_config` 中无解析、前端 `buildPipelinePayload` 无发送 → GUI 无法关闭内存单遍融合（磁盘 checkpoint/恢复路径不可达）。**处置：worker 解析 `payload.fusion`（默认 true），GUI 处理配置暴露「内存单遍流式融合」开关。**
- **[缺口 G2] 未暴露设置项（记录，默认值合理）**：`auto_tune`、`cleanup_intermediate`、`compute_workers`、`max_workers`、`output_storage`、`overwrite`、`reserve_memory_gib`、`source_storage`、`space_workers`、`statistics_policy`、`temporary_storage`、`tile_size`、`time_block`、`tuning_objective` —— 前端使用后端默认值，语义无破坏；`semantic_constraints` 为内部协议字段。高价值项（max_workers、调参预算等）随 GUI 重构按需暴露。
- **[判定 N1] 前端发送而 worker `_pipeline_config` 不消费的键**：`cache_path/engine/field_values/input_dir/mode/recursive/source_dimensions/template/time_rule/validation_mode` 由检查阶段 `_source_config` 消费；`inspection_kind/inspection_snapshot_path/validate_snapshot` 由 `_inspection_from_payload` 消费；`variable_resampling/variable_transforms` 由 worker helper（`_variable_resampling_options`/`_variable_transforms`）消费；`target_extent_enabled` 为前端内部标志（空间范围本身以 `lat_min` 等传递）。均非缺口。

### 6.4 capability 层：11 个操作缺中文标签
- `probe`、`raw.netcdf.inspect`、`raw.netcdf.convert`、`resample.nearest/bilinear/conservative/conservative_normed`、`zarr.rechunk_f32_codec`、`zarr.rechunk_f32_cancel`、`zarr.rechunk_f64_cancel`、`zarr.rechunk_multi` 未出现在前端 `OPERATION_LABELS` → 原生能力面板显示英文原名。**处置：补全中文标签（Phase 1）。**

### 6.5 非缺口确认
- 恢复入口：任务中心「恢复任务」卡片已提供 `inspectPipelineRecovery` + `resumePipeline` 交互（检查/恢复到新输出）✅。
- 纯转换入口：后端 pipeline 要求至少一个后续操作；raw 输入的「仅转换」场景需在交互重设计（Phase 3）由操作选择器显式支持 ✅（设计项）。
