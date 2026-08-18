# Fast NC Zarr v1.7.7 P2 评估与落地记录

- **文档更新**：2026-08-18
- **关联方案**：[v1.7.7 优化方案](v1.7.7-optimization-plan.md)
- **范围**：P2 前瞻性评估、桌面 UI 诊断、可维护性与 Python 静态检查

---

## 1. Native hyperslab reader 评估

### 1.1 当前状态

- Rust native 已支持：
  - Zarr v3 inspect / read / write / rechunk；
  - NetCDF-4/classic float32/float64 检查与转换；
  - 规则 float32 网格 nearest/bilinear 重采样。
- 尚未实现统一 HDF/NetCDF/TIFF typed hyperslab reader。

### 1.2 进入条件核对

| 条件 | 状态 |
|---|---|
| codec、缺失值、packed data、时间和 CRS 语义 fixture 完整 | ❌ 尚未完整 |
| 定义 dtype/shape/fill/attrs/内存预算/fallback ABI | 部分（模型层已有计划结构，未固化 HDF/GeoTIFF 映射） |
| 明确 HDF group/dataset 映射 | ❌ |
| 明确 GeoTIFF window/CRS/旋转/多 band 语义 | ❌ |
| Python window 读取 I/O 放大、打开次数、RSS、取消清理对比 | ❌ 无真实数据 A/B |

### 1.3 结论

**v1.7.7 不进入 native hyperslab reader 实现。** 当前能力矩阵和代表性 fixture 尚不足以支撑该能力的安全承诺；继续由 Python 兼容路径负责 HDF/GeoTIFF 复杂读取，并保持 `backend=auto` 的显式 fallback reason。

### 1.4 后续 ABI 建议（留档）

当进入实现时，建议协议字段包含：

- `source` / `dataset_path` / `variable`
- `dtype` / `shape` / `chunks`
- `fill_value` / `source_attrs`
- `memory_budget_bytes`
- `fallback_reason`
- `window`（starts / shape）
- `crs` / `transform` / `rotation`（GeoTIFF）

---

## 2. Conversion/resampling 融合评估

### 2.1 当前状态

- 当前 pipeline 保留 `source-crop.zarr` checkpoint。
- 转换和重采样是分阶段执行，每阶段有 staging、校验、manifest 和恢复信息。
- 已具备 `source-crop.zarr` 保留/清理测试。

### 2.2 评估结论

**v1.7.7 不进行 conversion/resampling 融合。** 原因：

1. 融合路径会削弱“转换 checkpoint 可恢复”的现有保证；
2. 尚未证明融合能降低 raw read bytes / source opens / write amplification 且不劣化失败恢复；
3. 在真实数据 A/B 完成前，不应为了减少一个中间目录牺牲原子发布、恢复和取消清理。

### 2.3 后续触发条件

只有在相同 source/target 上完成 staging 与 fused 路径 A/B，并同时满足：

- 可恢复 checkpoint；
- 原子发布；
- 可证明的读取放大上界；
- 取消与 checkpoint 清理不劣化；

才重新评估融合。

---

## 3. 桌面长任务 UI 与诊断

### 3.1 已落地

- 任务中心新增 **诊断详情** 卡片：
  - 显示请求后端、实际后端、回退原因；
  - 显示当前 checkpoint；
  - 显示 manifest 路径；
  - 显示逻辑字节、临时观测和 ETA；
  - 支持一键复制最近事件 JSON。
- 原有能力保持：
  - 实时事件流；
  - 恢复任务入口；
  - 任务 manifest 路径展示。

### 3.2 说明

- 事件 payload 来自 pipeline 的 `_stage_event_payload`，已包含 `requested_backend`、`resolved_backend`、`fallback_reason`、`stage_checkpoint`、`logical_bytes`、`temporary_bytes`、`eta_seconds`。
- UI 直接展示这些字段，避免把文件系统观测误当业务完成度。

---

## 4. 可维护性与 Python 静态检查

### 4.1 已落地

- Pixi 环境新增 `ruff`。
- 新增 `pixi run python-lint` 任务：
  - 规则：`F`（未使用/未定义等）、`E9`（运行时错误类）。
  - 范围：`src`、`tests`、`scripts`。
- 已修复当前 F/E9 问题：
  - 清理未使用 import / 未使用变量；
  - 补上 `resampling/engine.py` 缺失的 `GridInfo` 类型导入；
  - `application/__init__.py` TYPE_CHECKING 导入加 `noqa: F401`。
- CI workflow 增加 `pixi run python-lint` 步骤。

### 4.2 仍列为后续 backlog

- 大模块拆分（`resampling/engine.py`、`rechunking/engine.py`、`filename_mode.py`、`pipeline/engine.py`）不在 v1.7.7 内强制完成；
- 全量 ruff 规则与 `ruff format` 可作为后续质量门禁继续收紧；
- mypy/pyright 类型检查暂不引入，等待核心模块拆分后评估。

---

## 5. 结论

P2 中：

- **Native hyperslab reader**：评估完成，暂不实现；
- **Conversion/resampling 融合**：评估完成，暂不融合；
- **桌面 UI 诊断**：已实现；
- **Python 静态检查**：已落地；
- **大模块拆分**：保留为后续 backlog。
