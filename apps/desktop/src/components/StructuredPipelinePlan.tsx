import {
  asRecord,
  planAxisSummary,
  planDispositionLabel,
  planOperationLabel,
  planSeconds,
  planTuple,
  planValueText,
} from "../lib/format";
import type { PipelineTargetSummary } from "../lib/types";
import { Icon } from "./Icon";

export function StructuredPipelinePlan({ plan, target }: { plan: Record<string, unknown>; target: PipelineTargetSummary }) {
  const targetGrid = asRecord(plan.target_grid);
  const sourceWindow = asRecord(plan.source_read_window);
  const sourceSelection = asRecord(plan.source_selection);
  const inputInfo = asRecord(plan.input_info);
  const chunkPlan = asRecord(plan.final_chunk_plan);
  const compression = asRecord(plan.final_compression);
  const tuningBudgets = asRecord(plan.tuning_budgets);
  const outputLayout = asRecord(plan.output_layout);
  const variables = Array.isArray(outputLayout?.variables)
    ? outputLayout.variables.flatMap((item) => {
      const record = asRecord(item);
      return record ? [record] : [];
    })
    : [];
  const decisions = Array.isArray(plan.operation_decisions)
    ? plan.operation_decisions.flatMap((item) => {
      const record = asRecord(item);
      return record ? [record] : [];
    })
    : [];
  const dimensions = asRecord(inputInfo?.dimensions);
  const dataVariables = Array.isArray(inputInfo?.variables)
    ? inputInfo.variables.flatMap((item) => {
      const record = asRecord(item);
      return record && record.is_coord !== true ? [record] : [];
    })
    : [];
  const targetShape = targetGrid
    ? `${Array.isArray(targetGrid.lat) ? targetGrid.lat.length : "—"} × ${Array.isArray(targetGrid.lon) ? targetGrid.lon.length : "—"}`
    : planTuple(inputInfo?.shape || (dimensions ? Object.values(dimensions) : undefined));
  const planKind = sourceSelection ? "原始数据转换" : "现有 Zarr 处理";
  const warning = typeof plan.coverage_warning === "string" ? plan.coverage_warning : null;
  const axisReversals = Array.isArray(outputLayout?.axis_reversals) ? outputLayout.axis_reversals : [];
  return (
    <div className="structured-plan-report">
      <div className="inspection-report-grid">
        <div><span>计划类型</span><strong>{planKind}</strong><small>检查 ID · {planValueText(plan.inspection_id)}</small></div>
        <div><span>目标形状</span><strong>{targetShape}</strong><small>{targetGrid ? planValueText(targetGrid.extent, "自定义网格") : "沿用现有数据维度"}</small></div>
        <div><span>空间重采样</span><strong>{plan.needs_resample ? "需要执行" : "不需要"}</strong><small>{plan.needs_resample ? "将生成目标网格" : "保持源网格"}</small></div>
        <div><span>最终发布</span><strong>{plan.direct_finalization ? "直接发布" : plan.finalization_required ? "需要发布阶段" : "按计划完成"}</strong><small>{axisReversals.length ? `轴方向调整：${axisReversals.join("、")}` : "无轴方向调整"}</small></div>
      </div>
      <section className="report-section">
        <div className="report-section-heading"><strong>输出目标范围</strong><span>{target.automaticResample ? "自动规划重采样" : "按源网格裁剪"}</span></div>
        <div className="report-detail-grid">
          <div className="wide"><span>目标时间段</span><strong>{target.timeStart && target.timeEnd ? `${target.timeStart} → ${target.timeEnd}` : "全部时间"}</strong></div>
          <div><span>目标纬度范围</span><strong>{target.spatialEnabled ? `${target.latMin}° → ${target.latMax}°` : "源数据范围"}</strong></div>
          <div><span>目标经度范围</span><strong>{target.spatialEnabled ? `${target.lonMin}° → ${target.lonMax}°` : "源数据范围"}</strong></div>
          <div><span>目标空间分辨率</span><strong>{target.spatialEnabled || target.automaticResample ? `${planValueText(target.resolution)}°` : "源分辨率"}</strong></div>
        </div>
      </section>

      {targetGrid && (
        <section className="report-section">
          <div className="report-section-heading"><strong>目标空间网格</strong><span>{planValueText(targetGrid.extent, "自定义")}</span></div>
          <div className="report-detail-grid">
            <div><span>纬度点数</span><strong>{Array.isArray(targetGrid.lat) ? targetGrid.lat.length.toLocaleString() : "—"}</strong><small>{planValueText(targetGrid.lat_resolution)}° 分辨率</small></div>
            <div><span>经度点数</span><strong>{Array.isArray(targetGrid.lon) ? targetGrid.lon.length.toLocaleString() : "—"}</strong><small>{planValueText(targetGrid.lon_resolution)}° 分辨率</small></div>
            <div className="wide"><span>纬度范围</span><strong>{planAxisSummary(targetGrid.lat_bounds)}</strong></div>
            <div className="wide"><span>经度范围</span><strong>{planAxisSummary(targetGrid.lon_bounds)}</strong></div>
          </div>
        </section>
      )}

      <section className="report-section">
        <div className="report-section-heading"><strong>{sourceSelection ? "源数据读取窗口" : "输入 Zarr 结构"}</strong><span>{sourceSelection ? `${planValueText(sourceSelection.variables && Array.isArray(sourceSelection.variables) ? sourceSelection.variables.length : 0)} 个变量` : `${dataVariables.length} 个数据变量`}</span></div>
        <div className="report-detail-grid">
          {sourceSelection ? (
            <>
              <div><span>变量</span><strong>{planValueText(sourceSelection.variables)}</strong></div>
              <div><span>时间范围</span><strong>{planValueText(sourceSelection.time_start)} – {planValueText(sourceSelection.time_stop)}</strong></div>
              <div><span>纬度索引</span><strong>{planValueText(sourceSelection.lat_start)} – {planValueText(sourceSelection.lat_stop)}</strong></div>
              <div><span>经度索引</span><strong>{planValueText(sourceSelection.lon_start)} – {planValueText(sourceSelection.lon_stop)}</strong></div>
              {sourceWindow && <div className="wide"><span>窗口策略</span><strong>{planValueText(sourceWindow.method)} · {planValueText(sourceWindow.halo_description)}</strong></div>}
            </>
          ) : (
            <>
              <div><span>数据维度</span><strong>{dimensions ? Object.entries(dimensions).map(([key, value]) => `${key}=${planValueText(value)}`).join(" · ") : "—"}</strong></div>
              <div><span>输入形状</span><strong>{planTuple(inputInfo?.shape)}</strong></div>
              <div className="wide"><span>数据变量</span><strong>{dataVariables.map((item) => planValueText(item.name)).join("、") || "—"}</strong></div>
            </>
          )}
        </div>
      </section>

      <section className="report-section">
        <div className="report-section-heading"><strong>执行阶段</strong><span>{decisions.length} 项操作</span></div>
        {decisions.length ? (
          <div className="plan-decision-list">
            {decisions.map((decision) => {
              const operation = planValueText(decision.operation);
              const disposition = planValueText(decision.disposition);
              return <div className="plan-decision" key={operation}><div><strong>{planOperationLabel(operation)}</strong><small>{planValueText(decision.reason)}</small></div><span className={`plan-decision-badge ${disposition === "not_requested" ? "muted" : "active"}`}>{planDispositionLabel(disposition)}</span></div>;
            })}
          </div>
        ) : <p className="report-empty">计划没有返回操作决策。</p>}
      </section>

      <section className="report-section">
        <div className="report-section-heading"><strong>存储布局</strong><span>{variables.length || dataVariables.length} 个变量</span></div>
        <div className="plan-storage-summary">
          <div><span>转换 chunks</span><strong>{planTuple(plan.conversion_chunks)}</strong></div>
          <div><span>最终 chunks</span><strong>{planTuple(plan.final_chunks)}</strong></div>
          <div><span>压缩方案</span><strong>{planValueText(compression?.profile, "默认")}</strong><small>{planValueText(compression?.codec)} · {planValueText(compression?.shuffle)}</small></div>
          <div><span>分块策略</span><strong>{planValueText(chunkPlan?.strategy, "标准")}</strong><small>{planValueText(chunkPlan?.estimated_chunk_bytes)} bytes / chunk</small></div>
        </div>
        {variables.length ? <div className="plan-variable-list">{variables.map((variable) => <div className="plan-variable-card" key={planValueText(variable.output_name)}><div className="plan-variable-head"><strong>{planValueText(variable.output_name)}</strong><span>{variable.is_coord ? "坐标" : "数据"}</span></div><div><span>{planValueText(variable.dtype)}</span><span>维度 {planValueText(variable.dims)}</span><span>形状 {planTuple(variable.shape)}</span></div><small>chunks {planTuple(variable.chunks)}{asRecord(variable.codec) ? ` · ${planValueText(asRecord(variable.codec)?.kind)}` : ""}</small></div>)}</div> : <p className="report-empty">暂无变量布局详情。</p>}
      </section>

      {tuningBudgets && (
        <section className="report-section">
          <div className="report-section-heading"><strong>实际调参上限</strong><span>由当前 IPC payload 生效</span></div>
          <div className="report-detail-grid">
            <div><span>转换 / 重采样</span><strong>{planSeconds(tuningBudgets.tune_budget)}</strong></div>
            <div><span>重分块 worker</span><strong>{planSeconds(tuningBudgets.rechunk_tune_budget)}</strong></div>
            <div><span>压缩候选</span><strong>{planSeconds(tuningBudgets.compression_tune_budget)}</strong></div>
          </div>
        </section>
      )}

      {warning && <div className="plan-warning"><Icon name="activity" size={14} /><span>{warning}</span></div>}
      <details className="raw-plan-report"><summary>查看原始计划 JSON</summary><pre>{JSON.stringify(plan, null, 2)}</pre></details>
    </div>
  );
}