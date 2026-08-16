import { useEffect, useMemo, useState } from "react";
import {
  getBackendInfo,
  getNativeCapabilities,
  inspectPipelineRecovery,
  pickDirectory,
  pickSnapshotDestination,
  previewPipeline,
  resumePipeline,
  saveInspectionSnapshot,
  startInspection,
  startNativeTask,
  startPipeline,
  type BackendCapability,
  type BackendInfo,
  type FilenameFieldSummary,
  type InspectionResult,
  type InspectionTaskOperation,
  type PipelinePayload,
  type TaskEvent,
  type TaskSummary,
  type TimeDimensionSummary,
  type TimeFieldOption,
  type TimeInspection,
  type TimeRef,
  type TimeRule,
} from "./api";
import { useTaskEvents } from "./taskEvents";
import "./styles.css";

type TimeRuleMode = "full" | "doy" | "calendar";
type TimeComponent = "full" | "year" | "month" | "day" | "doy";

const TIME_COMPONENT_LABELS: Record<TimeComponent, string> = {
  full: "完整日期/时间",
  year: "年份",
  month: "月份",
  day: "日期",
  doy: "年内日序（DOY）",
};

function timeRefKey(ref: TimeRef | undefined): string {
  return ref ? `${ref.source}:${ref.component}:${ref.index}` : "";
}

function firstTimeRef(options: TimeFieldOption[], component: TimeComponent): TimeRef | undefined {
  return options.find((option) => option.ref.component === component)?.ref;
}

function timeRuleMode(rule: TimeRule | null): TimeRuleMode {
  if (rule?.full) return "full";
  if (rule?.doy) return "doy";
  return "calendar";
}

function initialTimeRule(
  inspection: TimeInspection,
  current: TimeRule | null,
): { rule: TimeRule | null; mode: TimeRuleMode } {
  if (current) return { rule: current, mode: timeRuleMode(current) };
  if (inspection.suggested_rule) {
    return { rule: inspection.suggested_rule, mode: timeRuleMode(inspection.suggested_rule) };
  }
  const full = firstTimeRef(inspection.options, "full");
  if (full) return { rule: { full }, mode: "full" };
  const year = firstTimeRef(inspection.options, "year");
  const doy = firstTimeRef(inspection.options, "doy");
  if (year && doy) return { rule: { year, doy }, mode: "doy" };
  const month = firstTimeRef(inspection.options, "month");
  const day = firstTimeRef(inspection.options, "day");
  if (year && month && day) return { rule: { year, month, day }, mode: "calendar" };
  return { rule: null, mode: "full" };
}

function validateTimeRule(mode: TimeRuleMode, rule: TimeRule | null): string | null {
  if (mode === "full" && !rule?.full) return "请选择一个完整日期或完整时间字段。";
  if (mode === "doy" && (!rule?.year || !rule.doy)) return "请选择年份字段和 DOY 字段。";
  if (mode === "calendar" && (!rule?.year || !rule.month || !rule.day)) {
    return "请选择年份、月份和日期字段。";
  }
  return null;
}

function timeRuleOptionLabel(options: TimeFieldOption[], ref: TimeRef | undefined): string {
  return options.find((option) => timeRefKey(option.ref) === timeRefKey(ref))?.label || "未选择";
}

function timeRuleSummary(rule: TimeRule | null, options: TimeFieldOption[]): string {
  if (!rule) return "尚未选择时间规则";
  if (rule.full) return `${TIME_COMPONENT_LABELS.full}：${timeRuleOptionLabel(options, rule.full)}`;
  const parts = (["year", "month", "day", "doy"] as const)
    .filter((component) => rule[component])
    .map((component) => `${TIME_COMPONENT_LABELS[component]}：${timeRuleOptionLabel(options, rule[component])}`);
  return parts.join("；");
}

type VariableDetail = {
  name: string;
  dtype: string;
  dims: string[];
  attrs: Record<string, unknown>;
};

type VariableTransformDraft = {
  fillValues: string;
  scaleFactor: string;
  addOffset: string;
  outputFill: string;
};

const EMPTY_VARIABLE_TRANSFORM: VariableTransformDraft = {
  fillValues: "",
  scaleFactor: "",
  addOffset: "",
  outputFill: "",
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function attributeText(attrs: Record<string, unknown>, key: string): string {
  const value = attrs[key];
  return value === undefined || value === null ? "" : String(value);
}

function parseOptionalNumber(value: string, label: string): number | "nan" | undefined {
  const text = value.trim();
  if (!text) return undefined;
  if (text.toLowerCase() === "nan") return "nan";
  const result = Number(text);
  if (!Number.isFinite(result)) throw new Error(`${label} 必须是有限数值或 nan。`);
  return result;
}
function parseCoordinateBound(value: string, label: string, minimum: number, maximum: number): number {
  const text = value.trim();
  if (!text) throw new Error(`请输入${label}。`);
  const result = Number(text);
  if (!Number.isFinite(result) || result < minimum || result > maximum) {
    throw new Error(`${label}必须位于 ${minimum} 到 ${maximum} 之间。`);
  }
  return result;
}
function parsePositiveInteger(value: string, label: string): number {
  const text = value.trim();
  const result = Number(text);
  if (!text || !Number.isInteger(result) || result <= 0) {
    throw new Error(`${label}必须是正整数。`);
  }
  return result;
}
function validateDateBoundary(value: string, label: string): string {
  const text = value.trim();
  const match = /^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$/.exec(text);
  if (!match) throw new Error(`${label}请输入 YYYY、YYYY-MM 或 YYYY-MM-DD。`);
  const year = Number(match[1]);
  const month = match[2] ? Number(match[2]) : null;
  const day = match[3] ? Number(match[3]) : null;
  if (year < 1 || (month !== null && (month < 1 || month > 12))) {
    throw new Error(`${label}不是有效日期。`);
  }
  if (day !== null) {
    const lastDay = new Date(Date.UTC(year, month || 1, 0)).getUTCDate();
    if (day < 1 || day > lastDay) throw new Error(`${label}不是有效日期。`);
  }
  return text;
}

function parseFillValues(value: string, label: string): Array<number | "nan"> | undefined {
  const text = value.trim();
  if (!text) return undefined;
  return text.split(",").map((item) => parseOptionalNumber(item, label)).filter(
    (item): item is number | "nan" => item !== undefined,
  );
}

type InspectionProgressStatus = "starting" | "running" | "cancelling" | "finished" | "failed" | "cancelled";
type InspectionProgressState = {
  taskId: string | null;
  operation: InspectionTaskOperation;
  label: string;
  message: string;
  completed: number;
  total: number;
  status: InspectionProgressStatus;
  startedAt: number;
};
type InputKind = "source" | "zarr";
type InspectionStage = "input" | "time" | "structure";
type View = "overview" | "inspection" | "pipeline" | "tasks" | "settings";
type BackendMode = "auto" | "rust";
type ChunkStrategy = "time" | "space" | "custom";
type IconName =
  | "activity"
  | "archive"
  | "arrow"
  | "chevron"
  | "clock"
  | "database"
  | "folder"
  | "grid"
  | "layers"
  | "play"
  | "refresh"
  | "settings"
  | "spark"
  | "tasks"
  | "terminal"
  | "upload";

const ICON_PATHS: Record<IconName, string> = {
  activity: "M3 12h4l2.2-7 4.2 14 2.2-7H21",
  archive: "M4 7h16v13H4zM3 4h18v3H3zM9 11h6",
  arrow: "M5 12h13M13 6l6 6-6 6",
  chevron: "M9 5l7 7-7 7",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zm0-14v5l3 2",
  database: "M5 6c0-1.7 3.1-3 7-3s7 1.3 7 3-3.1 3-7 3-7-1.3-7-3zm0 0v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6m-14 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6",
  folder: "M3 6.5A1.5 1.5 0 0 1 4.5 5H10l2 2h7.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z",
  grid: "M4 4h6v6H4zm10 0h6v6h-6zM4 14h6v6H4zm10 0h6v6h-6z",
  layers: "M12 3l9 5-9 5-9-5 9-5zm-9 9 9 5 9-5M3 17l9 5 9-5",
  play: "M8 5v14l11-7z",
  refresh: "M20 11a8 8 0 0 0-14.7-3L3 11m0 0V5m0 6h6M4 13a8 8 0 0 0 14.7 3L21 13m0 0v6m0-6h-6",
  settings: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zm0-5v3m0 8v3m0 5v-3m0-8V5m9 7h-3M8 12H5m14.4-6.4-2.1 2.1M7.7 16.3l-2.1 2.1m0-12.8 2.1 2.1m9.6 8.6 2.1 2.1",
  spark: "M12 2l1.7 6.3L20 10l-6.3 1.7L12 18l-1.7-6.3L4 10l6.3-1.7zM19 16l.7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7z",
  tasks: "M5 5h14v14H5zM8 9h8M8 13h5M8 17h3",
  terminal: "M5 7l5 5-5 5m7 0h7",
  upload: "M12 16V4m0 0L7 9m5-5 5 5M5 20h14",
};

const VIEW_TITLES: Record<View, string> = {
  overview: "工作台",
  inspection: "数据检查",
  pipeline: "处理流程",
  tasks: "任务中心",
  settings: "路径设置",
};

const OPERATION_LABELS: Record<string, string> = {
  "zarr.inspect": "Zarr 结构检查",
  "zarr.read_chunk_f32": "Float32 chunk 读取",
  "zarr.read_chunk_f64": "Float64 chunk 读取",
  "zarr.read_region_f32": "Float32 region 读取",
  "zarr.read_region_f64": "Float64 region 读取",
  "zarr.write_f32": "Float32 数组写入",
  "zarr.write_f64": "Float64 数组写入",
  "zarr.rechunk_f32": "Float32 重分块",
  "zarr.rechunk_f64": "Float64 重分块",
};

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={ICON_PATHS[name]} />
    </svg>
  );
}
function fieldValuesPreview(field: FilenameFieldSummary): string {
  const values = field.values.slice(0, 4);
  if (!values.length) return "暂无样例值";
  return values.join("、") + (field.values.length > values.length ? " ……" : "");
}

function timeValuesPreview(values: string[]): string {
  if (!values.length) return "未发现可解码值";
  const preview = values.slice(0, 3).join("、");
  return preview + (values.length > 3 ? " ……" : "");
}

function StructuredTimeInspection({ inspection }: { inspection: TimeInspection }) {
  const dimension: TimeDimensionSummary = inspection.time_dimension;
  const attributes = Object.entries(dimension.attrs).filter(([, value]) => value !== null && value !== "").slice(0, 4);
  return (
    <div className="structured-time-report">
      <div className="inspection-report-grid">
        <div><span>文件数量</span><strong>{inspection.files.length.toLocaleString()}</strong></div>
        <div><span>读取引擎</span><strong>{inspection.engine}</strong></div>
        <div><span>数据维度</span><strong>{inspection.dimensions.length || "—"}</strong><small>{inspection.dimensions.join(" · ") || "未识别"}</small></div>
        <div><span>坐标数量</span><strong>{inspection.coordinates.length || "—"}</strong><small>{inspection.coordinates.join(" · ") || "未识别"}</small></div>
      </div>
      <section className="report-section">
        <div className="report-section-heading"><strong>文件名时间字段</strong><span>{inspection.filename_fields.length} 个数字字段</span></div>
        {inspection.filename_fields.length ? (
          <div className="filename-field-list">
            {inspection.filename_fields.map((field) => (
              <div className="filename-field-card" key={field.index}>
                <div className="filename-field-head"><strong>字段 #{field.index}</strong><span className={field.changed ? "field-status changed" : "field-status"}>{field.changed ? "跨文件变化" : "稳定/未验证"}</span></div>
                <div className="filename-field-meta"><span>样例 <b>{field.sample}</b></span><span>位置 {field.start}–{field.start + field.length - 1}</span><span>长度 {field.length}</span></div>
                <small>{fieldValuesPreview(field)}</small>
              </div>
            ))}
          </div>
        ) : <p className="report-empty">未发现可用于时间解析的文件名数字字段。</p>}
      </section>
      <section className="report-section">
        <div className="report-section-heading"><strong>数据内时间维度</strong><span className={dimension.exists ? "report-status good" : "report-status warning"}>{dimension.exists ? "已发现" : "未发现"}</span></div>
        {dimension.exists ? (
          <div className="report-detail-grid">
            <div><span>维度名称</span><strong>{dimension.name || "未命名"}</strong></div>
            <div><span>格式</span><strong>{dimension.format_label || "未识别"}</strong></div>
            <div className="wide"><span>解码样例</span><strong>{timeValuesPreview(dimension.decoded_values)}</strong></div>
            {attributes.map(([key, value]) => <div key={key}><span>{key}</span><strong>{String(value)}</strong></div>)}
          </div>
        ) : <p className="report-empty">首个文件没有可识别的 time 维度，将依赖文件名规则生成时间轴。</p>}
      </section>
      <section className="report-section">
        <div className="report-section-heading"><strong>可用时间规则</strong><span>{inspection.options.length} 个候选</span></div>
        {inspection.options.length ? <div className="rule-option-list">{inspection.options.map((option) => <span key={`${option.ref.source}:${option.ref.component}:${option.ref.index}`}>{option.label}</span>)}</div> : <p className="report-empty">暂无可用候选，请检查输入文件的命名或时间维度。</p>}
      </section>
      <details className="raw-inspection-report"><summary>查看原始检查日志</summary><pre>{inspection.report}</pre></details>
    </div>
  );
}
function planValueText(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (Array.isArray(value)) return value.map((item) => planValueText(item, "")).join(" × ") || fallback;
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString() : fallback;
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function planTuple(value: unknown): string {
  return Array.isArray(value) ? value.map((item) => planValueText(item)).join(" × ") : planValueText(value);
}
function planAxisSummary(value: unknown): string {
  if (!Array.isArray(value) || value.length === 0) return "—";
  const first = planValueText(value[0]);
  const last = planValueText(value[value.length - 1]);
  return `${first} → ${last}（${value.length.toLocaleString()} 个边界）`;
}

type PipelineTargetSummary = {
  timeStart: string;
  timeEnd: string;
  spatialEnabled: boolean;
  latMin: string;
  latMax: string;
  lonMin: string;
  lonMax: string;
  resolution: number;
  automaticResample: boolean;
};

function planOperationLabel(operation: string): string {
  return {
    conversion: "格式转换",
    resampling: "空间重采样",
    rechunking: "重分块",
    recompression: "重压缩",
  }[operation] || operation;
}

function planDispositionLabel(disposition: string): string {
  return {
    executed_as_stage: "执行阶段",
    not_requested: "未请求",
    skipped: "跳过",
    reused: "复用已有结果",
  }[disposition] || disposition;
}

function StructuredPipelinePlan({ plan, target }: { plan: Record<string, unknown>; target: PipelineTargetSummary }) {
  const targetGrid = asRecord(plan.target_grid);
  const sourceWindow = asRecord(plan.source_read_window);
  const sourceSelection = asRecord(plan.source_selection);
  const inputInfo = asRecord(plan.input_info);
  const chunkPlan = asRecord(plan.final_chunk_plan);
  const compression = asRecord(plan.final_compression);
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

      {warning && <div className="plan-warning"><Icon name="activity" size={14} /><span>{warning}</span></div>}
      <details className="raw-plan-report"><summary>查看原始计划 JSON</summary><pre>{JSON.stringify(plan, null, 2)}</pre></details>
    </div>
  );
}


function progressStatusLabel(status: InspectionProgressStatus): string {
  return { starting: "正在启动", running: "后端执行中", cancelling: "正在取消", finished: "检查完成", failed: "检查失败", cancelled: "已取消" }[status];
}

function InspectionProgressCard({
  progress,
  nowMs,
  onCancel,
}: {
  progress: InspectionProgressState;
  nowMs: number;
  onCancel?: () => void;
}) {
  const determinate = progress.total > 0;
  const percentage = determinate ? Math.min(100, Math.max(0, Math.round((progress.completed / progress.total) * 100))) : 0;
  const elapsed = Math.max(0, Math.round((nowMs - progress.startedAt) / 1000));
  return (
    <div className={`inspection-progress-card ${progress.status}`} role="status" aria-live="polite">
      <div className="progress-icon"><Icon name={progress.status === "failed" ? "terminal" : "activity"} size={16} /></div>
      <div className="inspection-progress-content">
        <div className="inspection-progress-heading"><strong>{progress.label}</strong><span>{progressStatusLabel(progress.status)}</span></div>
        <p>{progress.message}</p>
        <div className="inspection-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={determinate ? percentage : undefined} aria-label={`${progress.label}进度`}>
          <span className={!determinate ? "indeterminate" : ""} style={determinate ? { width: `${percentage}%` } : undefined} />
        </div>
        <div className="inspection-progress-foot"><span>{determinate ? `${percentage}% · ${progress.completed}/${progress.total}` : "正在等待后端反馈"}</span><span>已用 {elapsed}s</span>{onCancel && (progress.status === "starting" || progress.status === "running") && <button type="button" onClick={onCancel}>取消</button>}</div>
      </div>
    </div>
  );
}


function reasonText(reason: unknown): string {
  if (reason instanceof Error) return reason.message;
  if (typeof reason === "string") return reason;
  if (reason && typeof reason === "object") {
    const value = reason as Record<string, unknown>;
    const message = typeof value.message === "string" ? value.message : null;
    const kind = typeof value.kind === "string" ? value.kind : null;
    const stage = typeof value.stage === "string" ? value.stage : null;
    if (message) {
      const context = [kind, stage].filter(Boolean).join(" / ");
      return context ? `${context}: ${message}` : message;
    }
    try {
      const serialized = JSON.stringify(reason);
      if (serialized) return serialized;
    } catch {
      // Keep the final fallback below for non-serializable rejection values.
    }
  }
  return String(reason);
}

function formatBytes(value: number): string {
  if (!value) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatTaskStatus(status: TaskSummary["status"]): string {
  return {
    running: "运行中",
    cancelling: "正在取消",
    finished: "已完成",
    failed: "失败",
    cancelled: "已取消",
  }[status];
}

function formatCommand(command: string): string {
  return {
    native_task: "原生任务",
    run_pipeline: "数据处理",
    resume_pipeline: "恢复处理",
    inspect_time_metadata: "时间轴检查",
    inspect_source: "结构检查",
    inspect_zarr: "Zarr 结构检查",
  }[command] || command;
}

function operationLabel(operation: string): string {
  return OPERATION_LABELS[operation] || operation;
}

function capabilityReason(item: BackendCapability["capabilities"][number]): string {
  if (item.supported) return "已就绪 · 原生执行";
  return item.reason || "当前走兼容执行";
}

function eventText(event: TaskEvent): string {
  const message = event.payload.message;
  if (typeof message === "string" && message) return message;
  if (event.event === "progress") {
    const completed = event.payload.completed;
    const total = event.payload.total;
    if (typeof completed === "number" && typeof total === "number" && total > 0) {
      return `${Math.round((completed / total) * 100)}%`;
    }
  }
  return event.stage || "任务事件";
}
function pathBaseName(value: string): string {
  const trimmed = value.replace(/[\\/]+$/, "");
  return trimmed.split(/[\\/]/).pop() || "output";
}

function pathParent(value: string): string {
  const trimmed = value.replace(/[\\/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index >= 0 ? trimmed.slice(0, index) || trimmed.slice(0, 1) : ".";
}

function joinPath(parent: string, child: string): string {
  const separator = parent.includes("\\") && !parent.includes("/") ? "\\" : "/";
  const trimmed = parent.replace(/[\\/]+$/, "");
  return `${trimmed}${trimmed ? separator : separator}${child}`;
}

function outputStoreName(inputPath: string, inputKind: InputKind): string {
  const base = pathBaseName(inputPath).replace(/\.zarr$/i, "") || "output";
  return inputKind === "zarr" ? `${base}-processed.zarr` : `${base}.zarr`;
}

function defaultOutputPath(inputPath: string, inputKind: InputKind): string {
  if (!inputPath) return "";
  return joinPath(pathParent(inputPath), outputStoreName(inputPath, inputKind));
}


function App() {
  const [backend, setBackend] = useState<BackendInfo | null>(null);
  const [nativeCapability, setNativeCapability] = useState<BackendCapability | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [inputKind, setInputKind] = useState<InputKind>("source");
  const [arrayPath, setArrayPath] = useState("/value");
  const [selectedVariables, setSelectedVariables] = useState<string[]>([]);
  const [variableNames, setVariableNames] = useState<Record<string, string>>({});
  const [variableTransforms, setVariableTransforms] = useState<Record<string, VariableTransformDraft>>({});
  const [inputPath, setInputPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [temporaryDir, setTemporaryDir] = useState("");
  const [recursive, setRecursive] = useState(false);
  const [engine, setEngine] = useState("auto");
  const [timeInspection, setTimeInspection] = useState<TimeInspection | null>(null);
  const [timeRule, setTimeRule] = useState<TimeRule | null>(null);
  const [timeRuleModalOpen, setTimeRuleModalOpen] = useState(false);
  const [timeRuleDraft, setTimeRuleDraft] = useState<TimeRule | null>(null);
  const [timeRuleMode, setTimeRuleMode] = useState<TimeRuleMode>("full");
  const [fullStructureValidation, setFullStructureValidation] = useState(false);
  const [timeRuleError, setTimeRuleError] = useState<string | null>(null);
  const [inspection, setInspection] = useState<InspectionResult | null>(null);
  const [stage, setStage] = useState<InspectionStage>("input");
  const [busy, setBusy] = useState(false);
  const [inspectionTaskId, setInspectionTaskId] = useState<string | null>(null);
  const [inspectionTaskOperation, setInspectionTaskOperation] = useState<InspectionTaskOperation | null>(null);
  const [inspectionProgress, setInspectionProgress] = useState<InspectionProgressState | null>(null);
  const [progressNowMs, setProgressNowMs] = useState(() => Date.now());
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("overview");
  const [plan, setPlan] = useState<Record<string, unknown> | null>(null);
  const [recoveryPath, setRecoveryPath] = useState("");
  const [recoveryInspection, setRecoveryInspection] = useState<InspectionResult | null>(null);
  const [resample, setResample] = useState(false);
  const [resampleMethod, setResampleMethod] = useState("bilinear");
  const [resolution, setResolution] = useState(0.1);
  const [targetTimeStart, setTargetTimeStart] = useState("");
  const [targetTimeEnd, setTargetTimeEnd] = useState("");
  const [targetSpatialEnabled, setTargetSpatialEnabled] = useState(false);
  const [targetLatMin, setTargetLatMin] = useState("");
  const [targetLatMax, setTargetLatMax] = useState("");
  const [targetLonMin, setTargetLonMin] = useState("");
  const [targetLonMax, setTargetLonMax] = useState("");
  const [skipna, setSkipna] = useState(true);
  const [naThres, setNaThres] = useState(1);
  const [computeDtype, setComputeDtype] = useState("source");
  const [beforeConditions, setBeforeConditions] = useState("");
  const [beforeResults, setBeforeResults] = useState("");
  const [pipelineTaskId, setPipelineTaskId] = useState<string | null>(null);
  const [afterConditions, setAfterConditions] = useState("");
  const [afterResults, setAfterResults] = useState("");
  const [rechunk, setRechunk] = useState(false);
  const [targetMib, setTargetMib] = useState(128);
  const [chunkStrategy, setChunkStrategy] = useState<ChunkStrategy>("time");
  const [customChunkTime, setCustomChunkTime] = useState("");
  const [customChunkLat, setCustomChunkLat] = useState("");
  const [customChunkLon, setCustomChunkLon] = useState("");
  const [recompress, setRecompress] = useState(false);
  const [compression, setCompression] = useState("auto");
  const [compressionCodec, setCompressionCodec] = useState("");
  const [compressionLevel, setCompressionLevel] = useState<number | "">("");
  const [compressionShuffle, setCompressionShuffle] = useState("auto");
  const [compressionObjective, setCompressionObjective] = useState("balanced");
  const [compressionTuneBudget, setCompressionTuneBudget] = useState(60);
  const [backendMode, setBackendMode] = useState<BackendMode>("auto");
  const [favorites, setFavorites] = useState<string[]>([]);
  const [recentPaths, setRecentPaths] = useState<string[]>([]);
  const { events, tasks, cancel } = useTaskEvents();

  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) {
      setBackend({ app: "fast-nc-zarr", version: "1.7.5", runtime: "browser-preview" });
      return;
    }
    void getBackendInfo().then(setBackend).catch((reason: unknown) => setBackendError(reasonText(reason)));
    void getNativeCapabilities().then(setNativeCapability).catch((reason: unknown) => setBackendError(reasonText(reason)));
  }, []);

  useEffect(() => {
    const event = events[events.length - 1];
    if (event?.event !== "finished" || event.payload.operation !== "zarr.inspect") return;
    const summary = event.payload.summary;
    if (!summary || typeof summary !== "object" || Array.isArray(summary)) return;
    setInspection({ kind: "zarr", path: inputPath, report: "原生 Zarr 检查完成", warnings: [], snapshot: summary as Record<string, unknown> });
    setStage("structure");
  }, [events, inputPath]);

  useEffect(() => {
    if (!busy || !inspectionProgress || !["starting", "running", "cancelling"].includes(inspectionProgress.status)) return;
    const timer = window.setInterval(() => setProgressNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [busy, inspectionProgress?.status]);

  useEffect(() => {
    if (!inspectionTaskId || !inspectionTaskOperation) return;
    const event = [...events].reverse().find((item) => item.task_id === inspectionTaskId);
    if (!event) return;
    const payloadMessage = typeof event.payload.message === "string" ? event.payload.message : null;
    const errorPayload = event.payload.error;
    const errorMessage = errorPayload && typeof errorPayload === "object" && !Array.isArray(errorPayload)
      ? reasonText(errorPayload)
      : typeof event.payload.reason === "string" ? event.payload.reason : null;
    setInspectionProgress((current) => {
      if (!current) return current;
      const next = { ...current };
      if (event.event === "accepted") {
        if (next.status !== "cancelling") {
          next.status = "starting";
          next.message = "已连接后端检查 worker，等待执行。";
        }
      } else if (event.event === "started") {
        if (next.status !== "cancelling") {
          next.status = "running";
          next.message = "后端已开始读取输入数据。";
        }
      } else if (event.event === "progress") {
        if (next.status !== "cancelling") {
          next.status = "running";
          next.message = payloadMessage || event.stage || "后端正在处理。";
        }
        if (typeof event.payload.completed === "number") next.completed = event.payload.completed;
        if (typeof event.payload.total === "number") next.total = event.payload.total;
      } else if (event.event === "log") {
        if (next.status !== "cancelling") {
          next.status = "running";
          next.message = payloadMessage || next.message;
        }
      } else if (event.event === "finished") {
        next.status = "finished";
        next.message = "后端检查完成，正在整理结果。";
        if (next.total <= 0) { next.completed = 1; next.total = 1; }
        else next.completed = next.total;
      } else if (event.event === "cancelled") {
        next.status = "cancelled";
        next.message = errorMessage || "检查已取消。";
      } else if (event.event === "failed") {
        next.status = "failed";
        next.message = errorMessage || "后端检查失败。";
      }
      return next;
    });
    if (event.event === "plan_ready" && inspectionTaskOperation === "preview_pipeline") {
      const planPayload = asRecord(event.payload.plan);
      if (planPayload) {
        setPlan(planPayload);
        setInspectionProgress((current) => current ? { ...current, message: "处理计划已生成。" } : current);
      }
    }
    if ((event.event === "inspection_ready" || event.event === "finished") && inspectionTaskOperation === "inspect_time_metadata" && Array.isArray(event.payload.files)) {
      const result = event.payload as unknown as TimeInspection;
      setTimeInspection(result);
      setTimeRule(result.suggested_rule || null);
      setStage("time");
    }
    if ((event.event === "inspection_ready" || event.event === "finished") && inspectionTaskOperation !== "inspect_time_metadata" && typeof event.payload.kind === "string" && typeof event.payload.path === "string") {
      setInspection({
        kind: event.payload.kind as InspectionResult["kind"],
        path: event.payload.path,
        report: typeof event.payload.report === "string" ? event.payload.report : "结构检查完成",
        warnings: Array.isArray(event.payload.warnings) ? event.payload.warnings.filter((item): item is string => typeof item === "string") : [],
        snapshot: event.payload.snapshot && typeof event.payload.snapshot === "object" && !Array.isArray(event.payload.snapshot) ? event.payload.snapshot as Record<string, unknown> : {},
        inspection_snapshot_path: typeof event.payload.inspection_snapshot_path === "string" ? event.payload.inspection_snapshot_path : undefined,
      });
      setStage("structure");
    }
    if (["finished", "failed", "cancelled"].includes(event.event)) {
      setBusy(false);
      if (event.event === "failed") setError(errorMessage || "后端检查失败。");
      setInspectionTaskId(null);
      setInspectionTaskOperation(null);
    }
  }, [events, inputPath, inspectionTaskId, inspectionTaskOperation]);
  useEffect(() => {
    if (!pipelineTaskId) return;
    const event = [...events].reverse().find((item) => item.task_id === pipelineTaskId);
    if (!event || !["finished", "failed", "cancelled"].includes(event.event)) return;
    if (event.event === "failed") {
      const failure = asRecord(event.payload.error);
      setError(failure ? reasonText(failure) : "数据处理失败，请打开任务中心查看后端事件。");
    }
    setPipelineTaskId(null);
  }, [events, pipelineTaskId]);

  useEffect(() => {
    try {
      setFavorites(JSON.parse(localStorage.getItem("fast-nc-zarr:favorites") || "[]") as string[]);
      setRecentPaths(JSON.parse(localStorage.getItem("fast-nc-zarr:recent") || "[]") as string[]);
    } catch {
      setFavorites([]);
      setRecentPaths([]);
    }
  }, []);

  const rememberPath = (path: string) => {
    setRecentPaths((current) => {
      const next = [path, ...current.filter((item) => item !== path)].slice(0, 8);
      localStorage.setItem("fast-nc-zarr:recent", JSON.stringify(next));
      return next;
    });
  };

  const toggleFavorite = (path: string) => {
    setFavorites((current) => {
      const next = current.includes(path) ? current.filter((item) => item !== path) : [path, ...current];
      localStorage.setItem("fast-nc-zarr:favorites", JSON.stringify(next));
      return next;
    });
  };

  const clearInspection = () => {
    setTimeInspection(null);
    setTimeRule(null);
    setTimeRuleModalOpen(false);
    setTimeRuleDraft(null);
    setTimeRuleError(null);
    setInspection(null);
    setSelectedVariables([]);
    setVariableNames({});
    setVariableTransforms({});
    setPlan(null);
    setTargetTimeStart("");
    setTargetTimeEnd("");
    setTargetSpatialEnabled(false);
    setTargetLatMin("");
    setTargetLatMax("");
    setTargetLonMin("");
    setTargetLonMax("");
    setPipelineTaskId(null);
    setInspectionTaskId(null);
    setInspectionTaskOperation(null);
    setInspectionProgress(null);
    setStage("input");

  };
  const toggleTargetSpatial = (enabled: boolean) => {
    setTargetSpatialEnabled(enabled);
    if (!enabled) return;
    const coordinates = asRecord(inspection?.snapshot.coordinates);
    setTargetLatMin((current) => current || String(coordinates?.lat_min ?? ""));
    setTargetLatMax((current) => current || String(coordinates?.lat_max ?? ""));
    setTargetLonMin((current) => current || String(coordinates?.lon_min ?? ""));
    setTargetLonMax((current) => current || String(coordinates?.lon_max ?? ""));
  };

  const chooseInput = async () => {
    setError(null);
    try {
      const selected = await pickDirectory();
      if (typeof selected === "string") {
        setInputPath(selected);
        rememberPath(selected);
        setOutputPath("");
        clearInspection();
        setView("inspection");
      }
    } catch (reason) {
      setError(reasonText(reason));
    }
  };
  const chooseTemporaryDirectory = async () => {
    setError(null);
    try {
      const selected = await pickDirectory();
      if (typeof selected === "string") setTemporaryDir(selected);
    } catch (reason) {
      setError(reasonText(reason));
    }
  };
  const chooseOutputDirectory = async () => {
    setError(null);
    try {
      const selected = await pickDirectory();
      if (typeof selected === "string") setOutputPath(joinPath(selected, outputStoreName(inputPath, inputKind)));
    } catch (reason) {
      setError(reasonText(reason));
    }
  };

  const inspectNativeZarr = async () => {
    if (!inputPath || inputKind !== "zarr") return;
    setBusy(true);
    setError(null);
    try {
      await startNativeTask({ operation: "zarr.inspect", payload: { path: inputPath, array_path: arrayPath } });
      setView("tasks");
    } catch (reason) {
      setError(reasonText(reason));
    } finally {
      setBusy(false);
    }
  };

  const startInspectionTask = async (operation: InspectionTaskOperation, payload: Record<string, unknown>, label: string) => {
    const startedAt = Date.now();
    setBusy(true);
    setError(null);
    setProgressNowMs(startedAt);
    setInspectionTaskOperation(operation);
    setInspectionProgress({ taskId: null, operation, label, message: "正在启动后端检查……", completed: 0, total: 0, status: "starting", startedAt });
    try {
      const taskId = operation === "preview_pipeline"
        ? await previewPipeline(payload as PipelinePayload)
        : await startInspection(operation, payload);
      setInspectionTaskId(taskId);
      setInspectionProgress((current) => current ? { ...current, taskId, status: "running" } : current);
    } catch (reason) {
      const message = reasonText(reason);
      setBusy(false);
      setError(message);
      setInspectionProgress((current) => current ? { ...current, status: "failed", message } : current);
    }
  };
  const cancelInspection = async () => {
    if (!inspectionTaskId) return;
    setInspectionProgress((current) => current ? { ...current, status: "cancelling", message: "已发送取消请求，等待后端停止当前读取。" } : current);
    try {
      await cancel(inspectionTaskId);
    } catch (reason) {
      const message = reasonText(reason);
      setError(message);
      setInspectionProgress((current) => current ? { ...current, status: "running", message: "取消请求未发送成功，后端仍在执行。" } : current);
    }
  };

  const inspectTime = async () => {
    if (!inputPath || inputKind !== "source" || busy) return;
    await startInspectionTask("inspect_time_metadata", { input_dir: inputPath, recursive, engine }, "检查时间轴");
  };
  const openTimeRuleModal = () => {
    if (!timeInspection) return;
    const initial = initialTimeRule(timeInspection, timeRule);
    setTimeRuleDraft(initial.rule);
    setTimeRuleMode(initial.mode);
    setTimeRuleError(null);
    setTimeRuleModalOpen(true);
  };

  const selectTimeRuleMode = (mode: TimeRuleMode) => {
    setTimeRuleMode(mode);
    setTimeRuleError(null);
    setTimeRuleDraft((current) => {
      const options = timeInspection?.options || [];
      const next: TimeRule = {};
      if (mode === "full") {
        const full = current?.full || firstTimeRef(options, "full");
        if (full) next.full = full;
      } else if (mode === "doy") {
        const year = current?.year || firstTimeRef(options, "year");
        const doy = current?.doy || firstTimeRef(options, "doy");
        if (year) next.year = year;
        if (doy) next.doy = doy;
      } else {
        const year = current?.year || firstTimeRef(options, "year");
        const month = current?.month || firstTimeRef(options, "month");
        const day = current?.day || firstTimeRef(options, "day");
        if (year) next.year = year;
        if (month) next.month = month;
        if (day) next.day = day;
      }
      return next;
    });
  };

  const selectTimeRuleComponent = (component: TimeComponent, value: string) => {
    const option = timeInspection?.options.find((item) => timeRefKey(item.ref) === value);
    setTimeRuleDraft((current) => {
      const next: TimeRule = { ...(current || {}) };
      if (option) next[component] = option.ref;
      else delete next[component];
      return next;
    });
    setTimeRuleError(null);
  };
  const confirmTimeRule = () => {
    const validation = validateTimeRule(timeRuleMode, timeRuleDraft);
    if (validation) {
      setTimeRuleError(validation);
      return;
    }
    setTimeRule(timeRuleDraft);
    setTimeRuleModalOpen(false);
    setTimeRuleError(null);
  };

  const inspectStructure = async () => {
    if (!inputPath || busy) return;
    if (inputKind === "zarr") {
      await startInspectionTask("inspect_zarr", { path: inputPath }, "读取 Zarr 结构");
      return;
    }
    await startInspectionTask(
      "inspect_source",
      { input_dir: inputPath, mode: "auto", recursive, engine, time_rule: timeRule, validation_mode: fullStructureValidation ? "full" : "fast" },
      "读取数据结构",
    );
  };

  const saveSnapshot = async () => {
    if (!inspection) return;
    const destination = await pickSnapshotDestination();
    if (typeof destination !== "string") return;
    setError(null);
    try {
      await saveInspectionSnapshot({ inspection_kind: inspection.kind, input_dir: inspection.path, destination });
    } catch (reason) {
      setError(reasonText(reason));
    }
  };

  const buildPipelinePayload = (): PipelinePayload | null => {
    if (!inspection || !inputPath) return null;
    if (variableDetails.length > 0 && selectedVariables.length === 0) {
      throw new Error("至少选择一个要处理的变量。");
    }
    const outputNames = Object.fromEntries(
      selectedVariables
        .map((name) => [name, (variableNames[name] || name).trim()] as const)
        .filter(([, name]) => name && name !== ""),
    );
    const outputNameValues = Object.values(outputNames);
    if (new Set(outputNameValues).size !== outputNameValues.length) {
      throw new Error("输出变量名不能重复。");
    }
    const timeStart = targetTimeStart.trim();
    const timeEnd = targetTimeEnd.trim();
    if (Boolean(timeStart) !== Boolean(timeEnd)) {
      throw new Error("目标时间段必须同时填写开始和结束日期。");
    }
    if (timeStart) validateDateBoundary(timeStart, "目标开始日期");
    if (timeEnd) validateDateBoundary(timeEnd, "目标结束日期");
    let latMin = -90;
    let latMax = 90;
    let lonMin = -180;
    let lonMax = 180;
    if (targetSpatialEnabled) {
      latMin = parseCoordinateBound(targetLatMin, "目标纬度下限", -90, 90);
      latMax = parseCoordinateBound(targetLatMax, "目标纬度上限", -90, 90);
      lonMin = parseCoordinateBound(targetLonMin, "目标经度下限", -180, 180);
      lonMax = parseCoordinateBound(targetLonMax, "目标经度上限", -180, 180);
      if (latMin >= latMax) throw new Error("目标纬度下限必须小于上限。");
      if (lonMin >= lonMax) throw new Error("目标经度下限必须小于上限。");
    }
    const effectiveResample = resample || automaticResample;
    const customChunks = rechunk && chunkStrategy === "custom"
      ? [
        parsePositiveInteger(customChunkTime, "时间 chunk"),
        parsePositiveInteger(customChunkLat, "纬度 chunk"),
        parsePositiveInteger(customChunkLon, "经度 chunk"),
      ]
      : null;
    const transforms = Object.fromEntries(
      selectedVariables.flatMap((name) => {
        const draft = variableTransforms[name] || EMPTY_VARIABLE_TRANSFORM;
        const fillValues = parseFillValues(draft.fillValues, `变量 ${name} 的缺失值`);
        const scaleFactor = parseOptionalNumber(draft.scaleFactor, `变量 ${name} 的缩放因子`);
        const addOffset = parseOptionalNumber(draft.addOffset, `变量 ${name} 的偏移量`);
        const outputFill = parseOptionalNumber(draft.outputFill, `变量 ${name} 的输出填充值`);
        const transform: Record<string, unknown> = {};
        if (fillValues !== undefined) transform.fill_values = fillValues;
        if (scaleFactor !== undefined) transform.scale_factor = scaleFactor;
        if (addOffset !== undefined) transform.add_offset = addOffset;
        if (outputFill !== undefined) transform.output_fill = outputFill;
        return Object.keys(transform).length ? [[name, transform] as const] : [];
      }),
    );
    return {
      output: outputPath || defaultOutputPath(inputPath, inputKind),
      temporary_dir: temporaryDir || null,
      input_dir: inputPath,
      input_kind: inputKind === "zarr" ? "zarr" : "raw",
      inspection_kind: inspection.kind,
      inspection_snapshot_path: inspection.inspection_snapshot_path || null,
      validate_snapshot: false,
      validation_mode: fullStructureValidation ? "full" : "fast",
      time_start: timeStart || null,
      time_end: timeEnd || null,
      lat_min: latMin,
      lat_max: latMax,
      lon_min: lonMin,
      lon_max: lonMax,
      target_extent_enabled: targetSpatialEnabled,
      time_rule: timeRule,
      recursive,
      engine,
      variables: selectedVariables,
      variable_names: outputNames,
      variable_transforms: transforms,
      compression_codec: recompress && compressionCodec ? compressionCodec : null,
      compression_level: recompress && compressionLevel !== "" ? compressionLevel : null,
      compression_shuffle: compressionShuffle,
      compression_objective: compressionObjective,
      compression_tune_budget: compressionTuneBudget,
      resample: effectiveResample,
      method: resampleMethod,
      resolution,
      skipna,
      na_thres: naThres,
      compute_dtype: computeDtype,
      before_conditions: beforeConditions,
      before_results: beforeResults,
      after_conditions: afterConditions,
      after_results: afterResults,
      rechunk,
      strategy: chunkStrategy,
      custom_chunks: customChunks,
      target_mib: targetMib,
      recompress,
      compression: recompress ? compression : "none",
      backend: backendMode,
      validate: true,
    };
  };

  const runPreview = async () => {
    let payload: PipelinePayload | null;
    try {
      payload = buildPipelinePayload();
    } catch (reason) {
      setError(reasonText(reason));
      return;
    }
    if (!payload) return;
    await startInspectionTask("preview_pipeline", payload, "生成处理计划");
  };

  const runPipeline = async () => {
    let payload: PipelinePayload | null;
    try {
      payload = buildPipelinePayload();
    } catch (reason) {
      setError(reasonText(reason));
      return;
    }
    if (!payload) return;
    setBusy(true);
    setError(null);
    try {
      const taskId = await startPipeline(payload);
      setPipelineTaskId(taskId);
      setView("tasks");
    } catch (reason) {
      setError(reasonText(reason));
    } finally {
      setBusy(false);
    }
  };

  const inspectRecovery = async () => {
    if (!recoveryPath) return;
    setBusy(true);
    setError(null);
    try {
      setRecoveryInspection(await inspectPipelineRecovery(recoveryPath));
    } catch (reason) {
      setError(reasonText(reason));
    } finally {
      setBusy(false);
    }
  };

  const resumeRecovery = async () => {
    if (!recoveryPath) return;
    setBusy(true);
    setError(null);
    try {
      await resumePipeline({
        output: outputPath || `${recoveryPath.replace(/[\\/]$/, "")}.recovered.zarr`,
        input_dir: recoveryPath,
        input_kind: "temporary",
        inspection_kind: "temporary",
        path: recoveryPath,
        backend: backendMode,
        validate: true,
      });
      setView("tasks");
    } catch (reason) {
      setError(reasonText(reason));
    } finally {
      setBusy(false);
    }
  };

  const stageIndex = stage === "input" ? 1 : stage === "time" ? 2 : 3;
  const automaticResample = targetSpatialEnabled;
  const effectiveResample = resample || automaticResample;
  const pipelineTarget: PipelineTargetSummary = {
    timeStart: targetTimeStart.trim(),
    timeEnd: targetTimeEnd.trim(),
    spatialEnabled: targetSpatialEnabled,
    latMin: targetLatMin.trim(),
    latMax: targetLatMax.trim(),
    lonMin: targetLonMin.trim(),
    lonMax: targetLonMax.trim(),
    resolution,
    automaticResample,
  };
  const canInspectTime = inputKind === "source" && Boolean(inputPath) && !busy;
  const hasTimeAxis = Boolean(timeInspection?.time_dimension?.exists);
  const canInspectStructure = Boolean(inputPath) && (inputKind === "zarr" || Boolean(timeRule) || (stage === "time" && hasTimeAxis)) && !busy;
  const runningTasks = useMemo(() => tasks.filter((task) => task.status === "running" || task.status === "cancelling"), [tasks]);
  const supported = (operation: string) => nativeCapability?.capabilities.find((item) => item.operation === operation);
  const variableDetails = useMemo<VariableDetail[]>(() => {
    const raw = inspection?.snapshot.variables;
    if (!Array.isArray(raw)) return [];
    return raw.flatMap((item) => {
      const record = asRecord(item);
      if (!record || record.is_coord === true || typeof record.name !== "string" || !record.name) return [];
      return [{
        name: record.name,
        dtype: typeof record.dtype === "string" ? record.dtype : "unknown",
        dims: Array.isArray(record.dims) ? record.dims.filter((value): value is string => typeof value === "string") : [],
        attrs: asRecord(record.attrs) || {},
      }];
    });
  }, [inspection]);
  const variableOptions = useMemo(() => variableDetails.map((item) => item.name), [variableDetails]);

  useEffect(() => {
    setSelectedVariables((current) => {
      const valid = current.filter((name) => variableOptions.includes(name));
      return valid.length ? valid : variableOptions;
    });
  }, [variableOptions]);
  useEffect(() => {
    setVariableNames((current) => Object.fromEntries(variableDetails.map((item) => [item.name, current[item.name] || item.name])));
    setVariableTransforms((current) => Object.fromEntries(variableDetails.map((item) => [item.name, current[item.name] || { ...EMPTY_VARIABLE_TRANSFORM }])));
  }, [variableDetails]);

  const updateVariableTransform = (name: string, patch: Partial<VariableTransformDraft>) => {
    setVariableTransforms((current) => ({
      ...current,
      [name]: { ...EMPTY_VARIABLE_TRANSFORM, ...(current[name] || {}), ...patch },
    }));
  };

  const capabilityItems = nativeCapability?.capabilities || [];
  const supportedCount = capabilityItems.filter((item) => item.supported).length;
  const lastEvent = events[events.length - 1];
  const statusLabel = busy ? "处理中" : runningTasks.length ? `${runningTasks.length} 个任务运行中` : inspection ? "检查已完成" : "系统就绪";

  const renderNavigation = (items: Array<{ view: View; label: string; icon: IconName; shortcut: string }>) => items.map((item) => (
    <button className={`nav-button ${view === item.view ? "active" : ""}`} type="button" key={item.view} onClick={() => setView(item.view)} disabled={item.view === "pipeline" && !inspection}>
      <Icon name={item.icon} />
      <span>{item.label}</span>
      <kbd>{item.shortcut}</kbd>
    </button>
  ));

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark"><Icon name="spark" size={21} /></div>
          <div><strong>Fast NC Zarr</strong><span>科学数据工作台</span></div>
        </div>
        <div className="workspace-switcher"><span className="workspace-dot" />默认工作区<Icon name="chevron" size={14} /></div>
        <div className="nav-label">工作区</div>
        <nav className="primary-nav" aria-label="主导航">
          {renderNavigation([
            { view: "overview", label: "总览", icon: "grid", shortcut: "01" },
            { view: "inspection", label: "数据检查", icon: "database", shortcut: "02" },
            { view: "pipeline", label: "处理流程", icon: "layers", shortcut: "03" },
            { view: "tasks", label: "任务中心", icon: "activity", shortcut: "04" },
          ])}
        </nav>
        <div className="nav-label nav-label-spaced">管理</div>
        <nav className="primary-nav" aria-label="管理导航">
          {renderNavigation([{ view: "settings", label: "路径设置", icon: "settings", shortcut: "05" }])}
        </nav>
        <div className="sidebar-capability">
          <div className="sidebar-capability-head"><span>原生执行能力</span><Icon name="arrow" size={14} /></div>
          <strong>{supportedCount}<small> / {capabilityItems.length || "—"} 项可用</small></strong>
          <div className="mini-progress"><span style={{ width: capabilityItems.length ? `${(supportedCount / capabilityItems.length) * 100}%` : "0%" }} /></div>
          <p>能力不足时自动切换兼容路径。</p>
        </div>
        <div className="sidebar-footer">
          <div className="connection-line"><span className={`connection-dot ${backend ? "online" : ""}`} />{backend ? "本地引擎已连接" : "正在连接本地引擎"}</div>
          <span>{backend ? `${backend.runtime} · v${backend.version}` : "等待运行时"}</span>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="breadcrumb"><span>FAST NC ZARR</span><Icon name="chevron" size={13} /><strong>{VIEW_TITLES[view]}</strong></div>
          <div className="topbar-actions">
            <div className={`status-pill ${busy ? "busy" : runningTasks.length ? "running" : ""}`}><span />{statusLabel}</div>
            <button className="icon-text-button" type="button" onClick={() => setView("tasks")}><Icon name="tasks" size={16} />任务中心{runningTasks.length > 0 && <b>{runningTasks.length}</b>}</button>
          </div>
        </header>

        <div className="page-content">
          {view === "overview" && (
            <>
              <section className="welcome-panel">
                <div className="welcome-copy">
                  <span className="eyebrow"><Icon name="spark" size={14} /> NATIVE-FIRST DATA WORKSPACE</span>
                  <h1>让每一次转换<br /><em>都可解释、可恢复。</em></h1>
                  <p>从原始科学数据到高性能 Zarr。检查结构、配置处理流程，并在一个清晰的工作台中追踪每个结果。</p>
                  <div className="button-row">
                    <button className="primary-button large" type="button" onClick={() => setView("inspection")}><Icon name="upload" size={17} />开始检查数据</button>
                    <button className="quiet-button large" type="button" onClick={() => setView("tasks")}><Icon name="activity" size={17} />查看任务</button>
                  </div>
                </div>
                <div className="welcome-visual" aria-hidden="true">
                  <div className="visual-grid" />
                  <div className="visual-orbit orbit-one" />
                  <div className="visual-orbit orbit-two" />
                  <div className="visual-core"><Icon name="database" size={32} /></div>
                  <div className="visual-chip chip-top"><span />ZARR V3</div>
                  <div className="visual-chip chip-bottom"><Icon name="activity" size={13} /> READY TO FLOW</div>
                </div>
              </section>

              <div className="metric-grid">
                <article className="metric-card"><div className="metric-icon blue"><Icon name="layers" /></div><div><span>原生能力</span><strong>{supportedCount}<small> / {capabilityItems.length || "—"}</small></strong><p>已通过能力矩阵</p></div></article>
                <article className="metric-card"><div className="metric-icon violet"><Icon name="activity" /></div><div><span>活动任务</span><strong>{runningTasks.length}</strong><p>{runningTasks.length ? "正在执行" : "当前没有运行任务"}</p></div></article>
                <article className="metric-card"><div className="metric-icon green"><Icon name="folder" /></div><div><span>最近路径</span><strong>{recentPaths.length}</strong><p>本地工作区记录</p></div></article>
                <article className="metric-card accent"><div className="metric-icon orange"><Icon name="spark" /></div><div><span>执行策略</span><strong>Auto</strong><p>按能力自动路由</p></div></article>
              </div>

              <div className="dashboard-grid">
                <article className="surface recent-card">
                  <div className="section-heading"><div><span className="section-kicker">QUICK ACCESS</span><h2>最近使用</h2></div><button className="link-button" type="button" onClick={() => setView("settings")}>管理路径 <Icon name="arrow" size={14} /></button></div>
                  {recentPaths.length ? <div className="recent-list">{recentPaths.slice(0, 4).map((path) => <button className="recent-item" type="button" key={path} onClick={() => { setInputPath(path); setView("inspection"); }}><span className="recent-icon"><Icon name="folder" size={16} /></span><span><strong>{path.split(/[\\/]/).pop() || path}</strong><small>{path}</small></span><Icon name="chevron" size={15} /></button>)}</div> : <div className="empty-inline"><Icon name="folder" size={22} /><p>还没有最近路径<br /><button type="button" onClick={() => setView("inspection")}>选择一个数据目录开始</button></p></div>}
                </article>
                <article className="surface route-card">
                  <div className="section-heading"><div><span className="section-kicker">EXECUTION ROUTE</span><h2>执行路线</h2></div><button className="link-button" type="button" onClick={() => setView("settings")}>查看设置 <Icon name="arrow" size={14} /></button></div>
                  <div className="route-list"><div className="route-item"><span className="route-number">01</span><div><strong>检查输入结构</strong><small>识别变量、维度和时间规则</small></div><span className="route-status ready">READY</span></div><div className="route-item"><span className="route-number">02</span><div><strong>按能力编排</strong><small>原生路径优先，兼容路径兜底</small></div><span className="route-status ready">AUTO</span></div><div className="route-item"><span className="route-number">03</span><div><strong>校验并发布</strong><small>staging 完成后原子发布结果</small></div><span className="route-status wait">SAFE</span></div></div>
                </article>
              </div>
            </>
          )}

          {view === "inspection" && (
            <>
              <div className="page-heading"><div><span className="section-kicker">STEP {String(stageIndex).padStart(2, "0")} / 03</span><h1>检查数据结构</h1><p>先确认输入的变量、时间和坐标，再生成一份可追踪的处理计划。</p></div><div className="step-progress"><span className={stageIndex >= 1 ? "active" : ""}>输入</span><i /><span className={stageIndex >= 2 ? "active" : ""}>时间</span><i /><span className={stageIndex >= 3 ? "active" : ""}>结构</span></div></div>
              <div className="inspection-layout">
                <section className="surface inspection-surface">
                  <div className="surface-title"><div><span className="section-kicker">INPUT SOURCE</span><h2>选择数据源</h2></div><span className="surface-number">01</span></div>
                  <div className="source-tabs" role="group" aria-label="输入类型"><button className={inputKind === "source" ? "selected" : ""} type="button" onClick={() => setInputKind("source")}><Icon name="archive" size={17} /><span><strong>原始数据目录</strong><small>NetCDF · HDF · TIFF</small></span></button><button className={inputKind === "zarr" ? "selected" : ""} type="button" onClick={() => setInputKind("zarr")}><Icon name="database" size={17} /><span><strong>现有 Zarr</strong><small>直接检查数组结构</small></span></button></div>
                  <label className="field-label" htmlFor="input-path">数据路径</label><div className="input-with-action"><Icon name="folder" size={17} /><input id="input-path" value={inputPath} onChange={(event) => setInputPath(event.target.value)} placeholder={inputKind === "source" ? "选择 NetCDF / HDF / TIFF 目录" : "选择 Zarr v3 目录"} /><button className="field-action" type="button" onClick={() => void chooseInput()}>浏览</button></div>
                  {inputKind === "zarr" && <div className="input-with-action secondary-input"><Icon name="layers" size={17} /><input aria-label="Zarr array path" value={arrayPath} onChange={(event) => setArrayPath(event.target.value)} placeholder="array path，例如 /value" /><button className="field-action" type="button" disabled={!inputPath || busy} onClick={() => void inspectNativeZarr()}>原生检查</button></div>}
                  {inputKind === "source" && <div className="inline-options"><label className="check-control"><input type="checkbox" checked={recursive} onChange={(event) => setRecursive(event.target.checked)} /><span className="fake-check" />递归扫描</label><label className="select-control">读取引擎<select value={engine} onChange={(event) => setEngine(event.target.value)}><option value="auto">自动选择</option><option value="h5netcdf">h5netcdf</option><option value="netcdf4">netcdf4</option><option value="rasterio">rasterio</option></select></label><label className="check-control"><input type="checkbox" checked={fullStructureValidation} onChange={(event) => setFullStructureValidation(event.target.checked)} /><span className="fake-check" />全量结构校验（慢）</label></div>}
                  {inspectionProgress && <InspectionProgressCard progress={inspectionProgress} nowMs={progressNowMs} onCancel={inspectionTaskId ? cancelInspection : undefined} />}
                  {timeInspection && (
                    <div className="inline-result time-inspection-result">
                      <div className="result-icon"><Icon name="clock" size={16} /></div>
                      <div>
                        <strong>时间规则待确认</strong>
                        <StructuredTimeInspection inspection={timeInspection} />
                        <div className="time-rule-action">
                          <span>{timeRuleSummary(timeRule, timeInspection.options)}</span>
                          <button className="quiet-button" type="button" onClick={openTimeRuleModal} disabled={busy}>
                            {timeRule ? "修改时间规则" : "选择时间规则"}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                  {inspection && <div className="inline-result success"><div className="result-icon"><Icon name="spark" size={16} /></div><div><strong>结构检查完成</strong><p>{inspection.report}</p><span>{inspection.warnings.length ? `${inspection.warnings.length} 条警告` : "未发现警告"}</span>{inspection.warnings.length > 0 && <div className="inspection-warning-list">{inspection.warnings.map((warning) => <div key={warning}><Icon name="activity" size={12} />{warning}</div>)}</div>}</div><button className="icon-button" type="button" title="保存检查快照" onClick={() => void saveSnapshot()}><Icon name="archive" size={16} /></button></div>}
                  {variableOptions.length > 0 && <div className="variable-block"><div className="variable-heading"><strong>参与处理的变量</strong><span>{selectedVariables.length} / {variableOptions.length}</span></div><div className="variable-list">{variableOptions.map((name) => <label key={name}><input type="checkbox" checked={selectedVariables.includes(name)} onChange={(event) => setSelectedVariables((current) => event.target.checked ? [...current, name] : current.filter((item) => item !== name))} /><span className="fake-check" />{name}</label>)}</div></div>}
                  <div className="surface-actions"><button className="quiet-button" type="button" disabled={!canInspectTime} onClick={() => void inspectTime()}><Icon name="clock" size={16} />检查时间轴</button><button className="primary-button" type="button" disabled={!canInspectStructure} onClick={() => void inspectStructure()}><Icon name="arrow" size={16} />读取结构</button></div>
                </section>
                <aside className="inspection-side">
                  <section className="surface profile-card"><div className="surface-title compact"><div><span className="section-kicker">DATA PROFILE</span><h2>数据概览</h2></div><Icon name="activity" size={18} /></div>{inspection ? <div className="profile-stats"><div><span>输入类型</span><strong>{inspection.kind === "zarr" ? "Zarr v3" : "原始数据"}</strong></div><div><span>变量数量</span><strong>{variableOptions.length || "—"}</strong></div><div><span>时间状态</span><strong>{timeRule ? "已确认" : "待检查"}</strong></div><div><span>警告</span><strong className={inspection.warnings.length ? "warning-text" : "good-text"}>{inspection.warnings.length}</strong></div></div> : <div className="profile-empty"><Icon name="database" size={24} /><strong>等待输入数据</strong><span>完成结构检查后，这里会显示变量和坐标摘要。</span></div>}</section>
                  <section className="surface capability-card"><div className="surface-title compact"><div><span className="section-kicker">CAPABILITY MATRIX</span><h2>原生能力</h2></div><span className="capability-count">{supportedCount}/{capabilityItems.length || "—"}</span></div><div className="capability-list">{capabilityItems.slice(0, 6).map((item) => <div className={`capability-row ${item.supported ? "supported" : "limited"}`} key={item.operation}><span className="capability-led" /><div><strong>{operationLabel(item.operation)}</strong><small>{capabilityReason(item)}</small></div></div>)}</div><button className="link-button full-link" type="button" onClick={() => setView("settings")}>查看执行设置 <Icon name="arrow" size={14} /></button></section>
                </aside>
              </div>
            </>
          )}

          {view === "pipeline" && (
            <>
              <div className="page-heading"><div><span className="section-kicker">PIPELINE BUILDER</span><h1>配置处理流程</h1><p>将已确认的数据检查结果编排为一条可恢复、可验证的执行路径。</p></div><span className="inspection-badge"><Icon name="spark" size={14} />检查结果已锁定</span></div>
              <div className="pipeline-layout">
                <section className="surface pipeline-surface">
                  <div className="surface-title"><div><span className="section-kicker">PROCESS FLOW</span><h2>处理阶段</h2></div><span className="surface-number">02</span></div>
                  {inspectionProgress && inspectionTaskOperation === "preview_pipeline" && <InspectionProgressCard progress={inspectionProgress} nowMs={progressNowMs} onCancel={inspectionTaskId ? cancelInspection : undefined} />}
                  <div className="stage-flow"><div className="flow-stage complete"><span>01</span><Icon name="database" size={16} /><div><strong>输入检查</strong><small>{inputPath.split(/[\\/]/).pop() || "已确认数据"}</small></div><Icon name="spark" size={14} /></div><div className="flow-connector" /><label className={`flow-stage toggle-stage ${effectiveResample ? "enabled" : ""}`}><span>02</span><Icon name="grid" size={16} /><div><strong>空间重采样</strong><small>{effectiveResample ? (automaticResample ? "按目标范围自动规划" : resampleMethod) : "未启用"}</small></div><input type="checkbox" checked={effectiveResample} disabled={automaticResample} onChange={(event) => setResample(event.target.checked)} /><span className="toggle-switch" /></label><div className="flow-connector" /><label className={`flow-stage toggle-stage ${rechunk ? "enabled" : ""}`}><span>03</span><Icon name="layers" size={16} /><div><strong>重分块</strong><small>{rechunk ? `目标 ${targetMib} MiB` : "未启用"}</small></div><input type="checkbox" checked={rechunk} onChange={(event) => setRechunk(event.target.checked)} /><span className="toggle-switch" /></label><div className="flow-connector" /><label className={`flow-stage toggle-stage ${recompress ? "enabled" : ""}`}><span>04</span><Icon name="spark" size={16} /><div><strong>重压缩</strong><small>{recompress ? compression : "未启用"}</small></div><input type="checkbox" checked={recompress} onChange={(event) => setRecompress(event.target.checked)} /><span className="toggle-switch" /></label></div>
                  <div className="advanced-panel target-range-panel">
                    <div className="panel-label"><Icon name="clock" size={15} />输出目标范围</div>
                    <p className="target-range-help">填写最终输出的时间段和空间范围。日期支持直接输入年份、年月或完整日期（YYYY、YYYY-MM、YYYY-MM-DD）；启用自定义经纬度后，系统会自动启用重采样，并根据目标网格计算读取窗口、chunks 和发布策略。</p>
                    <div className="advanced-fields target-time-fields">
                      <label>开始日期<input type="text" inputMode="numeric" value={targetTimeStart} disabled={inputKind === "zarr"} placeholder="YYYY / YYYY-MM / YYYY-MM-DD" onChange={(event) => setTargetTimeStart(event.target.value)} /></label>
                      <label>结束日期<input type="text" inputMode="numeric" value={targetTimeEnd} disabled={inputKind === "zarr"} placeholder="YYYY / YYYY-MM / YYYY-MM-DD" onChange={(event) => setTargetTimeEnd(event.target.value)} /></label>
                      {inputKind === "zarr" && <span className="target-range-note">现有 Zarr 流程当前按原始时间轴处理</span>}
                    </div>
                    <label className="check-control target-spatial-toggle"><input type="checkbox" checked={targetSpatialEnabled} onChange={(event) => toggleTargetSpatial(event.target.checked)} /><span className="fake-check" />启用自定义经纬度范围（自动重采样）</label>
                    {targetSpatialEnabled && (
                      <div className="advanced-fields target-spatial-fields">
                        <label>纬度下限<input type="number" min="-90" max="90" step="any" value={targetLatMin} onChange={(event) => setTargetLatMin(event.target.value)} /></label>
                        <label>纬度上限<input type="number" min="-90" max="90" step="any" value={targetLatMax} onChange={(event) => setTargetLatMax(event.target.value)} /></label>
                        <label>经度下限<input type="number" min="-180" max="180" step="any" value={targetLonMin} onChange={(event) => setTargetLonMin(event.target.value)} /></label>
                        <label>经度上限<input type="number" min="-180" max="180" step="any" value={targetLonMax} onChange={(event) => setTargetLonMax(event.target.value)} /></label>
                      </div>
                    )}
                  </div>
                  {effectiveResample && (
                    <div className="advanced-panel">
                      <div className="panel-label"><Icon name="grid" size={15} />重采样参数</div>
                      <div className="advanced-fields resample-fields">
                        <label>方法<select value={resampleMethod} onChange={(event) => setResampleMethod(event.target.value)}><option value="nearest_s2d">nearest_s2d</option><option value="nearest_d2s">nearest_d2s</option><option value="bilinear">bilinear</option><option value="conservative">conservative</option><option value="conservative_normed">conservative_normed</option><option value="patch">patch</option></select></label>
                        <label>目标分辨率<input type="number" min="0" step="any" value={resolution} onChange={(event) => setResolution(Number(event.target.value))} /></label>
                        <label>缺测阈值<input type="number" min="0" max="1" step="0.01" value={naThres} onChange={(event) => setNaThres(Number(event.target.value))} /></label>
                        <label>计算 dtype<select value={computeDtype} onChange={(event) => setComputeDtype(event.target.value)}><option value="source">保持源 dtype</option><option value="float32">浮点转 float32</option></select></label>
                        <label className="check-control"><input type="checkbox" checked={skipna} onChange={(event) => setSkipna(event.target.checked)} /><span className="fake-check" />忽略缺测值</label>
                        <label className="wide-field">采样前替换值<input value={beforeConditions} onChange={(event) => setBeforeConditions(event.target.value)} placeholder="例如 &lt;0, &gt;100" /></label>
                        <label className="wide-field">采样前替换结果<input value={beforeResults} onChange={(event) => setBeforeResults(event.target.value)} placeholder="例如 0, 100" /></label>
                        <label className="wide-field">采样后替换值<input value={afterConditions} onChange={(event) => setAfterConditions(event.target.value)} placeholder="例如 &lt;=median" /></label>
                        <label className="wide-field">采样后替换结果<input value={afterResults} onChange={(event) => setAfterResults(event.target.value)} placeholder="例如 100" /></label>
                      </div>
                    </div>
                  )}
                  {rechunk && (
                    <div className="advanced-panel">
                      <div className="panel-label"><Icon name="layers" size={15} />重分块参数</div>
                      <div className="advanced-fields rechunk-fields">
                        <label>重分块规则<select value={chunkStrategy} onChange={(event) => setChunkStrategy(event.target.value as ChunkStrategy)}><option value="time">时间连续</option><option value="space">空间连续</option><option value="custom">自定义</option></select></label>
                        <label>目标 chunk MiB<input type="number" min="1" step="1" value={targetMib} disabled={chunkStrategy === "custom"} onChange={(event) => setTargetMib(Number(event.target.value))} /></label>
                      </div>
                      {chunkStrategy === "time" && <p className="chunk-strategy-help">时间维度保持连续，纬度和经度按目标 chunk 大小自动规划。</p>}
                      {chunkStrategy === "space" && <p className="chunk-strategy-help">纬度和经度保持完整连续，时间维度按目标 chunk 大小自动规划。</p>}
                      {chunkStrategy === "custom" && (
                        <div className="advanced-fields custom-chunk-fields">
                          <label>时间 chunk<input type="number" min="1" step="1" value={customChunkTime} onChange={(event) => setCustomChunkTime(event.target.value)} placeholder="例如 10" /></label>
                          <label>纬度 chunk<input type="number" min="1" step="1" value={customChunkLat} onChange={(event) => setCustomChunkLat(event.target.value)} placeholder="例如 300" /></label>
                          <label>经度 chunk<input type="number" min="1" step="1" value={customChunkLon} onChange={(event) => setCustomChunkLon(event.target.value)} placeholder="例如 300" /></label>
                        </div>
                      )}
                    </div>
                  )}
                  {recompress && (
                    <div className="advanced-panel">
                      <div className="panel-label"><Icon name="spark" size={15} />重压缩参数</div>
                      <div className="advanced-fields compression-fields">
                        <label>压缩方案<select value={compression} onChange={(event) => setCompression(event.target.value)}><option value="auto">自动调优</option><option value="fast">快速</option><option value="balanced">平衡</option><option value="maximum">最高压缩率</option></select></label>
                        <label>压缩格式<select value={compressionCodec} onChange={(event) => { const value = event.target.value; setCompressionCodec(value); if ((value === "zstd" || value === "gzip") && !["auto", "noshuffle"].includes(compressionShuffle)) setCompressionShuffle("auto"); }}><option value="">按方案自动</option><option value="blosc-zstd">Blosc Zstd</option><option value="blosc-lz4">Blosc LZ4</option><option value="blosc-lz4hc">Blosc LZ4HC</option><option value="blosc-zlib">Blosc Zlib</option><option value="zstd">原生 Zstd</option><option value="gzip">原生 Gzip</option></select></label>
                        <label>压缩等级<input type="number" min={compressionCodec === "zstd" ? -7 : 0} max={compressionCodec === "zstd" ? 22 : 9} step="1" value={compressionLevel} onChange={(event) => setCompressionLevel(event.target.value === "" ? "" : Number(event.target.value))} placeholder="按方案" /></label>
                        <label>Shuffle<select value={compressionShuffle} onChange={(event) => setCompressionShuffle(event.target.value)}><option value="auto">按 dtype 自动</option><option value="noshuffle">不使用</option><option value="shuffle" disabled={compressionCodec === "zstd" || compressionCodec === "gzip"}>字节 shuffle</option><option value="bitshuffle" disabled={compressionCodec === "zstd" || compressionCodec === "gzip"}>bitshuffle</option></select></label>
                        <label>优化目标<select value={compressionObjective} onChange={(event) => setCompressionObjective(event.target.value)}><option value="speed">速度优先</option><option value="balanced">平衡</option><option value="compact">体积优先</option></select></label>
                        <label>调参预算(s)<input type="number" min="0" step="1" value={compressionTuneBudget} onChange={(event) => setCompressionTuneBudget(Number(event.target.value))} /></label>
                      </div>
                    </div>
                  )}
                  {variableDetails.length > 0 && (
                    <div className="advanced-panel variable-settings-panel">
                      <div className="panel-label"><Icon name="layers" size={15} />变量选择与处理参数</div>
                      <p className="variable-settings-help">可选择变量、重命名输出变量，并覆盖源文件中的缺失值、缩放因子和偏移量。留空表示沿用源元数据或默认行为。</p>
                      <div className="variable-settings-table">
                        <div className="variable-settings-head"><span>处理</span><span>源变量</span><span>输出变量名</span><span>缺失值</span><span>缩放因子</span><span>偏移量</span><span>输出填充值</span></div>
                        {variableDetails.map((detail) => {
                          const draft = variableTransforms[detail.name] || EMPTY_VARIABLE_TRANSFORM;
                          const sourceFill = attributeText(detail.attrs, "_FillValue") || attributeText(detail.attrs, "missing_value");
                          const sourceScale = attributeText(detail.attrs, "scale_factor");
                          const sourceOffset = attributeText(detail.attrs, "add_offset");
                          return (
                            <div className="variable-settings-row" key={detail.name}>
                              <label className="variable-enabled"><input type="checkbox" checked={selectedVariables.includes(detail.name)} onChange={(event) => setSelectedVariables((current) => event.target.checked ? [...current, detail.name] : current.filter((item) => item !== detail.name))} /><span className="fake-check" /></label>
                              <div className="variable-source"><strong>{detail.name}</strong><small>{detail.dtype} · {detail.dims.join(" × ") || "未知维度"}</small></div>
                              <input aria-label={`${detail.name} 输出变量名`} value={variableNames[detail.name] ?? detail.name} onChange={(event) => setVariableNames((current) => ({ ...current, [detail.name]: event.target.value }))} />
                              <input aria-label={`${detail.name} 缺失值`} value={draft.fillValues} onChange={(event) => updateVariableTransform(detail.name, { fillValues: event.target.value })} placeholder={sourceFill ? `源: ${sourceFill}` : "不处理"} />
                              <input aria-label={`${detail.name} 缩放因子`} value={draft.scaleFactor} onChange={(event) => updateVariableTransform(detail.name, { scaleFactor: event.target.value })} placeholder={sourceScale ? `源: ${sourceScale}` : "不额外缩放"} />
                              <input aria-label={`${detail.name} 偏移量`} value={draft.addOffset} onChange={(event) => updateVariableTransform(detail.name, { addOffset: event.target.value })} placeholder={sourceOffset ? `源: ${sourceOffset}` : "不额外偏移"} />
                              <input aria-label={`${detail.name} 输出填充值`} value={draft.outputFill} onChange={(event) => updateVariableTransform(detail.name, { outputFill: event.target.value })} placeholder="浮点默认 NaN" />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  <div className="output-block">
                    <div className="output-option">
                      <label className="field-label" htmlFor="temporary-path">临时处理目录</label>
                      <div className="input-with-action"><Icon name="layers" size={17} /><input id="temporary-path" value={temporaryDir} onChange={(event) => setTemporaryDir(event.target.value)} placeholder="留空：使用输出目录旁的临时目录" /><button className="field-action" type="button" onClick={() => void chooseTemporaryDirectory()}>浏览</button></div>
                      <p className="path-help">用于中间 Zarr、权重、缓存和 staging；建议选择空间充足的本地磁盘。</p>
                    </div>
                    <div className="output-option">
                      <label className="field-label" htmlFor="output-path">输出 Zarr 目录</label>
                      <div className="input-with-action"><Icon name="upload" size={17} /><input id="output-path" value={outputPath || defaultOutputPath(inputPath, inputKind)} onChange={(event) => setOutputPath(event.target.value)} placeholder="选择或填写最终 Zarr 目录" /><button className="field-action" type="button" onClick={() => void chooseOutputDirectory()}>选父目录</button></div>
                      <p className="path-help">浏览选择输出父目录，系统会生成“输入目录名.zarr”；也可直接填写最终 Zarr 目录。</p>
                    </div>
                  </div>
                  <div className="surface-actions"><button className="quiet-button" disabled={busy} type="button" onClick={() => void runPreview()}><Icon name="layers" size={16} />预览计划</button><button className="primary-button" disabled={busy} type="button" onClick={() => void runPipeline()}><Icon name="play" size={16} />启动处理</button></div>
                </section>
                <aside className="pipeline-side"><section className="surface execution-card"><div className="surface-title compact"><div><span className="section-kicker">EXECUTION POLICY</span><h2>执行策略</h2></div><Icon name="activity" size={18} /></div><label className="route-select-label">后端路由<select value={backendMode} onChange={(event) => setBackendMode(event.target.value as BackendMode)}><option value="auto">自动路由 · 推荐</option><option value="rust">原生强制 · 能力不足时失败</option></select></label><div className="policy-note"><span className="note-icon"><Icon name="spark" size={14} /></span><p><strong>Auto route</strong><br />优先使用已通过能力验证的原生操作，其他阶段保持结果语义不变。</p></div><div className="policy-list"><div><span className="policy-dot native" />原生能力优先</div><div><span className="policy-dot safe" />staging 校验后发布</div><div><span className="policy-dot trace" />manifest 全程留痕</div></div></section><section className="surface plan-card"><div className="surface-title compact"><div><span className="section-kicker">PLAN PREVIEW</span><h2>计划摘要</h2></div><span className="plan-state">{plan ? "READY" : "DRAFT"}</span></div>{plan ? <StructuredPipelinePlan plan={plan} target={pipelineTarget} /> : <div className="plan-empty"><Icon name="layers" size={22} /><span>点击“预览计划”<br />查看资源和阶段估算</span></div>}</section></aside>
              </div>
            </>
          )}

          {view === "tasks" && (
            <>
              <div className="page-heading"><div><span className="section-kicker">OBSERVABILITY</span><h1>任务中心</h1><p>所有任务、进度、资源和恢复点都集中在这里。</p></div><div className="task-summary"><strong>{tasks.length}</strong><span>历史任务</span></div></div>
              <div className="tasks-layout"><section className="surface tasks-surface"><div className="surface-title"><div><span className="section-kicker">RUN HISTORY</span><h2>执行记录</h2></div><span className="live-label"><span />LIVE</span></div>{tasks.length === 0 ? <div className="empty-state"><div className="empty-icon"><Icon name="activity" size={24} /></div><strong>还没有任务</strong><p>完成一次数据检查后，可以从处理流程启动任务。</p><button className="primary-button" type="button" onClick={() => setView("inspection")}>开始检查</button></div> : <div className="task-table"><div className="task-table-head"><span>任务</span><span>状态</span><span>资源</span><span>操作</span></div>{tasks.map((task) => <div className="task-table-row" key={task.taskId}><div className="task-name"><span className={`task-icon ${task.status}`}><Icon name={task.command === "native_task" ? "spark" : "layers"} size={15} /></span><div><strong>{formatCommand(task.command)}</strong><small>{task.taskId}</small>{task.manifest && <small className="manifest-path"><Icon name="archive" size={11} />{task.manifest}</small>}</div></div><span className={`task-state ${task.status}`}>{formatTaskStatus(task.status)}</span><span className="task-resource">{task.resource ? `${task.resource.logicalCpus} CPU · ${formatBytes(task.resource.memoryAvailableBytes)}` : "—"}</span><div>{(task.status === "running" || task.status === "cancelling") && <button className="table-action" type="button" onClick={() => void cancel(task.taskId)}><Icon name="refresh" size={14} />取消</button>}</div></div>)}</div>}</section><aside className="tasks-side"><section className="surface recovery-card"><div className="surface-title compact"><div><span className="section-kicker">CHECKPOINT RECOVERY</span><h2>恢复任务</h2></div><Icon name="refresh" size={18} /></div><p className="surface-description">输入保留的临时目录，检查 checkpoint 后恢复到新的输出位置。</p><label className="field-label" htmlFor="recovery-path">临时任务目录</label><div className="input-with-action"><Icon name="folder" size={16} /><input id="recovery-path" value={recoveryPath} onChange={(event) => setRecoveryPath(event.target.value)} placeholder="/tmp/.../pipeline-..." /><button className="field-action" type="button" disabled={!recoveryPath || busy} onClick={() => void inspectRecovery()}>检查</button></div>{recoveryInspection && <div className="recovery-result"><span className="result-icon"><Icon name="spark" size={14} /></span><div><strong>恢复点可读取</strong><p>{recoveryInspection.report}</p></div></div>}<button className="primary-button full-button" type="button" disabled={!recoveryInspection || busy} onClick={() => void resumeRecovery()}><Icon name="refresh" size={15} />恢复到新输出</button></section><section className="surface event-card"><div className="surface-title compact"><div><span className="section-kicker">EVENT STREAM</span><h2>实时事件</h2></div><span className="event-count">{events.length}</span></div><div className="event-stream">{events.length ? events.slice(-8).reverse().map((event) => <div className="event-item" key={`${event.request_id}-${event.sequence}`}><span className={`event-dot ${event.event}`} /><div><strong>{event.event}</strong><small>{eventText(event)}</small></div><time>{event.sequence.toString().padStart(2, "0")}</time></div>) : <p className="empty-event">暂无事件流</p>}</div></section></aside></div>
            </>
          )}

          {view === "settings" && (
            <>
              <div className="page-heading"><div><span className="section-kicker">WORKSPACE SETTINGS</span><h1>路径设置</h1><p>管理当前工作区的输入目录和快速访问记录。</p></div></div>
              <div className="settings-layout"><section className="surface settings-surface"><div className="surface-title"><div><span className="section-kicker">ACTIVE PATH</span><h2>当前输入路径</h2></div><span className="surface-number">05</span></div><p className="surface-description">路径只保存在当前桌面用户的本地工作区，不会写入项目文件。</p><label className="field-label" htmlFor="settings-input-path">输入目录</label><div className="input-with-action"><Icon name="folder" size={17} /><input id="settings-input-path" value={inputPath} onChange={(event) => setInputPath(event.target.value)} placeholder="输入或选择目录" /><button className="field-action" type="button" onClick={() => void chooseInput()}>浏览</button></div><div className="surface-actions"><button className="quiet-button" type="button" disabled={!inputPath} onClick={() => toggleFavorite(inputPath)}><Icon name="spark" size={15} />{inputPath && favorites.includes(inputPath) ? "取消收藏" : "收藏当前路径"}</button><button className="primary-button" type="button" disabled={!inputPath} onClick={() => { rememberPath(inputPath); setView("inspection"); }}><Icon name="arrow" size={15} />使用此路径</button></div></section><aside className="surface saved-paths"><div className="surface-title compact"><div><span className="section-kicker">SAVED LOCATIONS</span><h2>收藏路径</h2></div><Icon name="folder" size={18} /></div>{favorites.length ? <div className="saved-list">{favorites.map((path) => <div className="saved-item" key={path}><button type="button" onClick={() => { setInputPath(path); setView("inspection"); }}><Icon name="folder" size={15} /><span>{path}</span></button><button className="remove-button" type="button" onClick={() => toggleFavorite(path)} aria-label={`移除 ${path}`}>×</button></div>)}</div> : <div className="small-empty">还没有收藏路径</div>}<div className="saved-divider" /><div className="surface-title compact"><div><span className="section-kicker">RECENT</span><h2>最近目录</h2></div><Icon name="clock" size={18} /></div>{recentPaths.length ? <div className="saved-list">{recentPaths.slice(0, 5).map((path) => <button className="saved-item single" type="button" key={path} onClick={() => { setInputPath(path); setView("inspection"); }}><Icon name="clock" size={15} /><span>{path}</span><Icon name="chevron" size={14} /></button>)}</div> : <div className="small-empty">还没有最近目录</div>}</aside></div>
            </>
          )}
        </div>
        {timeRuleModalOpen && timeInspection && (
          <div
            className="modal-backdrop"
            role="presentation"
            onMouseDown={(event) => {
              if (event.target === event.currentTarget) setTimeRuleModalOpen(false);
            }}
          >
            <section className="time-rule-modal" role="dialog" aria-modal="true" aria-labelledby="time-rule-title">
              <div className="modal-heading">
                <div>
                  <span className="section-kicker">TIME RULE</span>
                  <h2 id="time-rule-title">选择时间规则</h2>
                </div>
                <button className="icon-button" type="button" aria-label="关闭" onClick={() => setTimeRuleModalOpen(false)}>×</button>
              </div>
              <p className="modal-description">请选择用于生成每日时间坐标的字段。完整日期适合文件名中的 YYYYDOY/YYYYMMDD；组合字段适合年份与 DOY 或年月日分开存储的产品。</p>
              <div className="rule-mode-tabs" role="tablist" aria-label="时间规则类型">
                <button className={timeRuleMode === "full" ? "active" : ""} type="button" onClick={() => selectTimeRuleMode("full")}>完整日期/时间</button>
                <button className={timeRuleMode === "doy" ? "active" : ""} type="button" onClick={() => selectTimeRuleMode("doy")}>年份 + DOY</button>
                <button className={timeRuleMode === "calendar" ? "active" : ""} type="button" onClick={() => selectTimeRuleMode("calendar")}>年份 + 月 + 日</button>
              </div>
              <div className="time-rule-fields">
                {timeRuleMode === "full" && (
                  <label className="time-rule-field">
                    <span>{TIME_COMPONENT_LABELS.full}</span>
                    <select
                      value={timeRefKey(timeRuleDraft?.full)}
                      onChange={(event) => selectTimeRuleComponent("full", event.target.value)}
                    >
                      <option value="">请选择字段</option>
                      {timeInspection.options.filter((option) => option.ref.component === "full").map((option) => (
                        <option key={timeRefKey(option.ref)} value={timeRefKey(option.ref)}>{option.label}</option>
                      ))}
                    </select>
                  </label>
                )}
                {timeRuleMode === "doy" && (["year", "doy"] as const).map((component) => (
                  <label className="time-rule-field" key={component}>
                    <span>{TIME_COMPONENT_LABELS[component]}</span>
                    <select
                      value={timeRefKey(timeRuleDraft?.[component])}
                      onChange={(event) => selectTimeRuleComponent(component, event.target.value)}
                    >
                      <option value="">请选择字段</option>
                      {timeInspection.options.filter((option) => option.ref.component === component).map((option) => (
                        <option key={timeRefKey(option.ref)} value={timeRefKey(option.ref)}>{option.label}</option>
                      ))}
                    </select>
                  </label>
                ))}
                {timeRuleMode === "calendar" && (["year", "month", "day"] as const).map((component) => (
                  <label className="time-rule-field" key={component}>
                    <span>{TIME_COMPONENT_LABELS[component]}</span>
                    <select
                      value={timeRefKey(timeRuleDraft?.[component])}
                      onChange={(event) => selectTimeRuleComponent(component, event.target.value)}
                    >
                      <option value="">请选择字段</option>
                      {timeInspection.options.filter((option) => option.ref.component === component).map((option) => (
                        <option key={timeRefKey(option.ref)} value={timeRefKey(option.ref)}>{option.label}</option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
              {timeRuleError && <div className="modal-error" role="alert">{timeRuleError}</div>}
              <div className="modal-preview"><span>当前选择</span><strong>{timeRuleSummary(timeRuleDraft, timeInspection.options)}</strong></div>
              <div className="modal-actions">
                <button className="quiet-button" type="button" onClick={() => setTimeRuleModalOpen(false)}>取消</button>
                <button className="primary-button" type="button" onClick={confirmTimeRule}>确认规则</button>
              </div>
            </section>
          </div>
        )}
        {(error || backendError) && <div className="error-toast" role="alert"><Icon name="terminal" size={17} /><span>{error || backendError}</span><button type="button" onClick={() => { setError(null); setBackendError(null); }}>×</button></div>}
      </main>
    </div>
  );
}

export default App;
