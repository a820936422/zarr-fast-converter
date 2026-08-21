import type {
  BackendCapability,
  FilenameFieldSummary,
  TaskEvent,
  TaskSummary,
  TimeInspection,
} from "../api";
import { OPERATION_LABELS } from "./constants";
import type { InputKind } from "./types";

export function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function attributeText(attrs: Record<string, unknown>, key: string): string {
  const value = attrs[key];
  return value === undefined || value === null ? "" : String(value);
}

export function fieldValuesPreview(field: FilenameFieldSummary): string {
  const values = field.values.slice(0, 4);
  if (!values.length) return "暂无样例值";
  return values.join("、") + (field.values.length > values.length ? " ……" : "");
}

export function timeValuesPreview(values: string[]): string {
  if (!values.length) return "未发现可解码值";
  const preview = values.slice(0, 3).join("、");
  return preview + (values.length > 3 ? " ……" : "");
}

export function planValueText(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (Array.isArray(value)) return value.map((item) => planValueText(item, "")).join(" × ") || fallback;
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString() : fallback;
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

export function planSeconds(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toLocaleString()} s`
    : "—";
}

export function planTuple(value: unknown): string {
  return Array.isArray(value) ? value.map((item) => planValueText(item)).join(" × ") : planValueText(value);
}

export function planAxisSummary(value: unknown): string {
  if (!Array.isArray(value) || value.length === 0) return "—";
  const first = planValueText(value[0]);
  const last = planValueText(value[value.length - 1]);
  return `${first} → ${last}（${value.length.toLocaleString()} 个边界）`;
}

export function planOperationLabel(operation: string): string {
  return (
    {
      conversion: "格式转换",
      resampling: "空间重采样",
      rechunking: "重分块",
      recompression: "重压缩",
    }[operation] || operation
  );
}

export function planDispositionLabel(disposition: string): string {
  return (
    {
      executed_as_stage: "执行阶段",
      not_requested: "未请求",
      skipped: "跳过",
      reused: "复用已有结果",
    }[disposition] || disposition
  );
}

export function operationLabel(operation: string): string {
  return OPERATION_LABELS[operation] || operation;
}

export function capabilityReason(item: BackendCapability["capabilities"][number]): string {
  if (item.supported) return "已就绪 · 原生执行";
  return item.reason || "当前走兼容执行";
}

export function reasonText(reason: unknown): string {
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

export function formatBytes(value: number): string {
  if (!value) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

export function formatTaskStatus(status: TaskSummary["status"]): string {
  return (
    {
      running: "运行中",
      cancelling: "正在取消",
      finished: "已完成",
      failed: "失败",
      cancelled: "已取消",
    }[status]
  );
}

export function formatCommand(command: string): string {
  return (
    {
      native_task: "原生任务",
      run_pipeline: "数据处理",
      resume_pipeline: "恢复处理",
      inspect_time_metadata: "时间轴检查",
      inspect_source: "结构检查",
      inspect_zarr: "Zarr 结构检查",
    }[command] || command
  );
}

export function eventText(event: TaskEvent): string {
  const message = event.payload.message;
  if (typeof message === "string" && message) return message;
  if (event.event === "progress") {
    const completed = event.payload.completed;
    const total = event.payload.total;
    const temporary = event.payload.temporary_bytes;
    const eta = event.payload.eta_seconds;
    const parts: string[] = [];
    if (typeof completed === "number" && typeof total === "number" && total > 0) {
      parts.push(`业务 ${Math.round((completed / total) * 100)}%`);
    }
    if (typeof temporary === "number") parts.push(`临时观测 ${formatBytes(temporary)}`);
    if (typeof eta === "number" && Number.isFinite(eta) && eta > 0) {
      parts.push(`预计 ${Math.ceil(eta)}s`);
    }
    if (parts.length) return parts.join(" · ");
  }
  return event.stage || "任务事件";
}

export function timeLabel(value: string): string {
  return value || "—";
}

export function pathBaseName(value: string): string {
  const trimmed = value.replace(/[\\/]+$/, "");
  return trimmed.split(/[\\/]/).pop() || "output";
}

export function pathParent(value: string): string {
  const trimmed = value.replace(/[\\/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index >= 0 ? trimmed.slice(0, index) || trimmed.slice(0, 1) : ".";
}

export function joinPath(parent: string, child: string): string {
  const separator = parent.includes("\\") && !parent.includes("/") ? "\\" : "/";
  const trimmed = parent.replace(/[\\/]+$/, "");
  return `${trimmed}${trimmed ? separator : separator}${child}`;
}

export function outputStoreName(inputPath: string, inputKind: InputKind): string {
  const base = pathBaseName(inputPath).replace(/\.zarr$/i, "") || "output";
  return inputKind === "zarr" ? `${base}-processed.zarr` : `${base}.zarr`;
}

export function defaultOutputPath(inputPath: string, inputKind: InputKind): string {
  if (!inputPath) return "";
  return joinPath(pathParent(inputPath), outputStoreName(inputPath, inputKind));
}

export function timeRuleSummaryText(rule: unknown): string {
  const record = asRecord(rule);
  if (!record) return "尚未选择时间规则";
  const full = record.full;
  if (full && typeof full === "object") return "已确认完整日期字段";
  const parts: string[] = [];
  for (const component of ["year", "month", "day", "doy"] as const) {
    const ref = record[component];
    if (ref && typeof ref === "object") parts.push(component);
  }
  return parts.length ? `已确认字段：${parts.join("、")}` : "尚未选择时间规则";
}

export function inspectionTitle(inspection: TimeInspection): string {
  const base = pathBaseName(inspection.input_dir);
  return base || "数据检查";
}