import type {
  FilenameFieldSummary,
  TimeFieldOption,
  TimeInspection,
  TimeRef,
  TimeRule,
} from "../api";
import { TIME_COMPONENT_LABELS } from "./constants";
import type { TimeComponent, TimeRuleMode } from "./types";

export function parseOptionalNumber(value: string, label: string): number | "nan" | undefined {
  const text = value.trim();
  if (!text) return undefined;
  if (text.toLowerCase() === "nan") return "nan";
  const result = Number(text);
  if (!Number.isFinite(result)) throw new Error(`${label} 必须是有限数值或 nan。`);
  return result;
}

export function parseCoordinateBound(value: string, label: string, minimum: number, maximum: number): number {
  const text = value.trim();
  if (!text) throw new Error(`请输入${label}。`);
  const result = Number(text);
  if (!Number.isFinite(result) || result < minimum || result > maximum) {
    throw new Error(`${label}必须位于 ${minimum} 到 ${maximum} 之间。`);
  }
  return result;
}

export function parsePositiveInteger(value: string, label: string): number {
  const text = value.trim();
  const result = Number(text);
  if (!text || !Number.isInteger(result) || result <= 0) {
    throw new Error(`${label}必须是正整数。`);
  }
  return result;
}

export function parseNonNegativeNumber(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${label}必须是有限非负数。`);
  }
  return value;
}

export function validateDateBoundary(value: string, label: string): string {
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

export function parseFillValues(value: string, label: string): Array<number | "nan"> | undefined {
  const text = value.trim();
  if (!text) return undefined;
  return text
    .split(",")
    .map((item) => parseOptionalNumber(item, label))
    .filter((item): item is number | "nan" => item !== undefined);
}

export function timeRefKey(ref: TimeRef | undefined): string {
  return ref ? `${ref.source}:${ref.component}:${ref.index}` : "";
}

export function firstTimeRef(options: TimeFieldOption[], component: TimeComponent): TimeRef | undefined {
  return options.find((option) => option.ref.component === component)?.ref;
}

export function timeRuleMode(rule: TimeRule | null): TimeRuleMode {
  if (rule?.full) return "full";
  if (rule?.doy) return "doy";
  return "calendar";
}

export function initialTimeRule(
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

export function validateTimeRule(mode: TimeRuleMode, rule: TimeRule | null): string | null {
  if (mode === "full" && !rule?.full) return "请选择一个完整日期或完整时间字段。";
  if (mode === "doy" && (!rule?.year || !rule.doy)) return "请选择年份字段和 DOY 字段。";
  if (mode === "calendar" && (!rule?.year || !rule.month || !rule.day)) {
    return "请选择年份、月份和日期字段。";
  }
  return null;
}

export function timeRuleOptionLabel(options: TimeFieldOption[], ref: TimeRef | undefined): string {
  return options.find((option) => timeRefKey(option.ref) === timeRefKey(ref))?.label || "未选择";
}

export function timeRuleSummary(rule: TimeRule | null, options: TimeFieldOption[]): string {
  if (!rule) return "尚未选择时间规则";
  if (rule.full) return `${TIME_COMPONENT_LABELS.full}：${timeRuleOptionLabel(options, rule.full)}`;
  const parts = (["year", "month", "day", "doy"] as const)
    .filter((component) => rule[component])
    .map((component) => `${TIME_COMPONENT_LABELS[component]}：${timeRuleOptionLabel(options, rule[component])}`);
  return parts.join("；");
}

export function fieldsText(field: FilenameFieldSummary): string {
  return `位置 ${field.start}–${field.start + field.length - 1} · 长度 ${field.length}`;
}