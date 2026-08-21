import type { InspectionTaskOperation } from "../api";

export type TimeRuleMode = "full" | "doy" | "calendar";
export type TimeComponent = "full" | "year" | "month" | "day" | "doy";

export type VariableDetail = {
  name: string;
  dtype: string;
  dims: string[];
  attrs: Record<string, unknown>;
  excluded?: boolean;
};

export type VariableTransformDraft = {
  fillValues: string;
  scaleFactor: string;
  addOffset: string;
  outputFill: string;
};

export type VariableResamplingDraft = {
  inherit: boolean;
  method: string;
  skipna: boolean;
  naThres: number;
  computeDtype: string;
};

export type InspectionProgressStatus =
  | "starting"
  | "running"
  | "cancelling"
  | "finished"
  | "failed"
  | "cancelled";
export type InspectionProgressState = {
  taskId: string | null;
  operation: InspectionTaskOperation;
  label: string;
  message: string;
  completed: number;
  total: number;
  status: InspectionProgressStatus;
  startedAt: number;
};

export type InputKind = "source" | "zarr";
export type InspectionStage = "input" | "time" | "structure";
export type View = "overview" | "inspection" | "pipeline" | "tasks" | "settings";
export type BackendMode = "auto" | "rust";
export type ChunkStrategy = "time" | "space" | "custom";

export type IconName =
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

export type PipelineTargetSummary = {
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