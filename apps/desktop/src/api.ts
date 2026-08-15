import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";

export type BackendInfo = {
  app: string;
  version: string;
  runtime: string;
};
export type OperationCapability = {
  operation: string;
  supported: boolean;
  reason: string | null;
  limitations: string[];
};

export type BackendCapability = {
  backend: string;
  protocol_version: number;
  crate_version: string | null;
  operations: string[];
  capabilities: OperationCapability[];
};

export type TimeRef = {
  source: "filename" | "time";
  component: "full" | "year" | "month" | "day" | "doy";
  index: number;
};

export type TimeRule = Partial<Record<"full" | "year" | "month" | "day" | "doy", TimeRef>>;

export type TimeFieldOption = {
  ref: TimeRef;
  label: string;
  sample: string;
};

export type InspectionRequest = {
  input_dir: string;
  mode?: "auto" | "complete" | "filename";
  recursive?: boolean;
  engine?: string;
  template?: string | null;
  field_values?: string[];
  source_dimensions?: [string, string, string];
  workers?: number;
  time_rule?: TimeRule | null;
  cache_path?: string | null;
};

export type FilenameFieldSummary = {
  index: number;
  start: number;
  length: number;
  sample: string;
  values: string[];
  changed: boolean;
};

export type TimeDimensionSummary = {
  exists: boolean;
  name: string | null;
  raw_values: string[];
  decoded_values: string[];
  attrs: Record<string, unknown>;
  format_label: string;
};

export type TimeInspection = {
  input_dir: string;
  files: string[];
  engine: string;
  dimensions: string[];
  coordinates: string[];
  filename_fields: FilenameFieldSummary[];
  time_dimension: TimeDimensionSummary;
  options: TimeFieldOption[];
  suggested_rule: TimeRule | null;
  report: string;
};

export type InspectionTaskOperation = "inspect_time_metadata" | "inspect_source" | "inspect_zarr";

export const startInspection = (operation: InspectionTaskOperation, payload: Record<string, unknown>) =>
  invoke<string>("start_inspection", { operation, payload });

export type InspectionResult = {
  kind: "source" | "zarr" | "temporary";
  path: string;
  report: string;
  warnings: string[];
  snapshot: Record<string, unknown>;
};

export type PipelinePayload = Record<string, unknown> & { output: string };

export type ResourceSnapshot = {
  capturedAtMs: number;
  logicalCpus: number;
  memoryTotalBytes: number;
  memoryAvailableBytes: number;
};

export type NativeInspectTaskRequest = {
  operation: "zarr.inspect";
  payload: { path: string; array_path: string };
};

export type NativeRechunkTaskRequest = {
  operation: "zarr.rechunk_f32";
  payload: Record<string, unknown>;
};
export type NativeWriteF64TaskRequest = {
  operation: "zarr.write_f64";
  payload: { path: string; array_path: string; shape: number[]; chunks: number[]; values: number[] };
};

export type NativeTaskRequest = NativeInspectTaskRequest | NativeRechunkTaskRequest | NativeWriteF64TaskRequest;

export type TaskSummary = {
  taskId: string;
  requestId: string;
  command: string;
  status: "running" | "cancelling" | "finished" | "failed" | "cancelled";
  manifest: string | null;
  error: Record<string, unknown> | null;
  cancellationFile: string | null;
  startedAt: number;
  resource: ResourceSnapshot | null;
};

export type TaskEvent = {
  protocol_version: 1;
  request_id: string;
  task_id: string | null;
  sequence: number;
  event: "accepted" | "started" | "inspection_ready" | "plan_ready" | "progress" | "resource" | "log" | "finished" | "failed" | "cancelled";
  stage?: string | null;
  payload: Record<string, unknown>;
};


export const getBackendInfo = () => invoke<BackendInfo>("get_backend_info");
export const inspectSource = (request: InspectionRequest) => invoke<InspectionResult>("inspect_source", { payload: request });
export const getNativeCapabilities = () => invoke<BackendCapability>("native_capabilities");
export const inspectZarr = (path: string) => invoke<InspectionResult>("inspect_zarr", { path });
export const inspectTimeMetadata = (inputDir: string, recursive = false, engine = "auto") =>
  invoke<TimeInspection>("inspect_time_metadata", { inputDir, recursive, engine });
export const saveInspectionSnapshot = (request: Record<string, unknown>) =>
  invoke<Record<string, unknown>>("save_inspection_snapshot", { payload: request });
export const previewPipeline = (payload: PipelinePayload) =>
  invoke<Record<string, unknown>>("preview_pipeline", { payload });
export const startPipeline = (payload: PipelinePayload) => invoke<string>("start_pipeline", { payload });
export const resumePipeline = (payload: PipelinePayload) => invoke<string>("resume_pipeline", { payload });
export const inspectPipelineRecovery = (path: string) => invoke<InspectionResult>("inspect_pipeline_recovery", { path });
export const getTask = (taskId: string) => invoke<TaskSummary | null>("get_task", { taskId });
export const startNativeTask = (request: NativeTaskRequest) => invoke<string>("start_native_task", { request });
export const listTasks = () => invoke<TaskSummary[]>("list_tasks");
export const cancelTask = (taskId: string) => invoke<void>("cancel_task", { taskId });
export const pickDirectory = () => open({ directory: true, multiple: false });
export const pickInputFile = () => open({ multiple: false, filters: [{ name: "遥感数据", extensions: ["nc", "hdf", "h5", "tif", "tiff", "zarr"] }] });
export const pickSnapshotDestination = () => save({ filters: [{ name: "检查快照", extensions: ["json"] }] });
