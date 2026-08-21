import type {
  IconName,
  TimeComponent,
  VariableResamplingDraft,
  VariableTransformDraft,
  View,
} from "./types";

export const TIME_COMPONENT_LABELS: Record<TimeComponent, string> = {
  full: "完整日期/时间",
  year: "年份",
  month: "月份",
  day: "日期",
  doy: "年内日序（DOY）",
};

export const DEFAULT_VARIABLE_RESAMPLING: VariableResamplingDraft = {
  inherit: true,
  method: "bilinear",
  skipna: true,
  naThres: 1,
  computeDtype: "source",
};

export const RESAMPLING_METHODS = [
  ["bilinear", "双线性插值"],
  ["conservative", "保守重采样"],
  ["conservative_normed", "归一化保守重采样"],
  ["nearest_s2d", "最近邻（源到目标）"],
  ["nearest_d2s", "最近邻（目标到源）"],
  ["patch", "Patch 插值"],
] as const;

export const EMPTY_VARIABLE_TRANSFORM: VariableTransformDraft = {
  fillValues: "",
  scaleFactor: "",
  addOffset: "",
  outputFill: "",
};

export const ICON_PATHS: Record<IconName, string> = {
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

export const VIEW_TITLES: Record<View, string> = {
  overview: "工作台",
  inspection: "数据检查",
  pipeline: "处理流程",
  tasks: "任务中心",
  settings: "路径设置",
};

export const OPERATION_LABELS: Record<string, string> = {
  probe: "环境探测",
  "raw.netcdf.inspect": "NetCDF 原生检查",
  "raw.netcdf.convert": "NetCDF 原生转换",
  "resample.nearest": "最近邻重采样",
  "resample.bilinear": "双线性重采样",
  "resample.conservative": "保守重采样",
  "resample.conservative_normed": "归一化保守重采样",
  "zarr.inspect": "Zarr 结构检查",
  "zarr.read_chunk_f32": "Float32 chunk 读取",
  "zarr.read_chunk_f64": "Float64 chunk 读取",
  "zarr.read_region_f32": "Float32 region 读取",
  "zarr.read_region_f64": "Float64 region 读取",
  "zarr.write_f32": "Float32 数组写入",
  "zarr.write_f64": "Float64 数组写入",
  "zarr.rechunk_f32": "Float32 重分块",
  "zarr.rechunk_f32_codec": "Float32 codec 重分块",
  "zarr.rechunk_f32_cancel": "Float32 重分块（可取消）",
  "zarr.rechunk_f64": "Float64 重分块",
  "zarr.rechunk_f64_cancel": "Float64 重分块（可取消）",
  "zarr.rechunk_multi": "多变量重分块",
};