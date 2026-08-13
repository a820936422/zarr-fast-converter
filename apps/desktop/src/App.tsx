import { useEffect, useMemo, useState } from "react";
import {
  getBackendInfo,
  inspectSource,
  inspectTimeMetadata,
  inspectZarr,
  pickDirectory,
  pickInputFile,
  pickSnapshotDestination,
  previewPipeline,
  saveInspectionSnapshot,
  startPipeline,
  type BackendInfo,
  type InspectionRequest,
  type InspectionResult,
  type PipelinePayload,
  type TimeInspection,
  type TimeRule,
} from "./api";
import { useTaskEvents } from "./taskEvents";
import "./styles.css";

type InputKind = "source" | "zarr";
type InspectionStage = "input" | "time" | "structure";
type View = "inspection" | "pipeline" | "tasks";

function reasonText(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

function App() {
  const [backend, setBackend] = useState<BackendInfo | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [inputKind, setInputKind] = useState<InputKind>("source");
  const [inputPath, setInputPath] = useState("");
  const [recursive, setRecursive] = useState(false);
  const [engine, setEngine] = useState("auto");
  const [timeInspection, setTimeInspection] = useState<TimeInspection | null>(null);
  const [timeRule, setTimeRule] = useState<TimeRule | null>(null);
  const [inspection, setInspection] = useState<InspectionResult | null>(null);
  const [stage, setStage] = useState<InspectionStage>("input");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("inspection");
  const [plan, setPlan] = useState<Record<string, unknown> | null>(null);
  const { events, tasks, cancel } = useTaskEvents();

  useEffect(() => {
    void getBackendInfo().then(setBackend).catch((reason: unknown) => setBackendError(reasonText(reason)));
  }, []);

  const chooseInput = async () => {
    setError(null);
    try {
      const selected = inputKind === "source" ? await pickDirectory() : await pickInputFile();
      if (typeof selected === "string") {
        setInputPath(selected);
        setTimeInspection(null);
        setTimeRule(null);
        setInspection(null);
        setPlan(null);
        setStage("input");
      }
    } catch (reason) {
      setError(reasonText(reason));
    }
  };

  const inspectTime = async () => {
    if (!inputPath || inputKind !== "source") return;
    setBusy(true);
    setError(null);
    try {
      const result = await inspectTimeMetadata(inputPath, recursive, engine);
      setTimeInspection(result);
      setTimeRule(result.suggested_rule);
      setStage("time");
    } catch (reason) {
      setError(reasonText(reason));
    } finally {
      setBusy(false);
    }
  };

  const inspectStructure = async () => {
    if (!inputPath) return;
    setBusy(true);
    setError(null);
    try {
      const result = inputKind === "zarr"
        ? await inspectZarr(inputPath)
        : await inspectSource({ input_dir: inputPath, mode: "auto", recursive, engine, time_rule: timeRule } satisfies InspectionRequest);
      setInspection(result);
      setStage("structure");
    } catch (reason) {
      setError(reasonText(reason));
    } finally {
      setBusy(false);
    }
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
    return {
      output: `${inputPath.replace(/[\\/]$/, "")}.zarr`,
      input_dir: inputPath,
      input_kind: inputKind === "zarr" ? "zarr" : "raw",
      inspection_kind: inspection.kind,
      time_rule: timeRule,
      recursive,
      engine,
      resample: false,
      rechunk: false,
      recompress: false,
      backend: "python",
      validate: true,
    };
  };

  const runPreview = async () => {
    const payload = buildPipelinePayload();
    if (!payload) return;
    setBusy(true);
    setError(null);
    try {
      const result = await previewPipeline(payload);
      setPlan(result);
    } catch (reason) {
      setError(reasonText(reason));
    } finally {
      setBusy(false);
    }
  };

  const runPipeline = async () => {
    const payload = buildPipelinePayload();
    if (!payload) return;
    setBusy(true);
    setError(null);
    try {
      await startPipeline(payload);
      setView("tasks");
    } catch (reason) {
      setError(reasonText(reason));
    } finally {
      setBusy(false);
    }
  };

  const stageIndex = stage === "input" ? 1 : stage === "time" ? 2 : 3;
  const canInspectTime = inputKind === "source" && Boolean(inputPath) && !busy;
  const canInspectStructure = Boolean(inputPath) && (inputKind === "zarr" || Boolean(timeRule) || stage === "time") && !busy;
  const runningTasks = useMemo(() => tasks.filter((task) => task.status === "running" || task.status === "cancelling"), [tasks]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">Z</span><div><strong>快速 Zarr 转换器</strong><small>桌面工作台</small></div></div>
        <nav aria-label="主导航">
          <button className={view === "inspection" ? "nav-item active" : "nav-item"} type="button" onClick={() => setView("inspection")}>数据检查</button>
          <button className={view === "pipeline" ? "nav-item active" : "nav-item"} type="button" disabled={!inspection} onClick={() => setView("pipeline")}>处理流程</button>
          <button className={view === "tasks" ? "nav-item active" : "nav-item"} type="button" onClick={() => setView("tasks")}>任务中心</button>
        </nav>
        <div className="sidebar-footer">{backend ? `${backend.runtime} · v${backend.version}` : "连接后端…"}</div>
      </aside>
      <section className="workspace">
        <header className="topbar">
          <div><span className="eyebrow">v1.7.1 / Tauri shell</span><h1>{view === "inspection" ? "数据检查" : view === "pipeline" ? "处理流程" : "任务中心"}</h1></div>
          <span className="status-badge">{busy ? "处理中" : runningTasks.length ? "任务运行中" : inspection ? "检查完成" : "准备就绪"}</span>
        </header>

        {view === "inspection" && <>
          <div className="stepper" aria-label="检查步骤">{["输入数据", "时间规则", "结构结果"].map((label, index) => <div className={index + 1 <= stageIndex ? "step active" : "step"} key={label}><span>{index + 1}</span>{label}</div>)}</div>
          <section className="content-grid">
            <article className="card hero-card">
              <span className="step-label">步骤 {stageIndex} / 3</span><h2>输入数据</h2>
              <div className="segmented" role="group" aria-label="输入类型"><button className={inputKind === "source" ? "selected" : ""} type="button" onClick={() => setInputKind("source")}>原始数据目录</button><button className={inputKind === "zarr" ? "selected" : ""} type="button" onClick={() => setInputKind("zarr")}>现有 Zarr</button></div>
              <label className="field-label" htmlFor="input-path">路径</label><div className="path-row"><input id="input-path" value={inputPath} onChange={(event) => setInputPath(event.target.value)} placeholder={inputKind === "source" ? "选择 NetCDF / HDF / TIFF 目录" : "选择 Zarr v3 目录"} /><button className="secondary-button" type="button" onClick={() => void chooseInput()}>浏览</button></div>
              <div className="options-row">{inputKind === "source" && <label><input type="checkbox" checked={recursive} onChange={(event) => setRecursive(event.target.checked)} />递归扫描</label>}{inputKind === "source" && <label>引擎 <select value={engine} onChange={(event) => setEngine(event.target.value)}><option value="auto">自动</option><option value="h5netcdf">h5netcdf</option><option value="netcdf4">netcdf4</option><option value="rasterio">rasterio</option></select></label>}</div>
              <div className="actions">{inputKind === "source" && <button className="secondary-button" disabled={!canInspectTime} type="button" onClick={() => void inspectTime()}>检查时间信息</button>}<button className="primary-button" disabled={!canInspectStructure} type="button" onClick={() => void inspectStructure()}>检查数据结构</button></div>
              {timeInspection && <div className="result-box"><strong>时间来源待确认</strong><p>{timeInspection.report}</p><select aria-label="时间规则" value={timeRule ? JSON.stringify(timeRule) : ""} onChange={(event) => setTimeRule(event.target.value ? JSON.parse(event.target.value) as TimeRule : null)}><option value="">请选择已确认规则</option><option value={timeInspection.suggested_rule ? JSON.stringify(timeInspection.suggested_rule) : ""}>使用建议规则</option></select></div>}
              {inspection && <div className="result-box"><strong>结构检查完成</strong><p>{inspection.report}</p><p>{inspection.warnings.length ? `警告：${inspection.warnings.join("；")}` : "无警告"}</p><button className="secondary-button" type="button" onClick={() => void saveSnapshot()}>保存检查快照</button></div>}
            </article>
            <article className="card checklist-card"><h2>迁移进度</h2><ul><li className="done">Tauri 应用壳</li><li className={inputPath ? "done" : ""}>输入路径</li><li className={timeRule || inputKind === "zarr" ? "done" : ""}>时间规则确认</li><li className={inspection ? "done" : ""}>结构检查</li></ul><div className="backend-state" role="status"><span className={backend ? "dot online" : "dot"} />{backend ? `${backend.app} · ${backend.runtime}` : "正在连接后端…"}</div></article>
          </section>
        </>}

        {view === "pipeline" && <section className="content-grid"><article className="card hero-card"><span className="step-label">已锁定检查结果</span><h2>一条龙处理</h2><p>输入：{inputPath}</p><p>输出：{inputPath ? `${inputPath.replace(/[\\/]$/, "")}.zarr` : "未设置"}</p><div className="actions"><button className="secondary-button" disabled={busy} type="button" onClick={() => void runPreview()}>预览计划</button><button className="primary-button" disabled={busy} type="button" onClick={() => void runPipeline()}>启动任务</button></div>{plan && <div className="result-box"><strong>计划已生成</strong><p>{JSON.stringify(plan, null, 2)}</p></div>}</article><article className="card checklist-card"><h2>请求操作</h2><ul><li className="done">源数据检查</li><li>空间重采样（未选择）</li><li>重分块（未选择）</li><li>重压缩（未选择）</li></ul></article></section>}

        {view === "tasks" && <section className="content-grid"><article className="card hero-card"><h2>任务历史</h2>{tasks.length === 0 ? <p>暂无任务。完成数据检查后可从处理流程启动。</p> : <div className="task-list">{tasks.map((task) => <div className="task-row" key={task.taskId}><div><strong>{task.command}</strong><small>{task.taskId}</small></div><span className={`task-status ${task.status}`}>{task.status}</span>{(task.status === "running" || task.status === "cancelling") && <button className="secondary-button" type="button" onClick={() => void cancel(task.taskId)}>取消</button>}{task.manifest && <span className="manifest-path">{task.manifest}</span>}</div>)}</div>}</article><article className="card checklist-card"><h2>实时日志</h2><div className="event-log">{events.slice(-12).map((event, index) => <p key={`${event.request_id}-${event.sequence}-${index}`}>[{event.stage ?? "worker"}] {event.event} {typeof event.payload.message === "string" ? event.payload.message : ""}</p>)}</div></article></section>}

        {(error || backendError) && <p className="error-text">{error || backendError}</p>}
      </section>
    </main>
  );
}

export default App;
