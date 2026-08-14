import { useEffect, useMemo, useState } from "react";
import {
  getBackendInfo,
  getNativeCapabilities,
  inspectPipelineRecovery,
  inspectSource,
  inspectTimeMetadata,
  inspectZarr,
  pickDirectory,
  pickSnapshotDestination,
  previewPipeline,
  saveInspectionSnapshot,
  startNativeTask,
  startPipeline,
  resumePipeline,
  type BackendCapability,
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
type View = "inspection" | "pipeline" | "tasks" | "settings";
type BackendMode = "python" | "auto" | "rust";

function reasonText(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

function App() {
  const [backend, setBackend] = useState<BackendInfo | null>(null);
  const [nativeCapability, setNativeCapability] = useState<BackendCapability | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [inputKind, setInputKind] = useState<InputKind>("source");
  const [arrayPath, setArrayPath] = useState("/value");
  const [selectedVariables, setSelectedVariables] = useState<string[]>([]);
  const [inputPath, setInputPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
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
  const [recoveryPath, setRecoveryPath] = useState("");
  const [recoveryInspection, setRecoveryInspection] = useState<InspectionResult | null>(null);
  const [resample, setResample] = useState(false);
  const [resampleMethod, setResampleMethod] = useState("bilinear");
  const [resolution, setResolution] = useState(0.1);
  const [rechunk, setRechunk] = useState(false);
  const [targetMib, setTargetMib] = useState(128);
  const [recompress, setRecompress] = useState(false);
  const [compression, setCompression] = useState("auto");
  const [backendMode, setBackendMode] = useState<BackendMode>("auto");
  const [favorites, setFavorites] = useState<string[]>([]);
  const [recentPaths, setRecentPaths] = useState<string[]>([]);
  const { events, tasks, cancel } = useTaskEvents();

  useEffect(() => {
    void getBackendInfo().then(setBackend).catch((reason: unknown) => setBackendError(reasonText(reason)));
    void getNativeCapabilities().then(setNativeCapability).catch((reason: unknown) => setBackendError(reasonText(reason)));
  }, []);

  useEffect(() => {
    const event = events[events.length - 1];
    if (event?.event !== "finished" || event.payload.operation !== "zarr.inspect") return;
    const summary = event.payload.summary;
    if (!summary || typeof summary !== "object" || Array.isArray(summary)) return;
    setInspection({ kind: "zarr", path: inputPath, report: "Rust native Zarr inspection complete", warnings: [], snapshot: summary as Record<string, unknown> });
    setStage("structure");
  }, [events, inputPath]);
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

  const chooseInput = async () => {
    setError(null);
    try {
      const selected = await pickDirectory();
      if (typeof selected === "string") {
        setInputPath(selected);
        rememberPath(selected);
        setOutputPath("");
        setTimeInspection(null);
        setTimeRule(null);
        setInspection(null);
        setSelectedVariables([]);
        setPlan(null);
        setStage("input");
      }
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
      output: outputPath || `${inputPath.replace(/[\\/]$/, "")}.zarr`,
      input_dir: inputPath,
      input_kind: inputKind === "zarr" ? "zarr" : "raw",
      inspection_kind: inspection.kind,
      time_rule: timeRule,
      recursive,
      engine,
      variables: selectedVariables,
      resample,
      method: resampleMethod,
      resolution,
      rechunk,
      strategy: "time",
      target_mib: targetMib,
      recompress,
      compression: recompress ? compression : "none",
      backend: backendMode,
      validate: true,
    };
  };

  const runPreview = async () => {
    const payload = buildPipelinePayload();
    if (!payload) return;
    setBusy(true);
    setError(null);
    try {
      setPlan(await previewPipeline(payload));
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
  const canInspectTime = inputKind === "source" && Boolean(inputPath) && !busy;
  const canInspectStructure = Boolean(inputPath) && (inputKind === "zarr" || Boolean(timeRule) || stage === "time") && !busy;
  const runningTasks = useMemo(() => tasks.filter((task) => task.status === "running" || task.status === "cancelling"), [tasks]);
  const supported = (operation: string) => nativeCapability?.capabilities.find((item) => item.operation === operation);
  const formatBytes = (value: number) => {
    if (!value) return "—";
    const units = ["B", "KiB", "MiB", "GiB"];
    const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
    return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
  };
  const variableOptions = useMemo(() => {
    const raw = inspection?.snapshot.variables;
    if (!Array.isArray(raw)) return [];
    return raw.flatMap((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) return [];
      const name = (item as Record<string, unknown>).name;
      return typeof name === "string" && name ? [name] : [];
    });
  }, [inspection]);

  useEffect(() => {
    setSelectedVariables((current) => {
      const valid = current.filter((name) => variableOptions.includes(name));
      return valid.length ? valid : variableOptions;
    });
  }, [variableOptions]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">Z</span><div><strong>快速 Zarr 转换器</strong><small>桌面工作台</small></div></div>
        <nav aria-label="主导航">
          <button className={view === "inspection" ? "nav-item active" : "nav-item"} type="button" onClick={() => setView("inspection")}>数据检查</button>
          <button className={view === "pipeline" ? "nav-item active" : "nav-item"} type="button" disabled={!inspection} onClick={() => setView("pipeline")}>处理流程</button>
          <button className={view === "tasks" ? "nav-item active" : "nav-item"} type="button" onClick={() => setView("tasks")}>任务中心</button>
          <button className={view === "settings" ? "nav-item active" : "nav-item"} type="button" onClick={() => setView("settings")}>路径设置</button>
        </nav>
        <div className="sidebar-footer">{backend ? `${backend.runtime} · v${backend.version}` : "连接后端…"}</div>
      </aside>
      <section className="workspace">
        <header className="topbar">
          <div><span className="eyebrow">v1.7.2 / Native-first Tauri shell</span><h1>{view === "inspection" ? "数据检查" : view === "pipeline" ? "处理流程" : view === "tasks" ? "任务中心" : "路径设置"}</h1></div>
          <span className="status-badge">{busy ? "处理中" : runningTasks.length ? "任务运行中" : inspection ? "检查完成" : "准备就绪"}</span>
        </header>

        {view === "inspection" && <>
          <div className="stepper" aria-label="检查步骤">{["输入数据", "时间规则", "结构结果"].map((label, index) => <div className={index + 1 <= stageIndex ? "step active" : "step"} key={label}><span>{index + 1}</span>{label}</div>)}</div>
          <section className="content-grid">
            <article className="card hero-card">
              <span className="step-label">步骤 {stageIndex} / 3</span><h2>输入数据</h2>
              <div className="segmented" role="group" aria-label="输入类型"><button className={inputKind === "source" ? "selected" : ""} type="button" onClick={() => setInputKind("source")}>原始数据目录</button><button className={inputKind === "zarr" ? "selected" : ""} type="button" onClick={() => setInputKind("zarr")}>现有 Zarr</button></div>
              <label className="field-label" htmlFor="input-path">路径</label><div className="path-row"><input id="input-path" value={inputPath} onChange={(event) => setInputPath(event.target.value)} placeholder={inputKind === "source" ? "选择 NetCDF / HDF / TIFF 目录" : "选择 Zarr v3 目录"} /><button className="secondary-button" type="button" onClick={() => void chooseInput()}>浏览</button></div>
              {inputKind === "zarr" && <div className="path-row"><input aria-label="Zarr array path" value={arrayPath} onChange={(event) => setArrayPath(event.target.value)} placeholder="array path，例如 /value" /><button className="secondary-button" type="button" disabled={!inputPath || busy} onClick={() => void inspectNativeZarr()}>Rust native 检查</button></div>}
              <div className="options-row">{inputKind === "source" && <label><input type="checkbox" checked={recursive} onChange={(event) => setRecursive(event.target.checked)} />递归扫描</label>}{inputKind === "source" && <label>引擎 <select value={engine} onChange={(event) => setEngine(event.target.value)}><option value="auto">自动</option><option value="h5netcdf">h5netcdf</option><option value="netcdf4">netcdf4</option><option value="rasterio">rasterio</option></select></label>}</div>
              <div className="actions">{inputKind === "source" && <button className="secondary-button" disabled={!canInspectTime} type="button" onClick={() => void inspectTime()}>检查时间信息</button>}<button className="primary-button" disabled={!canInspectStructure} type="button" onClick={() => void inspectStructure()}>检查数据结构</button></div>
              {timeInspection && <div className="result-box"><strong>时间来源待确认</strong><p>{timeInspection.report}</p><select aria-label="时间规则" value={timeRule ? JSON.stringify(timeRule) : ""} onChange={(event) => setTimeRule(event.target.value ? JSON.parse(event.target.value) as TimeRule : null)}><option value="">请选择已确认规则</option><option value={timeInspection.suggested_rule ? JSON.stringify(timeInspection.suggested_rule) : ""}>使用建议规则</option></select></div>}
              {inspection && <div className="result-box"><strong>结构检查完成</strong><p>{inspection.report}</p><p>{inspection.warnings.length ? `警告：${inspection.warnings.join("；")}` : "无警告"}</p><button className="secondary-button" type="button" onClick={() => void saveSnapshot()}>保存检查快照</button></div>}
              {variableOptions.length > 0 && <div className="variable-picker"><strong>参与处理的变量</strong>{variableOptions.map((name) => <label key={name}><input type="checkbox" checked={selectedVariables.includes(name)} onChange={(event) => setSelectedVariables((current) => event.target.checked ? [...current, name] : current.filter((item) => item !== name))} />{name}</label>)}</div>}
            </article>
            <article className="card checklist-card"><h2>Native 能力</h2><p className="helper-text">能力查询由 Tauri Rust 直接完成，不启动 Python worker。</p>{nativeCapability ? <div className="capability-list">{nativeCapability.capabilities.map((item) => <div className={`capability-item ${item.supported ? "supported" : "unavailable"}`} key={item.operation}><strong>{item.operation}</strong><small>{item.supported ? "可用" : item.reason || "使用 Python fallback"}</small></div>)}</div> : <p className="helper-text">正在读取 capability…</p>}<div className="backend-state" role="status"><span className={backend ? "dot online" : "dot"} />{backend ? `${backend.app} · ${backend.runtime}` : "正在连接后端…"}</div></article>
          </section>
        </>}

        {view === "pipeline" && <section className="content-grid"><article className="card hero-card"><span className="step-label">已锁定检查结果</span><h2>处理流程</h2><p>输入：{inputPath}</p><label className="field-label" htmlFor="output-path">输出 Zarr 目录</label><input id="output-path" value={outputPath || `${inputPath.replace(/[\\/]$/, "")}.zarr`} onChange={(event) => setOutputPath(event.target.value)} /><div className="options-row"><label><input type="checkbox" checked={resample} onChange={(event) => setResample(event.target.checked)} />空间重采样</label><label><input type="checkbox" checked={rechunk} onChange={(event) => setRechunk(event.target.checked)} />重分块</label><label><input type="checkbox" checked={recompress} onChange={(event) => setRecompress(event.target.checked)} />重压缩</label></div>{resample && <div className="options-row"><label>方法 <select value={resampleMethod} onChange={(event) => setResampleMethod(event.target.value)}><option value="nearest_s2d">nearest_s2d</option><option value="nearest_d2s">nearest_d2s</option><option value="bilinear">bilinear</option><option value="conservative">conservative（Python）</option></select></label><label>分辨率 <input className="small-input" type="number" min="0" step="any" value={resolution} onChange={(event) => setResolution(Number(event.target.value))} /></label></div>}{rechunk && <div className="options-row"><label>目标 chunk MiB <input className="small-input" type="number" min="1" step="1" value={targetMib} onChange={(event) => setTargetMib(Number(event.target.value))} /></label></div>}{recompress && <div className="options-row"><label>压缩 <select value={compression} onChange={(event) => setCompression(event.target.value)}><option value="auto">自动</option><option value="zstd">Zstd</option><option value="blosc-lz4">Blosc LZ4</option><option value="gzip">Gzip</option></select></label></div>}<div className="options-row"><label>执行 backend <select value={backendMode} onChange={(event) => setBackendMode(event.target.value as BackendMode)}><option value="python">Python reference</option><option value="auto">Auto（按 capability）</option><option value="rust">Rust native（不支持则失败）</option></select></label></div>{backendMode !== "python" && <p className="helper-text">Auto/Rust 只会启用已通过 capability 的最终化操作；原始格式、复杂重采样和 pipeline 仍由 Python 处理。</p>}<div className="actions"><button className="secondary-button" disabled={busy} type="button" onClick={() => void runPreview()}>预览计划</button><button className="primary-button" disabled={busy} type="button" onClick={() => void runPipeline()}>启动任务</button></div>{plan && <div className="result-box"><strong>计划已生成</strong><p>{JSON.stringify(plan, null, 2)}</p></div>}</article><article className="card checklist-card"><h2>请求能力</h2><ul><li className="done">源数据检查</li><li className={resample ? "done" : ""}>空间重采样 {resample ? `（${resampleMethod}）` : "（未选择）"}</li><li className={rechunk ? "done" : ""}>重分块 {rechunk ? "（已选择）" : "（未选择）"}</li><li className={recompress ? "done" : ""}>重压缩 {recompress ? `（${compression}）` : "（未选择）"}</li></ul>{resampleMethod && <p className="helper-text">native status：{supported(`resample.${resampleMethod}`)?.supported ? "supported" : "Python fallback"}</p>}</article></section>}

        {view === "tasks" && <section className="content-grid">
          <article className="card hero-card"><h2>任务历史</h2>{tasks.length === 0 ? <p>暂无任务。完成数据检查后可从处理流程启动。</p> : <div className="task-list">{tasks.map((task) => <div className="task-row" key={task.taskId}><div><strong>{task.command}</strong><small>{task.taskId}</small>{task.resource && <small>{task.resource.logicalCpus} CPU · 可用内存 {formatBytes(task.resource.memoryAvailableBytes)}</small>}</div><span className={`task-status ${task.status}`}>{task.status}</span>{(task.status === "running" || task.status === "cancelling") && <button className="secondary-button" type="button" onClick={() => void cancel(task.taskId)}>取消</button>}{task.manifest && <span className="manifest-path">{task.manifest}</span>}</div>)}</div>}</article>
          <article className="card checklist-card"><h2>任务恢复</h2><p className="helper-text">输入保留的 pipeline 临时目录，先检查 manifest/checkpoint，再恢复到新的输出目录。</p><label className="field-label" htmlFor="recovery-path">临时任务目录</label><div className="path-row"><input id="recovery-path" value={recoveryPath} onChange={(event) => setRecoveryPath(event.target.value)} placeholder="例如 /tmp/.../pipeline-..." /><button className="secondary-button" type="button" onClick={() => void inspectRecovery()}>检查</button></div><div className="actions"><button className="primary-button" type="button" disabled={!recoveryInspection || busy} onClick={() => void resumeRecovery()}>恢复任务</button></div>{recoveryInspection && <div className="result-box"><strong>恢复检查完成</strong><p>{recoveryInspection.report}</p></div>}</article>
          <article className="card checklist-card"><h2>实时日志</h2><div className="event-log">{events.slice(-20).map((eve) => <p key={`${eve.request_id}-${eve.sequence}`}><strong>{eve.event}</strong> · {eve.stage || "—"}<br />{JSON.stringify(eve.payload)}</p>)}</div></article>
        </section>}

        {view === "settings" && <section className="content-grid"><article className="card hero-card"><h2>路径设置</h2><p className="helper-text">路径只保存在当前桌面用户的 localStorage，不写入项目文件。</p><label className="field-label" htmlFor="settings-input-path">当前输入路径</label><div className="path-row"><input id="settings-input-path" value={inputPath} onChange={(event) => setInputPath(event.target.value)} placeholder="输入或选择目录" /><button className="secondary-button" type="button" onClick={() => void chooseInput()}>浏览</button></div><div className="actions"><button className="secondary-button" type="button" disabled={!inputPath} onClick={() => toggleFavorite(inputPath)}>{inputPath && favorites.includes(inputPath) ? "取消收藏" : "收藏当前路径"}</button><button className="primary-button" type="button" disabled={!inputPath} onClick={() => { rememberPath(inputPath); setView("inspection"); }}>使用此路径</button></div></article><article className="card checklist-card"><h2>收藏和最近目录</h2><div className="path-history"><strong>收藏</strong>{favorites.length ? favorites.map((path) => <div className="history-row" key={`favorite-${path}`}><button className="path-link" type="button" onClick={() => setInputPath(path)}>{path}</button><button className="icon-button" type="button" onClick={() => toggleFavorite(path)}>移除</button></div>) : <p className="helper-text">暂无收藏路径。</p>}<strong>最近使用</strong>{recentPaths.length ? recentPaths.map((path) => <div className="history-row" key={`recent-${path}`}><button className="path-link" type="button" onClick={() => setInputPath(path)}>{path}</button><button className="icon-button" type="button" onClick={() => toggleFavorite(path)}>{favorites.includes(path) ? "已收藏" : "收藏"}</button></div>) : <p className="helper-text">暂无最近目录。</p>}</div></article></section>}

        {(error || backendError) && <p className="error-text">{error || backendError}</p>}
      </section>
    </main>
  );
}

export default App;
