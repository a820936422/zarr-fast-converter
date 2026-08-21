import { useState } from "react";
import type { TaskEvent } from "../api";
import { asRecord, eventText, formatBytes } from "../lib/format";
import { Icon } from "./Icon";

export function TaskDiagnosticsCard({ event }: { event: TaskEvent | null }) {
  const [copied, setCopied] = useState(false);
  if (!event) {
    return (
      <section className="surface event-card">
        <div className="surface-title compact">
          <div><span className="section-kicker">TASK DIAGNOSTICS</span><h2>诊断详情</h2></div>
          <Icon name="terminal" size={18} />
        </div>
        <p className="surface-description">暂无事件，启动任务后这里会显示 backend、checkpoint 和进度指标。</p>
      </section>
    );
  }
  const payload = event.payload || {};
  const backend = asRecord(payload.backend);
  const requested = typeof backend?.requested === "string" ? backend.requested : typeof payload.requested_backend === "string" ? payload.requested_backend : null;
  const resolved = typeof backend?.resolved === "string" ? backend.resolved : typeof payload.resolved_backend === "string" ? payload.resolved_backend : null;
  const fallback = typeof backend?.fallback_reason === "string" ? backend.fallback_reason : typeof payload.fallback_reason === "string" ? payload.fallback_reason : null;
  const checkpoint = typeof payload.stage_checkpoint === "string" ? payload.stage_checkpoint : null;
  const manifest = typeof payload.manifest === "string" ? payload.manifest : null;
  const logical = typeof payload.logical_bytes === "number" ? payload.logical_bytes : null;
  const temporary = typeof payload.temporary_bytes === "number" ? payload.temporary_bytes : null;
  const eta = typeof payload.eta_seconds === "number" && Number.isFinite(payload.eta_seconds) ? payload.eta_seconds : null;

  const copyDiagnostics = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(event, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section className="surface event-card">
      <div className="surface-title compact">
        <div><span className="section-kicker">TASK DIAGNOSTICS</span><h2>诊断详情</h2></div>
        <button className="quiet-button" type="button" onClick={() => void copyDiagnostics()}>{copied ? "已复制" : "复制 JSON"}</button>
      </div>
      <div className="policy-list">
        <div><span className="policy-dot native" />请求后端：<strong>{requested || "—"}</strong></div>
        <div><span className="policy-dot safe" />实际后端：<strong>{resolved || "—"}</strong></div>
        {fallback ? <div><span className="policy-dot safe" />回退原因：<strong>{fallback}</strong></div> : null}
        {checkpoint ? <div><span className="policy-dot safe" />Checkpoint：<strong>{checkpoint}</strong></div> : null}
        {manifest ? <div><span className="policy-dot trace" />Manifest：<small className="manifest-path">{manifest}</small></div> : null}
        {logical !== null ? <div><span className="policy-dot trace" />逻辑字节：<strong>{formatBytes(logical)}</strong></div> : null}
        {temporary !== null ? <div><span className="policy-dot trace" />临时观测：<strong>{formatBytes(temporary)}</strong></div> : null}
        {eta !== null ? <div><span className="policy-dot trace" />ETA：<strong>{Math.ceil(eta)}s</strong></div> : null}
      </div>
      <p className="surface-description">{eventText(event)}</p>
    </section>
  );
}