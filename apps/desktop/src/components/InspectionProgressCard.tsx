import type { InspectionProgressState, InspectionProgressStatus } from "../lib/types";
import { Icon } from "./Icon";

export function progressStatusLabel(status: InspectionProgressStatus): string {
  return { starting: "正在启动", running: "后端执行中", cancelling: "正在取消", finished: "检查完成", failed: "检查失败", cancelled: "已取消" }[status];
}

export function InspectionProgressCard({
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


