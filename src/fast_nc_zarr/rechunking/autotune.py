from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Callable, Literal, Mapping

MIB = 1024**2
WorkerTrialStatus = Literal["ok", "failed", "skipped_budget", "skipped_after_failure"]

@dataclass(frozen=True)
class WorkerTrial:
    workers: int
    status: WorkerTrialStatus
    elapsed_seconds: float = 0.0
    logical_bytes: int = 0
    throughput_mib_s: float = 0.0
    peak_rss_bytes: int = 0
    failure: str | None = None
    def to_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class WorkerTuneReport:
    stage: str
    mode: Literal["auto", "explicit", "skipped"]
    safe_ceiling: int
    candidate_workers: tuple[int, ...]
    trials: tuple[WorkerTrial, ...]
    selected_workers: int
    selected_reason: str
    storage_reason: str
    sample_tasks: int
    sample_logical_bytes: int
    budget_seconds: float
    elapsed_seconds: float
    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_workers"] = list(self.candidate_workers)
        payload["candidates"] = [trial.to_dict() for trial in self.trials]
        payload.pop("trials", None)
        return payload

def worker_candidates(safe_ceiling: int) -> tuple[int, ...]:
    ceiling = max(1, int(safe_ceiling))
    values = {1, ceiling}
    value = 2
    while value < ceiling:
        values.add(value)
        value *= 2
    return tuple(sorted(values))

def select_worker_trial(trials: tuple[WorkerTrial, ...]) -> tuple[int, str]:
    successful = [trial for trial in trials if trial.status == "ok"]
    if not successful:
        return 1, "所有实测候选均失败或未执行，保守回退到 1 个 worker"
    best_speed = max(trial.throughput_mib_s for trial in successful)
    if not math.isfinite(best_speed) or best_speed <= 0:
        return 1, "实测吞吐无效，保守回退到 1 个 worker"
    near_best = [trial for trial in successful if trial.throughput_mib_s >= best_speed * 0.90]
    selected = min(near_best, key=lambda trial: trial.workers)
    fastest = max(successful, key=lambda trial: (trial.throughput_mib_s, -trial.workers))
    if selected.workers == 1:
        reason = "1 个 worker 已达到实测最佳吞吐的 90%，避免无收益的并发与内存开销"
    elif selected.workers == fastest.workers:
        reason = f"{selected.workers} 个 worker 在安全候选中实测吞吐最高"
    else:
        reason = f"{selected.workers} 个 worker 已达到最快候选的 90%，选择更小并发以降低 RSS 与文件系统压力"
    return selected.workers, reason

def benchmark_worker_candidates(stage: str, candidates: tuple[int, ...], runner: Callable[[int], Mapping[str, float | int]], *, safe_ceiling: int, storage_reason: str, sample_tasks: int, sample_logical_bytes: int, budget_seconds: float, cancel_event=None) -> WorkerTuneReport:
    ordered = tuple(sorted({max(1, min(int(safe_ceiling), int(candidate))) for candidate in candidates})) or (1,)
    started = time.perf_counter()
    budget = max(0.0, float(budget_seconds))
    trials: list[WorkerTrial] = []
    failure_seen = False
    for index, workers in enumerate(ordered):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("任务已取消。")
        if failure_seen:
            trials.append(WorkerTrial(workers=workers, status="skipped_after_failure", failure="较小并发候选失败，停止继续提高并发"))
            continue
        if index > 0 and time.perf_counter() - started >= budget:
            trials.append(WorkerTrial(workers=workers, status="skipped_budget", failure="达到本阶段实测预算"))
            continue
        trial_started = time.perf_counter()
        try:
            metrics = runner(workers)
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消。")
            elapsed = max(float(metrics.get("elapsed_seconds", time.perf_counter() - trial_started)), 1e-9)
            logical_bytes = max(0, int(metrics.get("logical_bytes", 0)))
            throughput = float(metrics.get("throughput_mib_s", logical_bytes / MIB / elapsed))
            if not math.isfinite(throughput) or throughput <= 0:
                raise RuntimeError("候选未产生有效吞吐")
            trials.append(WorkerTrial(workers=workers, status="ok", elapsed_seconds=elapsed, logical_bytes=logical_bytes, throughput_mib_s=throughput, peak_rss_bytes=max(0, int(metrics.get("peak_rss_bytes", 0)))))
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消。") from exc
            failure_seen = True
            trials.append(WorkerTrial(workers=workers, status="failed", elapsed_seconds=max(0.0, time.perf_counter() - trial_started), failure=f"{type(exc).__name__}: {exc}"[:1000]))
    trial_tuple = tuple(trials)
    selected, selected_reason = select_worker_trial(trial_tuple)
    return WorkerTuneReport(stage=stage, mode="auto", safe_ceiling=max(1, int(safe_ceiling)), candidate_workers=ordered, trials=trial_tuple, selected_workers=selected, selected_reason=selected_reason, storage_reason=storage_reason, sample_tasks=max(0, int(sample_tasks)), sample_logical_bytes=max(0, int(sample_logical_bytes)), budget_seconds=budget, elapsed_seconds=max(0.0, time.perf_counter() - started))

def explicit_worker_report(stage: str, workers: int, *, safe_ceiling: int, storage_reason: str, selected_reason: str) -> WorkerTuneReport:
    selected = max(1, min(int(workers), int(safe_ceiling)))
    return WorkerTuneReport(stage=stage, mode="explicit", safe_ceiling=max(1, int(safe_ceiling)), candidate_workers=(selected,), trials=(), selected_workers=selected, selected_reason=selected_reason, storage_reason=storage_reason, sample_tasks=0, sample_logical_bytes=0, budget_seconds=0.0, elapsed_seconds=0.0)

def skipped_worker_report(stage: str, reason: str) -> WorkerTuneReport:
    return WorkerTuneReport(stage=stage, mode="skipped", safe_ceiling=1, candidate_workers=(), trials=(), selected_workers=1, selected_reason=reason, storage_reason=reason, sample_tasks=0, sample_logical_bytes=0, budget_seconds=0.0, elapsed_seconds=0.0)
