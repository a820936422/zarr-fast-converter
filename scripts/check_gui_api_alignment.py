#!/usr/bin/env python3
"""Read-only GUI/backend interface alignment audit for v1.8.4.

Compares four contract layers between the desktop frontend and the backend:

- commands:  Tauri invoke handlers (lib.rs) vs api.ts invocations vs
             contracts/README.md command list;
- payload fields: keys consumed by the Python worker request parsers
             (desktop_worker/worker.py) vs keys emitted by the frontend
             (api.ts types + App.tsx pipeline payload builder);
- events:    worker/protocol event names vs the frontend task-event type
             and the contract event fixtures;
- capability: the golden capability fixture operations vs operation
             strings referenced by the frontend.

The script is read-only: it prints the delta tables and writes nothing
except an optional JSON report when --report is given.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_TS = ROOT / "apps/desktop/src/api.ts"
APP_TSX = ROOT / "apps/desktop/src/App.tsx"
TASK_EVENTS_TS = ROOT / "apps/desktop/src/taskEvents.ts"
LIB_RS = ROOT / "apps/desktop/src-tauri/src/lib.rs"
WORKER_PY = ROOT / "src/fast_nc_zarr/application/desktop_worker/worker.py"
PROTOCOL_PY = ROOT / "src/fast_nc_zarr/application/desktop_worker/protocol.py"
CONTRACTS_README = ROOT / "contracts/README.md"
EVENT_FIXTURES = ROOT / "contracts/fixtures"
CAPABILITY_FIXTURE = ROOT / "contracts/fixtures/capability-v1.json"
CONTRACTS_DIR = ROOT / "contracts"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def commands_rust(text: str) -> set[str]:
    names: set[str] = set()
    for module in ("commands", "tasks", "native", "pipeline"):
        names.update(re.findall(rf"{module}::(\w+),", text))
    names.add("get_backend_info")
    return names


def commands_ts(text: str) -> set[str]:
    return set(re.findall(r'invoke[^(]*\(\s*"(\w+)"', text))


def commands_contracts(text: str) -> set[str]:
    start = text.find("## Command layers")
    end = text.find("## Event envelope")
    block = text[start:end]
    return set(re.findall(r"^\s{0,4}- `(\w+)`$", block, re.MULTILINE))


def payload_keys_inspection_request(text: str) -> set[str]:
    block = text[text.find("export type InspectionRequest") :]
    block = block[: block.find("};")]
    return set(re.findall(r"^\s{2}(\w+)\??:", block, re.MULTILINE))


def payload_keys_pipeline_builder(text: str) -> set[str]:
    start = text.find("const buildPipelinePayload")
    if start < 0:
        return set()
    ret = text.find("return {", start)
    end = text.find("\n    };", ret)
    if ret < 0 or end < 0:
        return set()
    block = text[ret:end]
    keys = set(
        re.findall(r"^\s{6}([A-Za-z_][A-Za-z0-9_]*):", block, re.MULTILINE)
    )
    keys.update(
        re.findall(r"^\s{6}([A-Za-z_][A-Za-z0-9_]*),$", block, re.MULTILINE)
    )
    return keys


def _keys_between(text: str, start_marker: str, end_marker: str) -> set[str]:
    start = text.find(start_marker)
    if start < 0:
        return set()
    end = text.find(end_marker, start)
    block = text[start:end] if end > 0 else text[start:]
    keys = set(re.findall(r'payload\.get\("([A-Za-z_][A-Za-z0-9_]*)"', block))
    keys.update(re.findall(r'payload\["([A-Za-z_][A-Za-z0-9_]*)"\]', block))
    keys.update(re.findall(r'_(?:path|optional_path)\(payload, "([^"]+)"', block))
    return keys


def payload_keys_worker(text: str) -> set[str]:
    return _keys_between(text, "def _pipeline_config(", "\ndef ")


def event_names_protocol(text: str) -> set[str]:
    start = text.find("EVENTS = frozenset")
    block = text[start : text.find("})", start)]
    return set(re.findall(r'^\s*"(\w+)",?$', block, re.MULTILINE))


def event_names_fixtures(contracts: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted((contracts / "fixtures").glob("event-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict) and item.get("event"):
                names.add(str(item["event"]))
    schema = contracts / "event-v1.schema.json"
    if schema.is_file():
        schema_payload = json.loads(schema.read_text(encoding="utf-8"))
        names.update(schema_payload.get("properties", {}).get("event", {}).get("enum", []))
    return names


def capability_operations() -> set[str]:
    payload = json.loads(CAPABILITY_FIXTURE.read_text(encoding="utf-8"))
    return set(payload.get("operations", []))


def frontend_operation_refs(app_text: str) -> set[str]:
    block = app_text[app_text.find("const OPERATION_LABELS") :]
    block = block[: block.find("};")]
    return set(re.findall(r'"([A-Za-z0-9_.]+)"\s*:', block))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    api = _read(API_TS)
    app = _read(APP_TSX)
    tasks_ts = _read(TASK_EVENTS_TS)
    lib = _read(LIB_RS)
    worker = _read(WORKER_PY)
    protocol = _read(PROTOCOL_PY)
    contracts = _read(CONTRACTS_README)

    rust_commands = commands_rust(lib)
    ts_commands = commands_ts(api)
    contract_commands = commands_contracts(contracts)
    # Python-worker-only JSONL commands that are NOT Tauri commands (and
    # therefore must not be registered in Rust or invoked by the frontend).
    worker_only_commands = {"get_capabilities", "run_pipeline", "shutdown"}

    worker_keys = payload_keys_worker(worker)
    request_keys = payload_keys_inspection_request(api)
    pipeline_keys = payload_keys_pipeline_builder(app)

    event_protocol = event_names_protocol(protocol)
    event_fixtures = event_names_fixtures(CONTRACTS_DIR)
    event_frontend = set(re.findall(r'"((?:accepted|started|inspection_ready|plan_ready|progress|resource|log|finished|failed|cancelled))"', tasks_ts + api))

    capability = capability_operations()
    frontend_ops = frontend_operation_refs(app)

    sections: list[str] = []
    sections.append("## 1. 命令层")
    sections.append(
        "| 集合 | 缺失于前端 | 缺失于 Rust 注册 | 缺失于 contracts 文档 |"
    )
    sections.append("|---|---|---|---|")
    all_commands = rust_commands | ts_commands | contract_commands
    for name in sorted(all_commands):
        if name in worker_only_commands:
            sections.append(f"| `{name}` | worker-only（预期不注册不调用） | | |")
            continue
        sections.append(
            f"| `{name}` | {'❌' if name not in ts_commands else ''} | "
            f"{'❌' if name not in rust_commands else ''} | "
            f"{'❌' if name not in contract_commands else ''} |"
        )

    sections.append("")
    sections.append("## 2. 字段层（payload 键）")
    sections.append("| 键 | 来源 |")
    sections.append("|---|---|")
    all_keys = sorted(worker_keys | request_keys | pipeline_keys)
    for key in all_keys:
        sources = []
        if key in worker_keys:
            sources.append("worker")
        if key in request_keys:
            sources.append("api.InspectionRequest")
        if key in pipeline_keys:
            sources.append("App.buildPipelinePayload")
        sections.append(f"| `{key}` | {'、'.join(sources)} |")
    worker_only = sorted(worker_keys - request_keys - pipeline_keys)
    frontend_only = sorted((request_keys | pipeline_keys) - worker_keys)
    sections.append("")
    sections.append("### 仅 worker 消费（前端未发送）")
    sections.append("、".join(f"`{key}`" for key in worker_only) or "（无）")
    sections.append("")
    sections.append("### 仅前端发送（worker 未消费）")
    sections.append("、".join(f"`{key}`" for key in frontend_only) or "（无）")

    sections.append("")
    sections.append("## 3. 事件层")
    sections.append("| 事件 | protocol | fixtures | 前端类型 | 前端缺失 |")
    sections.append("|---|---|---|---|---|")
    all_events = sorted(event_protocol | event_fixtures | event_frontend)
    for name in all_events:
        missing = "❌" if name not in event_frontend else ""
        sections.append(
            f"| `{name}` | {'✅' if name in event_protocol else '❌'} | "
            f"{'✅' if name in event_fixtures else '❌'} | "
            f"{'✅' if name in event_frontend else '❌'} | {missing} |"
        )

    sections.append("")
    sections.append("## 4. capability 层")
    sections.append("| 操作 | 前端引用 |")
    sections.append("|---|---|")
    for name in sorted(capability | frontend_ops):
        sections.append(f"| `{name}` | {'✅' if name in frontend_ops else '❌（未在前端引用）'} |")
    sections.append("")
    sections.append("## 5. 审计结论（需人工确认）")
    findings = []
    if worker_only:
        findings.append("worker 消费但前端未发送的键：" + "、".join(worker_only))
    if frontend_only:
        findings.append("前端发送但 worker 未消费的键：" + "、".join(frontend_only))
    if not contract_commands.issubset(rust_commands | ts_commands):
        findings.append("contracts 文档命令集与实现不一致")
    sections.extend(f"- {item}" for item in findings or ["（无自动发现项）"])

    report_md = "\n".join(sections)
    print(report_md)
    if args.report is not None:
        args.report.write_text(report_md, encoding="utf-8")
        print(f"\n报告已写入：{args.report.resolve()}")
    if args.check:
        # Only structural consistency is enforced here; content gaps are
        # resolved by humans after reviewing the report.
        issues = bool(
            (contract_commands - rust_commands)
            or (ts_commands - rust_commands)
        )
        return 1 if issues else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())