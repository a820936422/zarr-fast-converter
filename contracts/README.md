# v1.8.1 IPC contract

Protocol version: `1`.

Transport boundaries:

- Tauri commands carry JSON-safe request and response objects.
- The Python worker uses JSON Lines over stdin/stdout.
- Human-readable diagnostics go to stderr; stdout is JSONL only.
- Large arrays and task products are addressed by filesystem paths, never embedded
  in an IPC response.
- `manifest.json` uses pipeline schema 7 and is independent of this protocol.

## Request envelope

```json
{
  "protocol_version": 1,
  "request_id": "request-001",
  "task_id": "task-001",
  "command": "inspect_source",
  "payload": {}
}
```

## Command layers

The project has two command surfaces. Tauri invoke commands are implemented by
`apps/desktop/src-tauri/src/lib.rs` and called by `apps/desktop/src/api.ts`:

- `get_backend_info`
- `native_capabilities`
- `inspect_source`
- `inspect_zarr`
- `inspect_time_metadata`
- `save_inspection_snapshot`
- `preview_pipeline`
- `start_pipeline`
- `start_inspection`
- `resume_pipeline`
- `inspect_pipeline_recovery`
- `start_native_task`
- `get_task`
- `list_tasks`
- `clear_task_history`
- `cancel_task`

The Python worker JSONL command set is narrower and is defined by
`fast_nc_zarr.application.desktop_worker.protocol.COMMANDS`:

- `get_capabilities`
- `inspect_source`
- `inspect_zarr`
- `inspect_time_metadata`
- `save_inspection_snapshot`
- `preview_pipeline`
- `run_pipeline`
- `resume_pipeline`
- `shutdown`

The shared request schema also reserves `native_task` and `cancel_task` for
desktop-side task handling. They MUST NOT be forwarded to the Python worker.
`start_native_task` is the public Tauri command that creates a native task;
`cancel_task` writes the cancellation marker consumed by the task registry.

## Event envelope

```json
{
  "protocol_version": 1,
  "request_id": "request-001",
  "task_id": "task-001",
  "sequence": 1,
  "event": "started",
  "stage": "inspection",
  "payload": {}
}
```

Event names:

- `accepted`
- `started`
- `inspection_ready`
- `plan_ready`
- `progress`
- `resource`
- `log`
- `finished`
- `failed`
- `cancelled`

A task has exactly one terminal event: `finished`, `failed`, or `cancelled`.
Sequences are monotonically increasing per task. Unknown payload fields are
forward-compatible; unknown command and event names are protocol errors.

## Error payload

```json
{
  "kind": "worker_protocol_error",
  "message": "invalid JSONL event",
  "retryable": false,
  "stage": "transport",
  "details": {},
  "manifest": null
}
```

Error kinds are `invalid_request`, `path_not_found`, `permission_denied`,
`input_invalid`, `backend_unavailable`, `worker_start_failed`,
`worker_protocol_error`, `cancelled`, `resource_budget_exceeded`,
`validation_failed`, `publication_failed`, and `unknown`.
## Backend capability report

The v1.7.7 native-first migration keeps protocol version `1` and adds a structured
capability report. Its canonical schema is
[`capability-v1.schema.json`](capability-v1.schema.json).

The report contains:

- `operations`: operation IDs currently supported by the reported backend;
- `capabilities`: supported and unsupported operation details;
- `reason`: a stable explanation when an operation is unavailable;
- `limitations`: user-visible constraints and fallback guidance.

`backend=python` always remains selectable. `backend=rust` fails explicitly when
the requested operation is unsupported. `backend=auto` may select Python only
when the report records an explainable fallback reason.

The native golden fixture is [`fixtures/capability-v1.json`](fixtures/capability-v1.json).
