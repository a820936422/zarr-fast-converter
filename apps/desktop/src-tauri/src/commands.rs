use fast_nc_zarr_model::BackendCapability;
use serde_json::{Map, Value};
use std::path::Path;
use tauri::State;

use crate::error::{AppError, ErrorKind};
use crate::protocol::{EventEnvelope, RequestEnvelope};
use crate::worker::{new_shared_worker, SharedWorker, WorkerProcess};

#[derive(Clone)]
pub struct AppState {
    pub worker: SharedWorker,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            worker: new_shared_worker(),
        }
    }
}

fn ensure_worker(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, Option<WorkerProcess>>, AppError> {
    let mut worker = state
        .worker
        .lock()
        .map_err(|_| AppError::new(ErrorKind::WorkerStartFailed, "worker state lock poisoned"))?;
    if worker.is_none() {
        *worker = Some(WorkerProcess::spawn()?);
    }
    Ok(worker)
}

fn request(
    command: String,
    payload: Map<String, Value>,
    task_id: Option<String>,
) -> RequestEnvelope {
    RequestEnvelope {
        protocol_version: 1,
        request_id: uuid::Uuid::new_v4().to_string(),
        task_id,
        command,
        payload,
    }
}

fn error_kind(value: Option<&str>) -> ErrorKind {
    match value {
        Some("invalid_request") => ErrorKind::InvalidRequest,
        Some("path_not_found") => ErrorKind::PathNotFound,
        Some("permission_denied") => ErrorKind::PermissionDenied,
        Some("input_invalid") => ErrorKind::InputInvalid,
        Some("backend_unavailable") => ErrorKind::BackendUnavailable,
        Some("worker_start_failed") => ErrorKind::WorkerStartFailed,
        Some("worker_protocol_error") => ErrorKind::WorkerProtocolError,
        Some("cancelled") => ErrorKind::Cancelled,
        Some("resource_budget_exceeded") => ErrorKind::ResourceBudgetExceeded,
        Some("validation_failed") => ErrorKind::ValidationFailed,
        Some("publication_failed") => ErrorKind::PublicationFailed,
        _ => ErrorKind::Unknown,
    }
}

fn terminal_error(event: &EventEnvelope) -> AppError {
    let wire = event.payload.get("error").and_then(Value::as_object);
    let kind = error_kind(
        wire.and_then(|value| value.get("kind"))
            .and_then(Value::as_str),
    );
    let message = wire
        .and_then(|value| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("worker task failed")
        .to_owned();
    let retryable = wire
        .and_then(|value| value.get("retryable"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let stage = wire
        .and_then(|value| value.get("stage"))
        .and_then(Value::as_str)
        .or(event.stage.as_deref())
        .map(str::to_owned);
    AppError {
        kind,
        message,
        retryable,
        stage,
    }
}

fn terminal_payload(events: Vec<EventEnvelope>) -> Result<Value, AppError> {
    let terminal = events.into_iter().last().ok_or_else(|| {
        AppError::new(ErrorKind::WorkerProtocolError, "worker returned no events")
    })?;
    match terminal.event.as_str() {
        "failed" => Err(terminal_error(&terminal)),
        "cancelled" => Err(AppError::new(
            ErrorKind::Cancelled,
            terminal
                .payload
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("worker task cancelled"),
        )
        .at_stage(terminal.stage.unwrap_or_else(|| "worker".to_string()))),
        _ => Ok(Value::Object(terminal.payload)),
    }
}

fn worker_request(state: &AppState, request: RequestEnvelope) -> Result<Value, AppError> {
    let mut worker = ensure_worker(state)?;
    let process = worker
        .as_mut()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerStartFailed, "worker was not initialized"))?;
    match process.send(&request) {
        Ok(events) => terminal_payload(events),
        Err(error) => {
            if matches!(
                error.kind,
                ErrorKind::WorkerProtocolError | ErrorKind::WorkerStartFailed
            ) {
                *worker = None;
            }
            Err(error)
        }
    }
}

#[tauri::command]
pub fn native_capabilities() -> Result<BackendCapability, AppError> {
    Ok(fast_nc_zarr_model::BackendCapability::desktop())
}

#[tauri::command]
pub fn inspect_zarr(state: State<'_, AppState>, path: String) -> Result<Value, AppError> {
    if !Path::new(&path).is_dir() {
        return Err(AppError::new(ErrorKind::PathNotFound, path).at_stage("inspection"));
    }
    let mut payload = Map::new();
    payload.insert("path".to_string(), Value::String(path));
    let request = request("inspect_zarr".to_string(), payload, None);
    worker_request(&state, request)
}

#[tauri::command]
pub fn inspect_time_metadata(
    state: State<'_, AppState>,
    input_dir: String,
    recursive: bool,
    engine: String,
) -> Result<Value, AppError> {
    if !Path::new(&input_dir).is_dir() {
        return Err(AppError::new(ErrorKind::PathNotFound, input_dir).at_stage("time_inspection"));
    }
    let mut payload = Map::new();
    payload.insert("input_dir".to_string(), Value::String(input_dir));
    payload.insert("recursive".to_string(), Value::Bool(recursive));
    payload.insert("engine".to_string(), Value::String(engine));
    let request = request("inspect_time_metadata".to_string(), payload, None);
    worker_request(&state, request)
}

#[tauri::command]
pub fn inspect_source(
    state: State<'_, AppState>,
    payload: Map<String, Value>,
) -> Result<Value, AppError> {
    let request = request("inspect_source".to_string(), payload, None);
    worker_request(&state, request)
}

#[tauri::command]
pub fn save_inspection_snapshot(
    state: State<'_, AppState>,
    payload: Map<String, Value>,
) -> Result<Value, AppError> {
    let request = request("save_inspection_snapshot".to_string(), payload, None);
    worker_request(&state, request)
}

#[cfg(test)]
mod tests {
    use super::terminal_payload;
    use crate::error::ErrorKind;
    use crate::protocol::EventEnvelope;
    use serde_json::json;

    fn event(name: &str, payload: serde_json::Value) -> EventEnvelope {
        EventEnvelope {
            protocol_version: 1,
            request_id: "request".to_string(),
            task_id: None,
            sequence: 0,
            event: name.to_string(),
            stage: Some("worker".to_string()),
            payload: payload.as_object().cloned().unwrap_or_default(),
        }
    }

    #[test]
    fn worker_failed_event_preserves_wire_error_kind() {
        let error = terminal_payload(vec![event(
            "failed",
            json!({
                "error": {
                    "kind": "resource_budget_exceeded",
                    "message": "too large",
                    "retryable": false,
                    "stage": "resources"
                }
            }),
        )])
        .expect_err("failed worker event must be an error");
        assert_eq!(error.kind, ErrorKind::ResourceBudgetExceeded);
        assert_eq!(error.message, "too large");
        assert_eq!(error.stage.as_deref(), Some("resources"));
    }

    #[test]
    fn worker_cancelled_event_is_not_reported_as_success() {
        let error = terminal_payload(vec![event(
            "cancelled",
            json!({"reason": "cancel requested"}),
        )])
        .expect_err("cancelled worker event must be an error");
        assert_eq!(error.kind, ErrorKind::Cancelled);
        assert_eq!(error.message, "cancel requested");
    }
}
