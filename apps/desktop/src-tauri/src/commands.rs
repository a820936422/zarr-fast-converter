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

fn terminal_payload(events: Vec<EventEnvelope>) -> Result<Value, AppError> {
    let terminal = events.into_iter().last().ok_or_else(|| {
        AppError::new(ErrorKind::WorkerProtocolError, "worker returned no events")
    })?;
    if terminal.event == "failed" {
        return Err(AppError::new(
            ErrorKind::Unknown,
            serde_json::to_string(&terminal.payload)
                .unwrap_or_else(|_| "worker failed".to_string()),
        ));
    }
    Ok(Value::Object(terminal.payload))
}

#[tauri::command]
pub fn worker_capabilities(state: State<'_, AppState>) -> Result<Value, AppError> {
    let request = request("get_capabilities".to_string(), Map::new(), None);
    let mut worker = ensure_worker(&state)?;
    let process = worker
        .as_mut()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerStartFailed, "worker was not initialized"))?;
    terminal_payload(process.send(&request)?)
}

#[tauri::command]
pub fn inspect_zarr(state: State<'_, AppState>, path: String) -> Result<Value, AppError> {
    if !Path::new(&path).is_dir() {
        return Err(AppError::new(ErrorKind::PathNotFound, path).at_stage("inspection"));
    }
    let mut payload = Map::new();
    payload.insert("path".to_string(), Value::String(path));
    let request = request("inspect_zarr".to_string(), payload, None);
    let mut worker = ensure_worker(&state)?;
    let process = worker
        .as_mut()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerStartFailed, "worker was not initialized"))?;
    terminal_payload(process.send(&request)?)
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
    let mut worker = ensure_worker(&state)?;
    let process = worker
        .as_mut()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerStartFailed, "worker was not initialized"))?;
    terminal_payload(process.send(&request)?)
}

#[tauri::command]
pub fn inspect_source(
    state: State<'_, AppState>,
    payload: Map<String, Value>,
) -> Result<Value, AppError> {
    let request = request("inspect_source".to_string(), payload, None);
    let mut worker = ensure_worker(&state)?;
    let process = worker
        .as_mut()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerStartFailed, "worker was not initialized"))?;
    terminal_payload(process.send(&request)?)
}

#[tauri::command]
pub fn save_inspection_snapshot(
    state: State<'_, AppState>,
    payload: Map<String, Value>,
) -> Result<Value, AppError> {
    let request = request("save_inspection_snapshot".to_string(), payload, None);
    let mut worker = ensure_worker(&state)?;
    let process = worker
        .as_mut()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerStartFailed, "worker was not initialized"))?;
    terminal_payload(process.send(&request)?)
}
