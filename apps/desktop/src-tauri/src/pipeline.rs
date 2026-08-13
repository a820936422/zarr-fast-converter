use serde_json::{Map, Value};
use tauri::{AppHandle, State};

use crate::commands::AppState;
use crate::error::{AppError, ErrorKind};
use crate::tasks::{
    cancellation_path, emit_task_event, new_request, now_seconds, payload_or_error, send_request,
    TaskRegistry, TaskSummary,
};

fn start_task(
    app: AppHandle,
    state: &AppState,
    registry: &TaskRegistry,
    command: &str,
    mut payload: Map<String, Value>,
) -> Result<String, AppError> {
    let task_id = uuid::Uuid::new_v4().to_string();
    let cancellation_file = cancellation_path(&task_id);
    payload.insert(
        "cancellation_file".to_string(),
        Value::String(cancellation_file.to_string_lossy().into_owned()),
    );
    let request = new_request(command, payload, Some(task_id.clone()));
    registry.insert(TaskSummary {
        task_id: task_id.clone(),
        request_id: request.request_id.clone(),
        command: request.command.clone(),
        status: "running".to_string(),
        manifest: None,
        error: None,
        cancellation_file: Some(cancellation_file.to_string_lossy().into_owned()),
        started_at: now_seconds(),
    })?;
    let worker = state.worker.clone();
    let registry = registry.clone();
    let task_key = task_id.clone();
    std::thread::Builder::new()
        .name(format!("{command}-{task_id}"))
        .spawn(move || {
            let events = match send_request(&worker, &request) {
                Ok(events) => events,
                Err(error) => {
                    let _ = registry.update_failure(&task_key, error);
                    return;
                }
            };
            for event in &events {
                let _ = emit_task_event(&app, event);
            }
            let _ = registry.update_terminal(&task_key, events.last());
        })
        .map_err(|error| AppError::new(ErrorKind::WorkerStartFailed, error.to_string()))?;
    Ok(task_id)
}

#[tauri::command]
pub fn preview_pipeline(
    state: State<'_, AppState>,
    payload: Map<String, Value>,
) -> Result<Value, AppError> {
    let request = new_request("preview_pipeline", payload, None);
    payload_or_error(send_request(&state.worker, &request)?)
}

#[tauri::command]
pub fn start_pipeline(
    app: AppHandle,
    state: State<'_, AppState>,
    registry: State<'_, TaskRegistry>,
    payload: Map<String, Value>,
) -> Result<String, AppError> {
    start_task(app, &state, &registry, "run_pipeline", payload)
}

#[tauri::command]
pub fn resume_pipeline(
    app: AppHandle,
    state: State<'_, AppState>,
    registry: State<'_, TaskRegistry>,
    mut payload: Map<String, Value>,
) -> Result<String, AppError> {
    payload.insert(
        "inspection_kind".to_string(),
        Value::String("temporary".to_string()),
    );
    start_task(app, &state, &registry, "resume_pipeline", payload)
}

#[tauri::command]
pub fn inspect_pipeline_recovery(
    state: State<'_, AppState>,
    path: String,
) -> Result<Value, AppError> {
    let payload = Map::from_iter([
        ("path".to_string(), Value::String(path)),
        (
            "input_kind".to_string(),
            Value::String("temporary".to_string()),
        ),
    ]);
    let request = new_request("inspect_zarr", payload, None);
    payload_or_error(send_request(&state.worker, &request)?)
}
