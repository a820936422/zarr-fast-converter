use serde_json::{Map, Value};
use tauri::{AppHandle, State};

use crate::commands::AppState;
use crate::error::{AppError, ErrorKind};
use crate::resource::snapshot;
use crate::tasks::{
    cancellation_path, emit_task_event, failed_event, new_request, now_seconds, payload_or_error,
    send_dedicated_streaming, send_request, TaskRegistry, TaskStatus, TaskSummary,
};

fn start_task(
    app: AppHandle,
    registry: &TaskRegistry,
    command: &str,
    mut payload: Map<String, Value>,
) -> Result<String, AppError> {
    let task_id = uuid::Uuid::new_v4().to_string();
    let cancellation_file = cancellation_path(&task_id);
    let resource = snapshot();
    payload.insert(
        "cancellation_file".to_string(),
        Value::String(cancellation_file.to_string_lossy().into_owned()),
    );
    payload.insert(
        "resource_snapshot".to_string(),
        serde_json::to_value(&resource)
            .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?,
    );
    let request = new_request(command, payload, Some(task_id.clone()));
    registry.insert(TaskSummary {
        task_id: task_id.clone(),
        request_id: request.request_id.clone(),
        command: request.command.clone(),
        status: TaskStatus::Running,
        manifest: None,
        error: None,
        cancellation_file: Some(cancellation_file.to_string_lossy().into_owned()),
        started_at: now_seconds(),
        resource: Some(resource),
    })?;
    let registry = registry.clone();
    let task_key = task_id.clone();
    std::thread::Builder::new()
        .name(format!("{command}-{task_id}"))
        .spawn(move || {
            let mut last_sequence = 0;
            let mut saw_terminal = false;
            let result = send_dedicated_streaming(&request, |event| {
                last_sequence = event.sequence.saturating_add(1);
                saw_terminal = event.is_terminal();
                if saw_terminal {
                    registry.update_terminal(&task_key, Some(event))?;
                }
                emit_task_event(&app, event)
            });
            if let Err(error) = result {
                if !saw_terminal {
                    let event = failed_event(&request, &error, last_sequence);
                    let _ = registry.update_terminal(&task_key, Some(&event));
                    let _ = emit_task_event(&app, &event);
                } else {
                    let _ = registry.update_failure(&task_key, error);
                }
            }
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
    registry: State<'_, TaskRegistry>,
    payload: Map<String, Value>,
) -> Result<String, AppError> {
    start_task(app, &registry, "run_pipeline", payload)
}

#[tauri::command]
pub fn resume_pipeline(
    app: AppHandle,
    registry: State<'_, TaskRegistry>,
    mut payload: Map<String, Value>,
) -> Result<String, AppError> {
    payload.insert(
        "inspection_kind".to_string(),
        Value::String("temporary".to_string()),
    );
    start_task(app, &registry, "resume_pipeline", payload)
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
