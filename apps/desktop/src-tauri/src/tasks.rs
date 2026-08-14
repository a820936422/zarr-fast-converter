use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tauri::{AppHandle, Emitter, State};

use crate::error::{AppError, ErrorKind};
use crate::protocol::{EventEnvelope, RequestEnvelope};
use crate::resource::ResourceSnapshot;
use crate::worker::{SharedWorker, WorkerProcess};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Running,
    Cancelling,
    Finished,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeTaskRequest {
    pub operation: String,
    #[serde(default)]
    pub payload: Map<String, Value>,
}
pub type TaskRequest = NativeTaskRequest;

pub type TaskEvent = EventEnvelope;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSummary {
    pub task_id: String,
    pub request_id: String,
    pub command: String,
    pub status: TaskStatus,
    pub manifest: Option<String>,
    pub error: Option<Value>,
    pub cancellation_file: Option<String>,
    pub started_at: u64,
    pub resource: Option<ResourceSnapshot>,
}

#[derive(Clone, Default)]
pub struct TaskRegistry {
    tasks: Arc<Mutex<HashMap<String, TaskSummary>>>,
}

impl TaskRegistry {
    pub fn insert(&self, task: TaskSummary) -> Result<(), AppError> {
        self.tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?
            .insert(task.task_id.clone(), task);
        Ok(())
    }

    pub fn get(&self, task_id: &str) -> Result<Option<TaskSummary>, AppError> {
        Ok(self
            .tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?
            .get(task_id)
            .cloned())
    }

    pub fn list(&self) -> Result<Vec<TaskSummary>, AppError> {
        Ok(self
            .tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?
            .values()
            .cloned()
            .collect())
    }

    pub fn cancellation_file(&self, task_id: &str) -> Result<Option<PathBuf>, AppError> {
        Ok(self.get(task_id)?.and_then(|task| {
            if matches!(task.status, TaskStatus::Running | TaskStatus::Cancelling) {
                task.cancellation_file.map(PathBuf::from)
            } else {
                None
            }
        }))
    }

    pub fn mark_cancelling(&self, task_id: &str) -> Result<(), AppError> {
        let mut tasks = self
            .tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?;
        let task = tasks.get_mut(task_id).ok_or_else(|| {
            AppError::new(ErrorKind::PathNotFound, format!("unknown task: {task_id}"))
        })?;
        if task.status == TaskStatus::Running {
            task.status = TaskStatus::Cancelling;
        }
        Ok(())
    }

    pub fn update_resource(
        &self,
        task_id: &str,
        resource: ResourceSnapshot,
    ) -> Result<(), AppError> {
        let mut tasks = self
            .tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?;
        if let Some(task) = tasks.get_mut(task_id) {
            task.resource = Some(resource);
        }
        Ok(())
    }
    pub fn update_failure(&self, task_id: &str, error: AppError) -> Result<(), AppError> {
        let mut tasks = self
            .tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?;
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = TaskStatus::Failed;
            task.error = Some(
                serde_json::to_value(error)
                    .unwrap_or_else(|_| Value::String("worker failed".to_string())),
            );
        }
        Ok(())
    }

    pub fn update_terminal(
        &self,
        task_id: &str,
        event: Option<&TaskEvent>,
    ) -> Result<(), AppError> {
        let mut tasks = self
            .tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?;
        let Some(task) = tasks.get_mut(task_id) else {
            return Ok(());
        };
        match event {
            None => {
                task.status = TaskStatus::Failed;
                task.error = Some(Value::String(
                    "worker returned no terminal event".to_string(),
                ));
            }
            Some(event) => match event.event.as_str() {
                "finished" => {
                    task.status = TaskStatus::Finished;
                    task.manifest = find_manifest(&Value::Object(event.payload.clone()));
                }
                "cancelled" => task.status = TaskStatus::Cancelled,
                "failed" => {
                    task.status = TaskStatus::Failed;
                    task.error = event.payload.get("error").cloned();
                }
                _ => task.status = TaskStatus::Failed,
            },
        }
        if let Some(path) = &task.cancellation_file {
            let _ = fs::remove_file(path);
        }
        Ok(())
    }
}

fn find_manifest(value: &Value) -> Option<String> {
    match value {
        Value::Object(object) => {
            if let Some(Value::String(path)) = object.get("manifest") {
                return Some(path.clone());
            }
            object.values().find_map(find_manifest)
        }
        Value::Array(items) => items.iter().find_map(find_manifest),
        _ => None,
    }
}

pub(crate) fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

pub(crate) fn new_request(
    command: &str,
    payload: Map<String, Value>,
    task_id: Option<String>,
) -> RequestEnvelope {
    RequestEnvelope {
        protocol_version: 1,
        request_id: uuid::Uuid::new_v4().to_string(),
        task_id,
        command: command.to_string(),
        payload,
    }
}

fn ensure_worker(
    shared: &SharedWorker,
) -> Result<std::sync::MutexGuard<'_, Option<WorkerProcess>>, AppError> {
    let mut worker = shared
        .lock()
        .map_err(|_| AppError::new(ErrorKind::WorkerStartFailed, "worker state lock poisoned"))?;
    if worker.is_none() {
        *worker = Some(WorkerProcess::spawn()?);
    }
    Ok(worker)
}

pub(crate) fn send_request(
    shared: &SharedWorker,
    request: &RequestEnvelope,
) -> Result<Vec<TaskEvent>, AppError> {
    let mut worker = ensure_worker(shared)?;
    worker
        .as_mut()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerStartFailed, "worker was not initialized"))?
        .send(request)
}

pub(crate) fn send_dedicated_streaming<F>(
    request: &RequestEnvelope,
    on_event: F,
) -> Result<TaskEvent, AppError>
where
    F: FnMut(&TaskEvent) -> Result<(), AppError>,
{
    let mut worker = WorkerProcess::spawn()?;
    worker.send_streaming(request, on_event)
}

fn terminal_event(events: &[TaskEvent]) -> Result<&TaskEvent, AppError> {
    events
        .last()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerProtocolError, "worker returned no events"))
}

pub(crate) fn payload_or_error(events: Vec<TaskEvent>) -> Result<Value, AppError> {
    let terminal = terminal_event(&events)?;
    if terminal.event == "failed" {
        return Err(AppError::new(
            ErrorKind::Unknown,
            serde_json::to_string(&terminal.payload)
                .unwrap_or_else(|_| "worker failed".to_string()),
        ));
    }
    Ok(Value::Object(terminal.payload.clone()))
}

pub(crate) fn failed_event(
    request: &RequestEnvelope,
    error: &AppError,
    sequence: u64,
) -> TaskEvent {
    let mut payload = Map::new();
    payload.insert(
        "error".to_string(),
        serde_json::to_value(error).unwrap_or_else(|_| Value::String("worker failed".to_string())),
    );
    TaskEvent {
        protocol_version: 1,
        request_id: request.request_id.clone(),
        task_id: request.task_id.clone(),
        sequence,
        event: "failed".to_string(),
        stage: Some(error.stage.clone().unwrap_or_else(|| "worker".to_string())),
        payload,
    }
}

pub(crate) fn emit_task_event(app: &AppHandle, event: &TaskEvent) -> Result<(), AppError> {
    app.emit("task-event", event)
        .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))
}

#[tauri::command]
pub fn get_task(
    task_id: String,
    registry: State<'_, TaskRegistry>,
) -> Result<Option<TaskSummary>, AppError> {
    registry.get(&task_id)
}

pub(crate) fn progress_path(task_id: &str) -> PathBuf {
    std::env::temp_dir()
        .join("fast-nc-zarr-tauri")
        .join(format!("{task_id}.progress.json"))
}
#[tauri::command]
pub fn list_tasks(registry: State<'_, TaskRegistry>) -> Result<Vec<TaskSummary>, AppError> {
    registry.list()
}

#[tauri::command]
pub fn cancel_task(
    task_id: String,
    app: AppHandle,
    registry: State<'_, TaskRegistry>,
) -> Result<(), AppError> {
    let path = registry.cancellation_file(&task_id)?.ok_or_else(|| {
        AppError::new(ErrorKind::InvalidRequest, "task has no cancellation handle")
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, b"cancel")?;
    registry.mark_cancelling(&task_id)?;
    app.emit("task-cancel-requested", &task_id)
        .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?;
    Ok(())
}

pub(crate) fn cancellation_path(task_id: &str) -> PathBuf {
    std::env::temp_dir()
        .join("fast-nc-zarr-tauri")
        .join(format!("{task_id}.cancel"))
}

#[cfg(test)]
mod tests {
    use super::{failed_event, TaskEvent, TaskRegistry, TaskStatus, TaskSummary};
    use crate::error::{AppError, ErrorKind};
    use crate::protocol::RequestEnvelope;
    use crate::resource::ResourceSnapshot;
    use serde_json::json;
    use std::fs;

    fn task(path: &str) -> TaskSummary {
        TaskSummary {
            task_id: "task-1".to_string(),
            request_id: "request-1".to_string(),
            command: "native_task".to_string(),
            status: TaskStatus::Running,
            manifest: None,
            error: None,
            cancellation_file: Some(path.to_string()),
            started_at: 1,
            resource: Some(ResourceSnapshot {
                captured_at_ms: 1,
                logical_cpus: 1,
                memory_total_bytes: 2,
                memory_available_bytes: 1,
            }),
        }
    }

    #[test]
    fn terminal_transition_cleans_cancellation_and_manifest() {
        let path =
            std::env::temp_dir().join(format!("fast-nc-zarr-task-test-{}", std::process::id()));
        fs::write(&path, b"cancel").expect("create cancellation file");
        let registry = TaskRegistry::default();
        registry
            .insert(task(path.to_str().expect("utf8 path")))
            .expect("insert task");
        let event = TaskEvent {
            protocol_version: 1,
            request_id: "request-1".to_string(),
            task_id: Some("task-1".to_string()),
            sequence: 3,
            event: "finished".to_string(),
            stage: Some("native".to_string()),
            payload: serde_json::from_value(json!({"manifest": "/tmp/manifest.json"}))
                .expect("payload object"),
        };
        registry
            .update_terminal("task-1", Some(&event))
            .expect("terminal update");
        let current = registry
            .get("task-1")
            .expect("get task")
            .expect("task exists");
        assert_eq!(current.status, TaskStatus::Finished);
        assert_eq!(current.manifest.as_deref(), Some("/tmp/manifest.json"));
        assert!(!path.exists());
        assert!(registry
            .cancellation_file("task-1")
            .expect("cancel handle")
            .is_none());
    }
    #[test]
    fn failed_event_preserves_sequence_and_wire_error() {
        let request = RequestEnvelope {
            protocol_version: 1,
            request_id: "request-1".to_string(),
            task_id: Some("task-1".to_string()),
            command: "run_pipeline".to_string(),
            payload: Default::default(),
        };
        let error = AppError::new(ErrorKind::WorkerProtocolError, "bad event").at_stage("worker");
        let event = failed_event(&request, &error, 4);
        assert_eq!(event.sequence, 4);
        assert_eq!(event.event, "failed");
        assert_eq!(event.stage.as_deref(), Some("worker"));
        assert_eq!(event.payload["error"]["kind"], "worker_protocol_error");
    }
}
