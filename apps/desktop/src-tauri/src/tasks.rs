use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tauri::{AppHandle, Emitter, State};

use crate::error::{AppError, ErrorKind};
use crate::protocol::{EventEnvelope, RequestEnvelope};
use crate::resource::ResourceSnapshot;
use crate::worker::{SharedWorker, WorkerProcess};

const TASK_STORE_SCHEMA_VERSION: u32 = 1;
const MAX_ACTIVE_TASKS: usize = 2;

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

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSummary {
    pub task_id: String,
    pub request_id: String,
    pub command: String,
    pub status: TaskStatus,
    #[serde(default)]
    pub manifest: Option<String>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub cancellation_file: Option<String>,
    pub started_at: u64,
    #[serde(default)]
    pub resource: Option<ResourceSnapshot>,
}

#[derive(Serialize, Deserialize)]
struct TaskStore {
    schema_version: u32,
    tasks: Vec<TaskSummary>,
}

#[derive(Clone)]
pub struct TaskRegistry {
    tasks: Arc<Mutex<HashMap<String, TaskSummary>>>,
    state_path: Arc<PathBuf>,
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::with_state_path(task_state_path())
    }
}

impl TaskRegistry {
    fn with_state_path(path: PathBuf) -> Self {
        let (tasks, normalized) = load_tasks(&path);
        if normalized {
            let _ = persist_tasks(&path, &tasks);
        }
        Self {
            tasks: Arc::new(Mutex::new(tasks)),
            state_path: Arc::new(path),
        }
    }

    fn update<F>(&self, update: F) -> Result<(), AppError>
    where
        F: FnOnce(&mut HashMap<String, TaskSummary>) -> Result<(), AppError>,
    {
        let mut tasks = self
            .tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?;
        let mut next = tasks.clone();
        update(&mut next)?;
        persist_tasks(self.state_path.as_ref(), &next)?;
        *tasks = next;
        Ok(())
    }

    pub fn insert(&self, task: TaskSummary) -> Result<(), AppError> {
        self.update(|tasks| {
            let active = tasks
                .values()
                .filter(|item| matches!(item.status, TaskStatus::Running | TaskStatus::Cancelling))
                .count();
            if active >= MAX_ACTIVE_TASKS
                && !tasks.contains_key(&task.task_id)
                && matches!(task.status, TaskStatus::Running | TaskStatus::Cancelling)
            {
                return Err(AppError::new(
                    ErrorKind::ResourceBudgetExceeded,
                    format!("最多同时运行 {MAX_ACTIVE_TASKS} 个桌面任务"),
                )
                .at_stage("resources"));
            }
            tasks.insert(task.task_id.clone(), task);
            Ok(())
        })
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
        let mut tasks: Vec<_> = self
            .tasks
            .lock()
            .map_err(|_| AppError::new(ErrorKind::Unknown, "task registry lock poisoned"))?
            .values()
            .cloned()
            .collect();
        tasks.sort_by(|left, right| {
            right
                .started_at
                .cmp(&left.started_at)
                .then_with(|| left.task_id.cmp(&right.task_id))
        });
        Ok(tasks)
    }

    pub fn cancellation_file(&self, task_id: &str) -> Result<Option<PathBuf>, AppError> {
        if !valid_task_id(task_id) {
            return Ok(None);
        }
        Ok(self.get(task_id)?.and_then(|task| {
            if matches!(task.status, TaskStatus::Running | TaskStatus::Cancelling) {
                Some(cancellation_path(task_id))
            } else {
                None
            }
        }))
    }

    pub fn mark_cancelling(&self, task_id: &str) -> Result<(), AppError> {
        self.update(|tasks| {
            let task = tasks.get_mut(task_id).ok_or_else(|| {
                AppError::new(ErrorKind::PathNotFound, format!("unknown task: {task_id}"))
            })?;
            if task.status == TaskStatus::Running {
                task.status = TaskStatus::Cancelling;
            }
            Ok(())
        })
    }

    pub fn update_resource(
        &self,
        task_id: &str,
        resource: ResourceSnapshot,
    ) -> Result<(), AppError> {
        self.update(|tasks| {
            if let Some(task) = tasks.get_mut(task_id) {
                task.resource = Some(resource);
            }
            Ok(())
        })
    }

    pub fn update_failure(&self, task_id: &str, error: AppError) -> Result<(), AppError> {
        let error = serde_json::to_value(error)
            .unwrap_or_else(|_| Value::String("worker failed".to_string()));
        self.update(|tasks| {
            if let Some(task) = tasks.get_mut(task_id) {
                task.status = TaskStatus::Failed;
                task.error = Some(error);
                task.cancellation_file = None;
            }
            Ok(())
        })
    }

    pub fn update_terminal(
        &self,
        task_id: &str,
        event: Option<&TaskEvent>,
    ) -> Result<(), AppError> {
        let cancellation_file = self
            .get(task_id)?
            .filter(|task| matches!(task.status, TaskStatus::Running | TaskStatus::Cancelling))
            .map(|_| cancellation_path(task_id));
        self.update(|tasks| {
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
            task.cancellation_file = None;
            Ok(())
        })?;
        if let Some(path) = cancellation_file {
            let _ = fs::remove_file(path);
        }
        Ok(())
    }
}

fn load_tasks(path: &Path) -> (HashMap<String, TaskSummary>, bool) {
    let Ok(contents) = fs::read_to_string(path) else {
        return (HashMap::new(), false);
    };
    let Ok(store) = serde_json::from_str::<TaskStore>(&contents) else {
        return (HashMap::new(), false);
    };
    if store.schema_version != TASK_STORE_SCHEMA_VERSION {
        return (HashMap::new(), false);
    }
    let mut normalized = false;
    let mut tasks = HashMap::new();
    for mut task in store.tasks {
        if matches!(task.status, TaskStatus::Running | TaskStatus::Cancelling) {
            let path = cancellation_path(&task.task_id);
            let _ = fs::remove_file(path);
            task.cancellation_file = None;
            task.status = TaskStatus::Failed;
            task.error = Some(application_restarted_error());
            normalized = true;
        }
        tasks.insert(task.task_id.clone(), task);
    }
    (tasks, normalized)
}

fn application_restarted_error() -> Value {
    let mut error = Map::new();
    error.insert(
        "kind".to_string(),
        Value::String("application_restarted".to_string()),
    );
    error.insert(
        "message".to_string(),
        Value::String(
            "桌面应用在任务完成前退出；请从 checkpoint recovery 检查并恢复。".to_string(),
        ),
    );
    error.insert("retryable".to_string(), Value::Bool(true));
    error.insert("stage".to_string(), Value::String("runtime".to_string()));
    Value::Object(error)
}

fn persist_tasks(path: &Path, tasks: &HashMap<String, TaskSummary>) -> Result<(), AppError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut values: Vec<_> = tasks.values().cloned().collect();
    values.sort_by(|left, right| {
        right
            .started_at
            .cmp(&left.started_at)
            .then_with(|| left.task_id.cmp(&right.task_id))
    });
    let contents = serde_json::to_vec_pretty(&TaskStore {
        schema_version: TASK_STORE_SCHEMA_VERSION,
        tasks: values,
    })
    .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("tasks.json");
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", uuid::Uuid::new_v4()));
    fs::write(&temporary, contents)?;
    match fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        Err(_) if path.exists() => {
            fs::remove_file(path)?;
            fs::rename(&temporary, path)?;
            Ok(())
        }
        Err(rename_error) => {
            let _ = fs::remove_file(&temporary);
            Err(rename_error.into())
        }
    }
}

fn explicit_state_path() -> Option<PathBuf> {
    std::env::var_os("FAST_NC_ZARR_TASK_STATE").map(PathBuf::from)
}

#[cfg(target_os = "windows")]
fn task_state_path() -> PathBuf {
    if let Some(path) = explicit_state_path() {
        return path;
    }
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .map(|home| PathBuf::from(home).join("AppData").join("Local"))
        })
        .unwrap_or_else(std::env::temp_dir)
        .join("fast-nc-zarr")
        .join("tasks.json")
}

#[cfg(target_os = "macos")]
fn task_state_path() -> PathBuf {
    if let Some(path) = explicit_state_path() {
        return path;
    }
    std::env::var_os("HOME")
        .map(|home| {
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
        })
        .unwrap_or_else(std::env::temp_dir)
        .join("fast-nc-zarr")
        .join("tasks.json")
}

#[cfg(all(unix, not(target_os = "macos")))]
fn task_state_path() -> PathBuf {
    if let Some(path) = explicit_state_path() {
        return path;
    }
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local").join("state"))
        })
        .unwrap_or_else(std::env::temp_dir)
        .join("fast-nc-zarr")
        .join("tasks.json")
}

#[cfg(not(any(unix, target_os = "windows", target_os = "macos")))]
fn task_state_path() -> PathBuf {
    explicit_state_path()
        .unwrap_or_else(std::env::temp_dir)
        .join("fast-nc-zarr")
        .join("tasks.json")
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
    let process = worker
        .as_mut()
        .ok_or_else(|| AppError::new(ErrorKind::WorkerStartFailed, "worker was not initialized"))?;
    match process.send(request) {
        Ok(events) => Ok(events),
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

fn wire_error_kind(value: Option<&str>) -> ErrorKind {
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

fn wire_error(event: &TaskEvent) -> AppError {
    let wire = event.payload.get("error").and_then(Value::as_object);
    AppError {
        kind: wire_error_kind(
            wire.and_then(|value| value.get("kind"))
                .and_then(Value::as_str),
        ),
        message: wire
            .and_then(|value| value.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("worker task failed")
            .to_owned(),
        retryable: wire
            .and_then(|value| value.get("retryable"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        stage: wire
            .and_then(|value| value.get("stage"))
            .and_then(Value::as_str)
            .or(event.stage.as_deref())
            .map(str::to_owned),
    }
}

pub(crate) fn payload_or_error(events: Vec<TaskEvent>) -> Result<Value, AppError> {
    let terminal = terminal_event(&events)?;
    match terminal.event.as_str() {
        "failed" => Err(wire_error(terminal)),
        "cancelled" => Err(AppError::new(
            ErrorKind::Cancelled,
            terminal
                .payload
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("worker task cancelled"),
        )
        .at_stage(
            terminal
                .stage
                .clone()
                .unwrap_or_else(|| "worker".to_string()),
        )),
        _ => Ok(Value::Object(terminal.payload.clone())),
    }
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
    let mut marker = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .or_else(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                Ok(fs::OpenOptions::new().write(true).open(&path)?)
            } else {
                Err(error)
            }
        })?;
    marker.write_all(b"cancel")?;
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

fn valid_task_id(task_id: &str) -> bool {
    !task_id.is_empty()
        && task_id
            .bytes()
            .all(|value| value.is_ascii_alphanumeric() || value == b'-' || value == b'_')
}

#[cfg(test)]
mod tests {
    use super::{
        cancellation_path, failed_event, TaskEvent, TaskRegistry, TaskStatus, TaskSummary,
    };
    use crate::error::{AppError, ErrorKind};
    use crate::protocol::RequestEnvelope;
    use crate::resource::ResourceSnapshot;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "fast-nc-zarr-task-{label}-{}-{}.json",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    fn task(task_id: &str, path: &str) -> TaskSummary {
        TaskSummary {
            task_id: task_id.to_string(),
            request_id: format!("request-{task_id}"),
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

    fn finished_event(task_id: &str) -> TaskEvent {
        TaskEvent {
            protocol_version: 1,
            request_id: format!("request-{task_id}"),
            task_id: Some(task_id.to_string()),
            sequence: 3,
            event: "finished".to_string(),
            stage: Some("native".to_string()),
            payload: serde_json::from_value(json!({"manifest": "/tmp/manifest.json"}))
                .expect("payload object"),
        }
    }

    #[test]
    fn terminal_transition_cleans_cancellation_and_manifest() {
        let task_id = "terminal-task";
        let state = temp_path("terminal");
        let path = cancellation_path(task_id);
        fs::create_dir_all(path.parent().expect("cancellation parent")).expect("create parent");
        fs::write(&path, b"cancel").expect("create cancellation file");
        let registry = TaskRegistry::with_state_path(state.clone());
        registry
            .insert(task(task_id, path.to_str().expect("utf8 path")))
            .expect("insert task");
        registry
            .update_terminal(task_id, Some(&finished_event(task_id)))
            .expect("terminal update");
        let current = registry
            .get(task_id)
            .expect("get task")
            .expect("task exists");
        assert_eq!(current.status, TaskStatus::Finished);
        assert_eq!(current.manifest.as_deref(), Some("/tmp/manifest.json"));
        assert!(current.cancellation_file.is_none());
        assert!(!path.exists());
        assert!(registry
            .cancellation_file(task_id)
            .expect("cancel handle")
            .is_none());
        let _ = fs::remove_file(state);
    }

    #[test]
    fn terminal_state_survives_registry_restart() {
        let task_id = "restart-finished-task";
        let state = temp_path("restart-finished");
        let path = cancellation_path(task_id);
        fs::create_dir_all(path.parent().expect("cancellation parent")).expect("create parent");
        fs::write(&path, b"cancel").expect("create cancellation file");
        {
            let registry = TaskRegistry::with_state_path(state.clone());
            registry
                .insert(task(task_id, path.to_str().expect("utf8 path")))
                .expect("insert task");
            registry
                .update_terminal(task_id, Some(&finished_event(task_id)))
                .expect("terminal update");
        }
        let restored = TaskRegistry::with_state_path(state.clone());
        let current = restored
            .get(task_id)
            .expect("get restored task")
            .expect("restored task exists");
        assert_eq!(current.status, TaskStatus::Finished);
        assert_eq!(current.manifest.as_deref(), Some("/tmp/manifest.json"));
        let stored: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&state).expect("read task store"))
                .expect("decode task store");
        assert_eq!(stored["schema_version"], 1);
        assert_eq!(stored["tasks"][0]["taskId"], task_id);
        assert!(!path.exists());
        let _ = fs::remove_file(state);
    }

    #[test]
    fn restart_normalizes_active_task_and_removes_cancel_handle() {
        let task_id = "restart-active-task";
        let state = temp_path("restart-active");
        let path = cancellation_path(task_id);
        fs::create_dir_all(path.parent().expect("cancellation parent")).expect("create parent");
        fs::write(&path, b"cancel").expect("create cancellation file");
        {
            let registry = TaskRegistry::with_state_path(state.clone());
            registry
                .insert(task(task_id, path.to_str().expect("utf8 path")))
                .expect("insert task");
        }
        let restored = TaskRegistry::with_state_path(state.clone());
        let current = restored
            .get(task_id)
            .expect("get restored task")
            .expect("restored task exists");
        assert_eq!(current.status, TaskStatus::Failed);
        assert_eq!(
            current.error.as_ref().expect("restart error")["kind"],
            "application_restarted"
        );
        assert!(current.cancellation_file.is_none());
        assert!(!path.exists());
        let _ = fs::remove_file(state);
    }

    #[test]
    fn corrupt_task_store_does_not_block_startup() {
        let state = temp_path("corrupt");
        fs::write(&state, b"not json").expect("write corrupt task store");
        let registry = TaskRegistry::with_state_path(state.clone());
        assert!(registry.list().expect("list tasks").is_empty());
        let _ = fs::remove_file(state);
    }

    #[test]
    fn active_task_quota_rejects_a_third_running_task() {
        let state = temp_path("quota");
        let registry = TaskRegistry::with_state_path(state.clone());
        registry
            .insert(task("quota-one", "/tmp/quota-one.cancel"))
            .expect("first task");
        registry
            .insert(task("quota-two", "/tmp/quota-two.cancel"))
            .expect("second task");
        let error = registry
            .insert(task("quota-three", "/tmp/quota-three.cancel"))
            .expect_err("third active task must be rejected");
        assert_eq!(error.kind, ErrorKind::ResourceBudgetExceeded);
        let _ = fs::remove_file(state);
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
