use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use fast_nc_zarr_model::{MultiRechunkExecutionPlan, RechunkExecutionPlan};
use fast_nc_zarr_zarr::{
    convert_netcdf_to_zarr, inspect_array, inspect_netcdf, rechunk_f32_array, rechunk_multi_array,
    resample_f32, write_f64_array, ResampleF32Request,
};
use serde_json::{Map, Value};
use tauri::{AppHandle, State};

use crate::error::{AppError, ErrorKind};
use crate::protocol::{EventEnvelope, RequestEnvelope};
use crate::resource::snapshot;
use crate::tasks::{
    cancellation_path, emit_task_event, new_request, now_seconds, progress_path, TaskRegistry,
    TaskRequest, TaskStatus, TaskSummary,
};
const NATIVE_OPERATIONS: &[&str] = &[
    "zarr.inspect",
    "raw.netcdf.inspect",
    "raw.netcdf.convert",
    "resample.nearest",
    "resample.bilinear",
    "zarr.rechunk_f32",
    "zarr.rechunk_multi",
    "zarr.write_f64",
];

#[tauri::command]
pub fn start_native_task(
    app: AppHandle,
    registry: State<'_, TaskRegistry>,
    request: TaskRequest,
) -> Result<String, AppError> {
    validate_operation(&request.operation)?;
    let task_id = uuid::Uuid::new_v4().to_string();
    let cancellation_file = cancellation_path(&task_id);
    let progress_file = (matches!(
        request.operation.as_str(),
        "zarr.rechunk_f32" | "zarr.rechunk_multi"
    ))
    .then(|| progress_path(&task_id));
    let mut payload = request.payload;
    payload.insert(
        "operation".to_string(),
        Value::String(request.operation.clone()),
    );
    payload.insert(
        "cancellation_file".to_string(),
        Value::String(cancellation_file.to_string_lossy().into_owned()),
    );
    if let Some(path) = &progress_file {
        payload.insert(
            "progress_file".to_string(),
            Value::String(path.to_string_lossy().into_owned()),
        );
    }
    let envelope = new_request("native_task", payload, Some(task_id.clone()));
    let resource = snapshot();
    registry.insert(TaskSummary {
        task_id: task_id.clone(),
        request_id: envelope.request_id.clone(),
        command: envelope.command.clone(),
        status: TaskStatus::Running,
        manifest: None,
        error: None,
        cancellation_file: Some(cancellation_file.to_string_lossy().into_owned()),
        started_at: now_seconds(),
        resource: Some(resource.clone()),
    })?;

    let registry = registry.inner().clone();
    let operation = request.operation;
    let task_key = task_id.clone();
    let thread_name = format!("native-{operation}-{task_id}");
    thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            run_native_task(
                app,
                registry,
                task_key,
                envelope,
                operation,
                cancellation_file,
                progress_file,
                resource,
            );
        })
        .map_err(|error| AppError::new(ErrorKind::WorkerStartFailed, error.to_string()))?;
    Ok(task_id)
}

fn validate_operation(operation: &str) -> Result<(), AppError> {
    if NATIVE_OPERATIONS.contains(&operation) {
        Ok(())
    } else {
        Err(AppError::new(
            ErrorKind::BackendUnavailable,
            format!("native operation is not available: {operation}"),
        )
        .at_stage("capabilities"))
    }
}

fn make_event(
    request: &RequestEnvelope,
    sequence: &mut u64,
    event: &str,
    stage: &str,
    payload: Map<String, Value>,
) -> EventEnvelope {
    let envelope = EventEnvelope {
        protocol_version: 1,
        request_id: request.request_id.clone(),
        task_id: request.task_id.clone(),
        sequence: *sequence,
        event: event.to_string(),
        stage: Some(stage.to_string()),
        payload,
    };
    *sequence = (*sequence).saturating_add(1);
    envelope
}

fn run_native_task(
    app: AppHandle,
    registry: TaskRegistry,
    task_id: String,
    request: RequestEnvelope,
    operation: String,
    cancellation_file: PathBuf,
    progress_file: Option<PathBuf>,
    resource: crate::resource::ResourceSnapshot,
) {
    let mut sequence = 0_u64;
    let mut emit = |event: &str,
                    stage: &str,
                    payload: Map<String, Value>|
     -> Result<EventEnvelope, AppError> {
        let envelope = make_event(&request, &mut sequence, event, stage, payload);
        emit_task_event(&app, &envelope)?;
        Ok(envelope)
    };

    let _ = emit("accepted", "transport", Map::new());
    let _ = emit("started", &operation, Map::new());
    let _ = registry.update_resource(&task_id, resource.clone());
    let result = match operation.as_str() {
        "zarr.inspect" => native_inspect(&request.payload),
        "raw.netcdf.inspect" => native_netcdf_inspect(&request.payload),
        "raw.netcdf.convert" => native_netcdf_convert(&request.payload),
        "resample.nearest" | "resample.bilinear" => native_resample(&request.payload),
        "zarr.write_f64" => native_write_f64(&request.payload),
        "zarr.rechunk_f32" => native_rechunk(
            &request.payload,
            &cancellation_file,
            progress_file.as_deref(),
            &mut emit,
        ),
        "zarr.rechunk_multi" => native_rechunk_multi(
            &request.payload,
            &cancellation_file,
            progress_file.as_deref(),
            &mut emit,
        ),
        _ => Err(AppError::new(
            ErrorKind::BackendUnavailable,
            format!("native operation is not available: {operation}"),
        )),
    };

    drop(emit);
    let terminal = match result {
        Ok(payload) => make_event(&request, &mut sequence, "finished", "native", payload),
        Err(error) if cancellation_file.is_file() || error.kind == ErrorKind::Cancelled => {
            let mut payload = Map::new();
            payload.insert("reason".to_string(), Value::String(error.message));
            make_event(&request, &mut sequence, "cancelled", "native", payload)
        }
        Err(error) => {
            let mut payload = Map::new();
            payload.insert(
                "error".to_string(),
                serde_json::to_value(error)
                    .unwrap_or_else(|_| Value::String("native task failed".to_string())),
            );
            make_event(&request, &mut sequence, "failed", "native", payload)
        }
    };
    let _ = registry.update_terminal(&task_id, Some(&terminal));
    let _ = emit_task_event(&app, &terminal);
    let _ = fs::remove_file(cancellation_file);
    if let Some(path) = progress_file {
        let _ = fs::remove_file(path);
    }
}

fn native_inspect(payload: &Map<String, Value>) -> Result<Map<String, Value>, AppError> {
    let root = required_string(payload, "path")?;
    let array_path = required_string(payload, "array_path")?;
    if !Path::new(&root).is_dir() {
        return Err(AppError::new(ErrorKind::PathNotFound, root).at_stage("inspection"));
    }
    let summary = inspect_array(&root, &array_path).map_err(|error| {
        AppError::new(ErrorKind::InputInvalid, error.to_string()).at_stage("inspection")
    })?;
    let mut output = Map::new();
    output.insert(
        "operation".to_string(),
        Value::String("zarr.inspect".to_string()),
    );
    output.insert(
        "summary".to_string(),
        serde_json::to_value(summary)
            .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?,
    );
    Ok(output)
}

fn native_netcdf_inspect(payload: &Map<String, Value>) -> Result<Map<String, Value>, AppError> {
    let path = required_string(payload, "path")?;
    if !Path::new(&path).is_file() {
        return Err(AppError::new(ErrorKind::PathNotFound, path).at_stage("inspection"));
    }
    let summary = inspect_netcdf(Path::new(&path))
        .map_err(|error| AppError::new(ErrorKind::InputInvalid, error).at_stage("inspection"))?;
    let mut output = Map::new();
    output.insert(
        "operation".to_string(),
        Value::String("raw.netcdf.inspect".to_string()),
    );
    output.insert(
        "summary".to_string(),
        serde_json::to_value(summary)
            .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?,
    );
    Ok(output)
}

fn native_netcdf_convert(payload: &Map<String, Value>) -> Result<Map<String, Value>, AppError> {
    let input = required_string(payload, "input")?;
    let output = required_string(payload, "output")?;
    if !Path::new(&input).is_file() {
        return Err(AppError::new(ErrorKind::PathNotFound, input).at_stage("conversion"));
    }
    let summary =
        convert_netcdf_to_zarr(Path::new(&input), Path::new(&output)).map_err(|error| {
            AppError::new(ErrorKind::PublicationFailed, error).at_stage("conversion")
        })?;
    let mut result = Map::new();
    result.insert(
        "operation".to_string(),
        Value::String("raw.netcdf.convert".to_string()),
    );
    result.insert(
        "summary".to_string(),
        serde_json::to_value(summary)
            .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?,
    );
    Ok(result)
}

fn native_resample(payload: &Map<String, Value>) -> Result<Map<String, Value>, AppError> {
    let request: ResampleF32Request = serde_json::from_value(Value::Object(payload.clone()))
        .map_err(|error| {
            AppError::new(ErrorKind::InvalidRequest, error.to_string()).at_stage("resampling")
        })?;
    let result = resample_f32(&request)
        .map_err(|error| AppError::new(ErrorKind::InputInvalid, error).at_stage("resampling"))?;
    let mut output = Map::new();
    output.insert(
        "operation".to_string(),
        Value::String(format!("resample.{}", result.method)),
    );
    output.insert(
        "result".to_string(),
        serde_json::to_value(result)
            .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?,
    );
    Ok(output)
}
fn native_write_f64(payload: &Map<String, Value>) -> Result<Map<String, Value>, AppError> {
    let root = required_string(payload, "path")?;
    let array_path = required_string(payload, "array_path")?;
    let shape = required_u64_vec(payload, "shape")?;
    let chunks = required_u64_vec(payload, "chunks")?;
    let values = required_f64_vec(payload, "values")?;
    if Path::new(&root).exists() {
        return Err(AppError::new(
            ErrorKind::PublicationFailed,
            format!("native task refuses an existing target: {root}"),
        )
        .at_stage("native"));
    }
    write_f64_array(&root, &array_path, &shape, &chunks, &values).map_err(|error| {
        AppError::new(ErrorKind::PublicationFailed, error.to_string()).at_stage("native")
    })?;
    let mut output = Map::new();
    output.insert(
        "operation".to_string(),
        Value::String("zarr.write_f64".to_string()),
    );
    output.insert("path".to_string(), Value::String(root));
    output.insert("array_path".to_string(), Value::String(array_path));
    output.insert(
        "shape".to_string(),
        Value::Array(shape.into_iter().map(Value::from).collect()),
    );
    output.insert(
        "chunks".to_string(),
        Value::Array(chunks.into_iter().map(Value::from).collect()),
    );
    Ok(output)
}

fn native_rechunk<F>(
    payload: &Map<String, Value>,
    cancellation_file: &Path,
    progress_file: Option<&Path>,
    emit: &mut F,
) -> Result<Map<String, Value>, AppError>
where
    F: FnMut(&str, &str, Map<String, Value>) -> Result<EventEnvelope, AppError>,
{
    let mut plan: RechunkExecutionPlan = serde_json::from_value(Value::Object(payload.clone()))
        .map_err(|error| {
            AppError::new(ErrorKind::InvalidRequest, error.to_string()).at_stage("native")
        })?;
    let target = PathBuf::from(&plan.target);
    if target.exists() {
        return Err(AppError::new(
            ErrorKind::PublicationFailed,
            format!(
                "native task refuses an existing target: {}",
                target.display()
            ),
        )
        .at_stage("native"));
    }
    plan.cancellation_file = Some(cancellation_file.to_string_lossy().into_owned());
    plan.progress_file = progress_file.map(|path| path.to_string_lossy().into_owned());
    let progress_path = progress_file.map(Path::to_path_buf);
    let join = thread::spawn(move || rechunk_f32_array(&plan));
    let mut last_progress = None;
    while !join.is_finished() {
        if let Some(progress) = read_progress(progress_path.as_deref()) {
            if last_progress != Some(progress) {
                last_progress = Some(progress);
                let mut payload = Map::new();
                payload.insert("completed".to_string(), Value::from(progress.0));
                payload.insert("total".to_string(), Value::from(progress.1));
                let _ = emit("progress", "native", payload);
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let result = join
        .join()
        .map_err(|_| AppError::new(ErrorKind::Unknown, "native task thread panicked"))?;
    if cancellation_file.is_file() || result.is_err() {
        let _ = fs::remove_dir_all(&target);
    }
    let metrics = result
        .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()).at_stage("native"))?;
    if let Some(progress) = read_progress(progress_path.as_deref()) {
        if last_progress != Some(progress) {
            let mut payload = Map::new();
            payload.insert("completed".to_string(), Value::from(progress.0));
            payload.insert("total".to_string(), Value::from(progress.1));
            let _ = emit("progress", "native", payload);
        }
    }
    let mut output = Map::new();
    output.insert(
        "operation".to_string(),
        Value::String("zarr.rechunk_f32".to_string()),
    );
    output.insert(
        "metrics".to_string(),
        serde_json::to_value(metrics)
            .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?,
    );
    Ok(output)
}
fn native_multi_staging_path(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    target.with_file_name(format!(".{name}.native-multi-{}.tmp", uuid::Uuid::new_v4()))
}

fn native_rechunk_multi<F>(
    payload: &Map<String, Value>,
    cancellation_file: &Path,
    progress_file: Option<&Path>,
    emit: &mut F,
) -> Result<Map<String, Value>, AppError>
where
    F: FnMut(&str, &str, Map<String, Value>) -> Result<EventEnvelope, AppError>,
{
    let mut plan: MultiRechunkExecutionPlan =
        serde_json::from_value(Value::Object(payload.clone())).map_err(|error| {
            AppError::new(ErrorKind::InvalidRequest, error.to_string()).at_stage("native")
        })?;
    let target = PathBuf::from(&plan.target);
    if target.exists() {
        return Err(AppError::new(
            ErrorKind::PublicationFailed,
            format!(
                "native task refuses an existing target: {}",
                target.display()
            ),
        )
        .at_stage("native"));
    }
    let staging = native_multi_staging_path(&target);
    plan.target = staging.to_string_lossy().into_owned();
    plan.cancellation_file = Some(cancellation_file.to_string_lossy().into_owned());
    plan.progress_file = progress_file.map(|path| path.to_string_lossy().into_owned());
    let progress_path = progress_file.map(Path::to_path_buf);
    let join = thread::spawn(move || rechunk_multi_array(&plan));
    let mut last_progress = None;
    while !join.is_finished() {
        if let Some(progress) = read_progress(progress_path.as_deref()) {
            if last_progress != Some(progress) {
                last_progress = Some(progress);
                let mut payload = Map::new();
                payload.insert("completed".to_string(), Value::from(progress.0));
                payload.insert("total".to_string(), Value::from(progress.1));
                let _ = emit("progress", "native", payload);
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let result = join
        .join()
        .map_err(|_| AppError::new(ErrorKind::Unknown, "native task thread panicked"))?;
    if cancellation_file.is_file() || result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    if cancellation_file.is_file() {
        return Err(AppError::new(ErrorKind::Cancelled, "任务已取消").at_stage("native"));
    }
    let mut metrics = result
        .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()).at_stage("native"))?;
    if target.exists() {
        let _ = fs::remove_dir_all(&staging);
        return Err(AppError::new(
            ErrorKind::PublicationFailed,
            format!(
                "native target appeared during publish: {}",
                target.display()
            ),
        )
        .at_stage("native"));
    }
    fs::rename(&staging, &target).map_err(|error| {
        let _ = fs::remove_dir_all(&staging);
        AppError::new(ErrorKind::PublicationFailed, error.to_string()).at_stage("native")
    })?;
    metrics.output = target.to_string_lossy().into_owned();
    for variable in &mut metrics.variables {
        variable.output = metrics.output.clone();
    }
    if let Some(progress) = read_progress(progress_path.as_deref()) {
        if last_progress != Some(progress) {
            let mut payload = Map::new();
            payload.insert("completed".to_string(), Value::from(progress.0));
            payload.insert("total".to_string(), Value::from(progress.1));
            let _ = emit("progress", "native", payload);
        }
    }
    let mut output = Map::new();
    output.insert(
        "operation".to_string(),
        Value::String("zarr.rechunk_multi".to_string()),
    );
    output.insert(
        "metrics".to_string(),
        serde_json::to_value(metrics)
            .map_err(|error| AppError::new(ErrorKind::Unknown, error.to_string()))?,
    );
    Ok(output)
}

fn required_string(payload: &Map<String, Value>, key: &str) -> Result<String, AppError> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            AppError::new(
                ErrorKind::InvalidRequest,
                format!("{key} must be a non-empty string"),
            )
        })
}

fn required_u64_vec(payload: &Map<String, Value>, key: &str) -> Result<Vec<u64>, AppError> {
    let values = payload.get(key).and_then(Value::as_array).ok_or_else(|| {
        AppError::new(ErrorKind::InvalidRequest, format!("{key} must be an array"))
    })?;
    let values = values
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                AppError::new(
                    ErrorKind::InvalidRequest,
                    format!("{key} must contain non-negative integers"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() || values.contains(&0) {
        return Err(AppError::new(
            ErrorKind::InvalidRequest,
            format!("{key} must contain positive values"),
        ));
    }
    Ok(values)
}

fn required_f64_vec(payload: &Map<String, Value>, key: &str) -> Result<Vec<f64>, AppError> {
    let values = payload.get(key).and_then(Value::as_array).ok_or_else(|| {
        AppError::new(ErrorKind::InvalidRequest, format!("{key} must be an array"))
    })?;
    values
        .iter()
        .map(|value| {
            value.as_f64().ok_or_else(|| {
                AppError::new(
                    ErrorKind::InvalidRequest,
                    format!("{key} must contain numbers"),
                )
            })
        })
        .collect()
}

fn read_progress(path: Option<&Path>) -> Option<(u64, u64)> {
    let path = path?;
    let contents = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&contents).ok()?;
    Some((
        value.get("completed")?.as_u64()?,
        value.get("total")?.as_u64()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        native_inspect, native_netcdf_inspect, native_rechunk, native_rechunk_multi,
        native_write_f64, validate_operation,
    };
    use crate::protocol::EventEnvelope;
    use fast_nc_zarr_zarr::write_f32_array;
    use serde_json::{json, Map, Value};
    use std::fs;
    use std::path::PathBuf;

    fn store(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "fast-nc-zarr-native-task-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn native_operation_gate_is_explicit() {
        assert!(validate_operation("zarr.inspect").is_ok());
        assert!(validate_operation("raw.netcdf.inspect").is_ok());
        assert!(validate_operation("raw.netcdf.convert").is_ok());
        assert!(validate_operation("resample.nearest").is_ok());
        assert!(validate_operation("resample.bilinear").is_ok());
    }

    #[test]
    fn native_write_f64_publishes_a_valid_zarr_store() {
        let target = store("float64-write.zarr");
        let payload: Map<String, Value> = serde_json::from_value(json!({
            "path": target,
            "array_path": "/value",
            "shape": [2, 2, 2],
            "chunks": [1, 2, 2],
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        }))
        .expect("write payload");
        let result = native_write_f64(&payload).expect("native write");
        assert_eq!(result["operation"], "zarr.write_f64");
        let summary = native_inspect(
            &serde_json::from_value(json!({"path": target, "array_path": "/value"}))
                .expect("inspect payload"),
        )
        .expect("inspect output");
        assert_eq!(summary["summary"]["data_type"], "float64");
        let _ = fs::remove_dir_all(target);
    }

    #[test]
    fn native_inspect_reads_zarr_without_python() {
        let source = store("inspect.zarr");
        write_f32_array(&source, "/value", &[2, 2, 2], &[1, 2, 2], &[0.0; 8])
            .expect("write source");
        let payload: Map<String, Value> = serde_json::from_value(json!({
            "path": source,
            "array_path": "/value"
        }))
        .expect("payload");
        let result = native_inspect(&payload).expect("inspect result");
        assert_eq!(result["summary"]["shape"], json!([2, 2, 2]));
        let _ = fs::remove_dir_all(source);
    }

    #[test]
    fn native_rechunk_emits_progress_and_publishes_metrics() {
        let source = store("source.zarr");
        let target = store("target.zarr");
        let progress = store("progress.json");
        let cancellation = store("cancel");
        write_f32_array(&source, "/value", &[2, 2, 2], &[1, 2, 2], &[0.0; 8])
            .expect("write source");
        let payload: Map<String, Value> = serde_json::from_value(json!({
            "source": source,
            "target": target,
            "array_path": "/value",
            "target_chunks": [2, 1, 2],
            "expected_dtype": "float32",
            "requested_workers": 1,
            "worker_ceiling": 1,
            "memory_budget_bytes": 1048576,
            "codec_concurrent_target": 1,
            "codec": "none",
            "codec_shuffle": "auto"
        }))
        .expect("payload");
        let mut events = Vec::new();
        let result = native_rechunk(
            &payload,
            &cancellation,
            Some(&progress),
            &mut |event, stage, payload| {
                events.push(event.to_string());
                Ok(EventEnvelope {
                    protocol_version: 1,
                    request_id: "request".to_string(),
                    task_id: Some("task".to_string()),
                    sequence: events.len() as u64,
                    event: event.to_string(),
                    stage: Some(stage.to_string()),
                    payload,
                })
            },
        )
        .expect("native rechunk result");
        assert_eq!(result["operation"], "zarr.rechunk_f32");
        assert!(
            result["metrics"]["target_chunk_count"]
                .as_u64()
                .unwrap_or(0)
                > 0
        );
        assert!(target.is_dir());
        assert!(events.iter().any(|event| event == "progress"));
        let _ = fs::remove_dir_all(source);
        let _ = fs::remove_dir_all(target);
        let _ = fs::remove_file(progress);
        let _ = fs::remove_file(cancellation);
    }

    #[test]
    fn native_multi_rechunk_publishes_after_staging() {
        let source = store("multi-source.zarr");
        let target = store("multi-target.zarr");
        let progress = store("multi-progress.json");
        let cancellation = store("multi-cancel");
        write_f32_array(&source, "/value", &[2, 2, 2], &[1, 2, 2], &[0.0; 8])
            .expect("write source");
        let payload: Map<String, Value> = serde_json::from_value(json!({
            "source": source,
            "target": target,
            "variables": [{
                "array_path": "/value",
                "expected_dtype": "float32",
                "target_chunks": [2, 1, 2]
            }],
            "requested_workers": 1,
            "worker_ceiling": 1,
            "memory_budget_bytes": 1048576,
            "codec_concurrent_target": 1,
            "codec": "none",
            "codec_shuffle": "auto"
        }))
        .expect("payload");
        let result = native_rechunk_multi(
            &payload,
            &cancellation,
            Some(&progress),
            &mut |_event, _stage, _payload| {
                Ok(EventEnvelope {
                    protocol_version: 1,
                    request_id: "request".to_string(),
                    task_id: Some("task".to_string()),
                    sequence: 0,
                    event: "progress".to_string(),
                    stage: Some("native".to_string()),
                    payload: Map::new(),
                })
            },
        )
        .expect("native multi rechunk result");
        assert_eq!(result["operation"], "zarr.rechunk_multi");
        assert_eq!(
            result["metrics"]["output"],
            target.to_string_lossy().as_ref()
        );
        assert!(target.is_dir());
        assert!(!target.parent().unwrap().join(".multi-target.zarr").exists());
        let _ = fs::remove_dir_all(source);
        let _ = fs::remove_dir_all(target);
        let _ = fs::remove_file(progress);
        let _ = fs::remove_file(cancellation);
    }

    #[test]
    fn native_rechunk_pre_cancelled_does_not_publish_target() {
        let source = store("cancel-source.zarr");
        let target = store("cancel-target.zarr");
        let cancellation = store("cancel-request");
        write_f32_array(&source, "/value", &[2, 2, 2], &[1, 2, 2], &[0.0; 8])
            .expect("write source");
        fs::write(&cancellation, b"cancel").expect("write cancellation");
        let payload: Map<String, Value> = serde_json::from_value(json!({
            "source": source,
            "target": target,
            "array_path": "/value",
            "target_chunks": [2, 1, 2],
            "expected_dtype": "float32",
            "requested_workers": 1,
            "worker_ceiling": 1,
            "memory_budget_bytes": 1048576,
            "codec_concurrent_target": 1,
            "codec": "none",
            "codec_shuffle": "auto"
        }))
        .expect("payload");
        let result = native_rechunk(
            &payload,
            &cancellation,
            None,
            &mut |_event, _stage, _payload| {
                Ok(EventEnvelope {
                    protocol_version: 1,
                    request_id: "request".to_string(),
                    task_id: Some("task".to_string()),
                    sequence: 0,
                    event: "progress".to_string(),
                    stage: Some("native".to_string()),
                    payload: Map::new(),
                })
            },
        );
        assert!(result.is_err());
        assert!(!target.exists());
        let _ = fs::remove_dir_all(source);
        let _ = fs::remove_dir_all(target);
        let _ = fs::remove_file(cancellation);
    }
}
