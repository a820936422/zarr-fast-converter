use crate::error::{AppError, ErrorKind};
use crate::protocol::{decode_event, EventEnvelope, RequestEnvelope};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

pub struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    sequences: HashMap<String, u64>,
    terminal_tasks: HashMap<String, bool>,
}
const MAX_WORKER_LINE_BYTES: usize = 1_048_576;

fn trusted_worker(path: &Path) -> bool {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    metadata.is_file() && !metadata.file_type().is_symlink()
}

impl WorkerProcess {
    pub fn spawn() -> Result<Self, AppError> {
        let project_root = env::var_os("FAST_NC_ZARR_PROJECT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."));
        let target = option_env!("TAURI_ENV_TARGET_TRIPLE").unwrap_or("");
        let source_worker = cfg!(debug_assertions)
            && env::var("FAST_NC_ZARR_SOURCE_WORKER").ok().as_deref() == Some("1");
        let mut candidates = Vec::new();
        if cfg!(debug_assertions) && !source_worker {
            if let Some(explicit) = env::var_os("FAST_NC_ZARR_WORKER") {
                candidates.push(PathBuf::from(explicit));
            }
        }
        if !source_worker {
            if let Ok(executable) = env::current_exe() {
                if let Some(directory) = executable.parent() {
                    candidates.push(directory.join("fast-nc-zarr-worker"));
                    if !target.is_empty() {
                        candidates.push(directory.join(format!("fast-nc-zarr-worker-{target}")));
                    }
                }
            }
        }
        if cfg!(debug_assertions) && !source_worker {
            candidates
                .push(project_root.join("apps/desktop/src-tauri/binaries/fast-nc-zarr-worker"));
            if !target.is_empty() {
                candidates.push(project_root.join(format!(
                    "apps/desktop/src-tauri/binaries/fast-nc-zarr-worker-{target}"
                )));
            }
        }
        let sidecar = if source_worker {
            None
        } else {
            candidates
                .into_iter()
                .find(|candidate| trusted_worker(candidate))
        };
        let use_sidecar = sidecar.is_some();
        if !use_sidecar && !cfg!(debug_assertions) {
            return Err(AppError::new(
                ErrorKind::WorkerStartFailed,
                "release worker sidecar is missing or not trusted",
            )
            .at_stage("worker_start"));
        }
        let mut command = sidecar.map(Command::new).unwrap_or_else(|| {
            Command::new(env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        });
        if !use_sidecar {
            let mut python_path = project_root.join("src").into_os_string();
            if let Some(existing) = env::var_os("PYTHONPATH") {
                python_path.push(":");
                python_path.push(existing);
            }
            command
                .env("PYTHONPATH", python_path)
                .args(["-m", "fast_nc_zarr.application.desktop_worker"]);
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|error| {
            AppError::new(ErrorKind::WorkerStartFailed, error.to_string()).at_stage("worker_start")
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            AppError::new(ErrorKind::WorkerStartFailed, "worker stdin unavailable")
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            AppError::new(ErrorKind::WorkerStartFailed, "worker stdout unavailable")
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            AppError::new(ErrorKind::WorkerStartFailed, "worker stderr unavailable")
        })?;
        let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(32)));
        let stderr_tail_writer = stderr_tail.clone();
        let _ = std::thread::Builder::new()
            .name("worker-stderr".to_string())
            .spawn(move || {
                for line in BufReader::new(stderr).lines().flatten() {
                    if let Ok(mut tail) = stderr_tail_writer.lock() {
                        tail.push_back(line);
                        while tail.len() > 32 {
                            tail.pop_front();
                        }
                    }
                }
            });
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr_tail,
            sequences: HashMap::new(),
            terminal_tasks: HashMap::new(),
        })
    }
    pub fn send(&mut self, request: &RequestEnvelope) -> Result<Vec<EventEnvelope>, AppError> {
        let mut events = Vec::new();
        self.send_streaming(request, |event| {
            events.push(event.clone());
            Ok(())
        })?;
        Ok(events)
    }

    pub fn send_streaming<F>(
        &mut self,
        request: &RequestEnvelope,
        mut on_event: F,
    ) -> Result<EventEnvelope, AppError>
    where
        F: FnMut(&EventEnvelope) -> Result<(), AppError>,
    {
        request
            .validate()
            .map_err(|error| AppError::new(ErrorKind::InvalidRequest, error).at_stage("request"))?;
        let line = serde_json::to_string(request)
            .map_err(|error| AppError::new(ErrorKind::WorkerProtocolError, error.to_string()))?;
        writeln!(self.stdin, "{line}")
            .map_err(|error| AppError::new(ErrorKind::WorkerStartFailed, error.to_string()))?;
        self.stdin
            .flush()
            .map_err(|error| AppError::new(ErrorKind::WorkerStartFailed, error.to_string()))?;
        loop {
            let mut line = String::new();
            let count = self.stdout.read_line(&mut line).map_err(|error| {
                AppError::new(ErrorKind::WorkerProtocolError, error.to_string())
            })?;
            if count == 0 {
                let status = self.child.try_wait().map_err(|error| {
                    AppError::new(ErrorKind::WorkerProtocolError, error.to_string())
                })?;
                let diagnostics = self.stderr_diagnostics();
                return Err(AppError::new(
                    ErrorKind::WorkerProtocolError,
                    format!("worker exited before terminal event: {status:?}{diagnostics}"),
                ));
            }
            if line.len() > MAX_WORKER_LINE_BYTES {
                return Err(AppError::new(
                    ErrorKind::WorkerProtocolError,
                    "worker event exceeds protocol byte limit",
                ));
            }
            let event = decode_event(line.trim()).map_err(|error| {
                let diagnostics = self.stderr_diagnostics();
                AppError::new(
                    ErrorKind::WorkerProtocolError,
                    format!("{error}{diagnostics}"),
                )
                .at_stage("worker_event")
            })?;
            if event.request_id != request.request_id {
                return Err(AppError::new(
                    ErrorKind::WorkerProtocolError,
                    "worker response request_id mismatch",
                ));
            }
            self.check_sequence(&event)?;
            let terminal = event.is_terminal();
            on_event(&event)?;
            if terminal {
                return Ok(event);
            }
        }
    }
    fn stderr_diagnostics(&self) -> String {
        let Ok(tail) = self.stderr_tail.lock() else {
            return String::new();
        };
        if tail.is_empty() {
            return String::new();
        }
        format!(
            "; stderr: {}",
            tail.iter().cloned().collect::<Vec<_>>().join(" | ")
        )
    }
    fn check_sequence(&mut self, event: &EventEnvelope) -> Result<(), AppError> {
        if self
            .terminal_tasks
            .get(&event.request_id)
            .copied()
            .unwrap_or(false)
        {
            return Err(AppError::new(
                ErrorKind::WorkerProtocolError,
                "event received after terminal event",
            ));
        }
        if let Some(previous) = self.sequences.get(&event.request_id) {
            if event.sequence <= *previous {
                return Err(AppError::new(
                    ErrorKind::WorkerProtocolError,
                    "worker event sequence is not increasing",
                ));
            }
        }
        self.sequences
            .insert(event.request_id.clone(), event.sequence);
        if event.is_terminal() {
            self.terminal_tasks.insert(event.request_id.clone(), true);
        }
        Ok(())
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub type SharedWorker = Arc<Mutex<Option<WorkerProcess>>>;

pub fn new_shared_worker() -> SharedWorker {
    Arc::new(Mutex::new(None))
}

#[cfg(all(test, unix))]
mod tests {
    use super::WorkerProcess;
    use crate::error::ErrorKind;
    use crate::protocol::RequestEnvelope;
    use std::env;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::{LazyLock, Mutex};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
        LOCK.lock().expect("worker env lock")
    }

    fn test_path(label: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "fast-nc-zarr-worker-{label}-{}",
            std::process::id()
        ))
    }

    fn with_worker_path(path: &PathBuf, action: impl FnOnce() -> Result<(), ErrorKind>) {
        let previous = env::var_os("FAST_NC_ZARR_WORKER");
        env::set_var("FAST_NC_ZARR_WORKER", path);
        let result = action();
        match previous {
            Some(value) => env::set_var("FAST_NC_ZARR_WORKER", value),
            None => env::remove_var("FAST_NC_ZARR_WORKER"),
        }
        result.expect("sidecar failure assertion");
    }

    #[test]
    fn non_executable_worker_is_typed_start_failure() {
        let _guard = env_lock();
        let path = test_path("permission");
        fs::write(&path, "#!/bin/sh\nexit 0\n").expect("write worker fixture");
        with_worker_path(&path, || {
            let error = match WorkerProcess::spawn() {
                Ok(_) => panic!("non-executable worker must fail"),
                Err(error) => error,
            };
            assert_eq!(error.kind, ErrorKind::WorkerStartFailed);
            Ok(())
        });
        let _ = fs::remove_file(path);
    }

    #[test]
    fn worker_exit_before_terminal_event_is_protocol_error() {
        let _guard = env_lock();
        let path = test_path("exit");
        fs::write(&path, "#!/bin/sh\nexit 0\n").expect("write worker fixture");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))
            .expect("chmod worker fixture");
        with_worker_path(&path, || {
            let mut worker = WorkerProcess::spawn().expect("executable worker should spawn");
            let request = RequestEnvelope {
                protocol_version: 1,
                request_id: "worker-exit-test".to_string(),
                task_id: None,
                command: "get_capabilities".to_string(),
                payload: Default::default(),
            };
            let error = worker
                .send(&request)
                .expect_err("early exit must fail protocol");
            assert_eq!(error.kind, ErrorKind::WorkerProtocolError);
            Ok(())
        });
        let _ = fs::remove_file(path);
    }
}
