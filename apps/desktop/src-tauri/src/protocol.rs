use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const PROTOCOL_VERSION: u32 = 1;

const COMMANDS: &[&str] = &[
    "get_capabilities",
    "inspect_source",
    "inspect_zarr",
    "inspect_time_metadata",
    "save_inspection_snapshot",
    "preview_pipeline",
    "run_pipeline",
    "resume_pipeline",
    "cancel_task",
    "shutdown",
];

const EVENTS: &[&str] = &[
    "accepted",
    "started",
    "inspection_ready",
    "plan_ready",
    "progress",
    "resource",
    "log",
    "finished",
    "failed",
    "cancelled",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RequestEnvelope {
    pub protocol_version: u32,
    pub request_id: String,
    #[serde(default)]
    pub task_id: Option<String>,
    pub command: String,
    pub payload: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EventEnvelope {
    pub protocol_version: u32,
    pub request_id: String,
    #[serde(default)]
    pub task_id: Option<String>,
    pub sequence: u64,
    pub event: String,
    #[serde(default)]
    pub stage: Option<String>,
    pub payload: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorPayload {
    pub kind: String,
    pub message: String,
    pub retryable: bool,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub details: Map<String, Value>,
    #[serde(default)]
    pub manifest: Option<String>,
}

impl RequestEnvelope {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(format!(
                "unsupported request protocol version: {}",
                self.protocol_version
            ));
        }
        if self.request_id.trim().is_empty() {
            return Err("request_id must be non-empty".to_string());
        }
        if !COMMANDS.contains(&self.command.as_str()) {
            return Err(format!("unsupported command: {}", self.command));
        }
        Ok(())
    }
}

impl EventEnvelope {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(format!(
                "unsupported event protocol version: {}",
                self.protocol_version
            ));
        }
        if self.request_id.trim().is_empty() {
            return Err("event request_id must be non-empty".to_string());
        }
        if !EVENTS.contains(&self.event.as_str()) {
            return Err(format!("unsupported event: {}", self.event));
        }
        Ok(())
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.event.as_str(), "finished" | "failed" | "cancelled")
    }
}

pub fn decode_event(line: &str) -> Result<EventEnvelope, String> {
    let event: EventEnvelope = serde_json::from_str(line)
        .map_err(|error| format!("invalid worker JSON event: {error}"))?;
    event.validate()?;
    Ok(event)
}

pub fn payload_value(payload: &Map<String, Value>, key: &str) -> Result<Value, String> {
    payload
        .get(key)
        .cloned()
        .ok_or_else(|| format!("missing payload field: {key}"))
}

#[cfg(test)]
mod tests {
    use super::{decode_event, RequestEnvelope};
    use serde_json::json;

    #[test]
    fn request_validation_rejects_unknown_command() {
        let request = RequestEnvelope {
            protocol_version: 1,
            request_id: "request".to_string(),
            task_id: None,
            command: "unknown".to_string(),
            payload: Default::default(),
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn event_decoder_identifies_terminal_event() {
        let event = decode_event(
            &json!({
                "protocol_version": 1,
                "request_id": "request",
                "sequence": 1,
                "event": "finished",
                "payload": {}
            })
            .to_string(),
        )
        .expect("valid event");
        assert!(event.is_terminal());
    }
}
