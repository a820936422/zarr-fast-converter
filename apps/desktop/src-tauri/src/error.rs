use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ErrorKind {
    InvalidRequest,
    PathNotFound,
    PermissionDenied,
    InputInvalid,
    BackendUnavailable,
    WorkerStartFailed,
    WorkerProtocolError,
    Cancelled,
    ResourceBudgetExceeded,
    ValidationFailed,
    PublicationFailed,
    Unknown,
}

impl ErrorKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::PathNotFound => "path_not_found",
            Self::PermissionDenied => "permission_denied",
            Self::InputInvalid => "input_invalid",
            Self::BackendUnavailable => "backend_unavailable",
            Self::WorkerStartFailed => "worker_start_failed",
            Self::WorkerProtocolError => "worker_protocol_error",
            Self::Cancelled => "cancelled",
            Self::ResourceBudgetExceeded => "resource_budget_exceeded",
            Self::ValidationFailed => "validation_failed",
            Self::PublicationFailed => "publication_failed",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppError {
    pub kind: ErrorKind,
    pub message: String,
    pub retryable: bool,
    pub stage: Option<String>,
}

impl AppError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retryable: matches!(kind, ErrorKind::WorkerStartFailed | ErrorKind::Unknown),
            stage: None,
        }
    }

    pub fn at_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for AppError {}

impl serde::Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut output = serializer.serialize_struct("AppError", 4)?;
        output.serialize_field("kind", self.kind.as_str())?;
        output.serialize_field("message", &self.message)?;
        output.serialize_field("retryable", &self.retryable)?;
        output.serialize_field("stage", &self.stage)?;
        output.end()
    }
}

impl From<std::io::Error> for AppError {
    fn from(error: std::io::Error) -> Self {
        let kind = match error.kind() {
            std::io::ErrorKind::NotFound => ErrorKind::PathNotFound,
            std::io::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
            _ => ErrorKind::Unknown,
        };
        Self::new(kind, error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{AppError, ErrorKind};

    #[test]
    fn error_serializes_wire_kind() {
        let error = AppError::new(ErrorKind::PermissionDenied, "cannot read");
        let value = serde_json::to_value(error).expect("serializable");
        assert_eq!(value["kind"], "permission_denied");
    }
}
