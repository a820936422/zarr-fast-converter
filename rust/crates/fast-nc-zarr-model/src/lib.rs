use serde::{Deserialize, Serialize};

pub const BACKEND_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapability {
    pub backend: String,
    pub protocol_version: u32,
    pub crate_version: String,
    pub operations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RechunkExecutionPlan {
    pub source: String,
    pub target: String,
    pub array_path: String,
    pub target_chunks: Vec<u64>,
    pub expected_dtype: String,
    pub requested_workers: u32,
    #[serde(default)]
    pub worker_ceiling: u32,
    #[serde(default)]
    pub memory_budget_bytes: u64,
    #[serde(default)]
    pub codec_concurrent_target: u32,
    #[serde(default)]
    pub codec: String,
    #[serde(default)]
    pub codec_level: Option<i32>,
    #[serde(default)]
    pub codec_shuffle: String,
    #[serde(default)]
    pub cancellation_file: Option<String>,
    #[serde(default)]
    pub progress_file: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RechunkVariablePlan {
    pub array_path: String,
    pub expected_dtype: String,
    pub target_chunks: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiRechunkExecutionPlan {
    pub source: String,
    pub target: String,
    pub variables: Vec<RechunkVariablePlan>,
    #[serde(default)]
    pub requested_workers: u32,
    #[serde(default)]
    pub worker_ceiling: u32,
    #[serde(default)]
    pub memory_budget_bytes: u64,
    #[serde(default)]
    pub codec_concurrent_target: u32,
    #[serde(default)]
    pub codec: String,
    #[serde(default)]
    pub codec_level: Option<i32>,
    #[serde(default)]
    pub codec_shuffle: String,
    #[serde(default)]
    pub cancellation_file: Option<String>,
    #[serde(default)]
    pub progress_file: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiRechunkMetrics {
    pub execution_path: String,
    pub output: String,
    pub variables: Vec<RechunkMetrics>,
    pub logical_bytes: u64,
    pub target_chunk_count: u64,
    pub resolved_workers: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RechunkMetrics {
    pub execution_path: String,
    pub output: String,
    pub source_shape: Vec<u64>,
    pub source_chunks: Vec<usize>,
    pub target_chunks: Vec<u64>,
    pub logical_bytes: u64,
    pub target_chunk_count: u64,
    pub resolved_workers: u32,
    pub worker_reason: String,
    pub peak_bytes_per_worker: u64,
    pub memory_budget_bytes: u64,
    pub codec_concurrent_target: u32,
}

impl BackendCapability {
    pub fn smoke() -> Self {
        Self {
            backend: "rust".to_owned(),
            protocol_version: BACKEND_PROTOCOL_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            operations: vec![
                "probe".to_owned(),
                "zarr.inspect".to_owned(),
                "zarr.read_chunk_f32".to_owned(),
                "zarr.read_region_f32".to_owned(),
                "zarr.write_f32".to_owned(),
                "zarr.rechunk_f32".to_owned(),
                "zarr.rechunk_f32_codec".to_owned(),
                "zarr.rechunk_f32_cancel".to_owned(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BackendCapability;

    #[test]
    fn smoke_capability_has_stable_protocol() {
        let capability = BackendCapability::smoke();
        assert_eq!(capability.backend, "rust");
        assert_eq!(capability.operations.len(), 8);
    }
}
