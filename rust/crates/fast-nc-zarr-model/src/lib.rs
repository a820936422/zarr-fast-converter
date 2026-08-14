use serde::{Deserialize, Serialize};

pub const BACKEND_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationCapability {
    pub operation: String,
    pub supported: bool,
    pub reason: Option<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
}

impl OperationCapability {
    fn supported(operation: &str, limitations: &[&str]) -> Self {
        Self {
            operation: operation.to_owned(),
            supported: true,
            reason: None,
            limitations: limitations.iter().map(|item| (*item).to_owned()).collect(),
        }
    }

    fn unsupported(operation: &str, reason: &str, limitations: &[&str]) -> Self {
        Self {
            operation: operation.to_owned(),
            supported: false,
            reason: Some(reason.to_owned()),
            limitations: limitations.iter().map(|item| (*item).to_owned()).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapability {
    pub backend: String,
    pub protocol_version: u32,
    pub crate_version: String,
    pub operations: Vec<String>,
    pub capabilities: Vec<OperationCapability>,
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
    #[serde(default)]
    pub is_coordinate: bool,
    #[serde(default)]
    pub dimension_names: Option<Vec<String>>,
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
        let supported = [
            ("probe", &[][..]),
            ("zarr.inspect", &["Zarr v3 directory store"][..]),
            ("zarr.read_chunk_f32", &["float32 arrays"][..]),
            ("zarr.read_region_f32", &["float32 arrays"][..]),
            ("zarr.write_f32", &["float32 arrays"][..]),
            ("zarr.read_chunk_f64", &["float64 arrays"][..]),
            ("zarr.read_region_f64", &["float64 arrays"][..]),
            ("zarr.write_f64", &["float64 arrays"][..]),
            (
                "zarr.rechunk_f64",
                &["single float64 data variable", "source codec preserved"][..],
            ),
            (
                "zarr.rechunk_f64_cancel",
                &["cooperative cancellation", "source codec preserved"][..],
            ),
            ("zarr.rechunk_f32", &["single float32 data variable"][..]),
            (
                "zarr.rechunk_f32_codec",
                &["single float32 data variable", "explicit lossless codec"][..],
            ),
            ("zarr.rechunk_f32_cancel", &["cooperative cancellation"][..]),
            (
                "zarr.rechunk_multi",
                &[
                    "float32, float64 and standard integer data variables",
                    "fill_value and CF attributes preserved",
                    "source or requested lossless codecs",
                    "atomic staging",
                    "cooperative cancellation",
                ][..],
            ),
            (
                "raw.netcdf.inspect",
                &[
                    "NetCDF-4/classic",
                    "time/lat/lon dimensions",
                    "numeric variables",
                ][..],
            ),
            (
                "raw.netcdf.convert",
                &[
                    "NetCDF-4/classic",
                    "float32/float64 variables",
                    "Zarr v3 output",
                ][..],
            ),
            (
                "resample.nearest",
                &[
                    "float32",
                    "regular latitude/longitude grids",
                    "NaN outside source bounds",
                ][..],
            ),
            (
                "resample.bilinear",
                &[
                    "float32",
                    "regular latitude/longitude grids",
                    "NaN outside source bounds",
                ][..],
            ),
        ];
        let operations = supported
            .iter()
            .map(|(operation, _)| (*operation).to_owned())
            .collect::<Vec<_>>();
        let mut capabilities = supported
            .iter()
            .map(|(operation, limitations)| OperationCapability::supported(operation, limitations))
            .collect::<Vec<_>>();
        capabilities.extend([OperationCapability::unsupported(
            "pipeline.native",
            "native pipeline runtime is not implemented in this phase",
            &["use Python fallback"],
        )]);
        Self {
            backend: "rust".to_owned(),
            protocol_version: BACKEND_PROTOCOL_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            operations,
            capabilities,
        }
    }

    pub fn operation(&self, operation: &str) -> Option<&OperationCapability> {
        self.capabilities
            .iter()
            .find(|item| item.operation == operation)
    }
}

#[cfg(test)]
mod tests {
    use super::BackendCapability;

    #[test]
    fn smoke_capability_has_stable_protocol_and_matrix() {
        let capability = BackendCapability::smoke();
        assert_eq!(capability.backend, "rust");
        assert_eq!(capability.operations.len(), 18);
        assert_eq!(capability.capabilities.len(), 19);
        let supported = capability
            .capabilities
            .iter()
            .filter(|item| item.supported)
            .map(|item| item.operation.clone())
            .collect::<Vec<_>>();
        assert_eq!(supported, capability.operations);
        assert!(
            capability
                .operation("raw.netcdf.convert")
                .unwrap()
                .supported
        );
        assert!(
            capability
                .operation("zarr.write_f64")
                .expect("f64 write")
                .supported
        );
        assert!(
            capability
                .operation("zarr.rechunk_f64")
                .expect("f64 rechunk")
                .supported
        );
        assert!(
            capability
                .operation("zarr.rechunk_multi")
                .expect("multi-variable rechunk")
                .supported
        );
    }
}
