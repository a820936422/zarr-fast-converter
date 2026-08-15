mod resample_native;

pub use resample_native::{resample_f32, ResampleF32Request, ResampleF32Response};

mod netcdf_native;

pub use netcdf_native::{
    convert_netcdf_to_zarr, inspect_netcdf, NetcdfConversionSummary, NetcdfDimensionSummary,
    NetcdfSummary, NetcdfVariableSummary,
};

use fast_nc_zarr_model::{
    MultiRechunkExecutionPlan, MultiRechunkMetrics, RechunkExecutionPlan, RechunkMetrics,
};
use ndarray::ArrayD;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{Map, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use zarrs::array::{
    data_type, Array, ArrayBuilder, ArrayMetadataOptions, ArraySubset, CodecOptions,
};
use zarrs::group::GroupBuilder;
use zarrs_filesystem::FilesystemStore;

#[derive(Debug, Error)]
#[error("{message}")]
pub struct ZarrError {
    message: String,
}

impl ZarrError {
    fn from_display(error: impl std::fmt::Display) -> Self {
        Self {
            message: error.to_string(),
        }
    }

    fn message(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

pub type Result<T, E = ZarrError> = std::result::Result<T, E>;

type Store = FilesystemStore;
type ArrayStore = Array<Store>;

#[derive(Debug, Clone, Serialize)]
pub struct ArraySummary {
    pub path: String,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<usize>,
    pub chunk_grid_shape: Vec<u64>,
    pub dimension_names: Option<Vec<String>>,
    pub attributes: Map<String, Value>,
    pub data_type: Option<Value>,
    pub fill_value: Option<Value>,
    pub codecs: Option<Value>,
    pub metadata: Value,
}

fn normalise_array_path(path: &str) -> String {
    if path.is_empty() {
        "/".to_owned()
    } else if path.starts_with('/') {
        path.to_owned()
    } else {
        format!("/{path}")
    }
}

fn open_array(root: &Path, array_path: &str) -> Result<ArrayStore> {
    if !root.is_dir() {
        return Err(ZarrError::message(format!(
            "Zarr store directory does not exist: {}",
            root.display()
        )));
    }
    array_relative_path(array_path)?;
    let store = Arc::new(FilesystemStore::new(root).map_err(ZarrError::from_display)?);
    Array::open(store, &normalise_array_path(array_path)).map_err(ZarrError::from_display)
}

fn serialised_metadata(array: &ArrayStore) -> Result<Value> {
    serde_json::to_value(array.metadata()).map_err(ZarrError::from_display)
}

fn array_relative_path(array_path: &str) -> Result<PathBuf> {
    let normalised = normalise_array_path(array_path);
    let relative = Path::new(normalised.trim_start_matches('/'));
    if relative.as_os_str().is_empty() {
        return Err(ZarrError::message(
            "array_path must identify an array below the root",
        ));
    }
    for component in relative.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return Err(ZarrError::message(
                "array_path must contain only normal path components",
            ));
        }
    }
    Ok(relative.to_path_buf())
}

fn canonical_path_for_overlap(path: &Path) -> PathBuf {
    let mut missing = Vec::new();
    let mut cursor = path.to_path_buf();
    while !cursor.exists() {
        if let Some(name) = cursor.file_name().map(|value| value.to_os_string()) {
            missing.push(name);
        }
        let Some(parent) = cursor.parent() else {
            break;
        };
        cursor = parent.to_path_buf();
    }
    let mut resolved = cursor.canonicalize().unwrap_or(cursor);
    for name in missing.iter().rev() {
        resolved.push(name);
    }
    resolved
}

fn stores_overlap(source: &Path, target: &Path) -> bool {
    let source = canonical_path_for_overlap(source);
    let target = canonical_path_for_overlap(target);
    source == target || source.starts_with(&target) || target.starts_with(&source)
}

fn cancellation_requested(plan: &RechunkExecutionPlan) -> bool {
    plan.cancellation_file
        .as_deref()
        .is_some_and(|path| Path::new(path).is_file())
}

fn validate_codec_shuffle(value: &str) -> Result<()> {
    if matches!(value, "" | "auto" | "noshuffle" | "shuffle" | "bitshuffle") {
        Ok(())
    } else {
        Err(ZarrError::message(format!(
            "unsupported Rust codec shuffle: {value}"
        )))
    }
}

fn report_progress(
    plan: &RechunkExecutionPlan,
    completed: u64,
    total: u64,
    lock: &std::sync::Mutex<()>,
) -> Result<()> {
    report_progress_with_total(plan, completed, total, lock)
}

fn report_progress_with_total(
    plan: &RechunkExecutionPlan,
    completed: u64,
    total: u64,
    lock: &std::sync::Mutex<()>,
) -> Result<()> {
    let Some(path) = plan.progress_file.as_deref() else {
        return Ok(());
    };
    let _guard = lock
        .lock()
        .map_err(|_| ZarrError::message("progress lock poisoned"))?;
    let payload = serde_json::json!({
        "completed": completed,
        "total": total,
        "fraction": if total == 0 { 1.0 } else { completed as f64 / total as f64 },
    });
    let temporary = format!("{path}.tmp");
    let mut file = fs::File::create(&temporary).map_err(ZarrError::from_display)?;
    file.write_all(payload.to_string().as_bytes())
        .map_err(ZarrError::from_display)?;
    file.sync_all().map_err(ZarrError::from_display)?;
    fs::rename(&temporary, path).map_err(ZarrError::from_display)?;
    Ok(())
}

fn copy_store_tree(
    source_root: &Path,
    source_dir: &Path,
    target_root: &Path,
    skipped_arrays: &[PathBuf],
) -> Result<()> {
    for entry in fs::read_dir(source_dir).map_err(ZarrError::from_display)? {
        let entry = entry.map_err(ZarrError::from_display)?;
        let source = entry.path();
        let relative = source
            .strip_prefix(source_root)
            .map_err(ZarrError::from_display)?;
        if skipped_arrays.iter().any(|path| relative == path) {
            continue;
        }
        let target = target_root.join(relative);
        let file_type = entry.file_type().map_err(ZarrError::from_display)?;
        if file_type.is_symlink() {
            return Err(ZarrError::message(format!(
                "source Zarr contains a symlink: {}",
                source.display()
            )));
        }
        if file_type.is_dir() {
            fs::create_dir_all(&target).map_err(ZarrError::from_display)?;
            copy_store_tree(source_root, &source, target_root, skipped_arrays)?;
        } else if file_type.is_file() {
            fs::copy(&source, &target).map_err(ZarrError::from_display)?;
        } else {
            return Err(ZarrError::message(format!(
                "source Zarr contains an unsupported filesystem entry: {}",
                source.display()
            )));
        }
    }
    Ok(())
}

fn copy_store_without_arrays(
    source_root: &Path,
    target_root: &Path,
    array_paths: &[String],
) -> Result<()> {
    if target_root.exists() {
        return Err(ZarrError::message(format!(
            "refusing to overwrite existing target store: {}",
            target_root.display()
        )));
    }
    fs::create_dir_all(target_root).map_err(ZarrError::from_display)?;
    let skipped_arrays = array_paths
        .iter()
        .map(|path| array_relative_path(path))
        .collect::<Result<Vec<_>>>()?;
    copy_store_tree(source_root, source_root, target_root, &skipped_arrays)
}

fn copy_store_without_array(
    source_root: &Path,
    target_root: &Path,
    array_path: &str,
) -> Result<()> {
    copy_store_without_arrays(source_root, target_root, &[array_path.to_owned()])
}

pub fn inspect_array(root: impl AsRef<Path>, array_path: &str) -> Result<ArraySummary> {
    let root = root.as_ref();
    let array = open_array(root, array_path)?;
    let metadata = serialised_metadata(&array)?;
    let first_chunk = vec![0_u64; array.shape().len()];
    let chunk_shape = array
        .chunk_shape_usize(&first_chunk)
        .map_err(ZarrError::from_display)?;
    let dimension_names = array.dimension_names().as_ref().map(|names| {
        names
            .iter()
            .map(|name| name.clone().unwrap_or_default())
            .collect::<Vec<_>>()
    });
    let attributes = array.attributes().clone();
    let data_type = metadata.get("data_type").cloned();
    let fill_value = metadata.get("fill_value").cloned();
    let codecs = metadata.get("codecs").cloned();
    Ok(ArraySummary {
        path: normalise_array_path(array_path),
        shape: array.shape().to_vec(),
        chunk_shape,
        chunk_grid_shape: array.chunk_grid_shape().to_vec(),
        dimension_names,
        attributes,
        data_type,
        fill_value,
        codecs,
        metadata,
    })
}

pub fn read_chunk_f32(
    root: impl AsRef<Path>,
    array_path: &str,
    chunk_indices: &[u64],
) -> Result<Vec<f32>> {
    let array = open_array(root.as_ref(), array_path)?;
    if chunk_indices.len() != array.shape().len() {
        return Err(ZarrError::message(format!(
            "chunk index dimensionality {} does not match array dimensionality {}",
            chunk_indices.len(),
            array.shape().len()
        )));
    }
    array
        .retrieve_chunk::<Vec<f32>>(chunk_indices)
        .map_err(ZarrError::from_display)
}

pub fn read_region_f32(
    root: impl AsRef<Path>,
    array_path: &str,
    starts: &[u64],
    shape: &[u64],
) -> Result<Vec<f32>> {
    let array = open_array(root.as_ref(), array_path)?;
    if starts.len() != array.shape().len() || shape.len() != array.shape().len() {
        return Err(ZarrError::message(
            "region starts and shape must match array dimensionality",
        ));
    }
    let subset = ArraySubset::new_with_start_shape(starts.to_vec(), shape.to_vec())
        .map_err(ZarrError::from_display)?;
    let values: ArrayD<f32> = array
        .retrieve_array_subset(&subset)
        .map_err(ZarrError::from_display)?;
    Ok(values.into_iter().collect())
}

pub fn read_chunk_f64(
    root: impl AsRef<Path>,
    array_path: &str,
    chunk_indices: &[u64],
) -> Result<Vec<f64>> {
    let array = open_array(root.as_ref(), array_path)?;
    if chunk_indices.len() != array.shape().len() {
        return Err(ZarrError::message(format!(
            "chunk index dimensionality {} does not match array dimensionality {}",
            chunk_indices.len(),
            array.shape().len()
        )));
    }
    array
        .retrieve_chunk::<Vec<f64>>(chunk_indices)
        .map_err(ZarrError::from_display)
}

pub fn read_region_f64(
    root: impl AsRef<Path>,
    array_path: &str,
    starts: &[u64],
    shape: &[u64],
) -> Result<Vec<f64>> {
    let array = open_array(root.as_ref(), array_path)?;
    if starts.len() != array.shape().len() || shape.len() != array.shape().len() {
        return Err(ZarrError::message(
            "region starts and shape must match array dimensionality",
        ));
    }
    let subset = ArraySubset::new_with_start_shape(starts.to_vec(), shape.to_vec())
        .map_err(ZarrError::from_display)?;
    let values: ArrayD<f64> = array
        .retrieve_array_subset(&subset)
        .map_err(ZarrError::from_display)?;
    Ok(values.into_iter().collect())
}

fn chunk_values_f64(
    values: &[f64],
    array_shape: &[u64],
    chunk_shape: &[usize],
    chunk_indices: &[u64],
) -> Result<Vec<f64>> {
    let expected = checked_product(array_shape)?;
    if values.len() != expected {
        return Err(ZarrError::message(format!(
            "values length {} does not match array element count {}",
            values.len(),
            expected
        )));
    }
    let strides = row_major_strides(array_shape)?;
    let mut output = Vec::with_capacity(chunk_shape.iter().product());
    let mut local = vec![0_usize; chunk_shape.len()];
    loop {
        let mut source_offset = 0_usize;
        let mut in_bounds = true;
        for axis in 0..chunk_shape.len() {
            let origin = usize::try_from(chunk_indices[axis])
                .ok()
                .and_then(|index| index.checked_mul(chunk_shape[axis]));
            let Some(origin) = origin else {
                return Err(ZarrError::message("chunk origin overflows usize"));
            };
            let coordinate = origin + local[axis];
            if coordinate >= usize::try_from(array_shape[axis]).unwrap_or(usize::MAX) {
                in_bounds = false;
                break;
            }
            source_offset = source_offset
                .checked_add(
                    coordinate
                        .checked_mul(strides[axis])
                        .ok_or_else(|| ZarrError::message("source offset overflows usize"))?,
                )
                .ok_or_else(|| ZarrError::message("source offset overflows usize"))?;
        }
        output.push(if in_bounds {
            values[source_offset]
        } else {
            f64::NAN
        });
        if !increment_index(&mut local, chunk_shape) {
            break;
        }
    }
    Ok(output)
}

pub fn write_f64_array(
    root: impl AsRef<Path>,
    array_path: &str,
    shape: &[u64],
    chunks: &[u64],
    values: &[f64],
) -> Result<()> {
    if shape.is_empty() || shape.len() != chunks.len() {
        return Err(ZarrError::message(
            "shape and chunks must have the same non-zero dimensionality",
        ));
    }
    if shape
        .iter()
        .zip(chunks)
        .any(|(shape, chunk)| *shape == 0 || *chunk == 0)
    {
        return Err(ZarrError::message("shape and chunks must be positive"));
    }
    let root = root.as_ref();
    array_relative_path(array_path)?;
    if root.exists() {
        if !root.is_dir() {
            return Err(ZarrError::message(format!(
                "output path is not a directory: {}",
                root.display()
            )));
        }
        if fs::read_dir(root)
            .map_err(ZarrError::from_display)?
            .next()
            .is_some()
        {
            return Err(ZarrError::message(format!(
                "refusing to overwrite non-empty Zarr store: {}",
                root.display()
            )));
        }
    } else {
        fs::create_dir_all(root).map_err(ZarrError::from_display)?;
    }

    let store = Arc::new(FilesystemStore::new(root).map_err(ZarrError::from_display)?);
    let group = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(ZarrError::from_display)?;
    group.store_metadata().map_err(ZarrError::from_display)?;

    let mut builder = ArrayBuilder::new(
        shape.to_vec(),
        chunks.to_vec(),
        data_type::float64(),
        f64::NAN,
    );
    let dimension_names = (0..shape.len())
        .map(|axis| format!("dim_{axis}"))
        .collect::<Vec<_>>();
    builder.dimension_names(Some(dimension_names));
    let array_path = normalise_array_path(array_path);
    let array = builder
        .build(store, &array_path)
        .map_err(ZarrError::from_display)?;
    array.store_metadata().map_err(ZarrError::from_display)?;

    let chunk_grid_shape = array.chunk_grid_shape().to_vec();
    let grid_shape = chunk_grid_shape
        .iter()
        .map(|value| {
            usize::try_from(*value).map_err(|_| ZarrError::message("chunk grid exceeds usize"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut chunk_indices = vec![0_usize; grid_shape.len()];
    loop {
        let chunk_indices_u64 = chunk_indices
            .iter()
            .map(|value| {
                u64::try_from(*value).map_err(|_| ZarrError::message("chunk index exceeds u64"))
            })
            .collect::<Result<Vec<_>>>()?;
        let chunk_shape = array
            .chunk_shape_usize(&chunk_indices_u64)
            .map_err(ZarrError::from_display)?;
        let chunk = chunk_values_f64(values, shape, &chunk_shape, &chunk_indices_u64)?;
        array
            .store_chunk(&chunk_indices_u64, chunk.as_slice())
            .map_err(ZarrError::from_display)?;
        if !increment_index(&mut chunk_indices, &grid_shape) {
            break;
        }
    }
    Ok(())
}

fn checked_product(values: &[u64]) -> Result<usize> {
    values.iter().try_fold(1_usize, |total, value| {
        let value = usize::try_from(*value)
            .map_err(|_| ZarrError::message("array shape exceeds this platform's usize"))?;
        total
            .checked_mul(value)
            .ok_or_else(|| ZarrError::message("array shape product overflows usize"))
    })
}

fn row_major_strides(shape: &[u64]) -> Result<Vec<usize>> {
    let mut strides = vec![1_usize; shape.len()];
    for index in (0..shape.len()).rev().skip(1) {
        let next = usize::try_from(shape[index + 1])
            .map_err(|_| ZarrError::message("array shape exceeds this platform's usize"))?;
        strides[index] = strides[index + 1]
            .checked_mul(next)
            .ok_or_else(|| ZarrError::message("array stride overflows usize"))?;
    }
    Ok(strides)
}

fn increment_index(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..index.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return true;
        }
        index[axis] = 0;
    }
    false
}

fn chunk_values(
    values: &[f32],
    array_shape: &[u64],
    chunk_shape: &[usize],
    chunk_indices: &[u64],
) -> Result<Vec<f32>> {
    let expected = checked_product(array_shape)?;
    if values.len() != expected {
        return Err(ZarrError::message(format!(
            "values length {} does not match array element count {}",
            values.len(),
            expected
        )));
    }
    let strides = row_major_strides(array_shape)?;
    let mut output = Vec::with_capacity(chunk_shape.iter().product());
    let mut local = vec![0_usize; chunk_shape.len()];
    loop {
        let mut source_offset = 0_usize;
        let mut in_bounds = true;
        for axis in 0..chunk_shape.len() {
            let origin = usize::try_from(chunk_indices[axis])
                .ok()
                .and_then(|index| index.checked_mul(chunk_shape[axis]));
            let Some(origin) = origin else {
                return Err(ZarrError::message("chunk origin overflows usize"));
            };
            let coordinate = origin + local[axis];
            if coordinate >= usize::try_from(array_shape[axis]).unwrap_or(usize::MAX) {
                in_bounds = false;
                break;
            }
            source_offset = source_offset
                .checked_add(
                    coordinate
                        .checked_mul(strides[axis])
                        .ok_or_else(|| ZarrError::message("source offset overflows usize"))?,
                )
                .ok_or_else(|| ZarrError::message("source offset overflows usize"))?;
        }
        output.push(if in_bounds {
            values[source_offset]
        } else {
            f32::NAN
        });
        if !increment_index(&mut local, chunk_shape) {
            break;
        }
    }
    Ok(output)
}

pub fn write_f32_array(
    root: impl AsRef<Path>,
    array_path: &str,
    shape: &[u64],
    chunks: &[u64],
    values: &[f32],
) -> Result<()> {
    if shape.is_empty() || shape.len() != chunks.len() {
        return Err(ZarrError::message(
            "shape and chunks must have the same non-zero dimensionality",
        ));
    }
    if shape
        .iter()
        .zip(chunks)
        .any(|(shape, chunk)| *shape == 0 || *chunk == 0)
    {
        return Err(ZarrError::message("shape and chunks must be positive"));
    }
    let root = root.as_ref();
    array_relative_path(array_path)?;
    if root.exists() {
        if !root.is_dir() {
            return Err(ZarrError::message(format!(
                "output path is not a directory: {}",
                root.display()
            )));
        }
        if fs::read_dir(root)
            .map_err(ZarrError::from_display)?
            .next()
            .is_some()
        {
            return Err(ZarrError::message(format!(
                "refusing to overwrite non-empty Zarr store: {}",
                root.display()
            )));
        }
    } else {
        fs::create_dir_all(root).map_err(ZarrError::from_display)?;
    }

    let store = Arc::new(FilesystemStore::new(root).map_err(ZarrError::from_display)?);
    let group = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(ZarrError::from_display)?;
    group.store_metadata().map_err(ZarrError::from_display)?;

    let mut builder = ArrayBuilder::new(
        shape.to_vec(),
        chunks.to_vec(),
        data_type::float32(),
        f32::NAN,
    );
    let dimension_names = (0..shape.len())
        .map(|axis| format!("dim_{axis}"))
        .collect::<Vec<_>>();
    builder.dimension_names(Some(dimension_names));
    let array_path = normalise_array_path(array_path);
    let array = builder
        .build(store, &array_path)
        .map_err(ZarrError::from_display)?;
    array.store_metadata().map_err(ZarrError::from_display)?;

    let chunk_grid_shape = array.chunk_grid_shape().to_vec();
    let grid_shape = chunk_grid_shape
        .iter()
        .map(|value| {
            usize::try_from(*value).map_err(|_| ZarrError::message("chunk grid exceeds usize"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut chunk_indices = vec![0_usize; grid_shape.len()];
    loop {
        let chunk_indices_u64 = chunk_indices
            .iter()
            .map(|value| {
                u64::try_from(*value).map_err(|_| ZarrError::message("chunk index exceeds u64"))
            })
            .collect::<Result<Vec<_>>>()?;
        let chunk_shape = array
            .chunk_shape_usize(&chunk_indices_u64)
            .map_err(ZarrError::from_display)?;
        let chunk = chunk_values(values, shape, &chunk_shape, &chunk_indices_u64)?;
        array
            .store_chunk(&chunk_indices_u64, chunk.as_slice())
            .map_err(ZarrError::from_display)?;
        if !increment_index(&mut chunk_indices, &grid_shape) {
            break;
        }
    }
    Ok(())
}

pub fn rechunk_f32_array(plan: &RechunkExecutionPlan) -> Result<RechunkMetrics> {
    array_relative_path(&plan.array_path)?;
    validate_codec_shuffle(&plan.codec_shuffle)?;
    if plan.expected_dtype != "float32" {
        return Err(ZarrError::message(format!(
            "P3 Rust rechunk backend only supports expected_dtype=float32, got {}",
            plan.expected_dtype
        )));
    }
    if plan.target_chunks.is_empty() || plan.target_chunks.contains(&0) {
        return Err(ZarrError::message(
            "target_chunks must contain positive values",
        ));
    }
    if !matches!(
        plan.codec.as_str(),
        "" | "none" | "zstd" | "blosc-zstd" | "blosc-lz4" | "blosc-lz4hc" | "blosc-zlib" | "gzip"
    ) {
        return Err(ZarrError::message(format!(
            "unsupported Rust codec: {}",
            plan.codec
        )));
    }
    if cancellation_requested(plan) {
        return Err(ZarrError::message("任务已取消"));
    }

    let source_root = Path::new(&plan.source);
    let source_array = open_array(source_root, &plan.array_path)?;
    let source_metadata = serialised_metadata(&source_array)?;
    if source_metadata.get("data_type") != Some(&Value::String("float32".to_owned())) {
        return Err(ZarrError::message(
            "P3 Rust rechunk backend only supports source data_type=float32",
        ));
    }
    if plan.target_chunks.len() != source_array.shape().len() {
        return Err(ZarrError::message(
            "target_chunks dimensionality does not match source array",
        ));
    }

    let source_shape = source_array.shape().to_vec();
    let source_chunk_shape = source_array
        .chunk_shape_usize(&vec![0_u64; source_shape.len()])
        .map_err(ZarrError::from_display)?;
    let source_chunk_elements = source_chunk_shape.iter().try_fold(1_u64, |total, value| {
        total
            .checked_mul(
                u64::try_from(*value)
                    .map_err(|_| ZarrError::message("source chunk shape exceeds u64"))?,
            )
            .ok_or_else(|| ZarrError::message("source chunk element count overflows u64"))
    })?;
    let target_chunk_elements = checked_product(&plan.target_chunks)? as u64;
    let peak_bytes_per_worker = source_chunk_elements
        .checked_add(target_chunk_elements)
        .and_then(|elements| elements.checked_mul(8))
        .ok_or_else(|| ZarrError::message("per-worker memory estimate overflows u64"))?;

    let requested_workers = u64::from(plan.requested_workers.max(1));
    let worker_ceiling = if plan.worker_ceiling == 0 {
        requested_workers
    } else {
        u64::from(plan.worker_ceiling).min(requested_workers)
    };
    let memory_limited_workers = if plan.memory_budget_bytes == 0 {
        worker_ceiling
    } else {
        (plan.memory_budget_bytes / peak_bytes_per_worker.max(1)).max(1)
    };

    let target_root = Path::new(&plan.target);
    if stores_overlap(source_root, target_root) {
        return Err(ZarrError::message(
            "source and target Zarr stores cannot overlap or nest",
        ));
    }
    copy_store_without_array(source_root, target_root, &plan.array_path)?;
    let target_store =
        Arc::new(FilesystemStore::new(target_root).map_err(ZarrError::from_display)?);

    let mut array_builder = ArrayBuilder::from_array(&source_array);
    array_builder.chunk_grid_metadata(plan.target_chunks.clone());
    if !matches!(plan.codec.as_str(), "" | "none") {
        let level = plan.codec_level.unwrap_or(1);
        let codecs: Vec<std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>> = match plan
            .codec
            .as_str()
        {
            "zstd" | "blosc-zstd" | "blosc-lz4" | "blosc-lz4hc" | "blosc-zlib" => {
                let codec = match plan.codec.as_str() {
                    "zstd" => {
                        std::sync::Arc::new(zarrs::array::codec::ZstdCodec::new(level, false))
                            as std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>
                    }
                    "blosc-zstd" | "blosc-lz4" | "blosc-lz4hc" | "blosc-zlib" => {
                        let compressor = match plan.codec.as_str() {
                            "blosc-lz4" => zarrs::array::codec::BloscCompressor::LZ4,
                            "blosc-lz4hc" => zarrs::array::codec::BloscCompressor::LZ4HC,
                            "blosc-zlib" => zarrs::array::codec::BloscCompressor::Zlib,
                            _ => zarrs::array::codec::BloscCompressor::Zstd,
                        };
                        let shuffle = match plan.codec_shuffle.as_str() {
                            "bitshuffle" => zarrs::array::codec::BloscShuffleMode::BitShuffle,
                            "shuffle" | "auto" => zarrs::array::codec::BloscShuffleMode::Shuffle,
                            "noshuffle" | "" => zarrs::array::codec::BloscShuffleMode::NoShuffle,
                            _ => {
                                return Err(ZarrError::message(format!(
                                    "unsupported Rust codec shuffle: {}",
                                    plan.codec_shuffle
                                )))
                            }
                        };
                        std::sync::Arc::new(
                            zarrs::array::codec::BloscCodec::new(
                                compressor,
                                u8::try_from(level)
                                    .map_err(|_| ZarrError::message("Blosc level must be 0-9"))?
                                    .try_into()
                                    .map_err(|_| ZarrError::message("Blosc level must be 0-9"))?,
                                None,
                                shuffle,
                                Some(4),
                            )
                            .map_err(ZarrError::from_display)?,
                        )
                            as std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>
                    }
                    _ => unreachable!(),
                };
                vec![codec]
            }
            "gzip" => vec![std::sync::Arc::new(
                zarrs::array::codec::GzipCodec::new(level.max(0) as u32)
                    .map_err(ZarrError::from_display)?,
            )
                as std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>],
            _ => unreachable!(),
        };
        array_builder.bytes_to_bytes_codecs(codecs);
    }
    let target_path = normalise_array_path(&plan.array_path);
    let target_array = array_builder
        .build(target_store, &target_path)
        .map_err(ZarrError::from_display)?;
    target_array
        .store_metadata_opt(&ArrayMetadataOptions::default().with_include_zarrs_metadata(false))
        .map_err(ZarrError::from_display)?;

    let target_grid_shape = target_array.chunk_grid_shape().to_vec();
    let target_grid_count = target_grid_shape.iter().try_fold(1_u64, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| ZarrError::message("target chunk count overflows u64"))
    })?;
    let progress_lock = std::sync::Mutex::new(());
    let resolved_workers = worker_ceiling
        .min(memory_limited_workers)
        .min(target_grid_count)
        .max(1);
    let resolved_workers = usize::try_from(resolved_workers)
        .map_err(|_| ZarrError::message("resolved worker count exceeds usize"))?;
    let codec_concurrent_target = if plan.codec_concurrent_target == 0 {
        1
    } else {
        usize::try_from(u64::from(plan.codec_concurrent_target).min(resolved_workers as u64))
            .map_err(|_| ZarrError::message("codec worker count exceeds usize"))?
            .max(1)
    };
    let codec_options = CodecOptions::default()
        .with_concurrent_target(codec_concurrent_target)
        .with_chunk_concurrent_minimum(1);
    let source_array = source_array.with_codec_options(codec_options);
    let target_array = target_array.with_codec_options(codec_options);
    let target_indices = ArraySubset::new_with_shape(target_grid_shape.clone()).indices();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(resolved_workers)
        .build()
        .map_err(ZarrError::from_display)?;
    let completed = std::sync::atomic::AtomicU64::new(0);
    pool.install(|| {
        target_indices
            .into_par_iter()
            .try_for_each(|chunk_indices| -> Result<()> {
                if cancellation_requested(plan) {
                    return Err(ZarrError::message("任务已取消"));
                }
                let subset = target_array
                    .chunk_subset_bounded(&chunk_indices)
                    .map_err(ZarrError::from_display)?;
                let values: ArrayD<f32> = source_array
                    .retrieve_array_subset_opt(&subset, &codec_options)
                    .map_err(ZarrError::from_display)?;
                target_array
                    .store_array_subset_opt(&subset, values, &codec_options)
                    .map_err(ZarrError::from_display)?;
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                report_progress(plan, done, target_grid_count, &progress_lock)
            })
    })?;
    if cancellation_requested(plan) {
        return Err(ZarrError::message("任务已取消"));
    }
    let logical_elements = checked_product(&source_shape)? as u64;
    let logical_bytes = logical_elements
        .checked_mul(4)
        .ok_or_else(|| ZarrError::message("logical byte count overflows u64"))?;
    Ok(RechunkMetrics {
        execution_path: if matches!(plan.codec.as_str(), "" | "none") {
            "rust-streaming-target-chunk".to_owned()
        } else {
            "rust-codec-target-chunk".to_owned()
        },
        output: plan.target.clone(),
        source_shape,
        source_chunks: source_chunk_shape,
        target_chunks: plan.target_chunks.clone(),
        logical_bytes,
        target_chunk_count: target_grid_count,
        resolved_workers: u32::try_from(resolved_workers)
            .map_err(|_| ZarrError::message("resolved worker count exceeds u32"))?,
        worker_reason: format!(
            "P3 bounded target-chunk pool; requested_workers={}, ceiling={}, memory_limited_workers={}, peak_bytes_per_worker={}, codec_concurrent_target={}",
            plan.requested_workers.max(1), plan.worker_ceiling, memory_limited_workers,
            peak_bytes_per_worker, codec_concurrent_target,
        ),
        peak_bytes_per_worker,
        memory_budget_bytes: plan.memory_budget_bytes,
        codec_concurrent_target: u32::try_from(codec_concurrent_target)
            .map_err(|_| ZarrError::message("codec worker count exceeds u32"))?,
    })
}

pub fn rechunk_f64_array(plan: &RechunkExecutionPlan) -> Result<RechunkMetrics> {
    array_relative_path(&plan.array_path)?;
    validate_codec_shuffle(&plan.codec_shuffle)?;
    if plan.expected_dtype != "float64" {
        return Err(ZarrError::message(format!(
            "P3 Rust float64 rechunk requires expected_dtype=float64, got {}",
            plan.expected_dtype
        )));
    }
    if !matches!(plan.codec.as_str(), "" | "none") {
        return Err(ZarrError::message(
            "P3 Rust float64 rechunk currently preserves the source codec and does not apply a new codec",
        ));
    }
    if plan.target_chunks.is_empty() || plan.target_chunks.contains(&0) {
        return Err(ZarrError::message(
            "target_chunks must contain positive values",
        ));
    }
    if cancellation_requested(plan) {
        return Err(ZarrError::message("任务已取消"));
    }

    let source_root = Path::new(&plan.source);
    let source_array = open_array(source_root, &plan.array_path)?;
    let source_metadata = serialised_metadata(&source_array)?;
    if source_metadata.get("data_type") != Some(&Value::String("float64".to_owned())) {
        return Err(ZarrError::message(
            "P3 Rust float64 rechunk only supports source data_type=float64",
        ));
    }
    if plan.target_chunks.len() != source_array.shape().len() {
        return Err(ZarrError::message(
            "target_chunks dimensionality does not match source array",
        ));
    }

    let source_shape = source_array.shape().to_vec();
    let source_chunk_shape = source_array
        .chunk_shape_usize(&vec![0_u64; source_shape.len()])
        .map_err(ZarrError::from_display)?;
    let source_chunk_elements = source_chunk_shape.iter().try_fold(1_u64, |total, value| {
        total
            .checked_mul(
                u64::try_from(*value)
                    .map_err(|_| ZarrError::message("source chunk shape exceeds u64"))?,
            )
            .ok_or_else(|| ZarrError::message("source chunk element count overflows u64"))
    })?;
    let target_chunk_elements = checked_product(&plan.target_chunks)? as u64;
    let peak_bytes_per_worker = source_chunk_elements
        .checked_add(target_chunk_elements)
        .and_then(|elements| elements.checked_mul(16))
        .ok_or_else(|| ZarrError::message("per-worker memory estimate overflows u64"))?;

    let requested_workers = u64::from(plan.requested_workers.max(1));
    let worker_ceiling = if plan.worker_ceiling == 0 {
        requested_workers
    } else {
        u64::from(plan.worker_ceiling).min(requested_workers)
    };
    let memory_limited_workers = if plan.memory_budget_bytes == 0 {
        worker_ceiling
    } else {
        (plan.memory_budget_bytes / peak_bytes_per_worker.max(1)).max(1)
    };
    let target_root = Path::new(&plan.target);
    if stores_overlap(source_root, target_root) {
        return Err(ZarrError::message(
            "source and target Zarr stores cannot overlap or nest",
        ));
    }
    if target_root.exists() {
        return Err(ZarrError::message(format!(
            "refusing to overwrite existing target store: {}",
            target_root.display()
        )));
    }
    copy_store_without_array(source_root, target_root, &plan.array_path)?;
    let target_store =
        Arc::new(FilesystemStore::new(target_root).map_err(ZarrError::from_display)?);
    let mut array_builder = ArrayBuilder::from_array(&source_array);
    array_builder.chunk_grid_metadata(plan.target_chunks.clone());
    let target_path = normalise_array_path(&plan.array_path);
    let target_array = array_builder
        .build(target_store, &target_path)
        .map_err(ZarrError::from_display)?;
    target_array
        .store_metadata_opt(&ArrayMetadataOptions::default().with_include_zarrs_metadata(false))
        .map_err(ZarrError::from_display)?;

    let target_grid_shape = target_array.chunk_grid_shape().to_vec();
    let target_grid_count = target_grid_shape.iter().try_fold(1_u64, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| ZarrError::message("target chunk count overflows u64"))
    })?;
    let progress_lock = std::sync::Mutex::new(());
    let resolved_workers = worker_ceiling
        .min(memory_limited_workers)
        .min(target_grid_count)
        .max(1);
    let resolved_workers = usize::try_from(resolved_workers)
        .map_err(|_| ZarrError::message("resolved worker count exceeds usize"))?;
    let codec_concurrent_target = if plan.codec_concurrent_target == 0 {
        1
    } else {
        usize::try_from(u64::from(plan.codec_concurrent_target).min(resolved_workers as u64))
            .map_err(|_| ZarrError::message("codec worker count exceeds usize"))?
            .max(1)
    };
    let codec_options = CodecOptions::default()
        .with_concurrent_target(codec_concurrent_target)
        .with_chunk_concurrent_minimum(1);
    let source_array = source_array.with_codec_options(codec_options);
    let target_array = target_array.with_codec_options(codec_options);
    let target_indices = ArraySubset::new_with_shape(target_grid_shape.clone()).indices();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(resolved_workers)
        .build()
        .map_err(ZarrError::from_display)?;
    let completed = std::sync::atomic::AtomicU64::new(0);
    pool.install(|| {
        target_indices
            .into_par_iter()
            .try_for_each(|chunk_indices| -> Result<()> {
                if cancellation_requested(plan) {
                    return Err(ZarrError::message("任务已取消"));
                }
                let subset = target_array
                    .chunk_subset_bounded(&chunk_indices)
                    .map_err(ZarrError::from_display)?;
                let values: ArrayD<f64> = source_array
                    .retrieve_array_subset_opt(&subset, &codec_options)
                    .map_err(ZarrError::from_display)?;
                target_array
                    .store_array_subset_opt(&subset, values, &codec_options)
                    .map_err(ZarrError::from_display)?;
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                report_progress(plan, done, target_grid_count, &progress_lock)
            })
    })?;
    if cancellation_requested(plan) {
        return Err(ZarrError::message("任务已取消"));
    }
    let logical_elements = checked_product(&source_shape)? as u64;
    let logical_bytes = logical_elements
        .checked_mul(8)
        .ok_or_else(|| ZarrError::message("logical byte count overflows u64"))?;
    Ok(RechunkMetrics {
        execution_path: "rust-streaming-target-chunk-f64".to_owned(),
        output: plan.target.clone(),
        source_shape,
        source_chunks: source_chunk_shape,
        target_chunks: plan.target_chunks.clone(),
        logical_bytes,
        target_chunk_count: target_grid_count,
        resolved_workers: u32::try_from(resolved_workers)
            .map_err(|_| ZarrError::message("resolved worker count exceeds u32"))?,
        worker_reason: format!(
            "P3 bounded float64 target-chunk pool; requested_workers={}, ceiling={}, memory_limited_workers={}, peak_bytes_per_worker={}, codec_concurrent_target={}",
            plan.requested_workers.max(1), plan.worker_ceiling, memory_limited_workers,
            peak_bytes_per_worker, codec_concurrent_target,
        ),
        peak_bytes_per_worker,
        memory_budget_bytes: plan.memory_budget_bytes,
        codec_concurrent_target: u32::try_from(codec_concurrent_target)
            .map_err(|_| ZarrError::message("codec worker count exceeds u32"))?,
    })
}
fn multi_target_chunk_count(shape: &[u64], chunks: &[u64]) -> Result<u64> {
    if shape.is_empty() || shape.len() != chunks.len() || chunks.contains(&0) {
        return Err(ZarrError::message(
            "multi-variable shape and target chunks must have matching positive dimensions",
        ));
    }
    shape
        .iter()
        .zip(chunks)
        .try_fold(1_u64, |total, (size, chunk)| {
            if *size == 0 {
                return Err(ZarrError::message(
                    "multi-variable arrays must have positive shapes",
                ));
            }
            let count = size
                .checked_add(*chunk - 1)
                .ok_or_else(|| ZarrError::message("target chunk grid overflows u64"))?
                / *chunk;
            total
                .checked_mul(count)
                .ok_or_else(|| ZarrError::message("target chunk count overflows u64"))
        })
}

fn variable_rechunk_plan(
    plan: &MultiRechunkExecutionPlan,
    variable: &fast_nc_zarr_model::RechunkVariablePlan,
) -> RechunkExecutionPlan {
    RechunkExecutionPlan {
        source: plan.source.clone(),
        target: plan.target.clone(),
        array_path: variable.array_path.clone(),
        target_chunks: variable.target_chunks.clone(),
        expected_dtype: variable.expected_dtype.clone(),
        requested_workers: plan.requested_workers,
        worker_ceiling: plan.worker_ceiling,
        memory_budget_bytes: plan.memory_budget_bytes,
        codec_concurrent_target: plan.codec_concurrent_target,
        codec: plan.codec.clone(),
        codec_level: plan.codec_level,
        codec_shuffle: plan.codec_shuffle.clone(),
        cancellation_file: plan.cancellation_file.clone(),
        progress_file: plan.progress_file.clone(),
    }
}

fn native_dtype_item_size(dtype: &str) -> Option<u64> {
    match dtype {
        "int8" | "uint8" => Some(1),
        "int16" | "uint16" => Some(2),
        "int32" | "uint32" | "float32" => Some(4),
        "int64" | "uint64" | "float64" => Some(8),
        _ => None,
    }
}

fn configure_rechunk_codecs(builder: &mut ArrayBuilder, plan: &RechunkExecutionPlan) -> Result<()> {
    if matches!(plan.codec.as_str(), "" | "none") {
        return Ok(());
    }
    if !matches!(
        plan.codec.as_str(),
        "zstd" | "blosc-zstd" | "blosc-lz4" | "blosc-lz4hc" | "blosc-zlib" | "gzip"
    ) {
        return Err(ZarrError::message(format!(
            "unsupported Rust codec: {}",
            plan.codec
        )));
    }
    let level = plan.codec_level.unwrap_or(1);
    let codecs: Vec<std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>> =
        match plan.codec.as_str() {
            "zstd" => vec![
                std::sync::Arc::new(zarrs::array::codec::ZstdCodec::new(level, false))
                    as std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>,
            ],
            "blosc-zstd" | "blosc-lz4" | "blosc-lz4hc" | "blosc-zlib" => {
                let compressor = match plan.codec.as_str() {
                    "blosc-lz4" => zarrs::array::codec::BloscCompressor::LZ4,
                    "blosc-lz4hc" => zarrs::array::codec::BloscCompressor::LZ4HC,
                    "blosc-zlib" => zarrs::array::codec::BloscCompressor::Zlib,
                    _ => zarrs::array::codec::BloscCompressor::Zstd,
                };
                let shuffle = match plan.codec_shuffle.as_str() {
                    "bitshuffle" => zarrs::array::codec::BloscShuffleMode::BitShuffle,
                    "shuffle" | "auto" => zarrs::array::codec::BloscShuffleMode::Shuffle,
                    "noshuffle" | "" => zarrs::array::codec::BloscShuffleMode::NoShuffle,
                    _ => {
                        return Err(ZarrError::message(format!(
                            "unsupported Rust codec shuffle: {}",
                            plan.codec_shuffle
                        )))
                    }
                };
                vec![std::sync::Arc::new(
                    zarrs::array::codec::BloscCodec::new(
                        compressor,
                        u8::try_from(level)
                            .map_err(|_| ZarrError::message("Blosc level must be 0-9"))?
                            .try_into()
                            .map_err(|_| ZarrError::message("Blosc level must be 0-9"))?,
                        None,
                        shuffle,
                        Some(4),
                    )
                    .map_err(ZarrError::from_display)?,
                )
                    as std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>]
            }
            "gzip" => vec![std::sync::Arc::new(
                zarrs::array::codec::GzipCodec::new(level.max(0) as u32)
                    .map_err(ZarrError::from_display)?,
            )
                as std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>],
            _ => unreachable!(),
        };
    builder.bytes_to_bytes_codecs(codecs);
    Ok(())
}

fn rechunk_array_into(
    source_array: ArrayStore,
    target_store: Arc<Store>,
    plan: &RechunkExecutionPlan,
    progress_start: u64,
    progress_total: u64,
) -> Result<RechunkMetrics> {
    let item_size = native_dtype_item_size(&plan.expected_dtype).ok_or_else(|| {
        ZarrError::message(format!(
            "P1 multi-variable native rechunk does not support dtype {}",
            plan.expected_dtype
        ))
    })?;
    if plan.target_chunks.is_empty() || plan.target_chunks.contains(&0) {
        return Err(ZarrError::message(
            "target_chunks must contain positive values",
        ));
    }
    let source_metadata = serialised_metadata(&source_array)?;
    if source_metadata.get("data_type") != Some(&Value::String(plan.expected_dtype.clone())) {
        return Err(ZarrError::message(format!(
            "source data_type does not match expected_dtype={} for {}",
            plan.expected_dtype, plan.array_path
        )));
    }
    let source_shape = source_array.shape().to_vec();
    if plan.target_chunks.len() != source_shape.len() {
        return Err(ZarrError::message(format!(
            "target_chunks dimensionality does not match {}",
            plan.array_path
        )));
    }
    let source_chunk_shape = source_array
        .chunk_shape_usize(&vec![0_u64; source_shape.len()])
        .map_err(ZarrError::from_display)?;
    let source_chunk_elements = source_chunk_shape.iter().try_fold(1_u64, |total, value| {
        total
            .checked_mul(
                u64::try_from(*value)
                    .map_err(|_| ZarrError::message("source chunk shape exceeds u64"))?,
            )
            .ok_or_else(|| ZarrError::message("source chunk element count overflows u64"))
    })?;
    let target_chunk_elements = checked_product(&plan.target_chunks)? as u64;
    let peak_bytes_per_worker = source_chunk_elements
        .checked_add(target_chunk_elements)
        .and_then(|elements| elements.checked_mul(item_size))
        .ok_or_else(|| ZarrError::message("per-worker memory estimate overflows u64"))?;
    let requested_workers = u64::from(plan.requested_workers.max(1));
    let worker_ceiling = if plan.worker_ceiling == 0 {
        requested_workers
    } else {
        u64::from(plan.worker_ceiling).min(requested_workers)
    };
    let memory_limited_workers = if plan.memory_budget_bytes == 0 {
        worker_ceiling
    } else {
        (plan.memory_budget_bytes / peak_bytes_per_worker.max(1)).max(1)
    };
    let mut array_builder = ArrayBuilder::from_array(&source_array);
    array_builder.chunk_grid_metadata(plan.target_chunks.clone());
    configure_rechunk_codecs(&mut array_builder, plan)?;
    let target_path = normalise_array_path(&plan.array_path);
    let target_array = array_builder
        .build(target_store, &target_path)
        .map_err(ZarrError::from_display)?;
    target_array
        .store_metadata_opt(&ArrayMetadataOptions::default().with_include_zarrs_metadata(false))
        .map_err(ZarrError::from_display)?;
    let target_metadata = serialised_metadata(&target_array)?;
    if source_metadata.get("fill_value") != target_metadata.get("fill_value") {
        return Err(ZarrError::message(format!(
            "fill_value changed for {}",
            plan.array_path
        )));
    }
    if source_array.attributes() != target_array.attributes() {
        return Err(ZarrError::message(format!(
            "CF and array attributes changed for {}",
            plan.array_path
        )));
    }
    let target_grid_shape = target_array.chunk_grid_shape().to_vec();
    let target_grid_count = target_grid_shape.iter().try_fold(1_u64, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| ZarrError::message("target chunk count overflows u64"))
    })?;
    let resolved_workers = worker_ceiling
        .min(memory_limited_workers)
        .min(target_grid_count)
        .max(1);
    let resolved_workers = usize::try_from(resolved_workers)
        .map_err(|_| ZarrError::message("resolved worker count exceeds usize"))?;
    let codec_concurrent_target = if plan.codec_concurrent_target == 0 {
        1
    } else {
        usize::try_from(u64::from(plan.codec_concurrent_target).min(resolved_workers as u64))
            .map_err(|_| ZarrError::message("codec worker count exceeds usize"))?
            .max(1)
    };
    let codec_options = CodecOptions::default()
        .with_concurrent_target(codec_concurrent_target)
        .with_chunk_concurrent_minimum(1);
    let source_array = source_array.with_codec_options(codec_options);
    let target_array = target_array.with_codec_options(codec_options);
    let target_indices = ArraySubset::new_with_shape(target_grid_shape).indices();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(resolved_workers)
        .build()
        .map_err(ZarrError::from_display)?;
    let progress_lock = std::sync::Mutex::new(());
    let completed = std::sync::atomic::AtomicU64::new(0);
    macro_rules! process_chunks {
        ($element:ty) => {{
            pool.install(|| {
                target_indices
                    .into_par_iter()
                    .try_for_each(|chunk_indices| -> Result<()> {
                        if cancellation_requested(plan) {
                            return Err(ZarrError::message("任务已取消"));
                        }
                        let subset = target_array
                            .chunk_subset_bounded(&chunk_indices)
                            .map_err(ZarrError::from_display)?;
                        let values: ArrayD<$element> = source_array
                            .retrieve_array_subset_opt(&subset, &codec_options)
                            .map_err(ZarrError::from_display)?;
                        target_array
                            .store_array_subset_opt(&subset, values, &codec_options)
                            .map_err(ZarrError::from_display)?;
                        let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        report_progress_with_total(
                            plan,
                            progress_start + done,
                            progress_total,
                            &progress_lock,
                        )
                    })
            })
        }};
    }
    match plan.expected_dtype.as_str() {
        "float32" => process_chunks!(f32),
        "float64" => process_chunks!(f64),
        "int8" => process_chunks!(i8),
        "int16" => process_chunks!(i16),
        "int32" => process_chunks!(i32),
        "int64" => process_chunks!(i64),
        "uint8" => process_chunks!(u8),
        "uint16" => process_chunks!(u16),
        "uint32" => process_chunks!(u32),
        "uint64" => process_chunks!(u64),
        _ => {
            return Err(ZarrError::message(format!(
                "unsupported dtype validated before chunk processing: {}",
                plan.expected_dtype
            )))
        }
    }?;
    if cancellation_requested(plan) {
        return Err(ZarrError::message("任务已取消"));
    }
    let logical_bytes = (checked_product(&source_shape)? as u64)
        .checked_mul(item_size)
        .ok_or_else(|| ZarrError::message("logical byte count overflows u64"))?;
    Ok(RechunkMetrics {
        execution_path: "rust-multi-variable-streaming".to_owned(),
        output: plan.target.clone(),
        source_shape,
        source_chunks: source_chunk_shape,
        target_chunks: plan.target_chunks.clone(),
        logical_bytes,
        target_chunk_count: target_grid_count,
        resolved_workers: u32::try_from(resolved_workers)
            .map_err(|_| ZarrError::message("resolved worker count exceeds u32"))?,
        worker_reason: format!(
            "P1 sequential variable orchestration with bounded target-chunk pools; requested_workers={}, ceiling={}, memory_limited_workers={}, peak_bytes_per_worker={}, codec_concurrent_target={}",
            plan.requested_workers.max(1), plan.worker_ceiling, memory_limited_workers,
            peak_bytes_per_worker, codec_concurrent_target
        ),
        peak_bytes_per_worker,
        memory_budget_bytes: plan.memory_budget_bytes,
        codec_concurrent_target: u32::try_from(codec_concurrent_target)
            .map_err(|_| ZarrError::message("codec worker count exceeds u32"))?,
    })
}

pub fn rechunk_multi_array(plan: &MultiRechunkExecutionPlan) -> Result<MultiRechunkMetrics> {
    if plan.variables.is_empty() {
        return Err(ZarrError::message(
            "multi-variable rechunk requires at least one variable",
        ));
    }
    validate_codec_shuffle(&plan.codec_shuffle)?;
    let source_root = Path::new(&plan.source);
    let target_root = Path::new(&plan.target);
    if !source_root.is_dir() {
        return Err(ZarrError::message(format!(
            "Zarr store directory does not exist: {}",
            source_root.display()
        )));
    }
    if source_root == target_root
        || source_root.starts_with(target_root)
        || target_root.starts_with(source_root)
    {
        return Err(ZarrError::message(
            "multi-variable source and target stores cannot overlap",
        ));
    }
    if target_root.exists() {
        return Err(ZarrError::message(format!(
            "refusing to overwrite existing target store: {}",
            target_root.display()
        )));
    }
    let mut array_paths = Vec::with_capacity(plan.variables.len());
    let mut source_arrays = Vec::with_capacity(plan.variables.len());
    let mut total_chunks = 0_u64;
    for variable in &plan.variables {
        if native_dtype_item_size(&variable.expected_dtype).is_none() {
            return Err(ZarrError::message(format!(
                "P1 multi-variable native rechunk does not support dtype {}",
                variable.expected_dtype
            )));
        }
        array_relative_path(&variable.array_path)?;
        let normalised = normalise_array_path(&variable.array_path);
        if array_paths.iter().any(|path| path == &normalised) {
            return Err(ZarrError::message(format!(
                "multi-variable plan contains duplicate array_path: {}",
                normalised
            )));
        }
        let source_array = open_array(source_root, &variable.array_path)?;
        let shape = source_array.shape().to_vec();
        if shape.len() < 2 {
            return Err(ZarrError::message(format!(
                "multi-variable plan cannot select coordinate array {}",
                variable.array_path
            )));
        }
        if variable.is_coordinate {
            return Err(ZarrError::message(format!(
                "multi-variable plan explicitly marks coordinate array {}",
                variable.array_path
            )));
        }
        let actual_dimension_names = source_array.dimension_names().as_ref().map(|names| {
            names
                .iter()
                .map(|name| name.clone().unwrap_or_default())
                .collect::<Vec<_>>()
        });
        if let Some(expected) = &variable.dimension_names {
            if actual_dimension_names.as_ref() != Some(expected) {
                return Err(ZarrError::message(format!(
                    "dimension names do not match expected metadata for {}",
                    variable.array_path
                )));
            }
        }
        let leaf = normalised
            .trim_start_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or("");
        let standard_name = source_array
            .attributes()
            .get("standard_name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let axis = source_array
            .attributes()
            .get("axis")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_uppercase();
        if shape.len() == 1
            || matches!(leaf, "time" | "lat" | "lon")
            || matches!(standard_name.as_str(), "time" | "latitude" | "longitude")
            || matches!(axis.as_str(), "T" | "X" | "Y")
        {
            return Err(ZarrError::message(format!(
                "multi-variable plan cannot select coordinate array {}",
                variable.array_path
            )));
        }
        let metadata = serialised_metadata(&source_array)?;
        if metadata.get("data_type") != Some(&Value::String(variable.expected_dtype.clone())) {
            return Err(ZarrError::message(format!(
                "source data_type does not match expected_dtype={} for {}",
                variable.expected_dtype, variable.array_path
            )));
        }
        total_chunks = total_chunks
            .checked_add(multi_target_chunk_count(&shape, &variable.target_chunks)?)
            .ok_or_else(|| ZarrError::message("multi-variable target chunk count overflows u64"))?;
        array_paths.push(normalised);
        source_arrays.push(source_array);
    }
    copy_store_without_arrays(source_root, target_root, &array_paths)?;
    let target_store =
        Arc::new(FilesystemStore::new(target_root).map_err(ZarrError::from_display)?);
    let result = (|| {
        let mut completed = 0_u64;
        let mut metrics = Vec::with_capacity(plan.variables.len());
        for (variable, source_array) in plan.variables.iter().zip(source_arrays) {
            let variable_plan = variable_rechunk_plan(plan, variable);
            let metric = rechunk_array_into(
                source_array,
                target_store.clone(),
                &variable_plan,
                completed,
                total_chunks,
            )?;
            completed = completed
                .checked_add(metric.target_chunk_count)
                .ok_or_else(|| ZarrError::message("multi-variable progress count overflows u64"))?;
            metrics.push(metric);
        }
        let logical_bytes = metrics.iter().try_fold(0_u64, |total, metric| {
            total
                .checked_add(metric.logical_bytes)
                .ok_or_else(|| ZarrError::message("multi-variable logical bytes overflow u64"))
        })?;
        let resolved_workers = metrics
            .iter()
            .map(|metric| metric.resolved_workers)
            .max()
            .unwrap_or(1);
        Ok(MultiRechunkMetrics {
            execution_path: "rust-multi-variable-streaming".to_owned(),
            output: plan.target.clone(),
            variables: metrics,
            logical_bytes,
            target_chunk_count: total_chunks,
            resolved_workers,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(target_root);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_paths_reject_parent_traversal_and_root() {
        assert!(array_relative_path("../escape").is_err());
        assert!(array_relative_path("/../escape").is_err());
        assert!(array_relative_path("/").is_err());
        assert_eq!(
            array_relative_path("temperature").unwrap(),
            PathBuf::from("temperature")
        );
    }

    #[test]
    fn rechunk_store_overlap_detects_nested_targets() {
        assert!(stores_overlap(
            Path::new("/tmp/fast-nc-zarr-source"),
            Path::new("/tmp/fast-nc-zarr-source/target"),
        ));
        assert!(!stores_overlap(
            Path::new("/tmp/fast-nc-zarr-source"),
            Path::new("/tmp/fast-nc-zarr-target"),
        ));
    }

    #[test]
    fn rechunk_rejects_unknown_codec_shuffle() {
        let plan = RechunkExecutionPlan {
            source: "/tmp/missing-source.zarr".to_string(),
            target: "/tmp/missing-target.zarr".to_string(),
            array_path: "/value".to_string(),
            target_chunks: vec![1, 1, 1],
            expected_dtype: "float32".to_string(),
            requested_workers: 1,
            worker_ceiling: 1,
            memory_budget_bytes: 0,
            codec_concurrent_target: 1,
            codec: "none".to_string(),
            codec_level: None,
            codec_shuffle: "typo".to_string(),
            cancellation_file: None,
            progress_file: None,
        };
        let error = rechunk_f32_array(&plan).expect_err("unknown shuffle must be rejected");
        assert!(error.to_string().contains("unsupported Rust codec shuffle"));
    }
}
