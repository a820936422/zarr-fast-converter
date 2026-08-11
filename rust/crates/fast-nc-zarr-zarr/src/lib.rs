use std::fs;
use std::path::Path;
use std::sync::Arc;

use ndarray::ArrayD;
use serde::Serialize;
use serde_json::{Map, Value};
use thiserror::Error;
use zarrs::array::{data_type, Array, ArrayBuilder, ArraySubset};
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

pub type Result<T> = std::result::Result<T, ZarrError>;

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
    let store = Arc::new(FilesystemStore::new(root).map_err(ZarrError::from_display)?);
    Array::open(store, &normalise_array_path(array_path)).map_err(ZarrError::from_display)
}

fn serialised_metadata(array: &ArrayStore) -> Result<Value> {
    serde_json::to_value(array.metadata()).map_err(ZarrError::from_display)
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
