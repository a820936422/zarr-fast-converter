use netcdf::types::{FloatType, IntType, NcVariableType};
use netcdf::{Attribute, AttributeValue};
use serde::Serialize;
use serde_json::{Map, Value};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use zarrs::array::Array;
use zarrs_filesystem::FilesystemStore;

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
pub struct NetcdfDimensionSummary {
    pub name: String,
    pub size: usize,
    pub unlimited: bool,
}

#[derive(Debug, Serialize)]
pub struct NetcdfVariableSummary {
    pub name: String,
    pub dimensions: Vec<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub numeric: bool,
    pub attributes: Map<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct NetcdfSummary {
    pub path: String,
    pub dimensions: Vec<NetcdfDimensionSummary>,
    pub variables: Vec<NetcdfVariableSummary>,
    pub attributes: Map<String, Value>,
    pub standard_coordinates: bool,
    pub supported_subset: bool,
    pub limitations: Vec<String>,
}

fn attribute_value(value: AttributeValue) -> Value {
    match value {
        AttributeValue::Uchar(v) => serde_json::json!(v),
        AttributeValue::Uchars(v) => serde_json::json!(v),
        AttributeValue::Schar(v) => serde_json::json!(v),
        AttributeValue::Schars(v) => serde_json::json!(v),
        AttributeValue::Ushort(v) => serde_json::json!(v),
        AttributeValue::Ushorts(v) => serde_json::json!(v),
        AttributeValue::Short(v) => serde_json::json!(v),
        AttributeValue::Shorts(v) => serde_json::json!(v),
        AttributeValue::Uint(v) => serde_json::json!(v),
        AttributeValue::Uints(v) => serde_json::json!(v),
        AttributeValue::Int(v) => serde_json::json!(v),
        AttributeValue::Ints(v) => serde_json::json!(v),
        AttributeValue::Ulonglong(v) => serde_json::json!(v),
        AttributeValue::Ulonglongs(v) => serde_json::json!(v),
        AttributeValue::Longlong(v) => serde_json::json!(v),
        AttributeValue::Longlongs(v) => serde_json::json!(v),
        AttributeValue::Float(v) => serde_json::json!(v),
        AttributeValue::Floats(v) => serde_json::json!(v),
        AttributeValue::Double(v) => serde_json::json!(v),
        AttributeValue::Doubles(v) => serde_json::json!(v),
        AttributeValue::Str(v) => serde_json::json!(v),
        AttributeValue::Strs(v) if v.len() == 1 => serde_json::json!(v[0]),
        AttributeValue::Strs(v) => serde_json::json!(v),
    }
}

fn attributes<'a>(
    owner: impl Iterator<Item = Attribute<'a>>,
) -> Result<Map<String, Value>, String> {
    let mut result = Map::new();
    for attribute in owner {
        let name = attribute.name().to_owned();
        let value = attribute_value(attribute.value().map_err(|error| error.to_string())?);
        if name == "_FillValue" && value.is_null() {
            continue;
        }
        result.insert(name, value);
    }
    Ok(result)
}

fn numeric_attribute(attributes: &Map<String, Value>, name: &str) -> Option<f64> {
    attributes.get(name).and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_i64().map(|item| item as f64))
            .or_else(|| value.as_u64().map(|item| item as f64))
    })
}
fn required_numeric_attribute(
    attributes: &Map<String, Value>,
    name: &str,
) -> Result<Option<f64>, String> {
    if !attributes.contains_key(name) {
        return Ok(None);
    }
    numeric_attribute(attributes, name)
        .ok_or_else(|| format!("NetCDF attribute {name} must be numeric"))
        .map(Some)
}

fn source_value_is_missing(value: f64, attributes: &Map<String, Value>) -> bool {
    ["_FillValue", "missing_value"].iter().any(|name| {
        let Some(marker) = numeric_attribute(attributes, name) else {
            return false;
        };
        if marker.is_nan() {
            value.is_nan()
        } else {
            value == marker
        }
    })
}

fn decode_float_value(value: f64, attributes: &Map<String, Value>) -> Result<f64, String> {
    let _ = required_numeric_attribute(attributes, "_FillValue")?;
    let _ = required_numeric_attribute(attributes, "missing_value")?;
    if source_value_is_missing(value, attributes) {
        return Ok(f64::NAN);
    }
    let scale = required_numeric_attribute(attributes, "scale_factor")?.unwrap_or(1.0);
    let offset = required_numeric_attribute(attributes, "add_offset")?.unwrap_or(0.0);
    if !scale.is_finite() || !offset.is_finite() {
        return Err("NetCDF scale_factor and add_offset must be finite".into());
    }
    Ok(value * scale + offset)
}

fn decode_f32_values(values: &[f32], attributes: &Map<String, Value>) -> Result<Vec<f32>, String> {
    values
        .iter()
        .map(|value| decode_float_value(f64::from(*value), attributes).map(|item| item as f32))
        .collect()
}

fn decode_f64_values(values: &[f64], attributes: &Map<String, Value>) -> Result<Vec<f64>, String> {
    values
        .iter()
        .map(|value| decode_float_value(*value, attributes))
        .collect()
}

fn decoded_float_attributes(attributes: &Map<String, Value>) -> Map<String, Value> {
    let mut result = attributes.clone();
    for name in [
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
        "valid_min",
        "valid_max",
        "valid_range",
        "actual_range",
    ] {
        if let Some(value) = result.remove(name) {
            let target = match name {
                "_FillValue" => "source_fill_value",
                "missing_value" => "source_missing_value",
                "scale_factor" => "source_scale_factor",
                "add_offset" => "source_add_offset",
                "valid_min" => "source_valid_min",
                "valid_max" => "source_valid_max",
                "valid_range" => "source_valid_range",
                "actual_range" => "source_actual_range",
                _ => unreachable!(),
            };
            result.insert(target.to_string(), value);
        }
    }
    result
}

fn dtype(value: &NcVariableType) -> (String, bool) {
    match value {
        NcVariableType::Float(FloatType::F32) => ("float32".into(), true),
        NcVariableType::Float(FloatType::F64) => ("float64".into(), true),
        NcVariableType::Int(IntType::I8) => ("int8".into(), true),
        NcVariableType::Int(IntType::I16) => ("int16".into(), true),
        NcVariableType::Int(IntType::I32) => ("int32".into(), true),
        NcVariableType::Int(IntType::I64) => ("int64".into(), true),
        NcVariableType::Int(IntType::U8) => ("uint8".into(), true),
        NcVariableType::Int(IntType::U16) => ("uint16".into(), true),
        NcVariableType::Int(IntType::U32) => ("uint32".into(), true),
        NcVariableType::Int(IntType::U64) => ("uint64".into(), true),
        NcVariableType::Char => ("char".into(), false),
        NcVariableType::String => ("string".into(), false),
        NcVariableType::Enum(_) => ("enum".into(), false),
        NcVariableType::Compound(_) => ("compound".into(), false),
        NcVariableType::Opaque(_) => ("opaque".into(), false),
        NcVariableType::Vlen(_) => ("vlen".into(), false),
    }
}

pub fn inspect_netcdf(path: &Path) -> Result<NetcdfSummary, String> {
    let file = netcdf::open(path)
        .map_err(|error| format!("cannot open NetCDF {}: {error}", path.display()))?;
    let dimensions = file
        .dimensions()
        .map(|dimension| NetcdfDimensionSummary {
            name: dimension.name(),
            size: dimension.len(),
            unlimited: dimension.is_unlimited(),
        })
        .collect::<Vec<_>>();
    let dimension_names = dimensions
        .iter()
        .map(|item| item.name.as_str())
        .collect::<Vec<_>>();
    let standard_coordinates = ["time", "lat", "lon"]
        .iter()
        .all(|name| dimension_names.contains(name));
    let mut variables = Vec::new();
    let mut limitations = Vec::new();
    for variable in file.variables() {
        let name = variable.name();
        let (dtype, numeric) = dtype(&variable.vartype());
        let variable_dimensions = variable
            .dimensions()
            .iter()
            .map(|dimension| dimension.name())
            .collect::<Vec<_>>();
        let variable_attributes = attributes(variable.attributes())?;
        let is_coordinate_name = matches!(name.as_str(), "time" | "lat" | "lon");
        let is_standard_coordinate =
            is_coordinate_name && variable_dimensions == [name.clone()] && numeric;
        let integer_coordinate_has_encoding = is_standard_coordinate
            && !matches!(dtype.as_str(), "float32" | "float64")
            && ["_FillValue", "missing_value", "scale_factor", "add_offset"]
                .iter()
                .any(|key| variable_attributes.contains_key(*key));
        if (is_coordinate_name && !is_standard_coordinate)
            || integer_coordinate_has_encoding
            || (!is_coordinate_name
                && (variable_dimensions
                    != vec!["time".to_owned(), "lat".to_owned(), "lon".to_owned()]
                    || !matches!(dtype.as_str(), "float32" | "float64")))
        {
            limitations.push(format!(
                "variable {name} is outside the native numeric time/lat/lon float subset"
            ));
        }
        variables.push(NetcdfVariableSummary {
            name: variable.name(),
            dimensions: variable_dimensions,
            shape: variable
                .dimensions()
                .iter()
                .map(|dimension| dimension.len())
                .collect(),
            dtype,
            numeric,
            attributes: variable_attributes,
        });
    }
    if !standard_coordinates {
        limitations.push("requires dimensions named time, lat and lon".into());
    }
    let supported_subset = standard_coordinates
        && variables.iter().any(|item| {
            item.dimensions.len() == 3 && matches!(item.dtype.as_str(), "float32" | "float64")
        })
        && limitations.is_empty();
    if !supported_subset && limitations.is_empty() {
        limitations.push("requires at least one float32/float64 three-dimensional variable".into());
    }
    Ok(NetcdfSummary {
        path: path.to_string_lossy().into_owned(),
        dimensions,
        variables,
        attributes: attributes(file.attributes())?,
        standard_coordinates,
        supported_subset,
        limitations,
    })
}

#[derive(Debug, Serialize)]
pub struct NetcdfConversionSummary {
    pub input: String,
    pub output: String,
    pub variables: Vec<String>,
    pub logical_bytes: u64,
}

fn native_values_budget_bytes() -> u64 {
    std::env::var("FAST_NC_ZARR_NATIVE_MEMORY_BUDGET_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(512 * 1024 * 1024)
}

fn ensure_native_values_budget(shape: &[u64], dtype: &str) -> Result<(), String> {
    let itemsize = match dtype {
        "float32" | "int32" | "uint32" => 4,
        "float64" | "int64" | "uint64" => 8,
        "int8" | "uint8" => 1,
        "int16" | "uint16" => 2,
        _ => return Err(format!("unsupported NetCDF dtype for budget: {dtype}")),
    };
    let elements = shape.iter().try_fold(1_u64, |total, value| {
        total
            .checked_mul(*value)
            .ok_or("array element count overflows u64")
    })?;
    let bytes = elements
        .checked_mul(itemsize)
        .ok_or("native NetCDF working set overflows u64")?;
    if bytes > native_values_budget_bytes() {
        return Err(format!(
            "resource_budget_exceeded: variable requires {bytes} bytes"
        ));
    }
    Ok(())
}

fn store_f32_array(
    store: std::sync::Arc<zarrs_filesystem::FilesystemStore>,
    name: &str,
    shape: &[u64],
    dimensions: &[String],
    values: &[f32],
    attrs: &Map<String, Value>,
    cancellation_file: Option<&Path>,
) -> Result<u64, String> {
    if cancellation_requested(cancellation_file) {
        return Err("任务已取消".into());
    }
    let chunks = shape
        .iter()
        .map(|value| (*value).clamp(1, 64))
        .collect::<Vec<_>>();
    let fill_value = numeric_attribute(attrs, "_FillValue")
        .map(|value| value as f32)
        .unwrap_or(f32::NAN);
    let mut builder = zarrs::array::ArrayBuilder::new(
        shape.to_vec(),
        chunks,
        zarrs::array::data_type::float32(),
        fill_value,
    );
    builder.dimension_names(Some(dimensions.to_vec()));
    let mut array = builder
        .build(store, &format!("/{name}"))
        .map_err(|error| error.to_string())?;
    for (key, value) in attrs {
        array.attributes_mut().insert(key.clone(), value.clone());
    }
    array
        .store_metadata_opt(
            &zarrs::array::ArrayMetadataOptions::default().with_include_zarrs_metadata(false),
        )
        .map_err(|error| error.to_string())?;
    let grid = array
        .chunk_grid_shape()
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let mut index = vec![0_usize; grid.len()];
    loop {
        if cancellation_requested(cancellation_file) {
            return Err("任务已取消".into());
        }
        let index_u64 = index.iter().map(|value| *value as u64).collect::<Vec<_>>();
        let chunk_shape = array
            .chunk_shape_usize(&index_u64)
            .map_err(|error| error.to_string())?;
        let chunk = super::chunk_values(values, shape, &chunk_shape, &index_u64)
            .map_err(|error| error.to_string())?;
        array
            .store_chunk(&index_u64, chunk.as_slice())
            .map_err(|error| error.to_string())?;
        if !super::increment_index(&mut index, &grid) {
            break;
        }
    }
    Ok(std::mem::size_of_val(values) as u64)
}
fn store_f64_array(
    store: std::sync::Arc<zarrs_filesystem::FilesystemStore>,
    name: &str,
    shape: &[u64],
    dimensions: &[String],
    values: &[f64],
    attrs: &Map<String, Value>,
    cancellation_file: Option<&Path>,
) -> Result<u64, String> {
    if cancellation_requested(cancellation_file) {
        return Err("任务已取消".into());
    }
    let chunks = shape
        .iter()
        .map(|value| (*value).clamp(1, 64))
        .collect::<Vec<_>>();
    let fill_value = numeric_attribute(attrs, "_FillValue").unwrap_or(f64::NAN);
    let mut builder = zarrs::array::ArrayBuilder::new(
        shape.to_vec(),
        chunks,
        zarrs::array::data_type::float64(),
        fill_value,
    );
    builder.dimension_names(Some(dimensions.to_vec()));
    let mut array = builder
        .build(store, &format!("/{name}"))
        .map_err(|error| error.to_string())?;
    for (key, value) in attrs {
        array.attributes_mut().insert(key.clone(), value.clone());
    }
    array
        .store_metadata_opt(
            &zarrs::array::ArrayMetadataOptions::default().with_include_zarrs_metadata(false),
        )
        .map_err(|error| error.to_string())?;
    let grid = array
        .chunk_grid_shape()
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let mut index = vec![0_usize; grid.len()];
    loop {
        if cancellation_requested(cancellation_file) {
            return Err("任务已取消".into());
        }
        let index_u64 = index.iter().map(|value| *value as u64).collect::<Vec<_>>();
        let chunk_shape = array
            .chunk_shape_usize(&index_u64)
            .map_err(|error| error.to_string())?;
        let chunk = super::chunk_values_f64(values, shape, &chunk_shape, &index_u64)
            .map_err(|error| error.to_string())?;
        array
            .store_chunk(&index_u64, chunk.as_slice())
            .map_err(|error| error.to_string())?;
        if !super::increment_index(&mut index, &grid) {
            break;
        }
    }
    Ok(std::mem::size_of_val(values) as u64)
}

fn integer_chunk_values<T: Copy + Default>(
    values: &[T],
    shape: &[u64],
    chunk_shape: &[usize],
    chunk_indices: &[u64],
) -> Result<Vec<T>, String> {
    let expected = super::checked_product(shape).map_err(|error| error.to_string())?;
    if values.len() != expected {
        return Err(format!(
            "values length {} does not match array element count {}",
            values.len(),
            expected
        ));
    }
    let strides = super::row_major_strides(shape).map_err(|error| error.to_string())?;
    let capacity = chunk_shape
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value))
        .ok_or_else(|| "chunk element count overflows usize".to_owned())?;
    let mut output = Vec::with_capacity(capacity);
    let mut local = vec![0_usize; chunk_shape.len()];
    loop {
        let mut source_offset = 0_usize;
        let mut in_bounds = true;
        for axis in 0..chunk_shape.len() {
            let origin = usize::try_from(chunk_indices[axis])
                .ok()
                .and_then(|index| index.checked_mul(chunk_shape[axis]))
                .ok_or_else(|| "chunk origin overflows usize".to_owned())?;
            let coordinate = origin
                .checked_add(local[axis])
                .ok_or_else(|| "chunk coordinate overflows usize".to_owned())?;
            let axis_size =
                usize::try_from(shape[axis]).map_err(|_| "array shape exceeds usize".to_owned())?;
            if coordinate >= axis_size {
                in_bounds = false;
                break;
            }
            source_offset = source_offset
                .checked_add(
                    coordinate
                        .checked_mul(strides[axis])
                        .ok_or_else(|| "source offset overflows usize".to_owned())?,
                )
                .ok_or_else(|| "source offset overflows usize".to_owned())?;
        }
        output.push(if in_bounds {
            values[source_offset]
        } else {
            T::default()
        });
        if !super::increment_index(&mut local, chunk_shape) {
            break;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn store_integer_array<
    T: Copy + Default + zarrs::array::Element + Into<zarrs::array::builder::ArrayBuilderFillValue>,
>(
    store: std::sync::Arc<zarrs_filesystem::FilesystemStore>,
    name: &str,
    shape: &[u64],
    dimensions: &[String],
    values: &[T],
    attrs: &Map<String, Value>,
    data_type_and_fill_value: (zarrs::array::builder::ArrayBuilderDataType, T),
    cancellation_file: Option<&Path>,
) -> Result<u64, String> {
    if cancellation_requested(cancellation_file) {
        return Err("任务已取消".into());
    }
    let (data_type, fill_value) = data_type_and_fill_value;
    let chunks = shape
        .iter()
        .map(|value| (*value).clamp(1, 64))
        .collect::<Vec<_>>();
    let mut builder =
        zarrs::array::ArrayBuilder::new(shape.to_vec(), chunks, data_type, fill_value);
    builder.dimension_names(Some(dimensions.to_vec()));
    let mut array = builder
        .build(store, &format!("/{name}"))
        .map_err(|error| error.to_string())?;
    for (key, value) in attrs {
        array.attributes_mut().insert(key.clone(), value.clone());
    }
    array
        .store_metadata_opt(
            &zarrs::array::ArrayMetadataOptions::default().with_include_zarrs_metadata(false),
        )
        .map_err(|error| error.to_string())?;
    let grid = array
        .chunk_grid_shape()
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let mut index = vec![0_usize; grid.len()];
    loop {
        if cancellation_requested(cancellation_file) {
            return Err("任务已取消".into());
        }
        let index_u64 = index.iter().map(|value| *value as u64).collect::<Vec<_>>();
        let chunk_shape = array
            .chunk_shape_usize(&index_u64)
            .map_err(|error| error.to_string())?;
        let chunk = integer_chunk_values(values, shape, &chunk_shape, &index_u64)?;
        array
            .store_chunk(&index_u64, chunk.as_slice())
            .map_err(|error| error.to_string())?;
        if !super::increment_index(&mut index, &grid) {
            break;
        }
    }
    Ok(std::mem::size_of_val(values) as u64)
}

fn path_entry_exists(path: &Path) -> bool {
    path.exists() || std::fs::symlink_metadata(path).is_ok()
}

fn source_identity(path: &Path) -> Result<(u64, u128), String> {
    let metadata = std::fs::metadata(path).map_err(|error| error.to_string())?;
    let modified = metadata
        .modified()
        .map_err(|error| error.to_string())?
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    Ok((metadata.len(), modified))
}

fn cancellation_requested(path: Option<&Path>) -> bool {
    path.is_some_and(|path| path.is_file())
}

struct StagingGuard {
    path: PathBuf,
    armed: bool,
}

impl StagingGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn fill_value_matches(metadata: &Value, dtype: &str, attrs: &Map<String, Value>) -> bool {
    let Some(actual) = metadata.get("fill_value") else {
        return false;
    };
    let expected = numeric_attribute(attrs, "_FillValue");
    match dtype {
        "float32" => {
            let expected = expected.map(|value| value as f32).unwrap_or(f32::NAN);
            if expected.is_nan() {
                actual.as_str() == Some("NaN") || actual.is_null()
            } else {
                actual
                    .as_f64()
                    .is_some_and(|value| value == f64::from(expected))
            }
        }
        "float64" => {
            let expected = expected.unwrap_or(f64::NAN);
            if expected.is_nan() {
                actual.as_str() == Some("NaN") || actual.is_null()
            } else {
                actual.as_f64().is_some_and(|value| value == expected)
            }
        }
        _ => expected.is_none() || actual.as_f64().is_some_and(|value| Some(value) == expected),
    }
}

fn validate_converted_output(output: &Path, summary: &NetcdfSummary) -> Result<(), String> {
    let root_metadata: Value = serde_json::from_str(
        &std::fs::read_to_string(output.join("zarr.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if root_metadata.get("zarr_format") != Some(&Value::from(3))
        || root_metadata.get("node_type") != Some(&Value::String("group".into()))
    {
        return Err("native conversion output is not a Zarr v3 group".into());
    }
    let expected_root_attrs = Value::Object(summary.attributes.clone());
    let actual_root_attrs = root_metadata
        .get("attributes")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    if actual_root_attrs != expected_root_attrs {
        return Err("native conversion root attributes changed".into());
    }

    let store = Arc::new(FilesystemStore::new(output).map_err(|error| error.to_string())?);
    let mut actual_names = Vec::new();
    for entry in std::fs::read_dir(output).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "zarr.json" {
            continue;
        }
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if !file_type.is_dir() || file_type.is_symlink() {
            return Err(format!(
                "native conversion output contains unsafe entry: {name}"
            ));
        }
        actual_names.push(name);
    }
    actual_names.sort();
    let mut expected_names = summary
        .variables
        .iter()
        .map(|item| item.name.clone())
        .collect::<Vec<_>>();
    expected_names.sort();
    if actual_names != expected_names {
        return Err(format!(
            "native conversion variables changed: expected {expected_names:?}, actual {actual_names:?}"
        ));
    }

    for item in &summary.variables {
        let array = Array::open(store.clone(), &format!("/{}", item.name))
            .map_err(|error| format!("cannot open converted variable {}: {error}", item.name))?;
        let metadata = serde_json::to_value(array.metadata()).map_err(|error| error.to_string())?;
        let actual_shape = metadata
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("converted variable {} has no shape", item.name))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| "invalid converted shape".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected_shape = item
            .shape
            .iter()
            .map(|value| *value as u64)
            .collect::<Vec<_>>();
        if actual_shape != expected_shape {
            return Err(format!("converted variable {} shape changed", item.name));
        }
        if metadata.get("data_type").and_then(Value::as_str) != Some(item.dtype.as_str()) {
            return Err(format!("converted variable {} dtype changed", item.name));
        }
        let expected_dimensions = item
            .dimensions
            .iter()
            .map(|value| Value::String(value.clone()))
            .collect::<Vec<_>>();
        if metadata.get("dimension_names").and_then(Value::as_array) != Some(&expected_dimensions) {
            return Err(format!(
                "converted variable {} dimensions changed",
                item.name
            ));
        }
        let expected_attributes = if matches!(item.dtype.as_str(), "float32" | "float64") {
            decoded_float_attributes(&item.attributes)
        } else {
            item.attributes.clone()
        };
        let actual_attributes = metadata
            .get("attributes")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        if actual_attributes != Value::Object(expected_attributes.clone()) {
            return Err(format!(
                "converted variable {} attributes changed",
                item.name
            ));
        }
        if !fill_value_matches(&metadata, &item.dtype, &expected_attributes) {
            return Err(format!(
                "converted variable {} fill_value changed",
                item.name
            ));
        }
    }
    Ok(())
}

pub fn convert_netcdf_to_zarr(
    input: &Path,
    output: &Path,
) -> Result<NetcdfConversionSummary, String> {
    convert_netcdf_to_zarr_with_cancellation(input, output, None)
}

pub fn convert_netcdf_to_zarr_with_cancellation(
    input: &Path,
    output: &Path,
    cancellation_file: Option<&Path>,
) -> Result<NetcdfConversionSummary, String> {
    if path_entry_exists(output) {
        return Err(format!(
            "refusing to overwrite existing output: {}",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("output has no parent directory: {}", output.display()))?;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    let staging = parent.join(format!(
        ".{name}.native-netcdf-{}-{timestamp}.tmp",
        std::process::id()
    ));
    std::fs::create_dir(&staging).map_err(|error| error.to_string())?;
    let mut staging_guard = StagingGuard::new(staging.clone());

    let result = convert_netcdf_to_zarr_inner(input, &staging, cancellation_file);
    match result {
        Ok(mut summary) => {
            if cancellation_requested(cancellation_file) {
                return Err("任务已取消".into());
            }
            if path_entry_exists(output) {
                return Err(format!(
                    "output appeared during native conversion: {}",
                    output.display()
                ));
            }
            if !staging.join("zarr.json").is_file() {
                return Err("native conversion did not produce a Zarr v3 group".into());
            }
            if let Err(error) = std::fs::rename(&staging, output) {
                return Err(error.to_string());
            }
            staging_guard.disarm();
            summary.output = output.to_string_lossy().into_owned();
            Ok(summary)
        }
        Err(error) => Err(error),
    }
}

fn convert_netcdf_to_zarr_inner(
    input: &Path,
    output: &Path,
    cancellation_file: Option<&Path>,
) -> Result<NetcdfConversionSummary, String> {
    if cancellation_requested(cancellation_file) {
        return Err("任务已取消".into());
    }
    let source_before = source_identity(input)?;
    let file = netcdf::open(input).map_err(|error| error.to_string())?;
    let summary = inspect_netcdf(input)?;
    if !summary.supported_subset {
        return Err(format!(
            "input is outside supported NetCDF subset: {:?}",
            summary.limitations
        ));
    }
    if summary.variables.iter().any(|item| {
        let coordinate = matches!(item.name.as_str(), "time" | "lat" | "lon")
            && item.dimensions == vec![item.name.clone()];
        !coordinate && !matches!(item.dtype.as_str(), "float32" | "float64")
    }) {
        return Err(
            "native conversion supports float32/float64 time-lat-lon data variables and numeric standard coordinates".into(),
        );
    }
    std::fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let store = std::sync::Arc::new(
        zarrs_filesystem::FilesystemStore::new(output).map_err(|error| error.to_string())?,
    );
    let mut group = zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|error| error.to_string())?;
    for (key, value) in &summary.attributes {
        group.attributes_mut().insert(key.clone(), value.clone());
    }
    group.store_metadata().map_err(|error| error.to_string())?;
    let mut names = Vec::new();
    let mut logical_bytes = 0_u64;
    for variable_summary in &summary.variables {
        if cancellation_requested(cancellation_file) {
            return Err("任务已取消".into());
        }
        let variable = file
            .variable(&variable_summary.name)
            .ok_or_else(|| format!("missing variable {}", variable_summary.name))?;
        let shape = variable_summary
            .shape
            .iter()
            .map(|value| *value as u64)
            .collect::<Vec<_>>();
        let attrs = variable_summary.attributes.clone();
        ensure_native_values_budget(&shape, &variable_summary.dtype)?;
        let bytes = match variable_summary.dtype.as_str() {
            "float32" => {
                let values = variable
                    .get_values::<f32, _>(..)
                    .map_err(|error| error.to_string())?;
                let values = decode_f32_values(&values, &attrs)?;
                let decoded_attrs = decoded_float_attributes(&attrs);
                store_f32_array(
                    store.clone(),
                    &variable_summary.name,
                    &shape,
                    &variable_summary.dimensions,
                    &values,
                    &decoded_attrs,
                    cancellation_file,
                )?
            }
            "float64" => {
                let values = variable
                    .get_values::<f64, _>(..)
                    .map_err(|error| error.to_string())?;
                let values = decode_f64_values(&values, &attrs)?;
                let decoded_attrs = decoded_float_attributes(&attrs);
                store_f64_array(
                    store.clone(),
                    &variable_summary.name,
                    &shape,
                    &variable_summary.dimensions,
                    &values,
                    &decoded_attrs,
                    cancellation_file,
                )?
            }
            "int8" => store_integer_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<i8, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
                (zarrs::array::data_type::int8().into(), 0_i8),
                cancellation_file,
            )?,
            "int16" => store_integer_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<i16, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
                (zarrs::array::data_type::int16().into(), 0_i16),
                cancellation_file,
            )?,
            "int32" => store_integer_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<i32, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
                (zarrs::array::data_type::int32().into(), 0_i32),
                cancellation_file,
            )?,
            "int64" => store_integer_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<i64, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
                (zarrs::array::data_type::int64().into(), 0_i64),
                cancellation_file,
            )?,
            "uint8" => store_integer_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<u8, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
                (zarrs::array::data_type::uint8().into(), 0_u8),
                cancellation_file,
            )?,
            "uint16" => store_integer_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<u16, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
                (zarrs::array::data_type::uint16().into(), 0_u16),
                cancellation_file,
            )?,
            "uint32" => store_integer_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<u32, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
                (zarrs::array::data_type::uint32().into(), 0_u32),
                cancellation_file,
            )?,
            "uint64" => store_integer_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<u64, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
                (zarrs::array::data_type::uint64().into(), 0_u64),
                cancellation_file,
            )?,
            _ => {
                return Err(format!(
                    "unsupported NetCDF dtype {} for variable {}",
                    variable_summary.dtype, variable_summary.name
                ));
            }
        };
        logical_bytes = logical_bytes.saturating_add(bytes);
        names.push(variable_summary.name.clone());
    }
    if cancellation_requested(cancellation_file) {
        return Err("任务已取消".into());
    }
    if source_identity(input)? != source_before {
        return Err("source NetCDF changed during native conversion".into());
    }
    validate_converted_output(output, &summary)?;
    if cancellation_requested(cancellation_file) {
        return Err("任务已取消".into());
    }
    Ok(NetcdfConversionSummary {
        input: input.to_string_lossy().into_owned(),
        output: output.to_string_lossy().into_owned(),
        variables: names,
        logical_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_budget_rejects_oversized_float64_variable() {
        let result = ensure_native_values_budget(&[512 * 1024 * 1024], "float64");
        assert!(result
            .expect_err("default native budget must reject a 4 GiB variable")
            .contains("resource_budget_exceeded"));
    }

    #[test]
    fn native_budget_detects_element_count_overflow() {
        let result = ensure_native_values_budget(&[u64::MAX, 2], "float32");
        assert!(result
            .expect_err("element multiplication must be checked")
            .contains("overflows u64"));
    }
}
