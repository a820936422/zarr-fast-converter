use netcdf::types::{FloatType, IntType, NcVariableType};
use netcdf::{Attribute, AttributeValue};
use serde::Serialize;
use serde_json::{Map, Value};
use std::path::Path;
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
        let is_coordinate_name = matches!(name.as_str(), "time" | "lat" | "lon");
        let is_standard_coordinate =
            is_coordinate_name && variable_dimensions == [name.clone()] && numeric;
        if (is_coordinate_name && !is_standard_coordinate)
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
            attributes: attributes(variable.attributes())?,
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
) -> Result<u64, String> {
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
) -> Result<u64, String> {
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
) -> Result<u64, String> {
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

pub fn convert_netcdf_to_zarr(
    input: &Path,
    output: &Path,
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

    let result = convert_netcdf_to_zarr_inner(input, &staging);
    match result {
        Ok(mut summary) => {
            if path_entry_exists(output) {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(format!(
                    "output appeared during native conversion: {}",
                    output.display()
                ));
            }
            if !staging.join("zarr.json").is_file() {
                let _ = std::fs::remove_dir_all(&staging);
                return Err("native conversion did not produce a Zarr v3 group".into());
            }
            if let Err(error) = std::fs::rename(&staging, output) {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(error.to_string());
            }
            summary.output = output.to_string_lossy().into_owned();
            Ok(summary)
        }
        Err(error) => {
            let _ = std::fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

fn convert_netcdf_to_zarr_inner(
    input: &Path,
    output: &Path,
) -> Result<NetcdfConversionSummary, String> {
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
    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|error| error.to_string())?
        .store_metadata()
        .map_err(|error| error.to_string())?;
    let mut names = Vec::new();
    let mut logical_bytes = 0_u64;
    for variable_summary in &summary.variables {
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
            "float32" => store_f32_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<f32, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
            )?,
            "float64" => store_f64_array(
                store.clone(),
                &variable_summary.name,
                &shape,
                &variable_summary.dimensions,
                &variable
                    .get_values::<f64, _>(..)
                    .map_err(|error| error.to_string())?,
                &attrs,
            )?,
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
    if source_identity(input)? != source_before {
        return Err("source NetCDF changed during native conversion".into());
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
