use netcdf::types::{FloatType, IntType, NcVariableType};
use netcdf::{Attribute, AttributeValue};
use serde::Serialize;
use serde_json::{Map, Value};
use std::path::Path;

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
        AttributeValue::Strs(v) => serde_json::json!(v),
    }
}

fn attributes<'a>(
    owner: impl Iterator<Item = Attribute<'a>>,
) -> Result<Map<String, Value>, String> {
    let mut result = Map::new();
    for attribute in owner {
        let name = attribute.name().to_owned();
        let value = attribute.value().map_err(|error| error.to_string())?;
        result.insert(name, attribute_value(value));
    }
    Ok(result)
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
        let (dtype, numeric) = dtype(&variable.vartype());
        let variable_dimensions = variable
            .dimensions()
            .iter()
            .map(|dimension| dimension.name())
            .collect::<Vec<_>>();
        if variable_dimensions.len() == 3 && !numeric {
            limitations.push(format!("variable {} is not numeric", variable.name()));
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
        && variables
            .iter()
            .any(|item| item.dimensions.len() == 3 && item.numeric)
        && limitations.is_empty();
    if !supported_subset && limitations.is_empty() {
        limitations.push("requires at least one numeric three-dimensional variable".into());
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
