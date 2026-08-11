use fast_nc_zarr_model::{BackendCapability, BACKEND_PROTOCOL_VERSION};
use fast_nc_zarr_zarr::{
    inspect_array, read_chunk_f32 as read_zarr_chunk_f32, read_region_f32 as read_zarr_region_f32,
    write_f32_array as write_zarr_f32_array,
};
use pyo3::prelude::*;

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
}

#[pyfunction]
fn capability_json() -> PyResult<String> {
    serde_json::to_string(&BackendCapability::smoke()).map_err(runtime_error)
}

#[pyfunction]
fn protocol_version() -> u32 {
    BACKEND_PROTOCOL_VERSION
}

#[pyfunction]
fn inspect_array_json(path: &str, array_path: &str) -> PyResult<String> {
    let summary = inspect_array(path, array_path).map_err(runtime_error)?;
    serde_json::to_string(&summary).map_err(runtime_error)
}

#[pyfunction]
fn read_chunk_f32(path: &str, array_path: &str, chunk_indices: Vec<u64>) -> PyResult<Vec<f32>> {
    read_zarr_chunk_f32(path, array_path, &chunk_indices).map_err(runtime_error)
}

#[pyfunction]
fn read_region_f32(
    path: &str,
    array_path: &str,
    starts: Vec<u64>,
    shape: Vec<u64>,
) -> PyResult<Vec<f32>> {
    read_zarr_region_f32(path, array_path, &starts, &shape).map_err(runtime_error)
}

#[pyfunction]
fn write_f32_array(
    path: &str,
    array_path: &str,
    shape: Vec<u64>,
    chunks: Vec<u64>,
    values: Vec<f32>,
) -> PyResult<()> {
    write_zarr_f32_array(path, array_path, &shape, &chunks, &values).map_err(runtime_error)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capability_json, module)?)?;
    module.add_function(wrap_pyfunction!(protocol_version, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_array_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_chunk_f32, module)?)?;
    module.add_function(wrap_pyfunction!(read_region_f32, module)?)?;
    module.add_function(wrap_pyfunction!(write_f32_array, module)?)?;
    Ok(())
}
