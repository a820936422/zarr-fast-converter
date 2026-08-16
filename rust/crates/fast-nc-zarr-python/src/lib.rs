use fast_nc_zarr_model::{
    BackendCapability, MultiRechunkExecutionPlan, RechunkExecutionPlan, BACKEND_PROTOCOL_VERSION,
};
use fast_nc_zarr_zarr::{
    convert_netcdf_to_zarr, inspect_array, inspect_netcdf, read_chunk_f32 as read_zarr_chunk_f32,
    read_chunk_f64 as read_zarr_chunk_f64, read_region_f32 as read_zarr_region_f32,
    read_region_f64 as read_zarr_region_f64, rechunk_f32_array, rechunk_f64_array,
    rechunk_multi_array, resample_f32, resample_f32_values,
    write_f32_array as write_zarr_f32_array, write_f64_array as write_zarr_f64_array,
    ResampleF32Request,
};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

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
fn read_chunk_f64(path: &str, array_path: &str, chunk_indices: Vec<u64>) -> PyResult<Vec<f64>> {
    read_zarr_chunk_f64(path, array_path, &chunk_indices).map_err(runtime_error)
}

#[pyfunction]
fn read_region_f64(
    path: &str,
    array_path: &str,
    starts: Vec<u64>,
    shape: Vec<u64>,
) -> PyResult<Vec<f64>> {
    read_zarr_region_f64(path, array_path, &starts, &shape).map_err(runtime_error)
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

#[pyfunction]
fn write_f64_array(
    path: &str,
    array_path: &str,
    shape: Vec<u64>,
    chunks: Vec<u64>,
    values: Vec<f64>,
) -> PyResult<()> {
    write_zarr_f64_array(path, array_path, &shape, &chunks, &values).map_err(runtime_error)
}

#[pyfunction]
fn rechunk_f32_json(py: Python<'_>, plan_json: &str) -> PyResult<String> {
    let plan: RechunkExecutionPlan = serde_json::from_str(plan_json).map_err(runtime_error)?;
    let metrics = py
        .detach(|| rechunk_f32_array(&plan))
        .map_err(runtime_error)?;
    serde_json::to_string(&metrics).map_err(runtime_error)
}

#[pyfunction]
fn rechunk_f64_json(py: Python<'_>, plan_json: &str) -> PyResult<String> {
    let plan: RechunkExecutionPlan = serde_json::from_str(plan_json).map_err(runtime_error)?;
    let metrics = py
        .detach(|| rechunk_f64_array(&plan))
        .map_err(runtime_error)?;
    serde_json::to_string(&metrics).map_err(runtime_error)
}

#[pyfunction]
fn rechunk_multi_json(py: Python<'_>, plan_json: &str) -> PyResult<String> {
    let plan: MultiRechunkExecutionPlan = serde_json::from_str(plan_json).map_err(runtime_error)?;
    let metrics = py
        .detach(|| rechunk_multi_array(&plan))
        .map_err(runtime_error)?;
    serde_json::to_string(&metrics).map_err(runtime_error)
}

#[pyfunction]
fn inspect_netcdf_json(path: &str) -> PyResult<String> {
    let summary = inspect_netcdf(std::path::Path::new(path)).map_err(runtime_error)?;
    serde_json::to_string(&summary).map_err(runtime_error)
}

#[pyfunction]
fn convert_netcdf_json(py: Python<'_>, input: &str, output: &str) -> PyResult<String> {
    let summary = py
        .detach(|| {
            convert_netcdf_to_zarr(std::path::Path::new(input), std::path::Path::new(output))
        })
        .map_err(runtime_error)?;
    serde_json::to_string(&summary).map_err(runtime_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn resample_f32_buffer(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    shape: Vec<usize>,
    source_lat: &Bound<'_, PyAny>,
    source_lon: &Bound<'_, PyAny>,
    target_lat: &Bound<'_, PyAny>,
    target_lon: &Bound<'_, PyAny>,
    method: &str,
) -> PyResult<(Py<PyBytes>, Vec<usize>)> {
    if shape.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "shape must contain exactly three dimensions",
        ));
    }
    let shape = [shape[0], shape[1], shape[2]];
    let values = PyBuffer::<f32>::get(values)?.to_vec(py)?;
    let source_lat = PyBuffer::<f32>::get(source_lat)?.to_vec(py)?;
    let source_lon = PyBuffer::<f32>::get(source_lon)?.to_vec(py)?;
    let target_lat = PyBuffer::<f32>::get(target_lat)?.to_vec(py)?;
    let target_lon = PyBuffer::<f32>::get(target_lon)?.to_vec(py)?;
    let method = method.to_owned();
    let output = py
        .detach(|| {
            resample_f32_values(
                &values,
                shape,
                &source_lat,
                &source_lon,
                &target_lat,
                &target_lon,
                &method,
            )
        })
        .map_err(runtime_error)?;
    let output_shape = vec![shape[0], target_lat.len(), target_lon.len()];
    let output_bytes = unsafe {
        std::slice::from_raw_parts(
            output.as_ptr().cast::<u8>(),
            output.len() * std::mem::size_of::<f32>(),
        )
    };
    Ok((PyBytes::new(py, output_bytes).unbind(), output_shape))
}

#[pyfunction]
fn resample_f32_json(py: Python<'_>, plan_json: &str) -> PyResult<String> {
    let plan: ResampleF32Request = serde_json::from_str(plan_json).map_err(runtime_error)?;
    let result = py.detach(|| resample_f32(&plan)).map_err(runtime_error)?;
    serde_json::to_string(&result).map_err(runtime_error)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(resample_f32_buffer, module)?)?;
    module.add_function(wrap_pyfunction!(resample_f32_json, module)?)?;
    module.add_function(wrap_pyfunction!(convert_netcdf_json, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_netcdf_json, module)?)?;
    module.add_function(wrap_pyfunction!(capability_json, module)?)?;
    module.add_function(wrap_pyfunction!(protocol_version, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_array_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_chunk_f32, module)?)?;
    module.add_function(wrap_pyfunction!(read_region_f32, module)?)?;
    module.add_function(wrap_pyfunction!(read_chunk_f64, module)?)?;
    module.add_function(wrap_pyfunction!(read_region_f64, module)?)?;
    module.add_function(wrap_pyfunction!(rechunk_f32_json, module)?)?;
    module.add_function(wrap_pyfunction!(write_f32_array, module)?)?;
    module.add_function(wrap_pyfunction!(rechunk_f64_json, module)?)?;
    module.add_function(wrap_pyfunction!(write_f64_array, module)?)?;
    module.add_function(wrap_pyfunction!(rechunk_multi_json, module)?)?;
    Ok(())
}
