use fast_nc_zarr_model::{BackendCapability, BACKEND_PROTOCOL_VERSION};
use pyo3::prelude::*;

#[pyfunction]
fn capability_json() -> PyResult<String> {
    serde_json::to_string(&BackendCapability::smoke())
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))
}

#[pyfunction]
fn protocol_version() -> u32 {
    BACKEND_PROTOCOL_VERSION
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capability_json, module)?)?;
    module.add_function(wrap_pyfunction!(protocol_version, module)?)?;
    Ok(())
}
