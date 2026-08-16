use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct ResampleF32Request {
    pub values: Vec<f32>,
    pub shape: [usize; 3],
    pub source_lat: Vec<f32>,
    pub source_lon: Vec<f32>,
    pub target_lat: Vec<f32>,
    pub target_lon: Vec<f32>,
    pub method: String,
}

#[derive(Debug, Serialize)]
pub struct ResampleF32Response {
    pub shape: [usize; 3],
    pub values: Vec<f32>,
    pub method: String,
}

#[derive(Clone, Copy)]
struct AxisBracket {
    low: usize,
    high: usize,
    weight: f32,
}

fn validate_axis(axis: &[f32], name: &str) -> Result<(), String> {
    if axis.len() < 2 {
        return Err(format!("{name} axis must contain at least two coordinates"));
    }
    if axis.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} axis must contain only finite coordinates"));
    }
    let direction = (axis[1] - axis[0]).signum();
    if direction == 0.0
        || axis
            .windows(2)
            .any(|pair| (pair[1] - pair[0]).signum() != direction)
    {
        return Err(format!("{name} axis must be strictly monotonic"));
    }
    Ok(())
}

fn axis_brackets(axis: &[f32], targets: &[f32]) -> Result<Vec<Option<AxisBracket>>, String> {
    let descending = axis[0] > axis[axis.len() - 1];
    let mut ordered = axis.to_vec();
    if descending {
        ordered.reverse();
    }
    let last = ordered.len() - 1;
    let mut result = Vec::with_capacity(targets.len());
    for &value in targets {
        if !value.is_finite() || value < ordered[0] || value > ordered[last] {
            result.push(None);
            continue;
        }
        let upper = ordered.partition_point(|item| *item < value);
        let (low, high, weight) = if upper == 0 {
            (0, 0, 0.0)
        } else if upper >= ordered.len() {
            (last, last, 0.0)
        } else {
            let low = upper - 1;
            let high = upper;
            (
                low,
                high,
                (value - ordered[low]) / (ordered[high] - ordered[low]),
            )
        };
        let map_index = |index: usize| {
            if descending {
                axis.len() - 1 - index
            } else {
                index
            }
        };
        result.push(Some(AxisBracket {
            low: map_index(low),
            high: map_index(high),
            weight,
        }));
    }
    Ok(result)
}

fn source_offset(time: usize, lat: usize, lon: usize, shape: [usize; 3]) -> usize {
    (time * shape[1] + lat) * shape[2] + lon
}

pub fn resample_f32_values(
    values: &[f32],
    shape: [usize; 3],
    source_lat: &[f32],
    source_lon: &[f32],
    target_lat: &[f32],
    target_lon: &[f32],
    method: &str,
) -> Result<Vec<f32>, String> {
    let expected_values = shape[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_mul(shape[2]))
        .ok_or_else(|| "source shape element count overflows usize".to_owned())?;
    if shape[0] == 0
        || shape[1] != source_lat.len()
        || shape[2] != source_lon.len()
        || values.len() != expected_values
    {
        return Err("source shape, axes and values are inconsistent".into());
    }
    validate_axis(source_lat, "source latitude")?;
    validate_axis(source_lon, "source longitude")?;
    if method != "nearest" && method != "bilinear" {
        return Err(format!("unsupported native resampling method: {method}"));
    }
    let lat_brackets = axis_brackets(source_lat, target_lat)?;
    let lon_brackets = axis_brackets(source_lon, target_lon)?;
    let output_values = shape[0]
        .checked_mul(target_lat.len())
        .and_then(|value| value.checked_mul(target_lon.len()))
        .ok_or_else(|| "target shape element count overflows usize".to_owned())?;
    let mut output = Vec::with_capacity(output_values);
    for time in 0..shape[0] {
        for lat in &lat_brackets {
            for lon in &lon_brackets {
                let value = match (lat, lon, method) {
                    (Some(lat), Some(lon), "nearest") => {
                        let source_lat = if lat.weight <= 0.5 { lat.low } else { lat.high };
                        let source_lon = if lon.weight <= 0.5 { lon.low } else { lon.high };
                        values[source_offset(time, source_lat, source_lon, shape)]
                    }
                    (Some(lat), Some(lon), "bilinear") => {
                        let a = values[source_offset(time, lat.low, lon.low, shape)];
                        let b = values[source_offset(time, lat.low, lon.high, shape)];
                        let c = values[source_offset(time, lat.high, lon.low, shape)];
                        let d = values[source_offset(time, lat.high, lon.high, shape)];
                        if [a, b, c, d].iter().any(|value| value.is_nan()) {
                            f32::NAN
                        } else {
                            a * (1.0 - lon.weight) * (1.0 - lat.weight)
                                + b * lon.weight * (1.0 - lat.weight)
                                + c * (1.0 - lon.weight) * lat.weight
                                + d * lon.weight * lat.weight
                        }
                    }
                    _ => f32::NAN,
                };
                output.push(value);
            }
        }
    }
    Ok(output)
}

pub fn resample_f32(request: &ResampleF32Request) -> Result<ResampleF32Response, String> {
    let values = resample_f32_values(
        &request.values,
        request.shape,
        &request.source_lat,
        &request.source_lon,
        &request.target_lat,
        &request.target_lon,
        &request.method,
    )?;
    Ok(ResampleF32Response {
        shape: [
            request.shape[0],
            request.target_lat.len(),
            request.target_lon.len(),
        ],
        values,
        method: request.method.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_handles_descending_latitude() {
        let response = resample_f32(&ResampleF32Request {
            values: vec![1.0, 2.0, 3.0, 4.0],
            shape: [1, 2, 2],
            source_lat: vec![1.0, 0.0],
            source_lon: vec![0.0, 1.0],
            target_lat: vec![0.1],
            target_lon: vec![0.9],
            method: "nearest".into(),
        })
        .unwrap();
        assert_eq!(response.values, vec![4.0]);
    }

    #[test]
    fn bilinear_interpolates_center() {
        let response = resample_f32(&ResampleF32Request {
            values: vec![0.0, 1.0, 2.0, 3.0],
            shape: [1, 2, 2],
            source_lat: vec![0.0, 1.0],
            source_lon: vec![0.0, 1.0],
            target_lat: vec![0.5],
            target_lon: vec![0.5],
            method: "bilinear".into(),
        })
        .unwrap();
        assert_eq!(response.values, vec![1.5]);
    }

    #[test]
    fn nearest_returns_nan_outside_source_bounds() {
        let response = resample_f32(&ResampleF32Request {
            values: vec![1.0, 2.0, 3.0, 4.0],
            shape: [1, 2, 2],
            source_lat: vec![0.0, 1.0],
            source_lon: vec![0.0, 1.0],
            target_lat: vec![2.0],
            target_lon: vec![2.0],
            method: "nearest".into(),
        })
        .unwrap();
        assert!(response.values[0].is_nan());
    }
    #[test]
    fn nearest_returns_nan_when_one_axis_is_outside() {
        let response = resample_f32(&ResampleF32Request {
            values: vec![1.0, 2.0, 3.0, 4.0],
            shape: [1, 2, 2],
            source_lat: vec![0.0, 1.0],
            source_lon: vec![0.0, 1.0],
            target_lat: vec![0.5],
            target_lon: vec![2.0],
            method: "nearest".into(),
        })
        .unwrap();
        assert!(response.values[0].is_nan());
    }

    #[test]
    fn rejects_non_monotonic_source_axis() {
        let error = resample_f32(&ResampleF32Request {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: [1, 2, 3],
            source_lat: vec![0.0, 1.0],
            source_lon: vec![0.0, 1.0, 0.5],
            target_lat: vec![0.5],
            target_lon: vec![0.5],
            method: "nearest".into(),
        })
        .expect_err("non-monotonic axes must be rejected");
        assert!(error.contains("strictly monotonic"));
    }
}
