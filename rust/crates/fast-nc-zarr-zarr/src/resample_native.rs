use ndarray::Array3;
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

fn bracket(axis: &[f32], value: f32) -> Option<(usize, usize, f32)> {
    if axis.is_empty()
        || value < axis.iter().copied().fold(f32::INFINITY, f32::min)
        || value > axis.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    {
        return None;
    }
    let descending = axis.first()? > axis.last()?;
    let mut ordered = axis.to_vec();
    if descending {
        ordered.reverse();
    }
    let upper = ordered.partition_point(|item| *item < value);
    if upper == 0 {
        return Some((0, 0, 0.0));
    }
    if upper >= ordered.len() {
        let last = ordered.len() - 1;
        return Some((last, last, 0.0));
    }
    let low = upper - 1;
    let high = upper;
    let weight = (value - ordered[low]) / (ordered[high] - ordered[low]);
    let map = |index: usize| {
        if descending {
            axis.len() - 1 - index
        } else {
            index
        }
    };
    Some((map(low), map(high), weight))
}

pub fn resample_f32(request: &ResampleF32Request) -> Result<ResampleF32Response, String> {
    if request.shape[0] == 0
        || request.shape[1] != request.source_lat.len()
        || request.shape[2] != request.source_lon.len()
        || request.values.len() != request.shape.iter().product::<usize>()
    {
        return Err("source shape, axes and values are inconsistent".into());
    }
    if request.method != "nearest" && request.method != "bilinear" {
        return Err(format!(
            "unsupported native resampling method: {}",
            request.method
        ));
    }
    let source = Array3::from_shape_vec(request.shape, request.values.clone())
        .map_err(|error| error.to_string())?;
    let mut output =
        Vec::with_capacity(request.shape[0] * request.target_lat.len() * request.target_lon.len());
    for time in 0..request.shape[0] {
        for &lat in &request.target_lat {
            for &lon in &request.target_lon {
                let in_bounds = |axis: &[f32], value: f32| {
                    axis.first()
                        .zip(axis.last())
                        .map(|(first, last)| {
                            let (minimum, maximum) = if first <= last {
                                (*first, *last)
                            } else {
                                (*last, *first)
                            };
                            value >= minimum && value <= maximum
                        })
                        .unwrap_or(false)
                };
                let nearest_in_bounds =
                    in_bounds(&request.source_lat, lat) && in_bounds(&request.source_lon, lon);

                let value = if request.method == "nearest" && !nearest_in_bounds {
                    f32::NAN
                } else if request.method == "nearest" {
                    let lat_index = request
                        .source_lat
                        .iter()
                        .enumerate()
                        .min_by(|(_, left), (_, right)| {
                            (f32::abs(**left - lat)).total_cmp(&f32::abs(**right - lat))
                        })
                        .map(|(index, _)| index);
                    let lon_index = request
                        .source_lon
                        .iter()
                        .enumerate()
                        .min_by(|(_, left), (_, right)| {
                            (f32::abs(**left - lon)).total_cmp(&f32::abs(**right - lon))
                        })
                        .map(|(index, _)| index);
                    match (lat_index, lon_index) {
                        (Some(y), Some(x)) => source[[time, y, x]],
                        _ => f32::NAN,
                    }
                } else {
                    let Some((y0, y1, wy)) = bracket(&request.source_lat, lat) else {
                        output.push(f32::NAN);
                        continue;
                    };
                    let Some((x0, x1, wx)) = bracket(&request.source_lon, lon) else {
                        output.push(f32::NAN);
                        continue;
                    };
                    let a = source[[time, y0, x0]];
                    let b = source[[time, y0, x1]];
                    let c = source[[time, y1, x0]];
                    let d = source[[time, y1, x1]];
                    if [a, b, c, d].iter().any(|value| value.is_nan()) {
                        f32::NAN
                    } else {
                        a * (1.0 - wx) * (1.0 - wy)
                            + b * wx * (1.0 - wy)
                            + c * (1.0 - wx) * wy
                            + d * wx * wy
                    }
                };
                output.push(value);
            }
        }
    }
    Ok(ResampleF32Response {
        shape: [
            request.shape[0],
            request.target_lat.len(),
            request.target_lon.len(),
        ],
        values: output,
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
}
