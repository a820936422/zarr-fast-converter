use serde::{Deserialize, Serialize};

fn default_skipna() -> bool {
    false
}

fn default_na_thres() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResampleF32Request {
    pub values: Vec<f32>,
    pub shape: [usize; 3],
    pub source_lat: Vec<f32>,
    pub source_lon: Vec<f32>,
    pub target_lat: Vec<f32>,
    pub target_lon: Vec<f32>,
    pub method: String,
    #[serde(default = "default_skipna")]
    pub skipna: bool,
    #[serde(default = "default_na_thres")]
    pub na_thres: f32,
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

fn validate_na_thres(na_thres: f32) -> Result<(), String> {
    if na_thres.is_finite() && (0.0..=1.0).contains(&na_thres) {
        Ok(())
    } else {
        Err("native na_thres must be finite and within [0, 1]".to_owned())
    }
}

fn weighted_value(
    values: &[f32],
    offsets: [usize; 4],
    weights: [f32; 4],
    count: usize,
    skipna: bool,
    na_thres: f32,
) -> f32 {
    let mut weighted = 0.0_f32;
    let mut valid_weight = 0.0_f32;
    let mut missing = false;
    for index in 0..count {
        let weight = weights[index];
        if weight == 0.0 {
            continue;
        }
        let value = values[offsets[index]];
        if value.is_nan() {
            missing = true;
        } else {
            weighted += value * weight;
            valid_weight += weight;
        }
    }
    if !skipna {
        return if missing { f32::NAN } else { weighted };
    }
    let minimum_valid_weight = (1.0 - na_thres).clamp(1.0e-6, 1.0 - 1.0e-6);
    if valid_weight < minimum_valid_weight {
        f32::NAN
    } else {
        weighted / valid_weight
    }
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
    resample_f32_values_with_options(
        values, shape, source_lat, source_lon, target_lat, target_lon, method, false, 1.0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resample_f32_values_with_options(
    values: &[f32],
    shape: [usize; 3],
    source_lat: &[f32],
    source_lon: &[f32],
    target_lat: &[f32],
    target_lon: &[f32],
    method: &str,
    skipna: bool,
    na_thres: f32,
) -> Result<Vec<f32>, String> {
    let output_values = shape[0]
        .checked_mul(target_lat.len())
        .and_then(|value| value.checked_mul(target_lon.len()))
        .ok_or_else(|| "target shape element count overflows usize".to_owned())?;
    let mut output = vec![0.0; output_values];
    resample_f32_values_into_with_options(
        values,
        shape,
        source_lat,
        source_lon,
        target_lat,
        target_lon,
        method,
        skipna,
        na_thres,
        &mut output,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn resample_f32_values_into(
    values: &[f32],
    shape: [usize; 3],
    source_lat: &[f32],
    source_lon: &[f32],
    target_lat: &[f32],
    target_lon: &[f32],
    method: &str,
    output: &mut [f32],
) -> Result<(), String> {
    resample_f32_values_into_with_options(
        values, shape, source_lat, source_lon, target_lat, target_lon, method, false, 1.0, output,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resample_f32_values_into_with_options(
    values: &[f32],
    shape: [usize; 3],
    source_lat: &[f32],
    source_lon: &[f32],
    target_lat: &[f32],
    target_lon: &[f32],
    method: &str,
    skipna: bool,
    na_thres: f32,
    output: &mut [f32],
) -> Result<(), String> {
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
    validate_na_thres(na_thres)?;
    if method != "nearest" && method != "bilinear" {
        return Err(format!("unsupported native resampling method: {method}"));
    }
    let lat_brackets = axis_brackets(source_lat, target_lat)?;
    let lon_brackets = axis_brackets(source_lon, target_lon)?;
    let output_values = shape[0]
        .checked_mul(target_lat.len())
        .and_then(|value| value.checked_mul(target_lon.len()))
        .ok_or_else(|| "target shape element count overflows usize".to_owned())?;
    if output.len() != output_values {
        return Err("native output buffer has an inconsistent element count".into());
    }
    let mut output_index = 0;
    for time in 0..shape[0] {
        for lat in &lat_brackets {
            for lon in &lon_brackets {
                let value = match (lat, lon, method) {
                    (Some(lat), Some(lon), "nearest") => {
                        let source_lat = if lat.weight <= 0.5 { lat.low } else { lat.high };
                        let source_lon = if lon.weight <= 0.5 { lon.low } else { lon.high };
                        let offset = source_offset(time, source_lat, source_lon, shape);
                        weighted_value(
                            values,
                            [offset, offset, offset, offset],
                            [1.0, 0.0, 0.0, 0.0],
                            1,
                            skipna,
                            na_thres,
                        )
                    }
                    (Some(lat), Some(lon), "bilinear") => {
                        let offsets = [
                            source_offset(time, lat.low, lon.low, shape),
                            source_offset(time, lat.low, lon.high, shape),
                            source_offset(time, lat.high, lon.low, shape),
                            source_offset(time, lat.high, lon.high, shape),
                        ];
                        let weights = [
                            (1.0 - lon.weight) * (1.0 - lat.weight),
                            lon.weight * (1.0 - lat.weight),
                            (1.0 - lon.weight) * lat.weight,
                            lon.weight * lat.weight,
                        ];
                        weighted_value(values, offsets, weights, 4, skipna, na_thres)
                    }
                    _ => f32::NAN,
                };
                output[output_index] = value;
                output_index += 1;
            }
        }
    }
    Ok(())
}

pub fn resample_f32(request: &ResampleF32Request) -> Result<ResampleF32Response, String> {
    let values = resample_f32_values_with_options(
        &request.values,
        request.shape,
        &request.source_lat,
        &request.source_lon,
        &request.target_lat,
        &request.target_lon,
        &request.method,
        request.skipna,
        request.na_thres,
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
            skipna: false,
            na_thres: 1.0,
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
            skipna: false,
            na_thres: 1.0,
        })
        .unwrap();
        assert_eq!(response.values, vec![1.5]);
    }
    #[test]
    fn writable_kernel_matches_allocating_kernel() {
        let request = ResampleF32Request {
            values: vec![0.0, 1.0, 2.0, 3.0],
            shape: [1, 2, 2],
            source_lat: vec![0.0, 1.0],
            source_lon: vec![0.0, 1.0],
            target_lat: vec![0.25, 0.75],
            target_lon: vec![0.25, 0.75],
            method: "bilinear".into(),
            skipna: false,
            na_thres: 1.0,
        };
        let expected = resample_f32(&request).unwrap().values;
        let mut output = vec![0.0; expected.len()];
        resample_f32_values_into(
            &request.values,
            request.shape,
            &request.source_lat,
            &request.source_lon,
            &request.target_lat,
            &request.target_lon,
            &request.method,
            &mut output,
        )
        .unwrap();
        assert_eq!(output, expected);
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
            skipna: false,
            na_thres: 1.0,
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
            skipna: false,
            na_thres: 1.0,
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
            skipna: false,
            na_thres: 1.0,
        })
        .expect_err("non-monotonic axes must be rejected");
        assert!(error.contains("strictly monotonic"));
    }

    #[test]
    fn bilinear_skipna_renormalizes_valid_weights() {
        let values = [f32::NAN, 2.0, 4.0, 6.0];
        let result = resample_f32_values_with_options(
            &values,
            [1, 2, 2],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.5],
            &[0.5],
            "bilinear",
            true,
            1.0,
        )
        .expect("skipna result");
        assert_eq!(result, vec![4.0]);

        let strict = resample_f32_values_with_options(
            &values,
            [1, 2, 2],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.5],
            &[0.5],
            "bilinear",
            true,
            0.0,
        )
        .expect("strict result");
        assert!(strict[0].is_nan());
    }

    #[test]
    fn skipna_returns_missing_for_all_missing_source_window() {
        let result = resample_f32_values_with_options(
            &[f32::NAN, f32::NAN, f32::NAN, f32::NAN],
            [1, 2, 2],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.5],
            &[0.5],
            "bilinear",
            true,
            1.0,
        )
        .expect("all missing result");
        assert!(result[0].is_nan());
    }
}
