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

#[derive(Clone, Copy)]
enum AxisMeasure {
    Latitude,
    Longitude,
}

#[derive(Clone, Copy)]
struct AxisOverlap {
    source_index: usize,
    measure: f64,
}

struct TargetAxisOverlap {
    overlaps: Vec<AxisOverlap>,
    target_measure: f64,
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

fn regular_axis_bounds(axis: &[f32], name: &str) -> Result<Vec<f64>, String> {
    validate_axis(axis, name)?;
    let coordinates = axis
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let resolution = (coordinates[1] - coordinates[0]).abs();
    let scale = coordinates
        .iter()
        .fold(resolution.max(1.0), |current, value| {
            current.max(value.abs())
        });
    let absolute_tolerance = (resolution * 1.0e-6)
        .max(f64::from(f32::EPSILON) * scale * 2.0)
        .max(1.0e-10);
    let tolerance = resolution * 1.0e-5 + absolute_tolerance;
    if coordinates
        .windows(2)
        .any(|pair| (pair[1] - pair[0] - coordinates[1] + coordinates[0]).abs() > tolerance)
    {
        return Err(format!("{name} axis must be regular"));
    }
    let mut bounds = Vec::with_capacity(coordinates.len() + 1);
    bounds.push(coordinates[0] - (coordinates[1] - coordinates[0]) / 2.0);
    bounds.extend(coordinates.windows(2).map(|pair| (pair[0] + pair[1]) / 2.0));
    let last = coordinates.len() - 1;
    bounds.push(coordinates[last] + (coordinates[last] - coordinates[last - 1]) / 2.0);
    Ok(bounds)
}

fn aligned_target_bounds(
    source_axis: &[f32],
    target_axis: &[f32],
    name: &str,
) -> Result<Vec<f64>, String> {
    let mut target_bounds = regular_axis_bounds(target_axis, name)?;
    let source_bounds = regular_axis_bounds(source_axis, name)?;
    let target_resolution = (f64::from(target_axis[1]) - f64::from(target_axis[0])).abs();
    let source_low = source_bounds.iter().copied().fold(f64::INFINITY, f64::min);
    let raw_low = target_bounds.iter().copied().fold(f64::INFINITY, f64::min);
    let aligned_low =
        source_low + ((raw_low - source_low) / target_resolution).round() * target_resolution;
    if (aligned_low - raw_low).abs() <= target_resolution * 1.0e-4 {
        let shift = aligned_low - raw_low;
        for bound in &mut target_bounds {
            *bound += shift;
        }
    }
    Ok(target_bounds)
}

fn interval_measure(low: f64, high: f64, measure: AxisMeasure) -> f64 {
    match measure {
        AxisMeasure::Latitude => (high.to_radians().sin() - low.to_radians().sin()).abs(),
        AxisMeasure::Longitude => (high - low).abs().to_radians(),
    }
}

fn build_axis_overlaps(
    source_bounds: &[f64],
    target_bounds: &[f64],
    measure: AxisMeasure,
) -> Vec<TargetAxisOverlap> {
    let mut source_cells = source_bounds
        .windows(2)
        .enumerate()
        .map(|(source_index, bounds)| {
            (
                bounds[0].min(bounds[1]),
                bounds[0].max(bounds[1]),
                source_index,
            )
        })
        .collect::<Vec<_>>();
    source_cells.sort_by(|left, right| left.0.total_cmp(&right.0));

    target_bounds
        .windows(2)
        .map(|bounds| {
            let target_low = bounds[0].min(bounds[1]);
            let target_high = bounds[0].max(bounds[1]);
            let tolerance = ((target_high - target_low).abs() * 1.0e-5).max(1.0e-7);
            let first = source_cells.partition_point(|cell| cell.1 < target_low - tolerance);
            let mut overlaps = Vec::new();
            for &(source_low, source_high, source_index) in &source_cells[first..] {
                if source_low > target_high + tolerance {
                    break;
                }
                let overlap_low = source_low.max(target_low);
                let overlap_high = source_high.min(target_high);
                let measure = if overlap_high > overlap_low {
                    interval_measure(overlap_low, overlap_high, measure)
                } else if (source_high - target_low).abs() <= tolerance
                    || (source_low - target_high).abs() <= tolerance
                {
                    0.0
                } else {
                    continue;
                };
                overlaps.push(AxisOverlap {
                    source_index,
                    measure,
                });
            }
            TargetAxisOverlap {
                overlaps,
                target_measure: interval_measure(target_low, target_high, measure),
            }
        })
        .collect()
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

#[allow(clippy::too_many_arguments)]
fn conservative_value(
    values: &[f32],
    time: usize,
    shape: [usize; 3],
    lat: &TargetAxisOverlap,
    lon: &TargetAxisOverlap,
    normed: bool,
    skipna: bool,
    na_thres: f32,
) -> f32 {
    let covered_lat = lat
        .overlaps
        .iter()
        .map(|overlap| overlap.measure)
        .sum::<f64>();
    let covered_lon = lon
        .overlaps
        .iter()
        .map(|overlap| overlap.measure)
        .sum::<f64>();
    let covered_area = covered_lat * covered_lon;
    let target_area = lat.target_measure * lon.target_measure;
    let denominator = if normed { covered_area } else { target_area };
    if covered_area <= 0.0 || denominator <= 0.0 {
        return f32::NAN;
    }

    let mut weighted = 0.0_f64;
    let mut valid_weight = 0.0_f64;
    let mut missing = false;
    let mut strict_boundary_missing = false;
    for lat_overlap in &lat.overlaps {
        for lon_overlap in &lon.overlaps {
            let value = values[source_offset(
                time,
                lat_overlap.source_index,
                lon_overlap.source_index,
                shape,
            )];
            let weight = lat_overlap.measure * lon_overlap.measure / denominator;
            if weight == 0.0 {
                // xESMF assigns boundary-touching cells a tiny nonzero weight;
                // with skipna disabled, a NaN there must still mask the output.
                if !skipna {
                    if value.is_nan() {
                        missing = true;
                    }
                } else if na_thres <= 0.0 && value.is_nan() {
                    strict_boundary_missing = true;
                }
                continue;
            }
            if value.is_nan() {
                missing = true;
            } else {
                weighted += f64::from(value) * weight;
                valid_weight += weight;
            }
        }
    }
    if !skipna {
        return if missing { f32::NAN } else { weighted as f32 };
    }
    if strict_boundary_missing {
        return f32::NAN;
    }
    let minimum_valid_fraction = f64::from((1.0 - na_thres).clamp(1.0e-6, 1.0 - 1.0e-6));
    let valid_fraction = valid_weight * denominator / covered_area;
    if valid_fraction < minimum_valid_fraction {
        f32::NAN
    } else {
        (weighted / valid_weight) as f32
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
    if !matches!(
        method,
        "nearest" | "bilinear" | "conservative" | "conservative_normed"
    ) {
        return Err(format!("unsupported native resampling method: {method}"));
    }
    let output_values = shape[0]
        .checked_mul(target_lat.len())
        .and_then(|value| value.checked_mul(target_lon.len()))
        .ok_or_else(|| "target shape element count overflows usize".to_owned())?;
    if output.len() != output_values {
        return Err("native output buffer has an inconsistent element count".into());
    }

    if matches!(method, "conservative" | "conservative_normed") {
        let source_lat_bounds = regular_axis_bounds(source_lat, "source latitude")?;
        let source_lon_bounds = regular_axis_bounds(source_lon, "source longitude")?;
        let target_lat_bounds = aligned_target_bounds(source_lat, target_lat, "target latitude")?;
        let target_lon_bounds = aligned_target_bounds(source_lon, target_lon, "target longitude")?;
        let lat_overlaps = build_axis_overlaps(
            &source_lat_bounds,
            &target_lat_bounds,
            AxisMeasure::Latitude,
        );
        let lon_overlaps = build_axis_overlaps(
            &source_lon_bounds,
            &target_lon_bounds,
            AxisMeasure::Longitude,
        );
        let normed = method == "conservative_normed";
        let mut output_index = 0;
        for time in 0..shape[0] {
            for lat in &lat_overlaps {
                for lon in &lon_overlaps {
                    output[output_index] =
                        conservative_value(values, time, shape, lat, lon, normed, skipna, na_thres);
                    output_index += 1;
                }
            }
        }
        return Ok(());
    }

    let lat_brackets = axis_brackets(source_lat, target_lat)?;
    let lon_brackets = axis_brackets(source_lon, target_lon)?;
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

    fn assert_close(left: f32, right: f32) {
        assert!(
            (left - right).abs() <= 1.0e-6,
            "expected {right}, received {left}"
        );
    }

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

    #[test]
    fn conservative_is_equivalent_for_ascending_and_descending_axes() {
        let ascending = resample_f32_values(
            &[1.0, 2.0, 3.0, 4.0],
            [1, 2, 2],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            "conservative",
        )
        .unwrap();
        let descending = resample_f32_values(
            &[4.0, 3.0, 2.0, 1.0],
            [1, 2, 2],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            "conservative",
        )
        .unwrap();
        assert_eq!(ascending, descending);
    }

    #[test]
    fn conservative_preserves_a_constant_field_when_coarsening() {
        let source_axis = [-1.5, -0.5, 0.5, 1.5];
        let target_axis = [-1.0, 1.0];
        let result = resample_f32_values(
            &[7.0; 16],
            [1, 4, 4],
            &source_axis,
            &source_axis,
            &target_axis,
            &target_axis,
            "conservative",
        )
        .unwrap();
        assert_eq!(result, vec![7.0; 4]);
    }

    #[test]
    fn conservative_coarsening_uses_spherical_latitude_area() {
        let source_lat = [0.5, 1.5, 2.5, 3.5];
        let source_lon = [0.5, 1.5, 2.5, 3.5];
        let target_axis = [1.0, 3.0];
        let values = [
            1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0,
        ];
        let result = resample_f32_values(
            &values,
            [1, 4, 4],
            &source_lat,
            &source_lon,
            &target_axis,
            &target_axis,
            "conservative",
        )
        .unwrap();
        let first_lat = 1.0_f64.to_radians().sin() - 0.0_f64.to_radians().sin();
        let second_lat = 2.0_f64.to_radians().sin() - 1.0_f64.to_radians().sin();
        let expected = ((first_lat + 3.0 * second_lat) / (first_lat + second_lat)) as f32;
        assert_close(result[0], expected);
    }

    #[test]
    fn conservative_normed_normalizes_partial_coverage() {
        let source_lat = [-0.5, 0.5];
        let source_lon = [0.5, 1.5];
        let target_lon = [-0.5, 1.5];
        let conservative = resample_f32_values(
            &[4.0; 4],
            [1, 2, 2],
            &source_lat,
            &source_lon,
            &source_lat,
            &target_lon,
            "conservative",
        )
        .unwrap();
        let normed = resample_f32_values(
            &[4.0; 4],
            [1, 2, 2],
            &source_lat,
            &source_lon,
            &source_lat,
            &target_lon,
            "conservative_normed",
        )
        .unwrap();
        assert_close(conservative[0], 1.0);
        assert_close(conservative[1], 3.0);
        assert_eq!(normed, vec![4.0; 4]);
    }

    #[test]
    fn conservative_missing_values_follow_skipna_thresholds() {
        let source_axis = [0.5, 1.5];
        let target_axis = [1.0, 3.0];
        let values = [f32::NAN, 4.0, 4.0, 4.0];
        for (threshold, missing) in [(0.0, true), (0.5, false), (1.0, false)] {
            let result = resample_f32_values_with_options(
                &values,
                [1, 2, 2],
                &source_axis,
                &source_axis,
                &target_axis,
                &target_axis,
                "conservative",
                true,
                threshold,
            )
            .unwrap();
            assert_eq!(result[0].is_nan(), missing);
            if !missing {
                assert_close(result[0], 4.0);
            }
        }

        let without_skipna = resample_f32_values_with_options(
            &values,
            [1, 2, 2],
            &source_axis,
            &source_axis,
            &target_axis,
            &target_axis,
            "conservative",
            false,
            1.0,
        )
        .unwrap();
        assert!(without_skipna[0].is_nan());

        let all_missing = resample_f32_values_with_options(
            &[f32::NAN; 4],
            [1, 2, 2],
            &source_axis,
            &source_axis,
            &target_axis,
            &target_axis,
            "conservative_normed",
            true,
            1.0,
        )
        .unwrap();
        assert!(all_missing[0].is_nan());
    }

    #[test]
    fn conservative_boundary_touch_nan_propagates_without_skipna() {
        let source_lat = [-6.925_f32, -6.875, -6.825, -6.775];
        let source_lon = [12.675_f32, 12.725, 12.775];
        let target_lat = [-6.95_f32, -6.85];
        let target_lon = [12.65_f32, 12.75];
        let mut values = [7.0_f32; 4 * 3];
        // A NaN on the source cell that only touches the target cell boundary.
        values[0] = f32::NAN;
        for method in ["conservative", "conservative_normed"] {
            let result = resample_f32_values_with_options(
                &values,
                [1, 4, 3],
                &source_lat,
                &source_lon,
                &target_lat,
                &target_lon,
                method,
                false,
                1.0,
            )
            .unwrap();
            assert!(
                result[3].is_nan(),
                "{method} must propagate boundary-touch NaN"
            );
        }

        values[0] = 9.0;
        for method in ["conservative", "conservative_normed"] {
            let result = resample_f32_values_with_options(
                &values,
                [1, 4, 3],
                &source_lat,
                &source_lon,
                &target_lat,
                &target_lon,
                method,
                false,
                1.0,
            )
            .unwrap();
            assert!(
                !result[3].is_nan(),
                "{method} finite touching cell must not mask the output"
            );
        }
    }

    #[test]
    fn conservative_returns_nan_without_source_coverage() {
        let result = resample_f32_values(
            &[1.0, 2.0, 3.0, 4.0],
            [1, 2, 2],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[10.0, 11.0],
            &[10.0, 11.0],
            "conservative",
        )
        .unwrap();
        assert!(result.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn conservative_writable_kernel_matches_allocating_kernel() {
        let source_axis = [0.5, 1.5];
        let target_axis = [1.0, 3.0];
        let values = [1.0, 2.0, 3.0, 4.0];
        let expected = resample_f32_values_with_options(
            &values,
            [1, 2, 2],
            &source_axis,
            &source_axis,
            &target_axis,
            &target_axis,
            "conservative_normed",
            true,
            0.5,
        )
        .unwrap();
        let mut output = vec![0.0; expected.len()];
        resample_f32_values_into_with_options(
            &values,
            [1, 2, 2],
            &source_axis,
            &source_axis,
            &target_axis,
            &target_axis,
            "conservative_normed",
            true,
            0.5,
            &mut output,
        )
        .unwrap();
        for (actual, expected) in output.iter().zip(&expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_close(*actual, *expected);
            }
        }
    }

    #[test]
    fn conservative_rejects_irregular_axes_and_invalid_options() {
        let error = resample_f32_values(
            &[1.0; 6],
            [1, 2, 3],
            &[0.0, 1.0],
            &[0.0, 1.0, 3.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            "conservative",
        )
        .expect_err("irregular source axes must be rejected");
        assert!(error.contains("regular"));

        let invalid_threshold = resample_f32_values_with_options(
            &[1.0; 4],
            [1, 2, 2],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            "conservative",
            true,
            1.1,
        )
        .expect_err("invalid thresholds must be rejected");
        assert!(invalid_threshold.contains("within [0, 1]"));

        let unknown_method = resample_f32_values(
            &[1.0; 4],
            [1, 2, 2],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            "conservative_fast",
        )
        .expect_err("unknown methods must be rejected");
        assert!(unknown_method.contains("unsupported native resampling method"));
    }
}
