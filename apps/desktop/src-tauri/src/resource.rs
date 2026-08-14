use serde::Serialize;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceSnapshot {
    pub captured_at_ms: u64,
    pub logical_cpus: usize,
    pub memory_total_bytes: u64,
    pub memory_available_bytes: u64,
}

pub fn snapshot() -> ResourceSnapshot {
    let (memory_total_bytes, memory_available_bytes) = memory_limits();
    ResourceSnapshot {
        captured_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
            .unwrap_or_default(),
        logical_cpus: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
        memory_total_bytes,
        memory_available_bytes,
    }
}

fn memory_limits() -> (u64, u64) {
    let Ok(contents) = fs::read_to_string("/proc/meminfo") else {
        return (0, 0);
    };
    let mut total = 0;
    let mut available = 0;
    for line in contents.lines() {
        let mut fields = line.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(value) = fields.next().and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        let bytes = value.saturating_mul(1024);
        match name {
            "MemTotal:" => total = bytes,
            "MemAvailable:" => available = bytes,
            _ => {}
        }
    }
    (total, available)
}

#[cfg(test)]
mod tests {
    use super::snapshot;

    #[test]
    fn snapshot_has_stable_cpu_and_timestamp_fields() {
        let resource = snapshot();
        assert!(resource.logical_cpus > 0);
        assert!(resource.captured_at_ms > 0);
    }
}
