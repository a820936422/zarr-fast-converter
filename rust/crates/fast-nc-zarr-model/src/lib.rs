use serde::{Deserialize, Serialize};

pub const BACKEND_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapability {
    pub backend: String,
    pub protocol_version: u32,
    pub crate_version: String,
    pub operations: Vec<String>,
}

impl BackendCapability {
    pub fn smoke() -> Self {
        Self {
            backend: "rust".to_owned(),
            protocol_version: BACKEND_PROTOCOL_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            operations: vec!["probe".to_owned()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendCapability, BACKEND_PROTOCOL_VERSION};

    #[test]
    fn smoke_capability_has_stable_protocol() {
        let capability = BackendCapability::smoke();
        assert_eq!(capability.backend, "rust");
        assert_eq!(capability.protocol_version, BACKEND_PROTOCOL_VERSION);
        assert_eq!(capability.operations, ["probe"]);
    }
}
