//! Content-bound preprocessing evidence. Unlisted datasets remain unaudited.
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    #[default]
    Unaudited,
    Retrospective,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Assessment {
    pub status: Status,
    pub evidence_path: Option<String>,
    pub evidence_sha256: Option<String>,
    pub description: String,
}
impl Default for Assessment {
    fn default() -> Self {
        Self {status:Status::Unaudited,evidence_path:None,evidence_sha256:None,
            description:"No content-matched upstream preprocessing audit is registered. Excluding future samples in model code alone does not establish end-to-end temporal causality.".into()}
    }
}
#[derive(Deserialize)]
struct Entry {
    dataset_hash: String,
    graph_hash: String,
    status: Status,
    evidence_path: String,
    evidence_sha256: String,
    description: String,
}
const REGISTRY: &str = include_str!("../../data/benchmark-preprocessing-audits.json");
pub fn assess(dataset_hash: &str, graph_hash: &str) -> crate::Result<Assessment> {
    let entries: Vec<Entry> = serde_json::from_str(REGISTRY).map_err(|e| e.to_string())?;
    for e in entries {
        if e.dataset_hash == dataset_hash && e.graph_hash == graph_hash {
            return Ok(Assessment {
                status: e.status,
                evidence_path: Some(e.evidence_path),
                evidence_sha256: Some(e.evidence_sha256),
                description: e.description,
            });
        }
    }
    Ok(Assessment::default())
}
#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn registered_evidence_matches_content_and_cannot_transfer_to_changed_data() {
        let entries: Vec<Entry> = serde_json::from_str(REGISTRY).unwrap();
        for e in entries {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let bytes = std::fs::read(root.join(&e.evidence_path)).unwrap();
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), e.evidence_sha256);
            let receipt: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(receipt["dataset_hash"].as_str().unwrap(), e.dataset_hash);
            assert_eq!(receipt["graph_hash"].as_str().unwrap(), e.graph_hash);
            assert_eq!(
                receipt["observed_status"],
                "retrospective_whole_recording_standardization"
            );
            assert!(
                receipt["animals"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|a| a["matches_whole_recording_zscore_at_1e_minus_10"] == true)
            );
            assert_eq!(
                assess(&e.dataset_hash, &e.graph_hash).unwrap().status,
                Status::Retrospective
            );
            assert_eq!(
                assess("changed dataset", &e.graph_hash).unwrap().status,
                Status::Unaudited
            );
            assert_eq!(
                assess(&e.dataset_hash, "changed graph").unwrap().status,
                Status::Unaudited
            );
        }
        let old: Assessment =
            serde_json::from_value(serde_json::to_value(Assessment::default()).unwrap()).unwrap();
        assert_eq!(old.status, Status::Unaudited);
    }
}
