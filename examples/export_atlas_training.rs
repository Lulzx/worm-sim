//! Export authoritative training-only sufficient statistics for external fitters.
use sha2::{Digest, Sha256};
use std::{fs, io::Write};
use wormsim::{
    Result,
    bench::{
        Dataset, Split, atlas, atlas_classification, atlas_level0::AtlasModel, atlas_training,
    },
    codec,
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 && a.len() != 7 {
        return Err(
            "usage: export_atlas_training GRAPH DATA SPLIT MODEL NEW_OUTPUT.json [EVIDENCE]".into(),
        );
    }
    let graph = codec::decode(&fs::read(&a[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&a[2])?;
    let split: Split = read(&a[3])?;
    let model: AtlasModel = read(&a[4])?;
    model.validate(&data, &graph, &split)?;
    let labels = if let Some(path) = a.get(6) {
        let evidence: atlas::Evidence = read(path)?;
        let labels = atlas_classification::training_labels(&evidence, &data, &graph, &split)?;
        if model.classification_evidence_hash.as_ref() != Some(&labels.evidence_hash) {
            return Err("evidence differs from checkpoint".into());
        }
        Some(labels)
    } else {
        None
    };
    if model.classifier.is_some() != labels.is_some() {
        return Err("classifier requires matching evidence".into());
    }
    let mut groups = vec![];
    for g in atlas_training::aggregate(&data, &graph, &split)? {
        let target = graph.neuron(&g.stimulated_neuron)?;
        groups.push(serde_json::json!({"target":target,"recording":g.recording,"sample_weight":g.sample_weight,"irreducible_mse":g.irreducible_mse,"training_trials":g.training_trials,"labels":labels.as_ref().and_then(|l|l.by_target.get(&target)).cloned().unwrap_or_default()}));
    }
    let probabilities = match &model.config.molecular_sign_priors {
        Some(p) => p.probabilities(&graph)?,
        None => graph.chemical.iter().map(|e| e.3).collect(),
    };
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"model_sha256":format!("{:x}",Sha256::digest(fs::read(&a[4]).map_err(|e|e.to_string())?)),"graph_hash":graph.hash,"dataset_hash":split.dataset_hash,"split_hash":split.content_hash()?,"names":graph.names,"training_trials":split.train,"sign_probabilities":probabilities,"classification_pairs":labels.as_ref().map_or(0,|l|l.pairs),"groups":groups});
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a[5])
        .map_err(|e| e.to_string())?
        .write_all(&serde_json::to_vec(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    Ok(())
}
