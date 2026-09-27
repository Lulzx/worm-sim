//! Training-only error decomposition for shared deterministic target responses.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs};
use wormsim::{
    Result,
    bench::{self, Dataset, Partition, Split, atlas_level0::AtlasModel, atlas_training},
    codec,
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: diagnose_atlas_training GRAPH DATA SPLIT FITTED_MODEL OUTPUT.json (training only)".into());
    }
    let graph = codec::decode(&fs::read(&a[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&a[2])?;
    let split: Split = read(&a[3])?;
    let model: AtlasModel = read(&a[4])?;
    let groups = atlas_training::aggregate(&data, &graph, &split)?;
    let total = groups.iter().map(|g| g.sample_weight).sum::<f64>();
    let floor = groups
        .iter()
        .map(|g| g.sample_weight * g.irreducible_mse)
        .sum::<f64>()
        / total;
    let mut mean_energy = 0.;
    let mut group_reports = vec![];
    for g in &groups {
        let mut error = 0.;
        let mut weight = 0.;
        for trace in &g.recording.traces {
            for v in trace.values.iter().flatten() {
                let w = trace.provenance.id_confidence;
                error += w * v * v;
                weight += w;
            }
        }
        let energy = error / weight;
        mean_energy += g.sample_weight * energy / total;
        group_reports.push(serde_json::json!({"stimulated_neuron":g.stimulated_neuron,"training_trials":g.training_trials.len(),"sample_weight":g.sample_weight,"within_target_trial_mse":g.irreducible_mse,"weighted_mean_trace_energy":energy}));
    }
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let mut direct_error = 0.;
    let mut direct_weight = 0.;
    for id in &split.train {
        for trace in &indexed[id].recording.traces {
            for v in trace.values.iter().flatten() {
                let w = trace.provenance.id_confidence;
                direct_error += w * v * v;
                direct_weight += w;
            }
        }
    }
    let zero = direct_error / direct_weight;
    let decomposition_error = (zero - floor - mean_energy).abs();
    if [zero, floor, mean_energy, total, direct_weight]
        .iter()
        .any(|v| !v.is_finite())
        || decomposition_error > 1e-10
        || (total - direct_weight).abs() > 1e-8 * total
    {
        return Err("training MSE decomposition failed independent raw-trial sum".into());
    }
    let prediction = model.predict(&data, &graph, &split, Partition::Train)?;
    let score = bench::evaluate(&data, &graph, &split, &prediction, Partition::Train)?;
    let mse = score.pooled_trace_scores.mse.ok_or("no training MSE")?;
    if !mse.is_finite() || mse + 1e-10 < floor {
        return Err("shared-response fit below its mean-trace lower bound".into());
    }
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"model_source_commit":model.source_commit,"model_sha256":format!("{:x}",Sha256::digest(fs::read(&a[4]).map_err(|e|e.to_string())?)),"dataset_hash":split.dataset_hash,"split_hash":split.content_hash()?,"partition":"train","training_trials":split.train.len(),"stimulated_targets":groups.len(),"sample_weight":total,"direct_zero_response_mse":zero,"shared_target_response_lower_bound_mse":floor,"weighted_mean_trace_energy":mean_energy,"decomposition_absolute_error":decomposition_error,"model_epoch":model.epoch,"model_mse":mse,"model_excess_above_lower_bound":mse-floor,"fraction_mean_trace_energy_captured":if mean_energy>0. {Some((zero-mse)/mean_energy)} else {None},"groups":group_reports,"scope":"Training-only confidence-weighted variance decomposition. The empirical per-target mean trace minimizes MSE among shared deterministic responses; within-target variation is not an irreducible biological noise claim and may be predictable from omitted trial/animal/state/input information. No held-out labels, calibration or model selection. This is not a deployable held-out-target baseline or a generalization score."});
    fs::write(
        &a[5],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "training MSE {mse}; zero {zero}; shared-target lower bound {floor}; mean-trace energy {mean_energy}; captured {:?}",
        report["fraction_mean_trace_energy_captured"]
    );
    Ok(())
}
