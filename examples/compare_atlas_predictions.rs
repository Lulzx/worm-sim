//! Paired cluster comparison of two frozen response prediction artifacts.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs};
use wormsim::{
    Result,
    bench::{self, Dataset, Partition, Predictions, Split, atlas, atlas_uncertainty},
    codec,
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 9 {
        return Err("usage: compare_atlas_predictions GRAPH DATA SPLIT EVIDENCE A_PREDICTIONS B_PREDICTIONS validation|test OUTPUT.json".into());
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&args[2])?;
    let split: Split = read(&args[3])?;
    let evidence: atlas::Evidence = read(&args[4])?;
    let a: Predictions = read(&args[5])?;
    let b: Predictions = read(&args[6])?;
    let partition = match args[7].as_str() {
        "validation" => Partition::Validation,
        "test" => Partition::Test,
        _ => return Err("invalid partition".into()),
    };
    let ar = bench::evaluate(&data, &graph, &split, &a, partition)?;
    let br = bench::evaluate(&data, &graph, &split, &b, partition)?;
    let ap = atlas::rank_responses(&evidence, &data, &graph, &split, &a, partition)?;
    let bp = atlas::rank_responses(&evidence, &data, &graph, &split, &b, partition)?;
    let pair_a = atlas::evaluate(&evidence, &data, &graph, &split, &ap, partition)?;
    let pair_b = atlas::evaluate(&evidence, &data, &graph, &split, &bp, partition)?;
    let trace = atlas_uncertainty::trace_difference(&data, &ar, &br, 42, 2000)?;
    let pair = atlas_uncertainty::pair_difference(&evidence, &ap, &bp, 42, 2000)?;
    let indexed: BTreeMap<_, _> = br
        .traces
        .iter()
        .map(|t| ((&t.trial, &t.neuron), t))
        .collect();
    let common = ar
        .traces
        .iter()
        .filter(|t| {
            t.scores.correlation.is_some()
                && indexed[&(&t.trial, &t.neuron)].scores.correlation.is_some()
        })
        .count();
    let hash = |p: &str| -> Result<String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(fs::read(p).map_err(|e| e.to_string())?)
        ))
    };
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"dataset_hash":split.dataset_hash,"split_hash":split.content_hash()?,"evidence_hash":evidence.content_hash()?,"partition":partition,"a":{"model":a.model,"source_commit":a.source_commit,"prediction_sha256":hash(&args[5])?,"free_parameters":a.free_parameters,"mse":ar.pooled_trace_scores.mse,"correlation":ar.macro_trace_correlation,"defined_trace_correlations":ar.defined_trace_correlations,"pair_auroc":pair_a.auroc.value},"b":{"model":b.model,"source_commit":b.source_commit,"prediction_sha256":hash(&args[6])?,"free_parameters":b.free_parameters,"mse":br.pooled_trace_scores.mse,"correlation":br.macro_trace_correlation,"defined_trace_correlations":br.defined_trace_correlations,"pair_auroc":pair_b.auroc.value},"common_defined_trace_correlations":common,"trace_difference_a_minus_b":trace,"pair_difference_a_minus_b":pair,"interpretation":"Paired percentile 95% cluster bootstrap with 2000 common draws and seed 42. Negative MSE difference favors A; positive correlation/AUROC difference favors A. Correlation differences use only traces defined for both models. Target and recording resampling are separate marginal analyses, not a crossed-cluster correction. Recording IDs are not verified animals. Conditional on frozen fits; no refitting in bootstrap. Previously inspected test cohort; exploratory comparison."});
    fs::write(
        &args[8],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
