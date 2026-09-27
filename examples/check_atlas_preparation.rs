//! Validation-only metric sensitivity to preparation duration and integration dt.
use sha2::{Digest, Sha256};
use std::fs;
use wormsim::{
    Result,
    bench::{self, Dataset, Partition, Split, atlas, atlas_level0::AtlasModel},
    codec, initial_state,
    model::Model,
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 {
        return Err("usage: check_atlas_preparation GRAPH DATA SPLIT EVIDENCE MODEL OUTPUT.json (validation only)".into());
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&args[2])?;
    let split: Split = read(&args[3])?;
    let evidence: atlas::Evidence = read(&args[4])?;
    let original: AtlasModel = read(&args[5])?;
    if original.config.preparation_seconds <= 0. {
        return Err("positive preparation required".into());
    }
    let network = Model::new(graph.clone())?;
    let params = original.parameters.expand(&network)?;
    let mut reports = vec![];
    for (duration_factor, dt_factor) in [(1., 1.), (2., 1.), (4., 1.), (4., 0.5)] {
        let mut model = original.clone();
        model.config.preparation_seconds *= duration_factor;
        model.config.dt *= dt_factor;
        let predictions = model.predict(&data, &graph, &split, Partition::Validation)?;
        let trace = bench::evaluate(&data, &graph, &split, &predictions, Partition::Validation)?;
        let pairs = atlas::rank_responses(
            &evidence,
            &data,
            &graph,
            &split,
            &predictions,
            Partition::Validation,
        )?;
        let classification = atlas::evaluate(
            &evidence,
            &data,
            &graph,
            &split,
            &pairs,
            Partition::Validation,
        )?;
        let (_, derivative) = initial_state::prepared_state(
            &network,
            &params,
            &model.initial,
            model.config.dt,
            model.config.preparation_seconds,
        )?;
        reports.push(serde_json::json!({"preparation_seconds":model.config.preparation_seconds,"dt":model.config.dt,"mse":trace.pooled_trace_scores.mse,"correlation":trace.macro_trace_correlation,"defined_correlations":trace.defined_trace_correlations,"pair_auroc":classification.auroc.value,"unforced_derivative_l2":derivative.iter().map(|v|v*v).sum::<f64>().sqrt()}));
    }
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"model_source_commit":original.source_commit,"model_sha256":format!("{:x}",Sha256::digest(fs::read(&args[5]).map_err(|e|e.to_string())?)),"epoch":original.epoch,"dataset_hash":split.dataset_hash,"split_hash":split.content_hash()?,"partition":"validation","settings":reports,"scope":"Frozen parameters, validation-only numerical sensitivity; no refitting or test predictions. Duration/dt overrides are diagnostics, not replacements for the selected checkpoint. Correlation and ranking can magnify tiny residual responses even when MSE is stable."});
    fs::write(
        &args[6],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
