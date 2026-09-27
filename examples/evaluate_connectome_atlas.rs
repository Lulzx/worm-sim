//! Evaluate a frozen atlas LDS, with separately clustered uncertainty.
use std::{collections::BTreeMap, fs, path::Path};
use wormsim::{
    Result,
    bench::{self, Dataset, Partition, Split, atlas, atlas_uncertainty, connectome_fit::Model},
    codec,
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn write(p: impl AsRef<Path>, v: &impl serde::Serialize) -> Result<()> {
    fs::write(p, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 8 {
        return Err("usage: evaluate_connectome_atlas GRAPH DATA SPLIT EVIDENCE MODEL validation|test NEW_OUTPUT_DIR".into());
    }
    let graph = codec::decode(&fs::read(&a[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&a[2])?;
    let split: Split = read(&a[3])?;
    let evidence: atlas::Evidence = read(&a[4])?;
    let model: Model = read(&a[5])?;
    let partition = match a[6].as_str() {
        "validation" => Partition::Validation,
        "test" => Partition::Test,
        _ => return Err("invalid partition".into()),
    };
    let prediction = model.predict(&data, &graph, &split, partition)?;
    let trace_report = bench::evaluate(&data, &graph, &split, &prediction, partition)?;
    evidence.validate(&data, &graph)?;
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let mut scores = BTreeMap::new();
    let grid = &prediction.trials.first().ok_or("empty partition")?.times;
    for trial in &prediction.trials {
        if &trial.times != grid {
            return Err("pair ranking requires the same response grid for all trials".into());
        }
        let target = indexed[&trial.id]
            .stimulated_neuron
            .as_ref()
            .ok_or("missing stimulus")?;
        for (neuron, values) in &trial.fluorescence {
            let value = values.iter().map(|v| v.abs()).sum::<f64>() * model.dynamics.sample_dt;
            if let Some(previous) = scores.insert((target.clone(), neuron.clone()), value)
                && previous != value
            {
                return Err("inconsistent impulse scores across trials".into());
            }
        }
    }
    let pairs = evidence
        .partition_pairs(&data, &split, partition)?
        .iter()
        .map(|p| {
            Ok(atlas::Prediction {
                stimulated: p.stimulated.clone(),
                responding: p.responding.clone(),
                score: *scores
                    .get(&(p.stimulated.clone(), p.responding.clone()))
                    .ok_or("missing pair score")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let pair_prediction = atlas::Predictions {
        evidence_hash: evidence.content_hash()?,
        split_hash: model.split_hash.clone(),
        model: prediction.model.clone(),
        free_parameters: model.free_parameters(),
        source_commit: model.source_commit.clone(),
        training_trials: model.training_trials.clone(),
        selection_trials: model.selection_trials.clone(),
        pairs,
    };
    let pair_report = atlas::evaluate(
        &evidence,
        &data,
        &graph,
        &split,
        &pair_prediction,
        partition,
    )?;
    let trace_bootstrap = atlas_uncertainty::traces(&data, &trace_report, 42, 2000)?;
    let pair_bootstrap = atlas_uncertainty::pairs(&evidence, &pair_prediction, 42, 2000)?;
    let out = Path::new(&a[7]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    write(out.join("pair-predictions.json"), &pair_prediction)?;
    write(out.join("pair-report.json"), &pair_report)?;
    write(out.join("trace-report.json"), &trace_report)?;
    let report = serde_json::json!({"schema_version":1,"evaluation_source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"model_source_commit":model.source_commit,"selected_iteration":model.iteration,"dataset_hash":model.dataset_hash,"split_hash":model.split_hash,"evidence_hash":evidence.content_hash()?,"partition":partition,"pair_score":"dt times sum of absolute predicted fluorescence on the common post-stimulation grid; one score per ordered non-self pair; no fitted classifier or q-based tuning","trace_bootstrap":trace_bootstrap,"pair_bootstrap":pair_bootstrap,"limitations":"Separate percentile 95% cluster bootstraps, conditional on the frozen fit. Target and recording dependence is crossed; separate marginal intervals do not account for both simultaneously. Recording IDs are not verified animal IDs. Aggregate published pair labels have no event-level recording decomposition. Undefined correlations and one-class AUROC draws are excluded and their defined counts are reported. No claim of biological acceptance or reproduction of original Creamer fitting."});
    write(out.join("uncertainty.json"), &report)?;
    println!(
        "MSE {:?}; trace correlation {:?}; pair AUROC {:?}",
        trace_report.pooled_trace_scores.mse,
        trace_report.macro_trace_correlation,
        pair_report.auroc.value
    );
    Ok(())
}
