//! Audit the common behavior protocol across every imported window, without model scoring.
use std::fs;
use wormsim::{
    Result,
    bench::{self, Dataset, Split, behavior},
    codec,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: audit_behavior_inputs GRAPH.wsc DATA.json SPLIT.json OUTPUT.json".into(),
        );
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let split: Split = serde_json::from_slice(&fs::read(&args[3]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let names = ["angular_velocity", "head_angle", "pumping", "velocity"].map(str::to_string);
    let model = behavior::fit(&data, &graph, &split, &names)?;
    let mut counts = std::collections::BTreeMap::new();
    let mut observed = vec![0usize; names.len()];
    let mut extrapolated = observed.clone();
    for trial in &data.trials {
        let inputs = model.inputs(trial)?;
        let origin = trial.forecast_origin.ok_or("missing origin")?;
        let mut changed = trial.clone();
        for values in changed.recording.behavior.values_mut() {
            for (t, y) in changed.recording.times.iter().zip(values) {
                if *t > origin {
                    *y = Some(99999.0);
                }
            }
        }
        if inputs != model.inputs(&changed)? {
            return Err("future behavior changed common inputs".into());
        }
        for (t, row) in inputs.iter().enumerate() {
            for i in 0..names.len() {
                if row[names.len() + i] == 1.0 {
                    observed[i] += 1;
                    if trial.recording.times[t] > origin {
                        return Err("post-origin observed behavior mask".into());
                    }
                } else {
                    extrapolated[i] += 1;
                }
            }
        }
        let partition = if split.train.contains(&trial.id) {
            "train"
        } else if split.validation.contains(&trial.id) {
            "validation"
        } else {
            "test"
        };
        *counts.entry(partition).or_insert(0) += 1;
    }
    let report = serde_json::json!({"schema_version":1,"model":model,"behavior_model_hash":model.content_hash()?,"free_parameters":model.free_parameters(),"input_dimensions":model.input_dim(),"windows_by_partition":counts,"observed_values_by_channel":observed,"extrapolated_or_initial_mean_values_by_channel":extrapolated,"future_behavior_replacement_preserves_all_inputs":true,"post_origin_observed_masks":0,"preprocessing_assessment":bench::preprocessing::assess(&split.dataset_hash,&graph.hash)?,"scope":"Protocol audit only, not neural forecast performance. Behavior AR coefficients and calibration use training animals only; actual future behavior never enters inputs."});
    fs::write(
        &args[4],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "audited {} windows; {} common covariates; {} behavior scalars",
        data.trials.len(),
        model.input_dim(),
        model.free_parameters()
    );
    Ok(())
}
