//! Reproducible atlas population fit. Output directory must not already exist.
use std::{fs, path::Path};
use wormsim::{
    Result,
    bench::{
        self, Dataset, Partition, Split,
        atlas_level0::{self, FitConfig},
    },
    codec,
};
fn read<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn write(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 && args.len() != 7 {
        return Err(
            "usage: fit_level0_atlas GRAPH DATA SPLIT CONFIG NEW_OUTPUT_DIR [PAIR_EVIDENCE]".into(),
        );
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&args[2])?;
    let split: Split = read(&args[3])?;
    let config: FitConfig = read(&args[4])?;
    let evidence = args
        .get(6)
        .map(|path| read::<bench::atlas::Evidence>(path))
        .transpose()?;
    split.validate(&data, &graph)?;
    let output = Path::new(&args[5]);
    fs::create_dir(output).map_err(|e| e.to_string())?;
    write(output.join("config.json"), &config)?;
    let (model, candidates) = atlas_level0::fit_select_with_evidence(
        &data,
        &graph,
        &split,
        config,
        evidence.as_ref(),
        |model, candidate| {
            write(
                output.join(format!("epoch-{}.json", candidate.epoch)),
                model,
            )?;
            write(
                output.join(format!("epoch-{}.report.json", candidate.epoch)),
                candidate,
            )?;
            println!(
                "epoch {} validation MSE {:.9}, correlation {:?}, epoch including validation seconds {:?}",
                candidate.epoch,
                candidate.validation_mse,
                candidate.validation_correlation,
                Some(candidate.elapsed_seconds)
            );
            Ok(())
        },
    )?;
    write(output.join("selected.json"), &model)?;
    write(output.join("selection.json"), &candidates)?;
    for (name, partition) in [
        ("validation", Partition::Validation),
        ("test", Partition::Test),
    ] {
        let prediction = model.predict(&data, &graph, &split, partition)?;
        let report = bench::evaluate(&data, &graph, &split, &prediction, partition)?;
        write(output.join(format!("{name}-predictions.json")), &prediction)?;
        write(output.join(format!("{name}-report.json")), &report)?;
        if name == "validation" {
            if model.config.preparation_seconds > 0.0 {
                let network = wormsim::model::Model::new(graph.clone())?;
                let params = model.parameters.expand(&network)?;
                let (state, derivative) = wormsim::initial_state::prepared_state(
                    &network,
                    &params,
                    &model.initial,
                    model.config.dt,
                    model.config.preparation_seconds,
                )?;
                let (long_state, long_derivative) = wormsim::initial_state::prepared_state(
                    &network,
                    &params,
                    &model.initial,
                    model.config.dt,
                    2.0 * model.config.preparation_seconds,
                )?;
                let mut longer = model.clone();
                longer.config.preparation_seconds *= 2.0;
                let longer_prediction = longer.predict(&data, &graph, &split, partition)?;
                let longer_report =
                    bench::evaluate(&data, &graph, &split, &longer_prediction, partition)?;
                let max_change = prediction
                    .trials
                    .iter()
                    .zip(&longer_prediction.trials)
                    .flat_map(|(a, b)| {
                        a.fluorescence.iter().flat_map(move |(name, values)| {
                            values
                                .iter()
                                .zip(&b.fluorescence[name])
                                .map(|(x, y)| (x - y).abs())
                        })
                    })
                    .fold(0.0, f64::max);
                write(
                    output.join("validation-preparation-check.json"),
                    &serde_json::json!({"source_commit":model.source_commit,"epoch":model.epoch,"preparation_seconds":model.config.preparation_seconds,"longer_seconds":longer.config.preparation_seconds,"derivative_l2":derivative.iter().map(|v|v*v).sum::<f64>().sqrt(),"longer_derivative_l2":long_derivative.iter().map(|v|v*v).sum::<f64>().sqrt(),"max_state_change":state.iter().zip(&long_state).map(|(a,b)|(a-b).abs()).fold(0.0,f64::max),"max_prediction_change":max_change,"selected_duration_mse":report.pooled_trace_scores.mse,"longer_duration_mse":longer_report.pooled_trace_scores.mse,"scope":"Validation-only preparation duration sensitivity and residual unforced derivative; not a proof of equilibrium uniqueness or a biological resting state."}),
                )?;
            }

            let fine = model.predict_dt(&data, &graph, &split, partition, model.config.dt / 2.0)?;
            let fine_report = bench::evaluate(&data, &graph, &split, &fine, partition)?;
            let max_change = prediction
                .trials
                .iter()
                .zip(&fine.trials)
                .flat_map(|(a, b)| {
                    a.fluorescence.iter().flat_map(move |(name, values)| {
                        values
                            .iter()
                            .zip(&b.fluorescence[name])
                            .map(|(x, y)| (x - y).abs())
                    })
                })
                .fold(0.0, f64::max);
            write(
                output.join("validation-half-step.json"),
                &serde_json::json!({"source_commit":model.source_commit,"epoch":model.epoch,"dt":model.config.dt,"fine_dt":model.config.dt/2.0,"max_prediction_change":max_change,"coarse_mse":report.pooled_trace_scores.mse,"fine_mse":fine_report.pooled_trace_scores.mse,"scope":"Validation-only numerical sensitivity at the selected checkpoint; not a convergence proof."}),
            )?;
        }

        println!(
            "selected epoch {} {name} MSE {:?}, correlation {:?}",
            model.epoch, report.pooled_trace_scores.mse, report.macro_trace_correlation
        );
    }
    Ok(())
}
