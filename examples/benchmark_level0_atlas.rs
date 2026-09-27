//! Measure one exact Level 0 response gradient after training-only aggregation.
use std::fs;
use wormsim::{
    Result,
    bench::{Dataset, Split, atlas_training},
    codec,
    initial_state::{self, Readout},
    model::Model,
    parameters::forecast_defaults,
};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: benchmark_level0_atlas GRAPH DATA SPLIT OUTPUT.json".into());
    }
    let graph = codec::decode(&fs::read(&a[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = serde_json::from_slice(&fs::read(&a[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let split: Split = serde_json::from_slice(&fs::read(&a[3]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    let groups = atlas_training::aggregate(&data, &graph, &split)?;
    let aggregation_seconds = started.elapsed().as_secs_f64();
    let group = groups.first().ok_or("no training groups")?;
    let model = Model::new(graph.clone())?;
    let params = forecast_defaults(&model);
    let initial = model.initial(&model.prepare(&params)?);
    let readout = Readout::identity(model.n());
    let target = graph.neuron(&group.stimulated_neuron)?;
    let times = &group.recording.times;
    let mut currents = vec![vec![0.; model.n()]; times.len()];
    for (t, row) in currents.iter_mut().enumerate().take(times.len() - 1) {
        row[target] = 0.2 * (-times[t] / 2.0).exp();
    }
    let started = std::time::Instant::now();
    let gradient = initial_state::response_gradient_with_currents(
        &model,
        &params,
        &group.recording,
        &readout,
        &initial,
        0.01,
        &currents,
    )?;
    let gradient_seconds = started.elapsed().as_secs_f64();
    let coarse = initial_state::response_with_currents(
        &model, &params, &initial, times, &readout, 0.01, &currents,
    )?;
    let fine = initial_state::response_with_currents(
        &model, &params, &initial, times, &readout, 0.005, &currents,
    )?;
    let max_half_step_change = coarse
        .iter()
        .flatten()
        .zip(fine.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"graph_hash":graph.hash,"dataset_hash":split.dataset_hash,"split_hash":split.content_hash()?,"training_trials":split.train.len(),"target_groups":groups.len(),"all_training_sample_weight":groups.iter().map(|g|g.sample_weight).sum::<f64>(),"aggregation_seconds":aggregation_seconds,"latent_neurons":model.n(),"raw_parameters":model.parameter_count(),"measured_target":group.stimulated_neuron,"measured_training_trials":group.training_trials,"frames":times.len(),"dt":0.01,"gradient_seconds":gradient_seconds,"group_mean_trace_mse":gradient.value,"group_original_trial_mse":gradient.value+group.irreducible_mse,"group_irreducible_mse":group.irreducible_mse,"parameter_gradient_norm":gradient.parameters.iter().map(|g|g*g).sum::<f64>().sqrt(),"current_gradient_norm":gradient.currents.iter().flatten().map(|g|g*g).sum::<f64>().sqrt(),"max_half_step_prediction_change":max_half_step_change,"scope":"One complete Level 0 training-target response gradient from forecast defaults with fixed assumed exponentially decaying stimulus current and shared initial state; not a population fit or calibrated optical stimulus. Aggregation timer includes dataset validation and hashing, not file loading. Gradient timer includes forward rollout, loss, and reverse state/parameter/readout/current derivatives. Half-step check uses this initial model only."});
    fs::write(
        &a[4],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{} groups; gradient {} seconds; half-step max change {}",
        groups.len(),
        gradient_seconds,
        max_half_step_change
    );
    Ok(())
}
