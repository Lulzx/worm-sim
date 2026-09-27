//! Measure one full-state training E/M step on a real training window before scheduling a fit.
use std::fs;
use wormsim::{
    Result,
    bench::{
        Axis, Dataset, Split,
        connectome_lds::{ConnectomeLds, StimulusSequence},
    },
    codec,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: benchmark_connectome_lds GRAPH.wsc DATA.json SPLIT.json OUTPUT.json".into(),
        );
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let split: Split = serde_json::from_slice(&fs::read(&args[3]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    split.validate(&data, &graph)?;
    if split.axis != Axis::StimulatedNeuron {
        return Err("expected held-out-neuron split".into());
    }
    let trial = data
        .trials
        .iter()
        .filter(|t| split.train.contains(&t.id))
        .min_by_key(|t| &t.id)
        .ok_or("empty training partition")?;
    let dt = trial.recording.times[1] - trial.recording.times[0];
    if trial
        .recording
        .times
        .windows(2)
        .any(|p| (p[1] - p[0] - dt).abs() > 1e-10)
    {
        return Err("nonuniform sample grid".into());
    }
    let frames = trial.recording.times.len();
    let mut observations = vec![vec![]; frames];
    for trace in &trial.recording.traces {
        let i = graph.neuron(&trace.neuron)?;
        for (frame, y) in observations.iter_mut().zip(&trace.values) {
            if let Some(y) = y
                && trace.provenance.id_confidence > 0.0
            {
                frame.push((i, *y, trace.provenance.id_confidence));
            }
        }
    }
    let sequence = StimulusSequence {
        target: graph.neuron(trial.stimulated_neuron.as_ref().ok_or("missing target")?)?,
        observations,
    };
    let model = ConnectomeLds::new(&graph, frames - 1, dt)?;
    eprintln!(
        "One E/M step: {} latent neurons, {} frames, trial {}",
        graph.names.len(),
        frames,
        trial.id
    );
    let (updated, step) = model.em_step(&[sequence], 1e-4, 0.995)?;
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"graph_hash":graph.hash,"dataset_hash":split.dataset_hash,"split_hash":split.content_hash()?,"training_trial":trial.id,"latent_neurons":graph.names.len(),"frames":frames,"observed_neurons":trial.recording.traces.len(),"kernel_lags":model.kernel.len(),"anatomical_weights":model.allowed.iter().map(Vec::len).sum::<usize>(),"step":step,"updated_kernel":updated.kernel,"scope":"One training-window full-covariance Gaussian E/M step from diagonal initialization; not a population fit or model evaluation. Timer excludes file loading, input preparation and initial model validation; includes smoothing, moment accumulation, constrained update, stability projection and final model validation. No biological acceptance claim."});
    fs::write(
        &args[4],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{} seconds", step.elapsed_seconds);
    Ok(())
}
