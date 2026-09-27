//! Task 2 state-inference audit; this does not claim a fitted population model.
use super::{Axis, Dataset, Split};
use crate::{
    Result,
    data::IndexedGraph,
    initial_state::{self, InferenceConfig, InferredState, Readout},
    model::Model,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Debug, Serialize, Deserialize)]
pub struct StateAudit {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub graph_hash: String,
    pub source_commit: String,
    pub trial: String,
    pub animal: String,
    pub partition: String,
    pub forecast_origin: f64,
    pub description: String,
    pub config: InferenceConfig,
    pub parameters: Vec<f64>,
    pub readout: Readout,
    pub readout_training_trials: Vec<String>,
    pub readout_unseen_neurons: Vec<String>,
    pub elapsed_seconds: f64,
    pub inferred: InferredState,
}
/// Training-only affine calibration: fluorescence mean maps to calcium 0.5,
/// one training standard deviation to 0.2 calcium. These constants are declared
/// initialization assumptions, not measured GCaMP kinetics or fitted readout gains.
pub fn training_readout(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
) -> Result<(Readout, Vec<String>)> {
    split.validate(data, graph)?;
    if split.axis != Axis::Animal {
        return Err("state inference requires animal split".into());
    }
    let mut stats: BTreeMap<&str, (f64, f64, f64)> = BTreeMap::new();
    let mut trials: Vec<_> = data
        .trials
        .iter()
        .filter(|t| split.train.contains(&t.id))
        .collect();
    trials.sort_by(|a, b| a.id.cmp(&b.id));
    for t in trials {
        for trace in &t.recording.traces {
            let w = trace.provenance.id_confidence;
            if w == 0.0 {
                continue;
            }
            for &x in trace.values.iter().flatten() {
                let s = stats.entry(&trace.neuron).or_default();
                s.0 += w;
                let d = x - s.1;
                s.1 += w / s.0 * d;
                s.2 += w * d * (x - s.1);
            }
        }
    }
    let mut readout = Readout::identity(graph.names.len());
    let mut unseen = vec![];
    for (i, name) in graph.names.iter().enumerate() {
        let (mean, sd) = if let Some(&(w, m, ss)) = stats.get(name.as_str()) {
            (m, (ss / w).max(0.0).sqrt().max(1e-8))
        } else {
            unseen.push(name.clone());
            (0.0, 1.0)
        };
        readout.gain[i] = 5.0 * sd;
        readout.offset[i] = mean - 2.5 * sd;
    }
    Ok((readout, unseen))
}
pub fn infer_trial(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    id: &str,
    cfg: InferenceConfig,
) -> Result<StateAudit> {
    let (readout, unseen) = training_readout(data, graph, split)?;
    let trial = data
        .trials
        .iter()
        .find(|t| t.id == id)
        .ok_or("unknown trial id")?;
    let origin = trial
        .forecast_origin
        .ok_or("trial has no forecast origin")?;
    let model = Model::new(graph.clone())?;
    let params = model.defaults();
    let start = std::time::Instant::now();
    let inferred = initial_state::infer(&model, &params, &trial.recording, origin, &readout, &cfg)?;
    Ok(StateAudit{schema_version:1,dataset_hash:split.dataset_hash.clone(),split_hash:split.content_hash()?,graph_hash:graph.hash.clone(),source_commit:option_env!("WORMSIM_COMMIT").unwrap_or("unversioned").into(),trial:id.into(),animal:trial.recording.animal_id.clone(),partition:if split.train.contains(&trial.id){"train"}else if split.validation.contains(&trial.id){"validation"}else{"test"}.into(),forecast_origin:origin,description:"History-only full 3N-state inference using frozen default Level 0 parameters, training-only affine calibration, and default-state shrinkage. This is an inference audit, not a population fit, identifiable hidden-state reconstruction, or benchmark success.".into(),config:cfg,parameters:params.raw,readout,readout_training_trials:split.train.clone(),readout_unseen_neurons:unseen,elapsed_seconds:start.elapsed().as_secs_f64(),inferred})
}
