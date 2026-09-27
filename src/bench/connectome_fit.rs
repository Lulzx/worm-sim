//! Population atlas fit with training-only EM and validation-only checkpoint selection.
use super::{
    Axis, Dataset, Partition, PredictedTrial, Predictions, Split, Trial,
    connectome_lds::{ConnectomeLds, StimulusSequence},
};
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfig {
    pub iterations: usize,
    pub kernel_lags: usize,
    pub ridge: f64,
    pub cap: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub graph_hash: String,
    pub source_commit: String,
    pub config: FitConfig,
    pub iteration: usize,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
    pub fitted_noise_outputs: usize,
    pub dynamics: ConnectomeLds,
}
#[derive(Debug, Serialize)]
pub struct Candidate {
    pub iteration: usize,
    pub validation_mse: f64,
    pub validation_correlation: Option<f64>,
    pub defined_trace_correlations: usize,
    pub training_step: Option<super::connectome_lds::StepReport>,
}
fn grid(trial: &Trial, dt: f64) -> Result<()> {
    if trial.forecast_origin.is_some()
        || trial.recording.times.len() < 2
        || trial.recording.times[0].abs() > 1e-10
        || trial
            .recording
            .times
            .windows(2)
            .any(|p| (p[1] - p[0] - dt).abs() > 1e-10)
    {
        return Err(
            "atlas LDS requires response windows on a uniform grid starting at stimulation zero"
                .into(),
        );
    }
    Ok(())
}
pub fn training_sequences(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    dt: f64,
) -> Result<Vec<StimulusSequence>> {
    split.validate(data, graph)?;
    if split.axis != Axis::StimulatedNeuron {
        return Err("atlas fit requires a stimulated-neuron split".into());
    }
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let mut ids = split.train.clone();
    ids.sort();
    ids.iter()
        .map(|id| {
            let trial = indexed[id];
            grid(trial, dt)?;
            let mut observations = vec![vec![]; trial.recording.times.len()];
            for trace in &trial.recording.traces {
                let i = graph.neuron(&trace.neuron)?;
                if trace.provenance.id_confidence > 0.0 {
                    for (frame, value) in observations.iter_mut().zip(&trace.values) {
                        if let Some(y) = value {
                            frame.push((i, *y, trace.provenance.id_confidence));
                        }
                    }
                }
            }
            // Canonical order makes equivalent source trace orders share a plan.
            for frame in &mut observations {
                frame.sort_by_key(|v| v.0);
            }
            Ok(StimulusSequence {
                target: graph.neuron(
                    trial
                        .stimulated_neuron
                        .as_deref()
                        .ok_or("missing stimulus")?,
                )?,
                observations,
            })
        })
        .collect()
}
impl Model {
    pub fn free_parameters(&self) -> usize {
        self.dynamics.allowed.iter().map(Vec::len).sum::<usize>()
            + self.dynamics.kernel.len()
            + 2 * self.dynamics.gaussian.dim
            + self.fitted_noise_outputs
    }
    pub fn predict(
        &self,
        data: &Dataset,
        graph: &IndexedGraph,
        split: &Split,
        partition: Partition,
    ) -> Result<Predictions> {
        split.validate(data, graph)?;
        self.dynamics.validate_for_graph(graph)?;
        if self.schema_version != 1
            || self.graph_hash != graph.hash
            || self.dataset_hash != split.dataset_hash
            || self.split_hash != split.content_hash()?
            || split.axis != Axis::StimulatedNeuron
            || self.training_trials != split.train
            || self.selection_trials != split.validation
            || self.source_commit.is_empty()
        {
            return Err("atlas LDS model lineage mismatch".into());
        }
        let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
        let mut cache = BTreeMap::new();
        let mut trials = vec![];
        for id in split.ids(partition) {
            let trial = indexed[id];
            grid(trial, self.dynamics.sample_dt)?;
            let target = graph.neuron(
                trial
                    .stimulated_neuron
                    .as_deref()
                    .ok_or("missing stimulus")?,
            )?;
            let frames = trial.recording.times.len();
            if let std::collections::btree_map::Entry::Vacant(e) = cache.entry((target, frames)) {
                e.insert(self.dynamics.impulse(target, frames)?);
            }
            let response = &cache[&(target, frames)];
            let mut fluorescence = BTreeMap::new();
            for trace in &trial.recording.traces {
                let i = graph.neuron(&trace.neuron)?;
                fluorescence.insert(
                    trace.neuron.clone(),
                    response.iter().map(|r| r[i]).collect::<Vec<_>>(),
                );
            }
            let response_scores = trial
                .response_labels
                .keys()
                .map(|name| {
                    (
                        name.clone(),
                        fluorescence[name].iter().map(|x| x.abs()).sum::<f64>()
                            * self.dynamics.sample_dt,
                    )
                })
                .collect();
            trials.push(PredictedTrial {
                id: id.clone(),
                times: trial.recording.times.clone(),
                fluorescence,
                response_scores,
            });
        }
        Ok(Predictions {
            schema_version: 1,
            dataset_hash: self.dataset_hash.clone(),
            split_hash: self.split_hash.clone(),
            model: "connectome-lds-shared-kernel".into(),
            free_parameters: self.free_parameters(),
            training_trials: self.training_trials.clone(),
            selection_trials: self.selection_trials.clone(),
            source_commit: self.source_commit.clone(),
            seed: split.seed,
            trials,
        })
    }
}
/// Select minimum pooled validation MSE, retaining the earlier checkpoint on ties.
/// Includes initialization; test outcomes and published pair labels are not read.
pub fn fit_select(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    config: FitConfig,
    mut checkpoint: impl FnMut(&Model, &Candidate) -> Result<()>,
) -> Result<(Model, Vec<Candidate>)> {
    split.validate(data, graph)?;
    if config.iterations == 0
        || config.iterations > 100
        || config.kernel_lags == 0
        || config.kernel_lags > 512
        || !config.ridge.is_finite()
        || config.ridge <= 0.0
        || !config.cap.is_finite()
        || config.cap <= 0.0
        || config.cap >= 1.0
        || split.validation.is_empty()
    {
        return Err("invalid atlas fit configuration".into());
    }
    let first = data
        .trials
        .iter()
        .find(|t| split.train.contains(&t.id))
        .ok_or("empty training set")?;
    if first.recording.times.len() < 2 {
        return Err("training window needs two frames".into());
    }
    let dt = first.recording.times[1] - first.recording.times[0];
    let sequences = training_sequences(data, graph, split, dt)?;
    let observed: std::collections::BTreeSet<_> = sequences
        .iter()
        .flat_map(|s| s.observations.iter().flatten().map(|v| v.0))
        .collect();
    let mut current = Model {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash()?,
        graph_hash: graph.hash.clone(),
        source_commit: option_env!("WORMSIM_COMMIT")
            .unwrap_or("unversioned")
            .into(),
        config: config.clone(),
        iteration: 0,
        training_trials: split.train.clone(),
        selection_trials: split.validation.clone(),
        fitted_noise_outputs: observed.len(),
        dynamics: ConnectomeLds::new(graph, config.kernel_lags, dt)?,
    };
    let mut selected = None;
    let mut best = f64::INFINITY;
    let mut candidates = vec![];
    for iteration in 0..=config.iterations {
        let training_step = if iteration > 0 {
            let (next, step) = current
                .dynamics
                .em_step(&sequences, config.ridge, config.cap)?;
            current.dynamics = next;
            Some(step)
        } else {
            None
        };
        current.iteration = iteration;
        let predictions = current.predict(data, graph, split, Partition::Validation)?;
        let score = super::evaluate(data, graph, split, &predictions, Partition::Validation)?;
        let mse = score
            .pooled_trace_scores
            .mse
            .ok_or("no validation observations")?;
        if !mse.is_finite() {
            return Err("nonfinite validation score".into());
        }
        let candidate = Candidate {
            iteration,
            validation_mse: mse,
            validation_correlation: score.macro_trace_correlation,
            defined_trace_correlations: score.defined_trace_correlations,
            training_step,
        };
        checkpoint(&current, &candidate)?;
        if mse < best {
            best = mse;
            selected = Some(current.clone());
        }
        candidates.push(candidate);
    }
    Ok((selected.ok_or("no selected atlas model")?, candidates))
}
