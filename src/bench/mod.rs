//! Versioned benchmark trials, group-disjoint splits, and common held-out scoring.
//! Prediction lineage is checked as declared metadata, not proof of how a model trained.
pub mod atlas;
pub mod atlas_level0;
pub mod atlas_training;
pub mod atlas_uncertainty;
pub mod behavior;
pub mod connectome_fit;
pub mod connectome_lds;
pub mod controls;
pub mod gru;
mod gru_cell;
pub mod lds;
mod lds_math;
pub mod level0;
pub mod linear;
pub mod metrics;
pub mod population;
pub mod preprocessing;
pub mod uncertainty;
use crate::{
    Result,
    data::{IndexedGraph, Recording},
};
use metrics::{Auroc, Moments, Scores};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

struct HashWriter(Sha256);
impl std::io::Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn hash(value: &impl Serialize) -> Result<String> {
    let mut writer = HashWriter(Sha256::new());
    serde_json::to_writer(&mut writer, value).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", writer.0.finalize()))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trial {
    pub id: String,
    pub stimulated_neuron: Option<String>,
    /// Forecasts are open-loop after this observed time, in recording seconds.
    pub forecast_origin: Option<f64>,
    pub recording: Recording,
    /// Measured response labels supplied by the dataset's published criterion.
    pub response_labels: BTreeMap<String, bool>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub schema_version: u32,
    pub name: String,
    pub graph_hash: String,
    /// Required source/version/preprocessing declaration. Synthetic data must say so.
    pub source: String,
    pub trials: Vec<Trial>,
}
impl Dataset {
    pub fn validate(&self, graph: &IndexedGraph) -> Result<()> {
        if self.schema_version != 1
            || self.name.trim().is_empty()
            || self.source.trim().is_empty()
            || self.graph_hash != graph.hash
            || self.trials.is_empty()
        {
            return Err("invalid benchmark dataset metadata or graph hash".into());
        }
        let mut ids = BTreeSet::new();
        for trial in &self.trials {
            if trial.id.is_empty() || !ids.insert(&trial.id) {
                return Err("empty or duplicate trial ID".into());
            }
            trial.recording.validate(graph)?;
            if trial.recording.traces.is_empty() {
                return Err("trial has no traces".into());
            }
            if let Some(name) = &trial.stimulated_neuron {
                graph.neuron(name)?;
            }
            if let Some(origin) = trial.forecast_origin
                && (!origin.is_finite() || !trial.recording.times.contains(&origin))
            {
                return Err("forecast origin must be on the observed time grid".into());
            }
            for name in trial.response_labels.keys() {
                if !trial
                    .recording
                    .traces
                    .iter()
                    .any(|trace| &trace.neuron == name)
                {
                    return Err("response label has no associated trace/confidence".into());
                }
            }
        }
        Ok(())
    }
    /// Trial/trace ordering does not alter identity; time/sample ordering does.
    pub fn content_hash(&self) -> Result<String> {
        // Sort references, not the potentially gigabytes of trace samples.
        // The versioned canonical tuple is streamed directly into SHA-256.
        let mut ordered: Vec<_> = self.trials.iter().collect();
        ordered.sort_by(|a, b| a.id.cmp(&b.id));
        let canonical: Vec<_> = ordered
            .into_iter()
            .map(|trial| {
                let r = &trial.recording;
                let mut traces: Vec<_> = r.traces.iter().collect();
                traces.sort_by(|a, b| a.neuron.cmp(&b.neuron));
                (
                    &trial.id,
                    &trial.stimulated_neuron,
                    trial.forecast_origin,
                    &r.dataset,
                    &r.animal_id,
                    &r.condition,
                    &r.times,
                    &r.behavior,
                    traces,
                    &trial.response_labels,
                )
            })
            .collect();
        hash(&(
            "wormsim-benchmark-data-v2",
            self.schema_version,
            &self.name,
            &self.graph_hash,
            &self.source,
            canonical,
        ))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    StimulatedNeuron,
    Animal,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Partition {
    Train,
    Validation,
    Test,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub graph_hash: String,
    pub axis: Axis,
    pub seed: u64,
    pub train: Vec<String>,
    pub validation: Vec<String>,
    pub test: Vec<String>,
}
fn group(trial: &Trial, axis: Axis) -> Result<&str> {
    match axis {
        Axis::Animal => Ok(&trial.recording.animal_id),
        Axis::StimulatedNeuron => trial
            .stimulated_neuron
            .as_deref()
            .ok_or("neuron split requires a stimulated neuron on every trial".into()),
    }
}
impl Split {
    /// Counts are groups, not trials. SHA-256 ranking avoids RNG/version drift.
    pub fn generate(
        data: &Dataset,
        graph: &IndexedGraph,
        axis: Axis,
        seed: u64,
        validation_groups: usize,
        test_groups: usize,
    ) -> Result<Self> {
        data.validate(graph)?;
        let mut groups = BTreeSet::new();
        for trial in &data.trials {
            groups.insert(group(trial, axis)?.to_string());
        }
        let held = validation_groups
            .checked_add(test_groups)
            .ok_or("split size overflow")?;
        if test_groups == 0 || held >= groups.len() {
            return Err("split requires nonempty training and test groups".into());
        }
        let mut ranked: Vec<_> = groups
            .into_iter()
            .map(|name| {
                let key = hash(&("wormsim-group-split-v1", axis, seed, &name))?;
                Ok((key, name))
            })
            .collect::<Result<_>>()?;
        ranked.sort();
        let assignment: BTreeMap<_, _> = ranked
            .into_iter()
            .enumerate()
            .map(|(i, (_, name))| {
                (
                    name,
                    if i < test_groups {
                        Partition::Test
                    } else if i < held {
                        Partition::Validation
                    } else {
                        Partition::Train
                    },
                )
            })
            .collect();
        let mut split = Self {
            schema_version: 1,
            dataset_hash: data.content_hash()?,
            graph_hash: graph.hash.clone(),
            axis,
            seed,
            train: vec![],
            validation: vec![],
            test: vec![],
        };
        for trial in &data.trials {
            match assignment[group(trial, axis)?] {
                Partition::Train => split.train.push(trial.id.clone()),
                Partition::Validation => split.validation.push(trial.id.clone()),
                Partition::Test => split.test.push(trial.id.clone()),
            }
        }
        split.train.sort();
        split.validation.sort();
        split.test.sort();
        split.validate(data, graph)?;
        Ok(split)
    }
    pub fn ids(&self, partition: Partition) -> &[String] {
        match partition {
            Partition::Train => &self.train,
            Partition::Validation => &self.validation,
            Partition::Test => &self.test,
        }
    }
    pub fn content_hash(&self) -> Result<String> {
        let mut canonical = self.clone();
        canonical.train.sort();
        canonical.validation.sort();
        canonical.test.sort();
        hash(&canonical)
    }
    pub fn validate(&self, data: &Dataset, graph: &IndexedGraph) -> Result<()> {
        data.validate(graph)?;
        if self.schema_version != 1
            || self.dataset_hash != data.content_hash()?
            || self.graph_hash != graph.hash
            || self.train.is_empty()
            || self.test.is_empty()
        {
            return Err(
                "invalid split version, content hash or empty training/test partition".into(),
            );
        }
        let trials: BTreeMap<_, _> = data.trials.iter().map(|t| (t.id.as_str(), t)).collect();
        let mut seen = BTreeSet::new();
        let mut owners = BTreeMap::new();
        for (partition, ids) in [&self.train, &self.validation, &self.test]
            .into_iter()
            .enumerate()
        {
            for id in ids {
                let trial = trials
                    .get(id.as_str())
                    .ok_or("split references unknown trial")?;
                if !seen.insert(id) {
                    return Err("trial occurs in multiple split positions".into());
                }
                let key = group(trial, self.axis)?;
                if let Some(previous) = owners.insert(key, partition)
                    && previous != partition
                {
                    return Err(format!(
                        "split leakage: group {key} occurs in multiple partitions"
                    ));
                }
            }
        }
        if seen.len() != data.trials.len() {
            return Err("split omits dataset trials".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictedTrial {
    pub id: String,
    pub times: Vec<f64>,
    pub fluorescence: BTreeMap<String, Vec<f64>>,
    pub response_scores: BTreeMap<String, f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Predictions {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub model: String,
    pub free_parameters: usize,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
    pub source_commit: String,
    pub seed: u64,
    pub trials: Vec<PredictedTrial>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceScore {
    pub trial: String,
    pub neuron: String,
    pub confidence: f64,
    pub scores: Scores,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronScore {
    pub neuron: String,
    pub scores: Scores,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HorizonScore {
    pub seconds: f64,
    pub unavailable_grid_or_missing_samples: usize,
    pub pooled: Scores,
    pub per_neuron: Vec<NeuronScore>,
    pub macro_neuron_r2: Option<f64>,
    pub defined_neuron_r2: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    pub dataset_hash: String,
    #[serde(default)]
    pub preprocessing_assessment: preprocessing::Assessment,
    pub graph_hash: String,
    pub split_hash: String,
    pub axis: Axis,
    pub partition: Partition,
    pub model: String,
    pub free_parameters: usize,
    pub prediction_source_commit: String,
    pub scorer_source_commit: String,
    pub seed: u64,
    pub trials: usize,
    pub missing_samples: usize,
    pub zero_confidence_traces: usize,
    pub pooled_trace_scores: Scores,
    pub macro_trace_correlation: Option<f64>,
    pub defined_trace_correlations: usize,
    pub traces: Vec<TraceScore>,
    pub response_auroc: Auroc,
    pub forecast_horizons: Vec<HorizonScore>,
    #[serde(default)]
    pub animal_bootstrap: Option<uncertainty::AnimalBootstrap>,
}
fn declared_subset(ids: &[String], allowed: &[String]) -> bool {
    ids.iter().collect::<BTreeSet<_>>().len() == ids.len()
        && ids.iter().all(|id| allowed.contains(id))
}
/// Evaluate exactly one complete partition. No silently dropped predictions.
/// Task 2 macro R² centers each neuron separately across forecast trials.
pub fn evaluate(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    predictions: &Predictions,
    partition: Partition,
) -> Result<Report> {
    split.validate(data, graph)?;
    let split_hash = split.content_hash()?;
    if predictions.schema_version != 1
        || predictions.dataset_hash != split.dataset_hash
        || predictions.split_hash != split_hash
        || predictions.model.trim().is_empty()
        || predictions.source_commit.trim().is_empty()
        || !declared_subset(&predictions.training_trials, &split.train)
        || !declared_subset(&predictions.selection_trials, &split.validation)
    {
        return Err("invalid prediction metadata or declared training/selection leakage".into());
    }
    let ids = split.ids(partition);
    if ids.is_empty() {
        return Err("cannot score an empty partition".into());
    }
    let predicted: BTreeMap<_, _> = predictions
        .trials
        .iter()
        .map(|t| (t.id.as_str(), t))
        .collect();
    if predicted.len() != predictions.trials.len()
        || predicted.len() != ids.len()
        || ids.iter().any(|id| !predicted.contains_key(id.as_str()))
    {
        return Err(
            "predictions must cover exactly the selected partition, without duplicates".into(),
        );
    }
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (t.id.as_str(), t)).collect();
    let mut pooled = Moments::default();
    let mut traces = vec![];
    let mut pairs = vec![];
    let mut missing_samples = 0;
    let mut zero_confidence_traces = 0;
    let mut correlation_sum = 0.0;
    let mut correlation_weight = 0.0;
    let mut defined_correlations = 0;
    let horizons = [1.0, 10.0, 30.0];
    let mut horizon_pooled = [Moments::default(), Moments::default(), Moments::default()];
    let mut horizon_neurons: Vec<BTreeMap<String, Moments>> = vec![BTreeMap::new(); 3];
    let mut horizon_missing = [0; 3];
    let mut ordered_ids = ids.to_vec();
    ordered_ids.sort();
    for id in &ordered_ids {
        let trial = indexed[id.as_str()];
        let pred = predicted[id.as_str()];
        let recording = &trial.recording;
        if pred.times != recording.times
            || pred.fluorescence.len() != recording.traces.len()
            || pred.response_scores.len() != trial.response_labels.len()
            || pred.response_scores.values().any(|v| !v.is_finite())
        {
            return Err(format!(
                "prediction time grid, trace set or response scores mismatch: {id}"
            ));
        }
        let origin = if split.axis == Axis::Animal {
            Some(
                trial
                    .forecast_origin
                    .ok_or("animal forecast trial requires an explicit origin")?,
            )
        } else {
            None
        };
        let mut ordered_traces: Vec<_> = recording.traces.iter().collect();
        ordered_traces.sort_by(|a, b| a.neuron.cmp(&b.neuron));
        for trace in ordered_traces {
            let values = pred
                .fluorescence
                .get(&trace.neuron)
                .ok_or("prediction omits a recorded neuron")?;
            if values.len() != recording.times.len() || values.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite prediction or wrong trace length".into());
            }
            let weight = trace.provenance.id_confidence;
            if weight == 0.0 {
                zero_confidence_traces += 1;
            }
            let mut moments = Moments::default();
            for (t, (target, &value)) in trace.values.iter().zip(values).enumerate() {
                if origin.is_some_and(|origin| recording.times[t] <= origin) {
                    continue;
                }
                if let Some(target) = target {
                    moments.push(*target, value, weight)?;
                    pooled.push(*target, value, weight)?;
                } else {
                    missing_samples += 1;
                }
            }
            let scores = moments.scores();
            if let Some(correlation) = scores.correlation {
                correlation_sum += weight * correlation;
                correlation_weight += weight;
                defined_correlations += 1;
            }
            traces.push(TraceScore {
                trial: id.clone(),
                neuron: trace.neuron.clone(),
                confidence: weight,
                scores,
            });
            if let Some(&label) = trial.response_labels.get(&trace.neuron) {
                let score = *pred
                    .response_scores
                    .get(&trace.neuron)
                    .ok_or("missing response ranking score")?;
                pairs.push((label, score, weight));
            }
            if let Some(origin) = origin {
                for (h, &seconds) in horizons.iter().enumerate() {
                    let index = recording
                        .times
                        .iter()
                        .position(|t| (*t - (origin + seconds)).abs() <= 1e-9);
                    if let Some(t) = index
                        && let Some(target) = trace.values[t]
                    {
                        horizon_pooled[h].push(target, values[t], weight)?;
                        horizon_neurons[h]
                            .entry(trace.neuron.clone())
                            .or_default()
                            .push(target, values[t], weight)?;
                    } else {
                        horizon_missing[h] += 1;
                    }
                }
            }
        }
    }
    let mut forecast_horizons = vec![];
    if split.axis == Axis::Animal {
        for (h, seconds) in horizons.into_iter().enumerate() {
            let per_neuron: Vec<_> = horizon_neurons[h]
                .iter()
                .map(|(name, m)| NeuronScore {
                    neuron: name.clone(),
                    scores: m.scores(),
                })
                .collect();
            let defined: Vec<_> = per_neuron.iter().filter_map(|n| n.scores.r2).collect();
            forecast_horizons.push(HorizonScore {
                seconds,
                unavailable_grid_or_missing_samples: horizon_missing[h],
                pooled: horizon_pooled[h].scores(),
                macro_neuron_r2: (!defined.is_empty())
                    .then(|| defined.iter().sum::<f64>() / defined.len() as f64),
                defined_neuron_r2: defined.len(),
                per_neuron,
            });
        }
    }
    Ok(Report {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        preprocessing_assessment: preprocessing::assess(&split.dataset_hash, &graph.hash)?,
        graph_hash: graph.hash.clone(),
        split_hash,
        axis: split.axis,
        partition,
        model: predictions.model.clone(),
        free_parameters: predictions.free_parameters,
        prediction_source_commit: predictions.source_commit.clone(),
        scorer_source_commit: option_env!("WORMSIM_COMMIT")
            .unwrap_or("unversioned")
            .into(),
        seed: predictions.seed,
        trials: ids.len(),
        missing_samples,
        zero_confidence_traces,
        pooled_trace_scores: pooled.scores(),
        macro_trace_correlation: (correlation_weight > 0.0)
            .then(|| correlation_sum / correlation_weight),
        defined_trace_correlations: defined_correlations,
        traces,
        response_auroc: metrics::auroc(&pairs)?,
        forecast_horizons,
        animal_bootstrap: if split.axis == Axis::Animal {
            Some(uncertainty::calculate(data, predictions, 42, 2000)?)
        } else {
            None
        },
    })
}

/// Zero-parameter forecasting control: last observed value at/before origin.
/// This reads no future target values and fits no parameters on any partition.
pub fn persistence(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    partition: Partition,
) -> Result<Predictions> {
    split.validate(data, graph)?;
    if split.axis != Axis::Animal {
        return Err("persistence control requires an animal forecast split".into());
    }
    let mut trials = vec![];
    for trial in data
        .trials
        .iter()
        .filter(|trial| split.ids(partition).contains(&trial.id))
    {
        if !trial.response_labels.is_empty() {
            return Err("persistence control does not predict response labels".into());
        }
        let origin = trial.forecast_origin.ok_or("missing forecast origin")?;
        let mut fluorescence = BTreeMap::new();
        for trace in &trial.recording.traces {
            let value = trial
                .recording
                .times
                .iter()
                .zip(&trace.values)
                .take_while(|(t, _)| **t <= origin)
                .filter_map(|(_, v)| *v)
                .last()
                .ok_or_else(|| {
                    format!(
                        "{} / {}: no observed history for persistence",
                        trial.id, trace.neuron
                    )
                })?;
            fluorescence.insert(
                trace.neuron.clone(),
                vec![value; trial.recording.times.len()],
            );
        }
        trials.push(PredictedTrial {
            id: trial.id.clone(),
            times: trial.recording.times.clone(),
            fluorescence,
            response_scores: BTreeMap::new(),
        });
    }
    trials.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(Predictions {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash()?,
        model: "last-observation persistence; no fitted parameters".into(),
        free_parameters: 0,
        training_trials: vec![],
        selection_trials: vec![],
        source_commit: option_env!("WORMSIM_COMMIT")
            .unwrap_or("unversioned")
            .into(),
        seed: 0,
        trials,
    })
}
