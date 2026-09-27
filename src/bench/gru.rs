//! Connectome-free masked GRU baseline with training-only scaling and open-loop BPTT.
pub use super::gru_cell::{Network, Observation};
use super::{Axis, Dataset, Partition, PredictedTrial, Predictions, Split, Trial};
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfig {
    pub hidden: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gradient_clip: f64,
    pub seed: u64,
}
impl Default for FitConfig {
    fn default() -> Self {
        Self {
            hidden: 6,
            epochs: 30,
            learning_rate: 0.003,
            weight_decay: 1e-4,
            gradient_clip: 1.0,
            seed: 42,
        }
    }
}
impl FitConfig {
    fn validate(&self) -> Result<()> {
        if self.hidden == 0
            || self.hidden > 128
            || self.epochs > 1000
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
            || self.learning_rate * self.weight_decay >= 1.0
            || !self.gradient_clip.is_finite()
            || self.gradient_clip <= 0.0
        {
            return Err("invalid GRU fit configuration".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GruModel {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub graph_hash: String,
    pub source_commit: String,
    pub sample_dt: f64,
    pub neurons: Vec<String>,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub network: Network,
    pub config: FitConfig,
    pub epoch: usize,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub epoch: usize,
    pub updates: usize,
    pub training_standardized_mse: Option<f64>,
    pub validation_horizon_r2: Vec<Option<f64>>,
    pub validation_criterion: f64,
    pub elapsed_seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectionReport {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub source_commit: String,
    pub config: FitConfig,
    pub criterion: String,
    pub candidates: Vec<Candidate>,
    pub selected_epoch: usize,
    pub trainable_parameters: usize,
    pub calibration_statistics: usize,
    pub free_parameters: usize,
}
impl GruModel {
    pub fn free_parameters(&self) -> usize {
        self.network.weights.len() + 2 * self.neurons.len()
    }
    fn validate(&self, data: &Dataset, graph: &IndexedGraph, split: &Split) -> Result<()> {
        split.validate(data, graph)?;
        self.network.validate()?;
        self.config.validate()?;
        let n = self.neurons.len();
        if self.schema_version != 1
            || split.axis != Axis::Animal
            || self.dataset_hash != split.dataset_hash
            || self.split_hash != split.content_hash()?
            || self.graph_hash != graph.hash
            || self.source_commit.trim().is_empty()
            || self.network.outputs != n
            || self.network.hidden != self.config.hidden
            || self.epoch > self.config.epochs
            || self.mean.len() != n
            || self.scale.len() != n
            || self.mean.iter().any(|v| !v.is_finite())
            || self.scale.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self.neurons.windows(2).any(|p| p[0] >= p[1])
            || !self.sample_dt.is_finite()
            || self.sample_dt <= 0.0
            || self.training_trials != split.train
            || self.selection_trials != split.validation
        {
            return Err("invalid GRU artifact or lineage".into());
        }
        for name in &self.neurons {
            graph.neuron(name)?;
        }
        Ok(())
    }
    fn sequence(&self, trial: &Trial, targets: bool) -> Result<(Vec<Vec<Observation>>, usize)> {
        if !trial.response_labels.is_empty() {
            return Err("GRU does not predict response labels".into());
        }
        let origin = trial.forecast_origin.ok_or("missing GRU forecast origin")?;
        let times = &trial.recording.times;
        let at = times
            .iter()
            .position(|t| *t == origin)
            .ok_or("off-grid GRU origin")?;
        if times
            .windows(2)
            .any(|p| (p[1] - p[0] - self.sample_dt).abs() > 1e-8)
        {
            return Err("GRU needs training sample interval".into());
        }
        let mut sequence = vec![vec![]; times.len()];
        for trace in &trial.recording.traces {
            let Ok(i) = self.neurons.binary_search(&trace.neuron) else {
                continue;
            };
            let w = trace.provenance.id_confidence;
            if w <= 0.0 {
                continue;
            }
            for (t, y) in trace.values.iter().enumerate() {
                if (targets || t <= at)
                    && let Some(y) = y
                {
                    sequence[t].push((i, (y - self.mean[i]) / self.scale[i], w));
                }
            }
        }
        for frame in &mut sequence {
            frame.sort_by_key(|x| x.0);
        }
        Ok((sequence, at))
    }
    pub fn predict(
        &self,
        data: &Dataset,
        graph: &IndexedGraph,
        split: &Split,
        partition: Partition,
    ) -> Result<Predictions> {
        self.validate(data, graph, split)?;
        let mut trials = vec![];
        let mut fallbacks = BTreeSet::new();
        let mut fallback_traces = 0;
        for trial in data
            .trials
            .iter()
            .filter(|t| split.ids(partition).contains(&t.id))
        {
            let (sequence, origin) = self.sequence(trial, false)?;
            let output = self.network.predict(&sequence, origin)?;
            let mut fluorescence = BTreeMap::new();
            for trace in &trial.recording.traces {
                let values = if let Ok(i) = self.neurons.binary_search(&trace.neuron) {
                    output
                        .iter()
                        .map(|y| self.mean[i] + self.scale[i] * y[i])
                        .collect::<Vec<_>>()
                } else {
                    let last = trace.values[..=origin]
                        .iter()
                        .filter_map(|y| *y)
                        .next_back()
                        .ok_or("unseen GRU neuron has no persistence history")?;
                    fallbacks.insert(trace.neuron.clone());
                    fallback_traces += 1;
                    vec![last; output.len()]
                };
                if values.iter().any(|v| !v.is_finite()) {
                    return Err("nonfinite GRU fluorescence".into());
                }
                fluorescence.insert(trace.neuron.clone(), values);
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
            dataset_hash: self.dataset_hash.clone(),
            split_hash: self.split_hash.clone(),
            model: format!(
                "connectome-free masked reset-before GRU; hidden={}; epoch={}; trainable={}; calibration={}; history only; unseen-neuron persistence fallback traces={fallback_traces}; neurons={fallbacks:?}",
                self.network.hidden,
                self.epoch,
                self.network.weights.len(),
                2 * self.neurons.len()
            ),
            free_parameters: self.free_parameters(),
            training_trials: self.training_trials.clone(),
            selection_trials: self.selection_trials.clone(),
            source_commit: option_env!("WORMSIM_COMMIT")
                .unwrap_or("unversioned")
                .into(),
            seed: self.config.seed,
            trials,
        })
    }
}
pub fn fit_select(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    cfg: FitConfig,
    mut checkpoint: impl FnMut(&GruModel, &Candidate) -> Result<()>,
) -> Result<(GruModel, SelectionReport)> {
    cfg.validate()?;
    split.validate(data, graph)?;
    if split.axis != Axis::Animal || split.train.is_empty() || split.validation.is_empty() {
        return Err("GRU needs training/validation animal partitions".into());
    }
    let mut training: Vec<_> = data
        .trials
        .iter()
        .filter(|t| split.train.contains(&t.id))
        .collect();
    training.sort_by(|a, b| a.id.cmp(&b.id));
    let mut statistics = BTreeMap::<String, (f64, f64, f64)>::new();
    let mut dt: Option<f64> = None;
    for trial in &training {
        for pair in trial.recording.times.windows(2) {
            let step = pair[1] - pair[0];
            if dt.is_some_and(|d| (d - step).abs() > 1e-8) {
                return Err("GRU needs common uniform training grid".into());
            }
            dt = Some(step);
        }
        for trace in &trial.recording.traces {
            let w = trace.provenance.id_confidence;
            if w <= 0.0 {
                continue;
            }
            for &y in trace.values.iter().flatten() {
                let s = statistics.entry(trace.neuron.clone()).or_default();
                s.0 += w;
                let delta = y - s.1;
                s.1 += w / s.0 * delta;
                s.2 += w * delta * (y - s.1);
            }
        }
    }
    let n = statistics.len();
    let network = Network::initialize(n, cfg.hidden, cfg.seed)?;
    let mut model = GruModel {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash()?,
        graph_hash: graph.hash.clone(),
        source_commit: option_env!("WORMSIM_COMMIT")
            .unwrap_or("unversioned")
            .into(),
        sample_dt: dt.ok_or("missing training intervals")?,
        neurons: statistics.keys().cloned().collect(),
        mean: statistics.values().map(|s| s.1).collect(),
        scale: statistics
            .values()
            .map(|s| (s.2 / s.0).max(0.0).sqrt().max(1e-8))
            .collect(),
        network,
        config: cfg.clone(),
        epoch: 0,
        training_trials: split.train.clone(),
        selection_trials: split.validation.clone(),
    };
    model.validate(data, graph, split)?;
    let sequences = training
        .iter()
        .map(|t| model.sequence(t, true))
        .collect::<Result<Vec<_>>>()?;
    let mut m = vec![0.0; model.network.weights.len()];
    let mut v = m.clone();
    let mut step = 0i32;
    let mut candidates = vec![];
    let mut best: Option<(f64, GruModel)> = None;
    for epoch in 0..=cfg.epochs {
        let start = std::time::Instant::now();
        let mut loss = 0.0;
        if epoch > 0 {
            let mut order: Vec<_> = (0..training.len()).collect();
            order.sort_by_cached_key(|&i| {
                let mut sha = Sha256::new();
                sha.update(cfg.seed.to_le_bytes());
                sha.update((epoch as u64).to_le_bytes());
                sha.update(training[i].id.as_bytes());
                sha.finalize().to_vec()
            });
            for i in order {
                let (value, mut grad) = model
                    .network
                    .loss_gradient(&sequences[i].0, sequences[i].1)?;
                loss += value;
                let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if !norm.is_finite() {
                    return Err("GRU gradient norm overflow".into());
                }
                let clip = (cfg.gradient_clip / norm.max(1e-30)).min(1.0);
                step += 1;
                for j in 0..grad.len() {
                    grad[j] *= clip;
                    m[j] = 0.9 * m[j] + 0.1 * grad[j];
                    v[j] = 0.999 * v[j] + 0.001 * grad[j] * grad[j];
                    model.network.weights[j] *= 1.0 - cfg.learning_rate * cfg.weight_decay;
                    model.network.weights[j] -= cfg.learning_rate
                        * (m[j] / (1.0 - 0.9f64.powi(step)))
                        / ((v[j] / (1.0 - 0.999f64.powi(step))).sqrt() + 1e-8);
                }
            }
        }
        model.epoch = epoch;
        let prediction = model.predict(data, graph, split, Partition::Validation)?;
        let evaluation = super::evaluate(data, graph, split, &prediction, Partition::Validation)?;
        let horizons: Vec<_> = evaluation
            .forecast_horizons
            .iter()
            .map(|h| h.macro_neuron_r2)
            .collect();
        if horizons.len() != 3 || horizons.iter().any(Option::is_none) {
            return Err("undefined GRU validation score".into());
        }
        let score = horizons.iter().flatten().sum::<f64>() / 3.0;
        if !score.is_finite() {
            return Err("nonfinite GRU validation score".into());
        }
        let report = Candidate {
            epoch,
            updates: if epoch == 0 { 0 } else { training.len() },
            training_standardized_mse: if epoch == 0 {
                None
            } else {
                Some(loss / training.len() as f64)
            },
            validation_horizon_r2: horizons,
            validation_criterion: score,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        };
        checkpoint(&model, &report)?;
        candidates.push(report);
        if best.as_ref().is_none_or(|(s, _)| score > *s) {
            best = Some((score, model.clone()));
        }
    }
    let (_, selected) = best.ok_or("no GRU candidate")?;
    let report=SelectionReport{schema_version:1,dataset_hash:split.dataset_hash.clone(),split_hash:split.content_hash()?,source_commit:model.source_commit,config:cfg,criterion:"Maximum mean validation macro-neuron R² at 1/10/30 s including epoch zero; ties retain earlier epoch. Training-only normalization; exact full-window BPTT through autoregressive feedback; no future teacher forcing or behavior input.".into(),candidates,selected_epoch:selected.epoch,trainable_parameters:selected.network.weights.len(),calibration_statistics:2*n,free_parameters:selected.free_parameters()};
    Ok((selected, report))
}
