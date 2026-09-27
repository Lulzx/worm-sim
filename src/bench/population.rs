//! Conditional-gradient population fitting with history-only latent state inference.
//! Each training window re-infers its state under current shared parameters.
//! Parameter updates hold that inferred state fixed; no implicit optimizer derivative.
use super::{Dataset, Partition, PredictedTrial, Predictions, Split, declared_subset};
use crate::{
    Result,
    data::IndexedGraph,
    initial_state::{self, InferenceConfig, Readout},
    model::Model,
    parameters::{Sharing, TiedParameters, forecast_defaults},
    solve::{self, Config, Method},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub prior_strength: f64,
    pub sign_prior_strength: f64,
    pub readout_prior_strength: f64,
    pub seed: u64,
    pub inference: InferenceConfig,
    pub sharing: Sharing,
}
impl Default for FitConfig {
    fn default() -> Self {
        Self {
            epochs: 2,
            learning_rate: 0.005,
            prior_strength: 0.01,
            sign_prior_strength: 0.01,
            readout_prior_strength: 0.01,
            seed: 42,
            inference: InferenceConfig {
                dt: 0.01,
                iterations: 8,
                ..Default::default()
            },
            sharing: Sharing::default(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopulationModel {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub graph_hash: String,
    pub source_commit: String,
    pub config: FitConfig,
    pub parameters: TiedParameters,
    pub readout: Readout,
    pub calibration: Readout,
    pub readout_fitted: Vec<bool>,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
    pub selected_epoch: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochReport {
    pub epoch: usize,
    pub updates: usize,
    pub training_conditional_mse: Option<f64>,
    pub validation_horizon_r2: Vec<Option<f64>>,
    pub validation_criterion: f64,
    pub elapsed_seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitReport {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub source_commit: String,
    pub config: FitConfig,
    pub criterion: String,
    pub epochs: Vec<EpochReport>,
    pub selected_epoch: usize,
    pub free_parameters: usize,
    pub calibration_statistics: usize,
    pub inferred_state_variables_per_trial: usize,
}
impl PopulationModel {
    pub fn free_parameters(&self) -> usize {
        self.parameters.free_parameters() + 2 * self.readout_fitted.iter().filter(|x| **x).count()
    }
    fn validate(&self, data: &Dataset, graph: &IndexedGraph, split: &Split) -> Result<()> {
        split.validate(data, graph)?;
        if self.schema_version != 1
            || self.dataset_hash != split.dataset_hash
            || self.split_hash != split.content_hash()?
            || self.graph_hash != graph.hash
            || self.source_commit.is_empty()
            || self.readout_fitted.len() != graph.names.len()
            || !declared_subset(&self.training_trials, &split.train)
            || !declared_subset(&self.selection_trials, &split.validation)
        {
            return Err("invalid population model lineage/dimensions".into());
        }
        self.config.validate()?;
        Ok(())
    }
    pub fn predict(
        &self,
        data: &Dataset,
        graph: &IndexedGraph,
        split: &Split,
        partition: Partition,
    ) -> Result<Predictions> {
        self.validate(data, graph, split)?;
        let model = Model::new(graph.clone())?;
        let params = self.parameters.expand(&model)?;
        let mut trials = vec![];
        for trial in data
            .trials
            .iter()
            .filter(|t| split.ids(partition).contains(&t.id))
        {
            if !trial.response_labels.is_empty() {
                return Err("population forecast does not classify stimulation labels".into());
            }
            let origin = trial.forecast_origin.ok_or("missing forecast origin")?;
            let inferred = initial_state::infer(
                &model,
                &params,
                &trial.recording,
                origin,
                &self.readout,
                &self.config.inference,
            )?;
            let times = &trial.recording.times;
            let dt = times[1] - times[0];
            if times.windows(2).any(|p| (p[1] - p[0] - dt).abs() > 1e-8) {
                return Err("population forecasts require a uniform save grid".into());
            }
            let origin_index = inferred.history_samples - 1;
            let cfg = Config {
                duration: times[times.len() - 1] - origin,
                dt: self.config.inference.dt,
                save_dt: dt,
                method: Method::Euler,
                events: vec![],
            };
            let trajectory =
                solve::simulate_from_state(&model, &params, &cfg, Some(&inferred.forecast_state))?;
            if trajectory.times.len() != times.len() - origin_index
                || trajectory
                    .times
                    .iter()
                    .zip(&times[origin_index..])
                    .any(|(a, b)| (*a + origin - b).abs() > 1e-8)
            {
                return Err("forecast output grid differs from recording".into());
            }
            let mut fluorescence = BTreeMap::new();
            for trace in &trial.recording.traces {
                let i = graph.neuron(&trace.neuron)?;
                let values: Vec<_> = inferred.history_predictions[..origin_index]
                    .iter()
                    .map(|row| row[i])
                    .chain(
                        trajectory
                            .fluorescence
                            .iter()
                            .map(|y| self.readout.offset[i] + self.readout.gain[i] * y[i]),
                    )
                    .collect();
                if values.iter().any(|v| !v.is_finite()) {
                    return Err("nonfinite population prediction".into());
                }
                fluorescence.insert(trace.neuron.clone(), values);
            }
            trials.push(PredictedTrial {
                id: trial.id.clone(),
                times: times.clone(),
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
                "Level 0 conditional-gradient population fit; selected epoch {}; full 3N history-only {:?} inference; neutral graph sign priors remain unannotated",
                self.selected_epoch, self.config.inference.method
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
impl FitConfig {
    fn validate(&self) -> Result<()> {
        if self.epochs == 0
            || self.epochs > 1000
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || [
                self.prior_strength,
                self.sign_prior_strength,
                self.readout_prior_strength,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid population fit configuration".into());
        }
        Ok(())
    }
}
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    step: usize,
}
impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
        }
    }
    fn update(&mut self, values: &mut [f64], gradient: &[f64], lr: f64) -> Result<()> {
        let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if !norm.is_finite() {
            return Err("nonfinite population gradient norm".into());
        }
        let scale = if norm > 10.0 { 10.0 / norm } else { 1.0 };
        self.step += 1;
        for i in 0..values.len() {
            let g = gradient[i] * scale;
            self.m[i] = 0.9 * self.m[i] + 0.1 * g;
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g;
            values[i] -= lr * (self.m[i] / (1.0 - 0.9f64.powf(self.step as f64)))
                / ((self.v[i] / (1.0 - 0.999f64.powf(self.step as f64))).sqrt() + 1e-8);
        }
        Ok(())
    }
}
fn validation(
    candidate: &PopulationModel,
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
) -> Result<(Vec<Option<f64>>, f64)> {
    let prediction = candidate.predict(data, graph, split, Partition::Validation)?;
    let report = super::evaluate(data, graph, split, &prediction, Partition::Validation)?;
    let values: Vec<_> = report
        .forecast_horizons
        .iter()
        .map(|h| h.macro_neuron_r2)
        .collect();
    if values.len() != 3 || values.iter().any(Option::is_none) {
        return Err("undefined population validation horizon R²".into());
    }
    let score = values.iter().flatten().sum::<f64>() / 3.0;
    if !score.is_finite() {
        return Err("nonfinite validation criterion".into());
    }
    Ok((values, score))
}
/// All training animals/windows are used. Epoch selection uses validation only;
/// the initial candidate is retained if every trained epoch performs worse.
pub fn fit(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    cfg: FitConfig,
    mut checkpoint: impl FnMut(&PopulationModel, &EpochReport) -> Result<()>,
) -> Result<(PopulationModel, FitReport)> {
    cfg.validate()?;
    if split.train.is_empty() || split.validation.is_empty() {
        return Err("fit requires nonempty training and validation partitions".into());
    }
    let (readout, unseen) = super::level0::training_readout(data, graph, split)?;
    let model = Model::new(graph.clone())?;
    let parameters = TiedParameters::new(&model, &forecast_defaults(&model), cfg.sharing.clone())?;
    let mut current = PopulationModel {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash()?,
        graph_hash: graph.hash.clone(),
        source_commit: option_env!("WORMSIM_COMMIT")
            .unwrap_or("unversioned")
            .into(),
        config: cfg.clone(),
        parameters,
        calibration: readout.clone(),
        readout,
        readout_fitted: graph
            .names
            .iter()
            .map(|name| !unseen.contains(name))
            .collect(),
        training_trials: split.train.clone(),
        selection_trials: split.validation.clone(),
        selected_epoch: 0,
    };
    let start = std::time::Instant::now();
    let (horizons, mut best_score) = validation(&current, data, graph, split)?;
    let mut best = current.clone();
    let mut epochs = vec![EpochReport {
        epoch: 0,
        updates: 0,
        training_conditional_mse: None,
        validation_horizon_r2: horizons,
        validation_criterion: best_score,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    }];
    checkpoint(&current, &epochs[0])?;
    let group_count = current.parameters.groups.len();
    let n = model.n();
    let mut optimizer = Adam::new(group_count + 2 * n);
    for epoch in 1..=cfg.epochs {
        let start = std::time::Instant::now();
        let mut loss = 0.0;
        let mut training: Vec<_> = data
            .trials
            .iter()
            .filter(|t| split.train.contains(&t.id))
            .map(|t| {
                let mut h = Sha256::new();
                h.update(cfg.seed.to_le_bytes());
                h.update((epoch as u64).to_le_bytes());
                h.update(t.id.as_bytes());
                (h.finalize(), t)
            })
            .collect();
        training.sort_by_key(|a| a.0);
        for (batch, (_, trial)) in training.iter().enumerate() {
            let params = current.parameters.expand(&model)?;
            let origin = trial.forecast_origin.ok_or("missing training origin")?;
            let inferred = initial_state::infer(
                &model,
                &params,
                &trial.recording,
                origin,
                &current.readout,
                &cfg.inference,
            )?;
            let index = trial
                .recording
                .times
                .iter()
                .position(|t| (*t - origin).abs() < 1e-9)
                .ok_or("origin absent from time grid")?;
            let mut future = trial.recording.clone();
            future.times = future.times[index..].iter().map(|t| t - origin).collect();
            for trace in &mut future.traces {
                trace.values = trace.values[index..].to_vec();
                trace.values[0] = None;
            }
            for values in future.behavior.values_mut() {
                *values = values[index..].to_vec();
            }
            let g = initial_state::parameter_gradient(
                &model,
                &params,
                &future,
                &current.readout,
                &inferred.forecast_state,
                cfg.inference.dt,
            )?;
            loss += g.value;
            let mut gradient = current.parameters.reduce_gradient(&g.parameters)?;
            let (_, prior) =
                current
                    .parameters
                    .prior(&model, cfg.prior_strength, cfg.sign_prior_strength)?;
            for (i, v) in gradient.iter_mut().enumerate() {
                *v += prior[i];
            }
            let fitted = current.readout_fitted.iter().filter(|v| **v).count().max(1) as f64;
            // Offsets are optimized in calibration-gain units, gains in log space.
            for i in 0..n {
                let delta = (current.readout.offset[i] - current.calibration.offset[i])
                    / current.calibration.gain[i];
                gradient.push(if current.readout_fitted[i] {
                    g.readout_offset[i] * current.calibration.gain[i]
                        + 2.0 * cfg.readout_prior_strength * delta / fitted
                } else {
                    0.0
                });
            }
            for i in 0..n {
                let delta = (current.readout.gain[i] / current.calibration.gain[i]).ln();
                gradient.push(if current.readout_fitted[i] {
                    g.readout_log_gain[i] + 2.0 * cfg.readout_prior_strength * delta / fitted
                } else {
                    0.0
                });
            }
            let mut values: Vec<_> = current.parameters.groups.iter().map(|g| g.value).collect();
            values.extend((0..n).map(|i| {
                (current.readout.offset[i] - current.calibration.offset[i])
                    / current.calibration.gain[i]
            }));
            values.extend(
                (0..n).map(|i| (current.readout.gain[i] / current.calibration.gain[i]).ln()),
            );
            optimizer.update(&mut values, &gradient, cfg.learning_rate)?;
            for (i, group) in current.parameters.groups.iter_mut().enumerate() {
                if group.trainable {
                    group.value = values[i];
                }
            }
            for i in 0..n {
                if current.readout_fitted[i] {
                    current.readout.offset[i] = current.calibration.offset[i]
                        + values[group_count + i] * current.calibration.gain[i];
                    current.readout.gain[i] =
                        current.calibration.gain[i] * values[group_count + n + i].exp();
                }
            }
            if (batch + 1) % 24 == 0 {
                eprintln!(
                    "epoch {epoch}: {}/{} windows; mean conditional training MSE {}",
                    batch + 1,
                    training.len(),
                    loss / (batch + 1) as f64
                );
            }
        }
        current.selected_epoch = epoch;
        let (horizons, score) = validation(&current, data, graph, split)?;
        let report = EpochReport {
            epoch,
            updates: training.len(),
            training_conditional_mse: Some(loss / training.len() as f64),
            validation_horizon_r2: horizons,
            validation_criterion: score,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        };
        checkpoint(&current, &report)?;
        epochs.push(report);
        if score > best_score {
            best_score = score;
            best = current.clone();
        }
    }
    let report=FitReport{schema_version:1,dataset_hash:split.dataset_hash.clone(),split_hash:split.content_hash()?,source_commit:current.source_commit.clone(),config:cfg,criterion:"Maximum mean validation macro-neuron R² at 1/10/30 s, including epoch 0; ties retain earlier epoch. Conditional gradients hold history-inferred state fixed. No test targets used.".into(),epochs,selected_epoch:best.selected_epoch,free_parameters:best.free_parameters(),calibration_statistics:2*best.readout_fitted.iter().filter(|x|**x).count(),inferred_state_variables_per_trial:model.state_len()};
    Ok((best, report))
}
