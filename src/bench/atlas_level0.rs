//! Deterministic Level 0 atlas response fitting with a shared positive input kernel.
use super::{
    Axis, Dataset, Partition, PredictedTrial, Predictions, Split, atlas,
    atlas_classification::{self, Classifier},
    atlas_correlation, atlas_training,
    population::Adam,
};
use crate::{
    Result,
    data::IndexedGraph,
    initial_state::{self, Readout},
    math::{Scalar, inverse_softplus},
    model::Model,
    parameters::{Sharing, TiedParameters, forecast_defaults},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfig {
    pub epochs: usize,
    pub dt: f64,
    #[serde(default)]
    pub preparation_seconds: f64,
    pub learning_rate: f64,
    #[serde(default)]
    pub learning_rate_schedule: super::optimization::LearningRateSchedule,
    #[serde(default)]
    pub optimizer: super::optimization::Optimizer,
    pub kernel_lags: usize,
    pub prior_strength: f64,
    pub sign_prior_strength: f64,
    pub kernel_prior_strength: f64,
    pub sharing: Sharing,
    #[serde(default)]
    pub classification: Option<ClassificationConfig>,
    #[serde(default)]
    pub molecular_sign_priors: Option<crate::molecular::SignPriors>,
    #[serde(default)]
    pub sign_initialization: Option<SignInitialization>,
    #[serde(default)]
    pub observation_gain: Option<ObservationGainConfig>,
    #[serde(default)]
    pub correlation: Option<CorrelationConfig>,
}
/// A declared optimization restart, not an inferred biological sign label.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignInitialization {
    pub seed: u64,
    pub reversal_magnitude: f64,
}
/// Optional pair-mean shape loss; validation selection remains trace MSE.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrelationConfig {
    pub weight: f64,
    pub epsilon: f64,
}
/// One global positive observation gain; calcium scales remain frozen.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationGainConfig {
    pub initial_gain: f64,
    /// Squared log-gain displacement from initialization, without neuron multiplicity.
    pub prior_strength: f64,
}
/// Fixed before fitting; validation selection continues to use trace MSE only.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationConfig {
    pub weight: f64,
    pub area_scale: f64,
    pub epsilon: f64,
}
impl FitConfig {
    fn validate(&self) -> Result<()> {
        if let Some(s) = &self.sign_initialization
            && (!s.reversal_magnitude.is_finite()
                || s.reversal_magnitude <= 0.
                || s.reversal_magnitude >= 1.)
        {
            return Err("invalid sign initialization magnitude".into());
        }
        self.learning_rate_schedule.validate()?;
        self.optimizer.validate()?;
        if let super::optimization::Optimizer::AdamW { weight_decay } = self.optimizer
            && (!(self.learning_rate * weight_decay).is_finite()
                || self.learning_rate * weight_decay > 1.)
        {
            return Err("AdamW base rate times decay must be at most one".into());
        }
        if let Some(c) = &self.correlation
            && (!c.weight.is_finite()
                || c.weight <= 0.
                || !c.epsilon.is_finite()
                || c.epsilon <= 0.
                || !c.epsilon.powi(2).is_finite()
                || c.epsilon.powi(2) <= 0.)
        {
            return Err("invalid correlation loss configuration".into());
        }
        if let Some(g) = &self.observation_gain
            && (!g.initial_gain.is_finite()
                || g.initial_gain <= 0.
                || !g.prior_strength.is_finite()
                || g.prior_strength < 0.)
        {
            return Err("invalid observation gain configuration".into());
        }
        if let Some(c) = &self.classification {
            if !c.weight.is_finite() || c.weight <= 0. {
                return Err("invalid classification weight".into());
            }
            Classifier {
                bias: 0.,
                raw_slope: 0.,
                area_scale: c.area_scale,
                epsilon: c.epsilon,
            }
            .validate()?;
        }
        if !self.preparation_seconds.is_finite()
            || self.preparation_seconds < 0.0
            || self.preparation_seconds > 300.0
            || self.epochs == 0
            || self.epochs > 1000
            || self.kernel_lags == 0
            || self.kernel_lags > 512
            || !self.dt.is_finite()
            || self.dt <= 0.
            || self.dt > 0.1
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || [
                self.prior_strength,
                self.sign_prior_strength,
                self.kernel_prior_strength,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
        {
            return Err("invalid Level 0 atlas fit configuration".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtlasModel {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub graph_hash: String,
    pub source_commit: String,
    pub config: FitConfig,
    pub epoch: usize,
    pub sample_dt: f64,
    pub parameters: TiedParameters,
    /// Shared fixed preparation seed. With zero preparation it is the response initial state;
    /// otherwise the response starts from its parameter-dependent unforced evolution.
    pub initial: Vec<f64>,
    /// Softplus coordinates; effective current is nonnegative and shared across targets.
    pub kernel_raw: Vec<f64>,
    pub kernel_prior: Vec<f64>,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
    #[serde(default)]
    pub classifier: Option<Classifier>,
    /// Identity of all supplied evidence; only train-target labels enter gradients.
    #[serde(default)]
    pub classification_evidence_hash: Option<String>,
    #[serde(default)]
    pub observation_log_gain: Option<f64>,
}
#[derive(Debug, Serialize)]
pub struct EpochReport {
    pub epoch: usize,
    pub applied_learning_rate: Option<f64>,
    pub preceding_training_mse: Option<f64>,
    pub preceding_penalty: Option<f64>,
    pub preceding_training_classification_bce: Option<f64>,
    pub preceding_training_pair_correlation_loss: Option<f64>,
    pub training_correlation_pairs: Option<usize>,
    pub validation_mse: f64,
    pub validation_correlation: Option<f64>,
    pub defined_trace_correlations: usize,
    pub elapsed_seconds: f64,
}
impl AtlasModel {
    pub fn readout(&self, n: usize) -> Result<Readout> {
        let gain = match (&self.config.observation_gain, self.observation_log_gain) {
            (None, None) => 1.,
            (Some(_), Some(log_gain)) if log_gain.is_finite() => log_gain.exp(),
            _ => return Err("observation gain/configuration mismatch".into()),
        };
        let readout = Readout {
            offset: vec![0.; n],
            gain: vec![gain; n],
        };
        readout.validate(n)?;
        Ok(readout)
    }
    pub fn free_parameters(&self) -> usize {
        self.parameters.free_parameters()
            + self.kernel_raw.len()
            + usize::from(self.observation_log_gain.is_some())
            + if self.classifier.is_some() { 2 } else { 0 }
    }
    fn currents(&self, target: usize, frames: usize, n: usize) -> Result<Vec<Vec<f64>>> {
        if target >= n || frames < 2 {
            return Err("invalid response target/grid".into());
        }
        let mut rows = vec![vec![0.; n]; frames];
        for (t, row) in rows.iter_mut().take(frames - 1).enumerate() {
            if t < self.kernel_raw.len() {
                row[target] = self.kernel_raw[t].softplus();
            }
        }
        Ok(rows)
    }
    /// Validate a saved checkpoint against the authoritative data and split.
    pub fn validate(&self, data: &Dataset, graph: &IndexedGraph, split: &Split) -> Result<Model> {
        split.validate(data, graph)?;
        self.config.validate()?;
        if let Some(priors) = &self.config.molecular_sign_priors {
            priors.probabilities(graph)?;
        }
        match (
            &self.config.classification,
            &self.classifier,
            &self.classification_evidence_hash,
        ) {
            (None, None, None) => {}
            (Some(c), Some(classifier), Some(hash))
                if hash.len() == 64 && hash.bytes().all(|v| v.is_ascii_hexdigit()) =>
            {
                classifier.validate()?;
                if classifier.area_scale != c.area_scale || classifier.epsilon != c.epsilon {
                    return Err("classifier scales differ from fit configuration".into());
                }
            }
            _ => return Err("invalid classification fit lineage".into()),
        }
        let model = Model::new(graph.clone())?;
        self.readout(model.n())?;
        if split.axis != Axis::StimulatedNeuron
            || self.schema_version != 1
            || self.graph_hash != graph.hash
            || self.dataset_hash != split.dataset_hash
            || self.split_hash != split.content_hash()?
            || self.source_commit.is_empty()
            || self.training_trials != split.train
            || self.selection_trials != split.validation
            || self.kernel_raw.len() != self.config.kernel_lags
            || self.kernel_prior.len() != self.kernel_raw.len()
            || self.initial.len() != model.state_len()
            || !self.sample_dt.is_finite()
            || self.sample_dt <= 0.
            || self
                .initial
                .iter()
                .chain(&self.kernel_raw)
                .chain(&self.kernel_prior)
                .any(|v| !v.is_finite())
        {
            return Err("invalid atlas Level 0 lineage or dimensions".into());
        }
        model.prepare(&self.parameters.expand(&model)?)?;
        Ok(model)
    }
    pub fn predict(
        &self,
        data: &Dataset,
        graph: &IndexedGraph,
        split: &Split,
        partition: Partition,
    ) -> Result<Predictions> {
        self.predict_dt(data, graph, split, partition, self.config.dt)
    }
    pub fn predict_dt(
        &self,
        data: &Dataset,
        graph: &IndexedGraph,
        split: &Split,
        partition: Partition,
        dt: f64,
    ) -> Result<Predictions> {
        let model = self.validate(data, graph, split)?;
        let params = self.parameters.expand(&model)?;
        let readout = self.readout(model.n())?;
        let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
        let mut cache = BTreeMap::new();
        let mut trials = vec![];
        for id in split.ids(partition) {
            let trial = indexed[id];
            let times = &trial.recording.times;
            check_grid(times, self.sample_dt)?;
            if trial.forecast_origin.is_some() {
                return Err("atlas responses cannot have forecast origins".into());
            }
            let target = graph.neuron(
                trial
                    .stimulated_neuron
                    .as_deref()
                    .ok_or("missing stimulus")?,
            )?;
            let frames = times.len();
            if let std::collections::btree_map::Entry::Vacant(e) = cache.entry((target, frames)) {
                let currents = self.currents(target, frames, model.n())?;
                e.insert(initial_state::prepared_response_with_currents(
                    &model,
                    &params,
                    &self.initial,
                    times,
                    &readout,
                    dt,
                    &currents,
                    self.config.preparation_seconds,
                )?);
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
                        fluorescence[name].iter().map(|v| v.abs()).sum::<f64>() * self.sample_dt,
                    )
                })
                .collect();
            trials.push(PredictedTrial {
                id: id.clone(),
                times: times.clone(),
                fluorescence,
                response_scores,
            });
        }
        Ok(Predictions {
            schema_version: 1,
            dataset_hash: self.dataset_hash.clone(),
            split_hash: self.split_hash.clone(),
            model: (if self.config.preparation_seconds == 0.0 {
                format!(
                    "level0-atlas-shared-positive-current-fixed-initial-state{}",
                    if self.classifier.is_some() {
                        "-joint-pair-bce"
                    } else {
                        ""
                    }
                )
            } else {
                format!(
                    "level0-atlas-shared-positive-current-preparation-{}s{}",
                    self.config.preparation_seconds,
                    if self.classifier.is_some() {
                        "-joint-pair-bce"
                    } else {
                        ""
                    }
                )
            }) + if self.config.molecular_sign_priors.is_some() {
                "-molecular-sign-priors"
            } else {
                ""
            } + if self.observation_log_gain.is_some() {
                "-learned-global-gain"
            } else {
                ""
            } + if self.config.correlation.is_some() {
                "-pair-mean-correlation"
            } else {
                ""
            } + if matches!(
                self.config.optimizer,
                super::optimization::Optimizer::AdamW { .. }
            ) {
                "-adamw"
            } else {
                ""
            } + &self
                .config
                .sign_initialization
                .as_ref()
                .map(|s| format!("-sign-seed-{}", s.seed))
                .unwrap_or_default(),
            free_parameters: self.free_parameters(),
            training_trials: self.training_trials.clone(),
            selection_trials: self.selection_trials.clone(),
            source_commit: self.source_commit.clone(),
            seed: self
                .config
                .sign_initialization
                .as_ref()
                .map_or(split.seed, |s| s.seed),
            trials,
        })
    }
}
fn check_grid(times: &[f64], dt: f64) -> Result<()> {
    if times.len() < 2
        || times[0].abs() > 1e-10
        || times.windows(2).any(|p| (p[1] - p[0] - dt).abs() > 1e-10)
    {
        return Err(
            "atlas Level 0 requires a shared uniform response grid starting at zero".into(),
        );
    }
    Ok(())
}
pub fn fit_select(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    config: FitConfig,
    checkpoint: impl FnMut(&AtlasModel, &EpochReport) -> Result<()>,
) -> Result<(AtlasModel, Vec<EpochReport>)> {
    fit_select_with_evidence(data, graph, split, config, None, checkpoint)
}
pub fn fit_select_with_evidence(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    config: FitConfig,
    evidence: Option<&atlas::Evidence>,
    mut checkpoint: impl FnMut(&AtlasModel, &EpochReport) -> Result<()>,
) -> Result<(AtlasModel, Vec<EpochReport>)> {
    config.validate()?;
    let sign_probabilities = config
        .molecular_sign_priors
        .as_ref()
        .map(|p| p.probabilities(graph))
        .transpose()?;
    let labels = match (&config.classification, evidence) {
        (Some(_), Some(evidence)) => Some(atlas_classification::training_labels(
            evidence, data, graph, split,
        )?),
        (None, None) => None,
        _ => {
            return Err(
                "classification configuration and evidence must be supplied together".into(),
            );
        }
    };
    let classifier = if let (Some(c), Some(labels)) = (&config.classification, &labels) {
        // Jeffreys smoothing uses train counts only, keeping one-class fits finite.
        let probability = (labels.detected as f64 + 0.5) / (labels.pairs as f64 + 1.);
        Some(Classifier {
            bias: (probability / (1. - probability)).ln(),
            raw_slope: inverse_softplus(1.),
            area_scale: c.area_scale,
            epsilon: c.epsilon,
        })
    } else {
        None
    };
    let groups = atlas_training::aggregate(data, graph, split)?;
    if groups.is_empty() || split.validation.is_empty() {
        return Err("atlas fit needs training and validation groups".into());
    }
    let correlation_pairs: usize = groups
        .iter()
        .map(|g| atlas_correlation::eligible_pairs(&g.recording))
        .sum();
    if config.correlation.is_some() && correlation_pairs == 0 {
        return Err("no nonconstant training pair-mean traces for correlation loss".into());
    }
    let times = &groups[0].recording.times;
    let sample_dt = times[1] - times[0];
    for g in &groups {
        check_grid(&g.recording.times, sample_dt)?;
        if labels.is_some() && g.recording.times != *times {
            return Err("joint atlas classification requires a common response grid".into());
        }
    }
    if config.kernel_lags
        >= groups
            .iter()
            .map(|g| g.recording.times.len())
            .max()
            .unwrap()
    {
        return Err("unrepresented atlas input kernel lag".into());
    }
    let model = Model::new(graph.clone())?;
    let raw = forecast_defaults(&model);
    let initial = model.initial(&model.prepare(&raw)?);
    let mut parameters = TiedParameters::new(&model, &raw, config.sharing.clone())?;
    if let Some(probabilities) = &sign_probabilities {
        parameters.initialize_sign_priors(&model, probabilities)?;
    }
    if let Some(s) = &config.sign_initialization {
        let probabilities = sign_probabilities
            .clone()
            .unwrap_or_else(|| model.graph.chemical.iter().map(|e| e.3).collect());
        parameters.initialize_sign_restart(&model, &probabilities, s.seed, s.reversal_magnitude)?;
    }
    let kernel_raw: Vec<_> = (0..config.kernel_lags)
        .map(|t| inverse_softplus(0.2 * (-(t as f64) * sample_dt / 2.).exp()))
        .collect();
    let mut current = AtlasModel {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash()?,
        graph_hash: graph.hash.clone(),
        source_commit: option_env!("WORMSIM_COMMIT")
            .unwrap_or("unversioned")
            .into(),
        config: config.clone(),
        epoch: 0,
        sample_dt,
        parameters,
        initial,
        kernel_prior: kernel_raw.clone(),
        kernel_raw,
        training_trials: split.train.clone(),
        selection_trials: split.validation.clone(),
        classifier,
        classification_evidence_hash: labels.as_ref().map(|l| l.evidence_hash.clone()),
        observation_log_gain: config
            .observation_gain
            .as_ref()
            .map(|g| g.initial_gain.ln()),
    };
    let pcount = current.parameters.groups.len();
    let classifier_index = pcount + config.kernel_lags;
    let gain_index = classifier_index + if labels.is_some() { 2 } else { 0 };
    let parameter_count = gain_index + usize::from(current.observation_log_gain.is_some());
    let mut optimizer = Adam::new(parameter_count);
    let mut trainable: Vec<_> = current
        .parameters
        .groups
        .iter()
        .map(|g| g.trainable)
        .collect();
    trainable.resize(parameter_count, true);
    let total_weight = groups.iter().map(|g| g.sample_weight).sum::<f64>();
    let mut best = f64::INFINITY;
    let mut selected = None;
    let mut reports = vec![];
    for epoch in 0..=config.epochs {
        let start = std::time::Instant::now();
        let rate = if epoch > 0 {
            Some(
                config
                    .learning_rate_schedule
                    .rate(config.learning_rate, epoch, config.epochs)?,
            )
        } else {
            None
        };
        let mut training = None;
        let mut penalty = None;
        let mut classification_bce = None;
        let mut correlation_loss = None;
        if epoch > 0 {
            let readout = current.readout(model.n())?;
            let raw = current.parameters.expand(&model)?;
            let mut gradient = vec![0.; parameter_count];
            let mut bce = 0.;
            let mut correlation_sum = 0.;
            let mut mse = 0.;
            for group in &groups {
                let target = graph.neuron(&group.stimulated_neuron)?;
                let currents = current.currents(target, group.recording.times.len(), model.n())?;
                let w = group.sample_weight / total_weight;
                let (g, gradient_weight) = if current.classifier.is_some()
                    || config.correlation.is_some()
                {
                    let g = initial_state::prepared_response_objective_gradient(
                        &model,
                        &raw,
                        &readout,
                        &current.initial,
                        &group.recording.times,
                        config.dt,
                        &currents,
                        config.preparation_seconds,
                        |response| {
                            let mut objective = 0.;
                            let mut fluorescence = vec![vec![0.; model.n()]; response.len()];
                            if let (Some(classifier), Some(labels), Some(c)) =
                                (&current.classifier, &labels, &config.classification)
                            {
                                let pair_labels = labels
                                    .by_target
                                    .get(&target)
                                    .map(Vec::as_slice)
                                    .unwrap_or(&[]);
                                let classification =
                                    classifier.loss(response, pair_labels, sample_dt)?;
                                let scale = c.weight / labels.pairs as f64;
                                bce += classification.value / labels.pairs as f64;
                                objective += scale * classification.value;
                                gradient[classifier_index] += scale * classification.bias_gradient;
                                gradient[classifier_index + 1] +=
                                    scale * classification.raw_slope_gradient;
                                for (a, b) in fluorescence
                                    .iter_mut()
                                    .flatten()
                                    .zip(classification.fluorescence.iter().flatten())
                                {
                                    *a += scale * b;
                                }
                            }
                            if let Some(c) = &config.correlation {
                                let shape = atlas_correlation::loss(
                                    &group.recording,
                                    graph,
                                    response,
                                    c.epsilon,
                                )?;
                                let scale = c.weight / correlation_pairs as f64;
                                correlation_sum += shape.value / correlation_pairs as f64;
                                objective += scale * shape.value;
                                for (a, b) in fluorescence
                                    .iter_mut()
                                    .flatten()
                                    .zip(shape.fluorescence.iter().flatten())
                                {
                                    *a += scale * b;
                                }
                            }
                            let weight = group
                                .recording
                                .traces
                                .iter()
                                .map(|trace| {
                                    trace.provenance.id_confidence
                                        * trace.values.iter().filter(|v| v.is_some()).count() as f64
                                })
                                .sum::<f64>();
                            if weight <= 0. {
                                return Err("empty aggregate trace objective".into());
                            }
                            let mut error = 0.;
                            for trace in &group.recording.traces {
                                let i = graph.neuron(&trace.neuron)?;
                                let scale = trace.provenance.id_confidence / weight;
                                for (t, value) in trace.values.iter().enumerate() {
                                    if let Some(value) = value {
                                        let delta = response[t][i] - value;
                                        error += scale * delta * delta;
                                        fluorescence[t][i] += w * 2. * scale * delta;
                                    }
                                }
                            }
                            mse += w * (error + group.irreducible_mse);
                            Ok((w * error + objective, fluorescence))
                        },
                    )?;
                    (g, 1.)
                } else {
                    let g = initial_state::prepared_response_gradient_with_currents(
                        &model,
                        &raw,
                        &group.recording,
                        &readout,
                        &current.initial,
                        config.dt,
                        &currents,
                        config.preparation_seconds,
                    )?;
                    mse += w * (g.value + group.irreducible_mse);
                    (g, w)
                };
                if current.observation_log_gain.is_some() {
                    gradient[gain_index] +=
                        gradient_weight * g.readout_log_gain.iter().sum::<f64>();
                }
                for (a, b) in gradient
                    .iter_mut()
                    .zip(current.parameters.reduce_gradient(&g.parameters)?)
                {
                    *a += gradient_weight * b;
                }
                for t in 0..config.kernel_lags.min(g.currents.len() - 1) {
                    gradient[pcount + t] +=
                        gradient_weight * g.currents[t][target] * current.kernel_raw[t].sigmoid();
                }
            }
            if labels.is_some() {
                classification_bce = Some(bce);
            }
            if config.correlation.is_some() {
                correlation_loss = Some(correlation_sum);
            }

            let (mut loss, prior) = current.parameters.prior_with_sign_probabilities(
                &model,
                config.prior_strength,
                config.sign_prior_strength,
                sign_probabilities.as_deref(),
            )?;
            for (a, b) in gradient.iter_mut().zip(prior) {
                *a += b;
            }
            for t in 0..config.kernel_lags {
                let delta = current.kernel_raw[t] - current.kernel_prior[t];
                let scale = config.kernel_prior_strength / config.kernel_lags as f64;
                loss += scale * delta * delta;
                gradient[pcount + t] += 2. * scale * delta;
            }
            let mut values: Vec<_> = current
                .parameters
                .groups
                .iter()
                .map(|g| g.value)
                .chain(current.kernel_raw.iter().copied())
                .collect();
            if let Some(classifier) = &current.classifier {
                values.extend([classifier.bias, classifier.raw_slope]);
            }
            if let (Some(log_gain), Some(g)) =
                (current.observation_log_gain, &config.observation_gain)
            {
                let delta = log_gain - g.initial_gain.ln();
                loss += g.prior_strength * delta * delta;
                gradient[gain_index] += 2. * g.prior_strength * delta;
                values.push(log_gain);
            }
            config.optimizer.update(
                &mut optimizer,
                &mut values,
                &gradient,
                &trainable,
                rate.ok_or("missing update learning rate")?,
            )?;
            if current.observation_log_gain.is_some() {
                current.observation_log_gain = Some(values[gain_index]);
            }
            for (i, g) in current.parameters.groups.iter_mut().enumerate() {
                if g.trainable {
                    g.value = values[i];
                }
            }
            current
                .kernel_raw
                .copy_from_slice(&values[pcount..classifier_index]);
            if let Some(classifier) = &mut current.classifier {
                classifier.bias = values[classifier_index];
                classifier.raw_slope = values[classifier_index + 1];
            }
            training = Some(mse);
            penalty = Some(loss);
        }
        current.epoch = epoch;
        let pred = current.predict(data, graph, split, Partition::Validation)?;
        let score = super::evaluate(data, graph, split, &pred, Partition::Validation)?;
        let mse = score
            .pooled_trace_scores
            .mse
            .ok_or("no validation observations")?;
        let report = EpochReport {
            epoch,
            applied_learning_rate: rate,
            preceding_training_mse: training,
            preceding_penalty: penalty,
            preceding_training_classification_bce: classification_bce,
            preceding_training_pair_correlation_loss: correlation_loss,
            training_correlation_pairs: config.correlation.as_ref().map(|_| correlation_pairs),
            validation_mse: mse,
            validation_correlation: score.macro_trace_correlation,
            defined_trace_correlations: score.defined_trace_correlations,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        };
        checkpoint(&current, &report)?;
        if mse < best {
            best = mse;
            selected = Some(current.clone());
        }
        reports.push(report);
    }
    Ok((selected.ok_or("no finite atlas checkpoint")?, reports))
}
