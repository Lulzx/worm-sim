//! Dense ridge-fitted linear dynamics in standardized fluorescence coordinates.
//! Normalization and dynamics use training animals only. Missing inputs use the
//! training mean; unobserved-in-training output neurons use persistence.
use super::{Axis, Dataset, Partition, PredictedTrial, Predictions, Split};
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearModel {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub source_commit: String,
    pub sample_dt: f64,
    pub ridge: f64,
    pub neurons: Vec<String>,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    /// Row-major [output, input], with the intercept as the final input.
    pub coefficients: Vec<f64>,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
    pub training_target_weight: Vec<f64>,
}
#[derive(Debug)]
pub struct Statistics {
    neurons: Vec<String>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    dt: f64,
    gram: Vec<f64>,
    cross: Vec<f64>,
    weight: Vec<f64>,
    data_hash: String,
    split_hash: String,
    training_trials: Vec<String>,
}
/// Statistics are reusable across ridge candidates; no validation outcomes enter.
pub fn training_statistics(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
) -> Result<Statistics> {
    split.validate(data, graph)?;
    if split.axis != Axis::Animal {
        return Err("linear forecast fitting requires animal splits".into());
    }
    let training: BTreeSet<_> = split.train.iter().map(String::as_str).collect();
    let mut ordered: Vec<_> = data
        .trials
        .iter()
        .filter(|t| training.contains(t.id.as_str()))
        .collect();
    ordered.sort_by(|a, b| (&a.recording.animal_id, &a.id).cmp(&(&b.recording.animal_id, &b.id)));
    let mut moments: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();
    let mut dt = None;
    for trial in &ordered {
        for pair in trial.recording.times.windows(2) {
            let step = pair[1] - pair[0];
            if dt.is_some_and(|d: f64| (step - d).abs() > 1e-8) {
                return Err("linear training requires a uniform shared time grid".into());
            }
            dt = Some(step);
        }
        for trace in &trial.recording.traces {
            let w = trace.provenance.id_confidence;
            if w == 0.0 {
                continue;
            }
            let m = moments.entry(trace.neuron.clone()).or_default();
            for &v in trace.values.iter().flatten() {
                let total = m.0 + w;
                let delta = v - m.1;
                m.1 += w / total * delta;
                m.2 += w * delta * (v - m.1);
                m.0 = total;
            }
        }
    }
    moments.retain(|_, v| v.0 > 0.0);
    let neurons: Vec<_> = moments.keys().cloned().collect();
    let n = neurons.len();
    let d = n + 1;
    if n == 0 || n > 1024 {
        return Err("linear fit needs 1..1024 observed training neurons".into());
    }
    let mean: Vec<_> = moments.values().map(|v| v.1).collect();
    let scale: Vec<_> = moments
        .values()
        .map(|v| (v.2 / v.0).max(0.0).sqrt().max(1e-8))
        .collect();
    let index: BTreeMap<_, _> = neurons
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let triangle = d * (d + 1) / 2;
    let mut gram = vec![0.0; n * triangle];
    let mut cross = vec![0.0; n * d];
    let mut weight = vec![0.0; n];
    // Share one Gram matrix for all fully observed output traces in each trial.
    // Missing targets get a correction so they never become zero-valued targets.
    let mut shared = vec![0.0; triangle];
    let mut x = vec![0.0; d];
    for trial in &ordered {
        shared.fill(0.0);
        let traces: Vec<_> = trial
            .recording
            .traces
            .iter()
            .filter_map(|trace| index.get(trace.neuron.as_str()).map(|&i| (i, trace)))
            .filter(|(_, t)| t.provenance.id_confidence > 0.0)
            .collect();
        let mut active: Vec<_> = traces
            .iter()
            .map(|(i, _)| *i)
            .chain(std::iter::once(n))
            .collect();
        active.sort_unstable();
        for t in 0..trial.recording.times.len().saturating_sub(1) {
            x.fill(0.0);
            x[n] = 1.0;
            for &(i, trace) in &traces {
                if let Some(v) = trace.values[t] {
                    x[i] = (v - mean[i]) / scale[i];
                }
            }
            for (position, &j) in active.iter().enumerate() {
                for &k in &active[..=position] {
                    shared[j * (j + 1) / 2 + k] += x[j] * x[k];
                }
            }
            for &(i, trace) in &traces {
                let w = trace.provenance.id_confidence;
                if let Some(y) = trace.values[t + 1] {
                    let y = (y - mean[i]) / scale[i];
                    weight[i] += w;
                    for &j in &active {
                        cross[i * d + j] += w * x[j] * y;
                    }
                } else {
                    let offset = i * triangle;
                    for (position, &j) in active.iter().enumerate() {
                        for &k in &active[..=position] {
                            gram[offset + j * (j + 1) / 2 + k] -= w * x[j] * x[k];
                        }
                    }
                }
            }
        }
        for &(i, trace) in &traces {
            let offset = i * triangle;
            let w = trace.provenance.id_confidence;
            for (position, &j) in active.iter().enumerate() {
                for &k in &active[..=position] {
                    let at = j * (j + 1) / 2 + k;
                    gram[offset + at] += w * shared[at];
                }
            }
        }
    }
    if weight.iter().any(|w| *w <= 0.0)
        || gram
            .iter()
            .chain(&cross)
            .chain(&mean)
            .chain(&scale)
            .any(|v| !v.is_finite())
    {
        return Err("nonfinite training statistics or neuron without observed transitions".into());
    }
    Ok(Statistics {
        neurons,
        mean,
        scale,
        dt: dt.ok_or("no training transitions")?,
        gram,
        cross,
        weight,
        data_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash()?,
        training_trials: split.train.clone(),
    })
}
/// Cholesky for a strictly ridge-regularized symmetric normal equation.
fn solve_spd(a: &mut [f64], b: &mut [f64]) -> Result<()> {
    let n = b.len();
    for i in 0..n {
        for j in 0..=i {
            let mut value = a[i * n + j];
            for k in 0..j {
                value -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if value <= 0.0 || !value.is_finite() {
                    return Err("non-positive ridge normal matrix".into());
                }
                a[i * n + j] = value.sqrt();
            } else {
                a[i * n + j] = value / a[j * n + j];
            }
        }
    }
    for i in 0..n {
        let mut v = b[i];
        for j in 0..i {
            v -= a[i * n + j] * b[j];
        }
        b[i] = v / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut v = b[i];
        for j in i + 1..n {
            v -= a[j * n + i] * b[j];
        }
        b[i] = v / a[i * n + i];
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite linear coefficients".into());
    }
    Ok(())
}
impl Statistics {
    pub fn fit(&self, ridge: f64) -> Result<LinearModel> {
        if !ridge.is_finite() || ridge <= 0.0 {
            return Err("ridge must be finite and strictly positive".into());
        }
        let n = self.neurons.len();
        let d = n + 1;
        let mut coefficients = vec![0.0; n * d];
        let mut a = vec![0.0; d * d];
        for i in 0..n {
            let triangle = d * (d + 1) / 2;
            for row in 0..d {
                for col in 0..=row {
                    a[row * d + col] =
                        self.gram[i * triangle + row * (row + 1) / 2 + col] / self.weight[i];
                }
            }
            for j in 0..d {
                a[j * d + j] += ridge;
                coefficients[i * d + j] = self.cross[i * d + j] / self.weight[i];
            }
            solve_spd(&mut a, &mut coefficients[i * d..(i + 1) * d])?;
        }
        Ok(LinearModel {
            schema_version: 1,
            dataset_hash: self.data_hash.clone(),
            split_hash: self.split_hash.clone(),
            source_commit: option_env!("WORMSIM_COMMIT")
                .unwrap_or("unversioned")
                .into(),
            sample_dt: self.dt,
            ridge,
            neurons: self.neurons.clone(),
            mean: self.mean.clone(),
            scale: self.scale.clone(),
            coefficients,
            training_trials: self.training_trials.clone(),
            selection_trials: vec![],
            training_target_weight: self.weight.clone(),
        })
    }
}
impl LinearModel {
    pub fn free_parameters(&self) -> usize {
        self.coefficients.len() + self.mean.len() + self.scale.len()
    }
    pub fn predict(
        &self,
        data: &Dataset,
        graph: &IndexedGraph,
        split: &Split,
        partition: Partition,
    ) -> Result<Predictions> {
        split.validate(data, graph)?;
        let n = self.neurons.len();
        let d = n + 1;
        if self.schema_version != 1
            || self.dataset_hash != split.dataset_hash
            || self.split_hash != split.content_hash()?
            || n == 0
            || n > 1024
            || self.training_target_weight.len() != n
            || self
                .training_target_weight
                .iter()
                .any(|w| !w.is_finite() || *w <= 0.0)
            || !self.ridge.is_finite()
            || self.ridge <= 0.0
            || self.source_commit.trim().is_empty()
            || self.mean.len() != n
            || self.scale.len() != n
            || self.coefficients.len() != n * d
            || self.neurons.windows(2).any(|p| p[0] >= p[1])
            || self.scale.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self
                .mean
                .iter()
                .chain(&self.coefficients)
                .any(|v| !v.is_finite())
            || !self.sample_dt.is_finite()
            || self.sample_dt <= 0.0
            || !super::declared_subset(&self.training_trials, &split.train)
            || !super::declared_subset(&self.selection_trials, &split.validation)
        {
            return Err("invalid fitted linear model or incompatible data/split".into());
        }
        for name in &self.neurons {
            graph.neuron(name)?;
        }
        let index: BTreeMap<_, _> = self
            .neurons
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();
        let mut trials = vec![];
        for trial in data
            .trials
            .iter()
            .filter(|t| split.ids(partition).contains(&t.id))
        {
            if !trial.response_labels.is_empty() {
                return Err("linear forecasting does not classify stimulation responses".into());
            }
            if trial
                .recording
                .times
                .windows(2)
                .any(|p| (p[1] - p[0] - self.sample_dt).abs() > 1e-8)
            {
                return Err("forecast time step differs from training".into());
            }
            let origin = trial.forecast_origin.ok_or("forecast origin required")?;
            let mut state = vec![0.0; n];
            let mut next = state.clone();
            let mut fluorescence: BTreeMap<String, Vec<f64>> = trial
                .recording
                .traces
                .iter()
                .map(|trace| {
                    (
                        trace.neuron.clone(),
                        Vec::with_capacity(trial.recording.times.len()),
                    )
                })
                .collect();
            let mut fallback = BTreeMap::new();
            for trace in &trial.recording.traces {
                let last = trial
                    .recording
                    .times
                    .iter()
                    .zip(&trace.values)
                    .take_while(|(t, _)| **t <= origin)
                    .filter_map(|(_, v)| *v)
                    .last()
                    .ok_or("no observed prefix for forecast output")?;
                fallback.insert(trace.neuron.as_str(), last);
            }
            for (t, &time) in trial.recording.times.iter().enumerate() {
                if t > 0 {
                    for (i, value) in next.iter_mut().enumerate() {
                        let row = &self.coefficients[i * d..(i + 1) * d];
                        *value =
                            row[n] + row[..n].iter().zip(&state).map(|(a, b)| a * b).sum::<f64>();
                    }
                    std::mem::swap(&mut state, &mut next);
                }
                if time <= origin {
                    for trace in &trial.recording.traces {
                        if trace.provenance.id_confidence > 0.0
                            && let Some(&i) = index.get(trace.neuron.as_str())
                            && let Some(value) = trace.values[t]
                        {
                            state[i] = (value - self.mean[i]) / self.scale[i];
                        }
                    }
                }
                for trace in &trial.recording.traces {
                    let value = match index.get(trace.neuron.as_str()) {
                        Some(&i) => state[i] * self.scale[i] + self.mean[i],
                        None => fallback[trace.neuron.as_str()],
                    };
                    if !value.is_finite() {
                        return Err("linear forecast diverged".into());
                    }
                    fluorescence.get_mut(&trace.neuron).unwrap().push(value);
                }
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
            model: format!("dense linear fluorescence dynamics; ridge={}", self.ridge),
            free_parameters: self.free_parameters(),
            training_trials: self.training_trials.clone(),
            selection_trials: self.selection_trials.clone(),
            source_commit: self.source_commit.clone(),
            seed: split.seed,
            trials,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub ridge: f64,
    pub validation_mean_horizon_r2: Option<f64>,
    pub horizon_r2: Vec<Option<f64>>,
    pub elapsed_seconds: f64,
    pub error: Option<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct SelectionReport {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub criterion: String,
    pub training_statistics_seconds: f64,
    pub candidates: Vec<Candidate>,
    pub selected_ridge: f64,
    pub free_parameters: usize,
}
pub fn fit_select(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    ridges: &[f64],
) -> Result<(LinearModel, SelectionReport)> {
    use std::time::Instant;
    if split.validation.is_empty()
        || ridges.is_empty()
        || ridges.iter().any(|r| !r.is_finite() || *r <= 0.0)
    {
        return Err(
            "model selection requires validation animals and positive ridge candidates".into(),
        );
    }
    let start = Instant::now();
    let stats = training_statistics(data, graph, split)?;
    let statistics_seconds = start.elapsed().as_secs_f64();
    let mut candidates = vec![];
    let mut best: Option<(f64, LinearModel)> = None;
    for &ridge in ridges {
        let start = Instant::now();
        let attempt = (|| -> Result<_> {
            let model = stats.fit(ridge)?;
            let predictions = model.predict(data, graph, split, Partition::Validation)?;
            let report = super::evaluate(data, graph, split, &predictions, Partition::Validation)?;
            let horizons: Vec<_> = report
                .forecast_horizons
                .iter()
                .map(|h| h.macro_neuron_r2)
                .collect();
            if horizons.len() != 3 || horizons.iter().any(Option::is_none) {
                return Err("undefined validation horizon R²".into());
            }
            let score = horizons.iter().flatten().sum::<f64>() / 3.0;
            if !score.is_finite() {
                return Err("nonfinite validation criterion".into());
            }
            Ok((model, score, horizons))
        })();
        match attempt {
            Ok((model, score, horizons)) => {
                candidates.push(Candidate {
                    ridge,
                    validation_mean_horizon_r2: Some(score),
                    horizon_r2: horizons,
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                    error: None,
                });
                if best.as_ref().is_none_or(|(previous, _)| score > *previous) {
                    best = Some((score, model));
                }
            }
            Err(error) => candidates.push(Candidate {
                ridge,
                validation_mean_horizon_r2: None,
                horizon_r2: vec![],
                elapsed_seconds: start.elapsed().as_secs_f64(),
                error: Some(error),
            }),
        }
    }
    let (_, mut model) = best.ok_or("all linear candidates failed validation")?;
    model.selection_trials = split.validation.clone();
    let report=SelectionReport {schema_version:1,dataset_hash:split.dataset_hash.clone(),split_hash:split.content_hash()?,criterion:"Maximum arithmetic mean of macro per-neuron R² at 1, 10, 30 seconds on validation animals; ties retain earlier declared candidate".into(),training_statistics_seconds:statistics_seconds,candidates,selected_ridge:model.ridge,free_parameters:model.free_parameters()};
    Ok((model, report))
}
