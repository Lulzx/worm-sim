//! Stable latent Gaussian LDS: masked Kalman filtering, RTS smoothing and
//! constrained EM moment updates. Forecasts condition on observed history only.
use super::{Axis, Dataset, Partition, PredictedTrial, Predictions, Split, lds_math::*};
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianLds {
    pub dim: usize,
    pub outputs: usize,
    pub transition: Vec<f64>,
    pub observation: Vec<f64>,
    pub process_cov: Vec<f64>,
    pub noise: Vec<f64>,
    pub initial_cov: Vec<f64>,
}
/// (observed output index, standardized value, positive identity confidence).
pub type Observation = (usize, f64, f64);
#[derive(Debug)]
pub struct Posterior {
    pub means: Vec<Vec<f64>>,
    pub covariances: Vec<Vec<f64>>,
    pub lag_covariances: Vec<Vec<f64>>,
    pub negative_log_likelihood: f64,
    pub observations: usize,
}
impl GaussianLds {
    pub fn validate(&self) -> Result<()> {
        let k = self.dim;
        let n = self.outputs;
        if k == 0
            || k > 64
            || n == 0
            || n > 1024
            || self.transition.len() != k * k
            || self.observation.len() != n * k
            || self.process_cov.len() != k * k
            || self.initial_cov.len() != k * k
            || self.noise.len() != n
            || self
                .transition
                .iter()
                .chain(&self.observation)
                .chain(&self.process_cov)
                .chain(&self.initial_cov)
                .chain(&self.noise)
                .any(|v| !v.is_finite())
            || self.noise.iter().any(|v| *v <= 0.0)
        {
            return Err("invalid Gaussian LDS shapes/values".into());
        }
        for a in [&self.process_cov, &self.initial_cov] {
            if (0..k).any(|i| (0..k).any(|j| (a[i * k + j] - a[j * k + i]).abs() > 1e-9)) {
                return Err("asymmetric LDS covariance".into());
            }
            chol(a, k)?;
        }
        Ok(())
    }
    pub fn transition_norm_bound(&self) -> Result<f64> {
        self.validate()?;
        norm_bound(&self.transition, self.dim)
    }
    fn advance(&self, mean: &[f64], cov: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let k = self.dim;
        let next = mv(&self.transition, mean, k, k);
        let mut p = mm(
            &mm(&self.transition, cov, k, k, k),
            &transpose(&self.transition, k, k),
            k,
            k,
            k,
        );
        for (v, q) in p.iter_mut().zip(&self.process_cov) {
            *v += q;
        }
        (next, p)
    }
    fn observe(
        &self,
        mean: &mut [f64],
        cov: &mut [f64],
        values: &[Observation],
    ) -> Result<(f64, usize)> {
        let k = self.dim;
        let mut nll = 0.0;
        let mut count = 0;
        for &(i, y, w) in values {
            if i >= self.outputs || !y.is_finite() || !w.is_finite() || w <= 0.0 || w > 1.0 {
                return Err("invalid LDS observation".into());
            }
            let row = &self.observation[i * k..(i + 1) * k];
            let u = mv(cov, row, k, k);
            let variance = row.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>() + self.noise[i] / w;
            if !variance.is_finite() || variance <= 0.0 {
                return Err("invalid innovation variance".into());
            }
            let residual = y - row.iter().zip(mean.iter()).map(|(a, b)| a * b).sum::<f64>();
            nll += 0.5
                * ((2.0 * std::f64::consts::PI * variance).ln() + residual * residual / variance);
            count += 1;
            for a in 0..k {
                mean[a] += u[a] * residual / variance;
                for b in 0..k {
                    cov[a * k + b] -= u[a] * u[b] / variance;
                }
            }
        }
        symmetrize(cov, k);
        if mean.iter().chain(cov.iter()).any(|v| !v.is_finite()) || !nll.is_finite() {
            return Err("nonfinite Kalman update".into());
        }
        Ok((nll, count))
    }
    /// Smoothing is for training only; prediction never invokes this method.
    pub fn smooth(&self, sequence: &[Vec<Observation>]) -> Result<Posterior> {
        self.validate()?;
        if sequence.is_empty() {
            return Err("empty LDS sequence".into());
        }
        let k = self.dim;
        let mut mean = vec![0.0; k];
        let mut cov = self.initial_cov.clone();
        let mut means = vec![];
        let mut covs = vec![];
        let mut predicted_means = vec![];
        let mut predicted_covs = vec![];
        let mut nll = 0.0;
        let mut count = 0;
        for (t, values) in sequence.iter().enumerate() {
            if t > 0 {
                (mean, cov) = self.advance(&mean, &cov);
            }
            predicted_means.push(mean.clone());
            predicted_covs.push(cov.clone());
            let (l, n) = self.observe(&mut mean, &mut cov, values)?;
            nll += l;
            count += n;
            means.push(mean.clone());
            covs.push(cov.clone());
        }
        let mut lag = vec![vec![0.0; k * k]; sequence.len().saturating_sub(1)];
        let at = transpose(&self.transition, k, k);
        for t in (0..sequence.len() - 1).rev() {
            let j = mm(
                &mm(&covs[t], &at, k, k, k),
                &inverse(&predicted_covs[t + 1], k)?,
                k,
                k,
                k,
            );
            let delta: Vec<_> = means[t + 1]
                .iter()
                .zip(&predicted_means[t + 1])
                .map(|(a, b)| a - b)
                .collect();
            let adjustment = mv(&j, &delta, k, k);
            for (i, v) in means[t].iter_mut().enumerate() {
                *v += adjustment[i];
            }
            lag[t] = mm(&covs[t + 1], &transpose(&j, k, k), k, k, k);
            let diff: Vec<_> = covs[t + 1]
                .iter()
                .zip(&predicted_covs[t + 1])
                .map(|(a, b)| a - b)
                .collect();
            let update = mm(&mm(&j, &diff, k, k, k), &transpose(&j, k, k), k, k, k);
            for (v, u) in covs[t].iter_mut().zip(update) {
                *v += u;
            }
            symmetrize(&mut covs[t], k);
        }
        Ok(Posterior {
            means,
            covariances: covs,
            lag_covariances: lag,
            negative_log_likelihood: nll,
            observations: count,
        })
    }
    /// Fit one masked EM moment update. Stability projection and ridge/floors mean
    /// likelihood monotonicity is not guaranteed; every training NLL is reported.
    pub fn em_step(
        &mut self,
        sequences: &[Vec<Vec<Observation>>],
        cap: f64,
        ridge: f64,
    ) -> Result<f64> {
        self.validate()?;
        if sequences.is_empty()
            || !cap.is_finite()
            || cap <= 0.0
            || cap >= 1.0
            || !ridge.is_finite()
            || ridge < 0.0
        {
            return Err("invalid LDS EM settings".into());
        }
        let k = self.dim;
        let n = self.outputs;
        let mut s00 = vec![0.0; k * k];
        let mut s11 = s00.clone();
        let mut s10 = s00.clone();
        let mut initial = s00.clone();
        let mut emission = vec![0.0; n * k * k];
        let mut cross = vec![0.0; n * k];
        let mut square = vec![0.0; n];
        let mut count = vec![0usize; n];
        let mut transitions = 0;
        let mut nll = 0.0;
        let mut observed = 0;
        for sequence in sequences {
            let p = self.smooth(sequence)?;
            nll += p.negative_log_likelihood;
            observed += p.observations;
            for (t, values) in sequence.iter().enumerate() {
                let mut moment = p.covariances[t].clone();
                for a in 0..k {
                    for b in 0..k {
                        moment[a * k + b] += p.means[t][a] * p.means[t][b];
                    }
                }
                if t == 0 {
                    for (v, m) in initial.iter_mut().zip(&moment) {
                        *v += m;
                    }
                }
                if t + 1 < sequence.len() {
                    transitions += 1;
                    for (v, m) in s00.iter_mut().zip(&moment) {
                        *v += m;
                    }
                }
                if t > 0 {
                    for (v, m) in s11.iter_mut().zip(&moment) {
                        *v += m;
                    }
                    for a in 0..k {
                        for b in 0..k {
                            s10[a * k + b] += p.lag_covariances[t - 1][a * k + b]
                                + p.means[t][a] * p.means[t - 1][b];
                        }
                    }
                }
                for &(i, y, w) in values {
                    count[i] += 1;
                    square[i] += w * y * y;
                    for a in 0..k {
                        cross[i * k + a] += w * y * p.means[t][a];
                        for b in 0..k {
                            emission[i * k * k + a * k + b] += w * moment[a * k + b];
                        }
                    }
                }
            }
        }
        if transitions == 0 || observed == 0 || count.contains(&0) {
            return Err("LDS EM has no transitions or an unobserved training output".into());
        }
        let mut regularized = s00.clone();
        for i in 0..k {
            regularized[i * k + i] += ridge * transitions as f64;
        }
        let mut a = mm(&s10, &inverse(&regularized, k)?, k, k, k);
        stabilize(&mut a, k, cap)?;
        let as10t = mm(&a, &transpose(&s10, k, k), k, k, k);
        let as00at = mm(&mm(&a, &s00, k, k, k), &transpose(&a, k, k), k, k, k);
        let mut q = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                q[i * k + j] = (s11[i * k + j] - as10t[i * k + j] - as10t[j * k + i]
                    + as00at[i * k + j])
                    / transitions as f64;
            }
        }
        for i in 0..k {
            q[i * k + i] += 1e-6;
        }
        symmetrize(&mut q, k);
        chol(&q, k)?;
        let mut c = vec![0.0; n * k];
        let mut noise = vec![0.0; n];
        for i in 0..n {
            let raw = &emission[i * k * k..(i + 1) * k * k];
            let mut gram = raw.to_vec();
            for j in 0..k {
                gram[j * k + j] += ridge * count[i] as f64;
            }
            let row = solve(&chol(&gram, k)?, &cross[i * k..(i + 1) * k], k);
            let dot = row
                .iter()
                .zip(&cross[i * k..(i + 1) * k])
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let projected = mv(raw, &row, k, k);
            let error =
                square[i] - 2.0 * dot + row.iter().zip(projected).map(|(a, b)| a * b).sum::<f64>();
            noise[i] = (error / count[i] as f64).max(1e-4);
            c[i * k..(i + 1) * k].copy_from_slice(&row);
        }
        for v in &mut initial {
            *v /= sequences.len() as f64;
        }
        for i in 0..k {
            initial[i * k + i] += 1e-6;
        }
        symmetrize(&mut initial, k);
        self.transition = a;
        self.observation = c;
        self.process_cov = q;
        self.noise = noise;
        self.initial_cov = initial;
        self.validate()?;
        Ok(nll / observed as f64)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfig {
    pub ranks: Vec<usize>,
    pub iterations: usize,
    pub ridge: f64,
    pub transition_cap: f64,
}
impl Default for FitConfig {
    fn default() -> Self {
        Self {
            ranks: vec![4, 8, 16, 32],
            iterations: 8,
            ridge: 1e-4,
            transition_cap: 0.995,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentModel {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub graph_hash: String,
    pub source_commit: String,
    pub sample_dt: f64,
    pub neurons: Vec<String>,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub gaussian: GaussianLds,
    pub transition_cap: f64,
    pub ridge: f64,
    pub iteration: usize,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub rank: usize,
    pub iteration: usize,
    pub preceding_training_nll_per_observation: Option<f64>,
    pub validation_horizon_r2: Vec<Option<f64>>,
    pub validation_criterion: f64,
    pub transition_norm_bound: f64,
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
    pub selected_rank: usize,
    pub selected_iteration: usize,
    pub free_parameters: usize,
    pub training_preparation_seconds: f64,
}
struct Training {
    neurons: Vec<String>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    dt: f64,
    sequences: Vec<Vec<Vec<Observation>>>,
    covariance: Vec<f64>,
}
fn prepare(data: &Dataset, graph: &IndexedGraph, split: &Split) -> Result<Training> {
    split.validate(data, graph)?;
    if split.axis != Axis::Animal {
        return Err("latent LDS requires animal splits".into());
    }
    let mut trials: Vec<_> = data
        .trials
        .iter()
        .filter(|t| split.train.contains(&t.id))
        .collect();
    trials.sort_by(|a, b| a.id.cmp(&b.id));
    let mut statistics: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();
    let mut dt: Option<f64> = None;
    for trial in &trials {
        for t in trial.recording.times.windows(2) {
            let step = t[1] - t[0];
            if dt.is_some_and(|d| (d - step).abs() > 1e-8) {
                return Err("LDS needs a common uniform time grid".into());
            }
            dt = Some(step);
        }
        for trace in &trial.recording.traces {
            let w = trace.provenance.id_confidence;
            if w == 0.0 {
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
    let neurons: Vec<_> = statistics.keys().cloned().collect();
    let n = neurons.len();
    if n == 0 || n > 1024 {
        return Err("LDS needs 1..1024 training-observed neurons".into());
    }
    let mean: Vec<_> = statistics.values().map(|s| s.1).collect();
    let scale: Vec<_> = statistics
        .values()
        .map(|s| (s.2 / s.0).max(0.0).sqrt().max(1e-8))
        .collect();
    let index: BTreeMap<_, _> = neurons
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let mut sequences = vec![];
    let mut covariance = vec![0.0; n * n];
    let mut frames = 0;
    for trial in trials {
        let mut sequence = vec![vec![]; trial.recording.times.len()];
        for trace in &trial.recording.traces {
            let Some(&i) = index.get(trace.neuron.as_str()) else {
                continue;
            };
            let w = trace.provenance.id_confidence;
            if w == 0.0 {
                continue;
            }
            for (t, &y) in trace.values.iter().enumerate() {
                if let Some(y) = y {
                    sequence[t].push((i, (y - mean[i]) / scale[i], w));
                }
            }
        }
        for values in &mut sequence {
            values.sort_by_key(|x| x.0);
            frames += 1;
            for &(i, y, w) in values.iter() {
                for &(j, z, v) in values.iter() {
                    covariance[i * n + j] += w.sqrt() * v.sqrt() * y * z;
                }
            }
        }
        sequences.push(sequence);
    }
    for v in &mut covariance {
        *v /= frames as f64;
    }
    Ok(Training {
        neurons,
        mean,
        scale,
        dt: dt.ok_or("no LDS training intervals")?,
        sequences,
        covariance,
    })
}
fn initialize(training: &Training, rank: usize, cap: f64) -> Result<GaussianLds> {
    let n = training.neurons.len();
    if rank == 0 || rank > n || rank > 64 {
        return Err("latent rank exceeds available outputs or size limit".into());
    }
    let (d, v) = eigen(&training.covariance, n)?;
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| d[b * n + b].total_cmp(&d[a * n + a]));
    let mut c = vec![0.0; n * rank];
    for (j, &column) in order.iter().take(rank).enumerate() {
        let magnitude = (d[column * n + column] - 0.05).max(0.01).sqrt();
        let pivot = (0..n)
            .max_by(|&a, &b| v[a * n + column].abs().total_cmp(&v[b * n + column].abs()))
            .unwrap();
        let sign = if v[pivot * n + column] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for i in 0..n {
            c[i * rank + j] = sign * magnitude * v[i * n + column];
        }
    }
    let noise = (0..n)
        .map(|i| {
            (1.0 - c[i * rank..(i + 1) * rank]
                .iter()
                .map(|v| v * v)
                .sum::<f64>())
            .max(0.1)
        })
        .collect();
    let gaussian = GaussianLds {
        dim: rank,
        outputs: n,
        transition: eye(rank, 0.95f64.min(cap)),
        observation: c,
        process_cov: eye(rank, 0.1),
        noise,
        initial_cov: eye(rank, 1.0),
    };
    gaussian.validate()?;
    Ok(gaussian)
}
impl LatentModel {
    pub fn free_parameters(&self) -> usize {
        let k = self.gaussian.dim;
        let n = self.neurons.len();
        k * k + n * k + k * (k + 1) + 3 * n
    }
    fn validate(&self, data: &Dataset, graph: &IndexedGraph, split: &Split) -> Result<()> {
        split.validate(data, graph)?;
        self.gaussian.validate()?;
        let n = self.neurons.len();
        if self.schema_version != 1
            || split.axis != Axis::Animal
            || self.dataset_hash != split.dataset_hash
            || self.split_hash != split.content_hash()?
            || self.graph_hash != graph.hash
            || self.source_commit.trim().is_empty()
            || self.gaussian.outputs != n
            || self.mean.len() != n
            || self.scale.len() != n
            || self.mean.iter().any(|x| !x.is_finite())
            || self.scale.iter().any(|x| !x.is_finite() || *x <= 0.0)
            || self.neurons.windows(2).any(|p| p[0] >= p[1])
            || !self.sample_dt.is_finite()
            || self.sample_dt <= 0.0
            || !self.transition_cap.is_finite()
            || self.transition_cap <= 0.0
            || self.transition_cap >= 1.0
            || self.gaussian.transition_norm_bound()? > self.transition_cap + 1e-9
            || !super::declared_subset(&self.training_trials, &split.train)
            || !super::declared_subset(&self.selection_trials, &split.validation)
        {
            return Err("invalid latent LDS artifact or data lineage".into());
        }
        for name in &self.neurons {
            graph.neuron(name)?;
        }
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
        let k = self.gaussian.dim;
        let n = self.neurons.len();
        let index: BTreeMap<_, _> = self
            .neurons
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();
        let mut trials = vec![];
        let mut fallbacks = std::collections::BTreeSet::new();
        let mut fallback_traces = 0;
        for trial in data
            .trials
            .iter()
            .filter(|t| split.ids(partition).contains(&t.id))
        {
            if !trial.response_labels.is_empty() {
                return Err("latent LDS does not score response labels".into());
            }
            let origin = trial.forecast_origin.ok_or("missing forecast origin")?;
            let times = &trial.recording.times;
            if times
                .windows(2)
                .any(|p| (p[1] - p[0] - self.sample_dt).abs() > 1e-8)
            {
                return Err("LDS prediction grid differs from training interval".into());
            }
            let mut observations = vec![vec![]; times.len()];
            let mut fluorescence = BTreeMap::new();
            for trace in &trial.recording.traces {
                if let Some(&i) = index.get(trace.neuron.as_str()) {
                    let w = trace.provenance.id_confidence;
                    for (t, &y) in trace.values.iter().enumerate() {
                        if times[t] <= origin
                            && w > 0.0
                            && let Some(y) = y
                        {
                            observations[t].push((i, (y - self.mean[i]) / self.scale[i], w));
                        }
                    }
                    fluorescence.insert(trace.neuron.clone(), vec![0.0; times.len()]);
                } else {
                    let last = times
                        .iter()
                        .zip(&trace.values)
                        .take_while(|(t, _)| **t <= origin)
                        .filter_map(|(_, y)| *y)
                        .last()
                        .ok_or("unseen LDS neuron has no history for persistence fallback")?;
                    fluorescence.insert(trace.neuron.clone(), vec![last; times.len()]);
                    fallbacks.insert(trace.neuron.clone());
                    fallback_traces += 1;
                }
            }
            let mut mean = vec![0.0; k];
            let mut cov = self.gaussian.initial_cov.clone();
            for (t, &time) in times.iter().enumerate() {
                if t > 0 {
                    if time <= origin {
                        (mean, cov) = self.gaussian.advance(&mean, &cov);
                    } else {
                        mean = mv(&self.gaussian.transition, &mean, k, k);
                    }
                }
                if time <= origin {
                    observations[t].sort_by_key(|x| x.0);
                    self.gaussian
                        .observe(&mut mean, &mut cov, &observations[t])?;
                }
                let y = mv(&self.gaussian.observation, &mean, n, k);
                for trace in &trial.recording.traces {
                    if let Some(&i) = index.get(trace.neuron.as_str()) {
                        let value = self.mean[i] + self.scale[i] * y[i];
                        if !value.is_finite() {
                            return Err("nonfinite latent LDS forecast".into());
                        }
                        fluorescence.get_mut(&trace.neuron).unwrap()[t] = value;
                    }
                }
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
                "stable latent Gaussian LDS; rank={k}; iteration={}; history-only Kalman filtering; unseen-neuron persistence fallback traces={fallback_traces}; neurons={fallbacks:?}",
                self.iteration
            ),
            free_parameters: self.free_parameters(),
            training_trials: self.training_trials.clone(),
            selection_trials: self.selection_trials.clone(),
            source_commit: option_env!("WORMSIM_COMMIT")
                .unwrap_or("unversioned")
                .into(),
            seed: 0,
            trials,
        })
    }
}
pub fn fit_select(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    cfg: FitConfig,
    mut checkpoint: impl FnMut(&LatentModel, &Candidate) -> Result<()>,
) -> Result<(LatentModel, SelectionReport)> {
    if split.validation.is_empty()
        || cfg.ranks.is_empty()
        || cfg.iterations > 100
        || !cfg.ridge.is_finite()
        || cfg.ridge < 0.0
        || !cfg.transition_cap.is_finite()
        || cfg.transition_cap <= 0.0
        || cfg.transition_cap >= 1.0
    {
        return Err("invalid latent LDS selection configuration".into());
    }
    let start = std::time::Instant::now();
    let training = prepare(data, graph, split)?;
    let preparation = start.elapsed().as_secs_f64();
    let mut candidates = vec![];
    let mut best: Option<(f64, LatentModel)> = None;
    for &rank in &cfg.ranks {
        let mut candidate = LatentModel {
            schema_version: 1,
            dataset_hash: split.dataset_hash.clone(),
            split_hash: split.content_hash()?,
            graph_hash: graph.hash.clone(),
            source_commit: option_env!("WORMSIM_COMMIT")
                .unwrap_or("unversioned")
                .into(),
            sample_dt: training.dt,
            neurons: training.neurons.clone(),
            mean: training.mean.clone(),
            scale: training.scale.clone(),
            gaussian: initialize(&training, rank, cfg.transition_cap)?,
            transition_cap: cfg.transition_cap,
            ridge: cfg.ridge,
            iteration: 0,
            training_trials: split.train.clone(),
            selection_trials: split.validation.clone(),
        };
        for iteration in 0..=cfg.iterations {
            let start = std::time::Instant::now();
            let preceding = if iteration > 0 {
                Some(candidate.gaussian.em_step(
                    &training.sequences,
                    cfg.transition_cap,
                    cfg.ridge,
                )?)
            } else {
                None
            };
            candidate.iteration = iteration;
            let prediction = candidate.predict(data, graph, split, Partition::Validation)?;
            let evaluation =
                super::evaluate(data, graph, split, &prediction, Partition::Validation)?;
            let horizons: Vec<_> = evaluation
                .forecast_horizons
                .iter()
                .map(|h| h.macro_neuron_r2)
                .collect();
            if horizons.len() != 3 || horizons.iter().any(Option::is_none) {
                return Err("undefined latent LDS validation score".into());
            }
            let score = horizons.iter().flatten().sum::<f64>() / 3.0;
            if !score.is_finite() {
                return Err("nonfinite LDS validation criterion".into());
            }
            let report = Candidate {
                rank,
                iteration,
                preceding_training_nll_per_observation: preceding,
                validation_horizon_r2: horizons,
                validation_criterion: score,
                transition_norm_bound: candidate.gaussian.transition_norm_bound()?,
                elapsed_seconds: start.elapsed().as_secs_f64(),
            };
            checkpoint(&candidate, &report)?;
            candidates.push(report);
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, candidate.clone()));
            }
        }
    }
    let (_, model) = best.ok_or("no latent LDS candidate")?;
    let report=SelectionReport{schema_version:1,dataset_hash:split.dataset_hash.clone(),split_hash:split.content_hash()?,source_commit:model.source_commit.clone(),config:cfg,criterion:"Maximum mean validation macro-neuron R² at 1/10/30 s; ties retain earlier declared rank/iteration. All normalization, latent initialization and EM moments use training animals only.".into(),candidates,selected_rank:model.gaussian.dim,selected_iteration:model.iteration,free_parameters:model.free_parameters(),training_preparation_seconds:preparation};
    Ok((model, report))
}
