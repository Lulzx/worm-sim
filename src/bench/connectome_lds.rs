//! Anatomically constrained Gaussian LDS with a shared stimulation kernel.
//! The shared kernel is necessary when whole stimulated identities are held out.
use super::lds::prepared::observation_pattern;
use super::{
    lds::{GaussianLds, Observation},
    lds_math::*,
};
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StimulusSequence {
    pub target: usize,
    pub observations: Vec<Vec<Observation>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectomeLds {
    pub gaussian: GaussianLds,
    /// Sorted source-state columns for each target-state row, including self.
    pub allowed: Vec<Vec<usize>>,
    /// Shared across target neurons. kernel[t] drives transition t -> t+1.
    pub kernel: Vec<f64>,
    pub sample_dt: f64,
}
#[derive(Debug, Serialize)]
pub struct StepReport {
    pub preceding_negative_log_likelihood: f64,
    pub observations: usize,
    pub transitions: usize,
    pub fitted_noise_outputs: usize,
    pub unprojected_transition_norm: f64,
    pub elapsed_seconds: f64,
}
fn anatomy(graph: &IndexedGraph) -> Vec<Vec<usize>> {
    let mut rows: Vec<_> = (0..graph.names.len())
        .map(|i| BTreeSet::from([i]))
        .collect();
    for &(pre, post, _, _) in &graph.chemical {
        rows[post].insert(pre);
    }
    for &(a, b, _) in &graph.gaps {
        rows[a].insert(b);
        rows[b].insert(a);
    }
    rows.into_iter().map(|r| r.into_iter().collect()).collect()
}
impl ConnectomeLds {
    pub fn new(graph: &IndexedGraph, lags: usize, sample_dt: f64) -> Result<Self> {
        let n = graph.names.len();
        if n == 0
            || n > 512
            || lags == 0
            || lags > 512
            || !sample_dt.is_finite()
            || sample_dt <= 0.0
        {
            return Err("invalid connectome LDS dimensions or timebase".into());
        }
        let model = Self {
            gaussian: GaussianLds {
                dim: n,
                outputs: n,
                transition: eye(n, 0.8),
                input_dim: n,
                input_weights: eye(n, 1.0),
                observation: eye(n, 1.0),
                process_cov: eye(n, 0.1),
                noise: vec![0.1; n],
                initial_cov: eye(n, 0.1),
            },
            allowed: anatomy(graph),
            kernel: (0..lags)
                .map(|t| 0.2 * (-(t as f64) * sample_dt / 2.0).exp())
                .collect(),
            sample_dt,
        };
        model.validate()?;
        Ok(model)
    }
    pub fn validate_for_graph(&self, graph: &IndexedGraph) -> Result<()> {
        self.validate()?;
        if self.allowed != anatomy(graph) {
            return Err("LDS support differs from the anatomical graph".into());
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        self.gaussian.validate()?;
        let n = self.gaussian.dim;
        if self.gaussian.outputs != n
            || self.gaussian.input_dim != n
            || self.allowed.len() != n
            || self.kernel.is_empty()
            || self.kernel.len() > 512
            || self.kernel.iter().any(|x| !x.is_finite())
            || !self.sample_dt.is_finite()
            || self.sample_dt <= 0.0
        {
            return Err("invalid connectome LDS dimensions or kernel".into());
        }
        for i in 0..n {
            if !self.allowed[i].contains(&i)
                || self.allowed[i].iter().any(|j| *j >= n)
                || self.allowed[i].windows(2).any(|p| p[0] >= p[1])
            {
                return Err("invalid LDS anatomical support".into());
            }
            for j in 0..n {
                let identity = f64::from(i == j);
                if self.gaussian.observation[i * n + j] != identity
                    || self.gaussian.input_weights[i * n + j] != identity
                    || (i != j
                        && (self.gaussian.process_cov[i * n + j] != 0.0
                            || self.gaussian.initial_cov[i * n + j] != 0.0))
                    || (!self.allowed[i].contains(&j) && self.gaussian.transition[i * n + j] != 0.0)
                {
                    return Err(
                        "connectome LDS requires masked A, identity C/input routing, diagonal Q/P0"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }
    pub fn inputs(&self, target: usize, frames: usize) -> Result<Vec<Vec<f64>>> {
        let n = self.gaussian.dim;
        if target >= n || frames < 2 {
            return Err("invalid stimulation target or frame count".into());
        }
        let mut inputs = vec![vec![0.0; n]; frames];
        for (t, row) in inputs.iter_mut().take(frames - 1).enumerate() {
            if t < self.kernel.len() {
                row[target] = self.kernel[t];
            }
        }
        Ok(inputs)
    }
    /// Zero-mean initial state; no held-out fluorescence is used for response prediction.
    pub fn impulse(&self, target: usize, frames: usize) -> Result<Vec<Vec<f64>>> {
        self.validate()?;
        if target >= self.gaussian.dim || frames < 2 {
            return Err("invalid impulse target/grid".into());
        }
        let n = self.gaussian.dim;
        let mut means = vec![vec![0.0; n]; frames];
        for t in 0..frames - 1 {
            for i in 0..n {
                means[t + 1][i] = self.allowed[i]
                    .iter()
                    .map(|j| self.gaussian.transition[i * n + j] * means[t][*j])
                    .sum();
            }
            if t < self.kernel.len() {
                means[t + 1][target] += self.kernel[t];
            }
        }
        if means.iter().flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite impulse response".into());
        }
        Ok(means)
    }
    /// Exact masked Gaussian smoothing, followed by a jointly constrained normal
    /// solve for A and the global kernel. An active A norm projection is followed
    /// by a conditional kernel refit. Noise updates use the final projected means.
    pub fn em_step(
        &self,
        sequences: &[StimulusSequence],
        ridge: f64,
        cap: f64,
    ) -> Result<(Self, StepReport)> {
        self.validate()?;
        if sequences.is_empty()
            || !ridge.is_finite()
            || ridge <= 0.0
            || !cap.is_finite()
            || cap <= 0.0
            || cap >= 1.0
        {
            return Err("invalid constrained EM settings".into());
        }
        let start = std::time::Instant::now();
        let n = self.gaussian.dim;
        let l = self.kernel.len();
        let mut s00 = vec![0.0; n * n];
        let mut s10 = vec![0.0; n * n];
        let mut s11 = vec![0.0; n];
        // For row i, a one-hot lag feature is present only when i is stimulated.
        let mut x_input = vec![0.0; n * n * l];
        let mut y_input = vec![0.0; n * l];
        let mut input_count = vec![0.0; n * l];
        let mut initial = vec![0.0; n];
        let mut noise = vec![0.0; n];
        let mut counts = vec![0usize; n];
        let mut transitions = 0;
        let mut observations = 0;
        let mut nll = 0.0;
        let mut groups = BTreeMap::new();
        for sequence in sequences {
            groups
                .entry(observation_pattern(&sequence.observations))
                .or_insert_with(Vec::new)
                .push(sequence);
        }
        for group in groups.values() {
            let plan = self.gaussian.prepare_smoother(&group[0].observations)?;
            for sequence in group {
                let inputs = self.inputs(sequence.target, sequence.observations.len())?;
                let posterior = plan.smooth(&sequence.observations, &inputs)?;
                nll += posterior.negative_log_likelihood;
                observations += posterior.observations;
                for (i, value) in initial.iter_mut().enumerate() {
                    *value += posterior.covariances[0][i * n + i] + posterior.means[0][i].powi(2);
                }
                for (t, frame) in sequence.observations.iter().enumerate() {
                    for &(i, y, w) in frame {
                        noise[i] += w
                            * ((y - posterior.means[t][i]).powi(2)
                                + posterior.covariances[t][i * n + i]);
                        counts[i] += 1;
                    }
                    if t + 1 == sequence.observations.len() {
                        continue;
                    }
                    transitions += 1;
                    for i in 0..n {
                        s11[i] += posterior.covariances[t + 1][i * n + i]
                            + posterior.means[t + 1][i].powi(2);
                        for j in 0..n {
                            s00[i * n + j] += posterior.covariances[t][i * n + j]
                                + posterior.means[t][i] * posterior.means[t][j];
                            s10[i * n + j] += posterior.lag_covariances[t][i * n + j]
                                + posterior.means[t + 1][i] * posterior.means[t][j];
                        }
                    }
                    if t < l {
                        let i = sequence.target;
                        input_count[i * l + t] += 1.0;
                        y_input[i * l + t] += posterior.means[t + 1][i];
                        for j in 0..n {
                            x_input[(i * n + j) * l + t] += posterior.means[t][j];
                        }
                    }
                }
            }
        }
        if transitions == 0
            || observations == 0
            || (0..l).any(|t| (0..n).all(|i| input_count[i * l + t] == 0.0))
        {
            return Err("missing transitions/observations or unidentifiable kernel lag".into());
        }
        let penalty = ridge * transitions as f64;
        let mut schur = eye(l, penalty);
        let mut rhs = vec![0.0; l];
        let mut conditional = vec![];
        for i in 0..n {
            let mask = &self.allowed[i];
            let d = mask.len();
            let precision = 1.0 / self.gaussian.process_cov[i * n + i];
            let mut gram = vec![0.0; d * d];
            let mut cross = vec![0.0; d * l];
            let mut target = vec![0.0; d];
            for (a, &j) in mask.iter().enumerate() {
                target[a] = precision * s10[i * n + j];
                for (b, &k) in mask.iter().enumerate() {
                    gram[a * d + b] = precision * s00[j * n + k];
                }
                gram[a * d + a] += penalty;
                for t in 0..l {
                    cross[a * l + t] = precision * x_input[(i * n + j) * l + t];
                }
            }
            let inv = inverse(&gram, d)?;
            let intercept = mv(&inv, &target, d, d);
            let gain = mm(&inv, &cross, d, d, l);
            for t in 0..l {
                schur[t * l + t] += precision * input_count[i * l + t];
                rhs[t] += precision * y_input[i * l + t]
                    - (0..d).map(|a| cross[a * l + t] * intercept[a]).sum::<f64>();
                for u in 0..l {
                    schur[t * l + u] -= (0..d)
                        .map(|a| cross[a * l + t] * gain[a * l + u])
                        .sum::<f64>();
                }
            }
            conditional.push((intercept, gain));
        }
        symmetrize(&mut schur, l);
        let kernel = solve(&chol(&schur, l)?, &rhs, l);
        let mut next = self.clone();
        next.kernel = kernel;
        for (i, (intercept, gain)) in conditional.iter().enumerate() {
            for (a, &j) in self.allowed[i].iter().enumerate() {
                next.gaussian.transition[i * n + j] = intercept[a]
                    - (0..l)
                        .map(|t| gain[a * l + t] * next.kernel[t])
                        .sum::<f64>();
            }
        }
        let unprojected = stabilize(&mut next.gaussian.transition, n, cap)?;
        if unprojected > cap {
            // Lag features are one-hot, so this conditional normal solve is diagonal.
            for t in 0..l {
                let mut numerator = 0.0;
                let mut denominator = penalty;
                for i in 0..n {
                    let precision = 1.0 / self.gaussian.process_cov[i * n + i];
                    numerator += precision
                        * (y_input[i * l + t]
                            - self.allowed[i]
                                .iter()
                                .map(|j| {
                                    next.gaussian.transition[i * n + j]
                                        * x_input[(i * n + j) * l + t]
                                })
                                .sum::<f64>());
                    denominator += precision * input_count[i * l + t];
                }
                next.kernel[t] = numerator / denominator;
            }
        }
        for i in 0..n {
            let a = &next.gaussian.transition[i * n..(i + 1) * n];
            let mut residual = s11[i]
                - 2.0
                    * self.allowed[i]
                        .iter()
                        .map(|j| a[*j] * s10[i * n + j])
                        .sum::<f64>();
            for &j in &self.allowed[i] {
                for &k in &self.allowed[i] {
                    residual += a[j] * s00[j * n + k] * a[k];
                }
            }
            for t in 0..l {
                let b = next.kernel[t];
                residual += b * b * input_count[i * l + t] - 2.0 * b * y_input[i * l + t]
                    + 2.0
                        * b
                        * self.allowed[i]
                            .iter()
                            .map(|j| a[*j] * x_input[(i * n + j) * l + t])
                            .sum::<f64>();
            }
            if !residual.is_finite() {
                return Err("nonfinite process-noise moment".into());
            }
            next.gaussian.process_cov[i * n + i] = (residual / transitions as f64).max(1e-6);
            next.gaussian.initial_cov[i * n + i] = (initial[i] / sequences.len() as f64).max(1e-6);
            if counts[i] > 0 {
                next.gaussian.noise[i] = (noise[i] / counts[i] as f64).max(1e-6);
            }
        }
        next.validate()?;
        Ok((
            next,
            StepReport {
                preceding_negative_log_likelihood: nll,
                observations,
                transitions,
                fitted_noise_outputs: counts.iter().filter(|c| **c > 0).count(),
                unprojected_transition_norm: unprojected,
                elapsed_seconds: start.elapsed().as_secs_f64(),
            },
        ))
    }
}
