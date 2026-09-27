//! Model-bound exact covariance reuse for sequences with identical observation patterns.
use super::*;

pub type ObservationPattern = Vec<Vec<(usize, u64)>>;
pub fn observation_pattern(sequence: &[Vec<Observation>]) -> ObservationPattern {
    sequence
        .iter()
        .map(|f| f.iter().map(|&(i, _, w)| (i, w.to_bits())).collect())
        .collect()
}
struct Update {
    output: usize,
    projected_covariance: Vec<f64>,
    variance: f64,
}
/// Borrows its model, preventing parameter mutation while covariance plans exist.
/// Stores one pattern only; callers should process and release groups sequentially.
pub struct PreparedSmoother<'a> {
    model: &'a GaussianLds,
    pattern: ObservationPattern,
    updates: Vec<Vec<Update>>,
    gains: Vec<Vec<f64>>,
    covariances: Vec<Vec<f64>>,
    lag_covariances: Vec<Vec<f64>>,
}
pub struct PreparedPosterior<'a> {
    pub means: Vec<Vec<f64>>,
    pub covariances: &'a [Vec<f64>],
    pub lag_covariances: &'a [Vec<f64>],
    pub negative_log_likelihood: f64,
    pub observations: usize,
}
impl GaussianLds {
    pub fn prepare_smoother(&self, sequence: &[Vec<Observation>]) -> Result<PreparedSmoother<'_>> {
        self.validate()?;
        if sequence.is_empty() {
            return Err("empty LDS sequence".into());
        }
        let k = self.dim;
        let at = transpose(&self.transition, k, k);
        let mut cov = self.initial_cov.clone();
        let mut predicted = Vec::new();
        let mut covs = Vec::new();
        let mut updates = Vec::new();
        for (t, frame) in sequence.iter().enumerate() {
            if t > 0 {
                cov = mm(&mm(&self.transition, &cov, k, k, k), &at, k, k, k);
                for (v, q) in cov.iter_mut().zip(&self.process_cov) {
                    *v += q;
                }
            }
            predicted.push(cov.clone());
            let mut frame_updates = Vec::new();
            for &(i, y, w) in frame {
                if i >= self.outputs || !y.is_finite() || !w.is_finite() || w <= 0.0 || w > 1.0 {
                    return Err("invalid LDS observation".into());
                }
                let row = &self.observation[i * k..(i + 1) * k];
                let u = mv(&cov, row, k, k);
                let variance =
                    row.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>() + self.noise[i] / w;
                if !variance.is_finite() || variance <= 0.0 {
                    return Err("invalid innovation variance".into());
                }
                for a in 0..k {
                    for b in 0..k {
                        cov[a * k + b] -= u[a] * u[b] / variance;
                    }
                }
                frame_updates.push(Update {
                    output: i,
                    projected_covariance: u,
                    variance,
                });
            }
            symmetrize(&mut cov, k);
            if cov.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite Kalman covariance".into());
            }
            updates.push(frame_updates);
            covs.push(cov.clone());
        }
        let mut gains = vec![vec![]; sequence.len() - 1];
        let mut lag = vec![vec![]; sequence.len() - 1];
        for t in (0..sequence.len() - 1).rev() {
            let j = mm(
                &mm(&covs[t], &at, k, k, k),
                &inverse(&predicted[t + 1], k)?,
                k,
                k,
                k,
            );
            let jt = transpose(&j, k, k);
            lag[t] = mm(&covs[t + 1], &jt, k, k, k);
            let diff: Vec<_> = covs[t + 1]
                .iter()
                .zip(&predicted[t + 1])
                .map(|(a, b)| a - b)
                .collect();
            let update = mm(&mm(&j, &diff, k, k, k), &jt, k, k, k);
            for (v, u) in covs[t].iter_mut().zip(update) {
                *v += u;
            }
            symmetrize(&mut covs[t], k);
            gains[t] = j;
        }
        Ok(PreparedSmoother {
            model: self,
            pattern: observation_pattern(sequence),
            updates,
            gains,
            covariances: covs,
            lag_covariances: lag,
        })
    }
}
impl PreparedSmoother<'_> {
    pub fn smooth(
        &self,
        sequence: &[Vec<Observation>],
        inputs: &[Vec<f64>],
    ) -> Result<PreparedPosterior<'_>> {
        if observation_pattern(sequence) != self.pattern {
            return Err("LDS observation pattern differs from prepared smoother".into());
        }
        self.model.check_inputs(sequence.len(), inputs)?;
        let k = self.model.dim;
        let mut mean = vec![0.0; k];
        let mut means = Vec::new();
        let mut predicted = Vec::new();
        let mut nll = 0.0;
        let mut count = 0;
        for (t, frame) in sequence.iter().enumerate() {
            if t > 0 {
                mean = self.model.advance_mean(&mean, &inputs[t - 1]);
            }
            predicted.push(mean.clone());
            for ((_, y, _), update) in frame.iter().zip(&self.updates[t]) {
                if !y.is_finite() {
                    return Err("nonfinite LDS observation".into());
                }
                let row = &self.model.observation[update.output * k..(update.output + 1) * k];
                let residual = y - row.iter().zip(&mean).map(|(a, b)| a * b).sum::<f64>();
                nll += 0.5
                    * ((2.0 * std::f64::consts::PI * update.variance).ln()
                        + residual * residual / update.variance);
                for (m, u) in mean.iter_mut().zip(&update.projected_covariance) {
                    *m += u * residual / update.variance;
                }
                count += 1;
            }
            means.push(mean.clone());
        }
        for t in (0..sequence.len() - 1).rev() {
            let delta: Vec<_> = means[t + 1]
                .iter()
                .zip(&predicted[t + 1])
                .map(|(a, b)| a - b)
                .collect();
            let adjustment = mv(&self.gains[t], &delta, k, k);
            for (m, a) in means[t].iter_mut().zip(adjustment) {
                *m += a;
            }
        }
        if !nll.is_finite() || means.iter().flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite prepared posterior".into());
        }
        Ok(PreparedPosterior {
            means,
            covariances: &self.covariances,
            lag_covariances: &self.lag_covariances,
            negative_log_likelihood: nll,
            observations: count,
        })
    }
}
