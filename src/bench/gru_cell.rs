//! Dense reset-before GRU and exact discrete BPTT, including forecast feedback.
use crate::Result;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Network {
    pub outputs: usize,
    pub hidden: usize,
    /// Three rows per hidden unit (reset, retain, candidate), then affine readout.
    pub weights: Vec<f64>,
}
#[derive(Clone)]
struct Step {
    x: Vec<f64>,
    previous: Vec<f64>,
    reset: Vec<f64>,
    retain: Vec<f64>,
    candidate: Vec<f64>,
    hidden: Vec<f64>,
}
/// Sparse standardized observation: index, value, identity confidence.
pub type Observation = (usize, f64, f64);
impl Network {
    fn row(&self) -> usize {
        2 * self.outputs + self.hidden + 1
    }
    fn readout(&self) -> usize {
        3 * self.hidden * self.row()
    }
    pub fn initialize(outputs: usize, hidden: usize, seed: u64) -> Result<Self> {
        if outputs == 0 || outputs > 1024 || hidden == 0 || hidden > 128 {
            return Err("invalid GRU dimensions".into());
        }
        let mut s = seed;
        let mut uniform = || {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((z ^ (z >> 31)) >> 11) as f64 / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let row = 2 * outputs + hidden + 1;
        let mut weights = vec![0.0; 3 * hidden * row + outputs * (hidden + 1)];
        let scale = (6.0 / (2 * outputs + 2 * hidden) as f64).sqrt();
        for r in 0..3 * hidden {
            for j in 0..row - 1 {
                weights[r * row + j] = uniform() * scale;
            }
        }
        let offset = 3 * hidden * row;
        for i in 0..outputs {
            for j in 0..hidden {
                weights[offset + i * (hidden + 1) + j] =
                    uniform() * (6.0 / (hidden + outputs) as f64).sqrt();
            }
        }
        Ok(Self {
            outputs,
            hidden,
            weights,
        })
    }
    pub fn validate(&self) -> Result<()> {
        if self.outputs == 0
            || self.outputs > 1024
            || self.hidden == 0
            || self.hidden > 128
            || self.weights.len() != self.readout() + self.outputs * (self.hidden + 1)
            || self.weights.iter().any(|x| !x.is_finite())
        {
            return Err("invalid GRU weights/dimensions".into());
        }
        Ok(())
    }
    fn step(&self, x: Vec<f64>, previous: Vec<f64>) -> Step {
        let d = 2 * self.outputs;
        let h = self.hidden;
        let mut gates = vec![vec![0.0; h]; 3];
        for g in 0..3 {
            for i in 0..h {
                let w = &self.weights[(g * h + i) * self.row()..(g * h + i + 1) * self.row()];
                let mut v = w[d + h];
                for j in 0..d {
                    v += w[j] * x[j];
                }
                for j in 0..h {
                    v += w[d + j] * previous[j] * if g == 2 { gates[0][j] } else { 1.0 };
                }
                gates[g][i] = if g == 2 {
                    v.tanh()
                } else if v >= 0.0 {
                    1.0 / (1.0 + (-v).exp())
                } else {
                    let e = v.exp();
                    e / (1.0 + e)
                };
            }
        }
        let hidden = (0..h)
            .map(|i| gates[1][i] * previous[i] + (1.0 - gates[1][i]) * gates[2][i])
            .collect();
        Step {
            x,
            previous,
            reset: gates[0].clone(),
            retain: gates[1].clone(),
            candidate: gates[2].clone(),
            hidden,
        }
    }
    fn decode(&self, hidden: &[f64]) -> Vec<f64> {
        let h = self.hidden;
        (0..self.outputs)
            .map(|i| {
                let w =
                    &self.weights[self.readout() + i * (h + 1)..self.readout() + (i + 1) * (h + 1)];
                w[h] + w[..h].iter().zip(hidden).map(|(a, b)| a * b).sum::<f64>()
            })
            .collect()
    }
    fn check_sequence(&self, sequence: &[Vec<Observation>], origin: usize) -> Result<()> {
        self.validate()?;
        if sequence.len() < 2 || origin >= sequence.len() - 1 {
            return Err("GRU needs history and future samples".into());
        }
        for frame in sequence {
            let mut seen = vec![false; self.outputs];
            for &(i, y, w) in frame {
                if i >= self.outputs
                    || seen[i]
                    || !y.is_finite()
                    || !w.is_finite()
                    || w <= 0.0
                    || w > 1.0
                {
                    return Err("invalid GRU observations".into());
                }
                seen[i] = true;
            }
        }
        Ok(())
    }
    fn forward(&self, sequence: &[Vec<Observation>], origin: usize) -> (Vec<Vec<f64>>, Vec<Step>) {
        let n = self.outputs;
        let mut hidden = vec![0.0; self.hidden];
        let mut output = vec![self.decode(&hidden)];
        let mut tape = vec![];
        for t in 0..sequence.len() - 1 {
            let mut x = vec![0.0; 2 * n];
            x[..n].copy_from_slice(&output[t]);
            if t <= origin {
                for &(i, y, w) in &sequence[t] {
                    x[i] = y;
                    x[n + i] = w;
                }
            }
            let step = self.step(x, hidden);
            hidden = step.hidden.clone();
            output.push(self.decode(&hidden));
            tape.push(step);
        }
        (output, tape)
    }
    pub fn predict(&self, sequence: &[Vec<Observation>], origin: usize) -> Result<Vec<Vec<f64>>> {
        self.check_sequence(sequence, origin)?;
        let (out, _) = self.forward(sequence, origin);
        if out.iter().flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite GRU prediction".into());
        }
        Ok(out)
    }
    pub fn loss_gradient(
        &self,
        sequence: &[Vec<Observation>],
        origin: usize,
    ) -> Result<(f64, Vec<f64>)> {
        self.check_sequence(sequence, origin)?;
        let (output, tape) = self.forward(sequence, origin);
        let n = self.outputs;
        let h = self.hidden;
        let d = 2 * n;
        let denom: f64 = sequence[origin + 1..].iter().flatten().map(|x| x.2).sum();
        if denom <= 0.0 {
            return Err("no GRU forecast training targets".into());
        }
        let mut loss = 0.0;
        let mut dy = vec![vec![0.0; n]; sequence.len()];
        for t in origin + 1..sequence.len() {
            for &(i, y, w) in &sequence[t] {
                let e = output[t][i] - y;
                loss += w * e * e / denom;
                dy[t][i] = 2.0 * w * e / denom;
            }
        }
        let mut grad = vec![0.0; self.weights.len()];
        let mut dh = vec![0.0; h];
        for t in (1..sequence.len()).rev() {
            let s = &tape[t - 1];
            for (i, &delta) in dy[t].iter().enumerate() {
                let off = self.readout() + i * (h + 1);
                for j in 0..h {
                    grad[off + j] += delta * s.hidden[j];
                    dh[j] += delta * self.weights[off + j];
                }
                grad[off + h] += delta;
            }
            let mut dp = vec![0.0; h];
            let mut dx = vec![0.0; d];
            let mut dr = vec![0.0; h];
            let mut dz = vec![0.0; h];
            let mut dc = vec![0.0; h];
            for i in 0..h {
                dp[i] = dh[i] * s.retain[i];
                dz[i] =
                    dh[i] * (s.previous[i] - s.candidate[i]) * s.retain[i] * (1.0 - s.retain[i]);
                dc[i] = dh[i] * (1.0 - s.retain[i]) * (1.0 - s.candidate[i] * s.candidate[i]);
            }
            for (i, &delta) in dc.iter().enumerate() {
                let off = (2 * h + i) * self.row();
                for j in 0..d {
                    grad[off + j] += delta * s.x[j];
                    dx[j] += delta * self.weights[off + j];
                }
                for j in 0..h {
                    grad[off + d + j] += delta * s.reset[j] * s.previous[j];
                    let q = delta * self.weights[off + d + j];
                    dr[j] += q * s.previous[j];
                    dp[j] += q * s.reset[j];
                }
                grad[off + d + h] += delta;
            }
            for (delta, &reset) in dr.iter_mut().zip(&s.reset) {
                *delta *= reset * (1.0 - reset);
            }
            for (g, delta) in [&dr, &dz].iter().enumerate() {
                for i in 0..h {
                    let off = (g * h + i) * self.row();
                    for j in 0..d {
                        grad[off + j] += delta[i] * s.x[j];
                        dx[j] += delta[i] * self.weights[off + j];
                    }
                    for j in 0..h {
                        grad[off + d + j] += delta[i] * s.previous[j];
                        dp[j] += delta[i] * self.weights[off + d + j];
                    }
                    grad[off + d + h] += delta[i];
                }
            }
            let mut observed = vec![false; n];
            if t - 1 <= origin {
                for &(i, _, _) in &sequence[t - 1] {
                    observed[i] = true;
                }
            }
            for i in 0..n {
                if !observed[i] {
                    dy[t - 1][i] += dx[i];
                }
            }
            dh = dp;
        }
        // Output zero is the affine readout of the fixed zero initial state.
        for i in 0..n {
            grad[self.readout() + i * (h + 1) + h] += dy[0][i];
        }
        if !loss.is_finite() || grad.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite GRU loss/gradient".into());
        }
        Ok((loss, grad))
    }
}
