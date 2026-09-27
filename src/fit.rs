//! Exact forward AD reference gradients, intended for gradient audits and small
//! circuits. Full gradients cost one simulation per parameter, not a GPU fit.
use crate::{
    Result,
    data::Recording,
    math::{Dual, Scalar},
    model::{Model, Parameters},
    solve::{Config, Trajectory, simulate},
};
pub fn loss<S: Scalar>(model: &Model, pred: &Trajectory<S>, recording: &Recording) -> Result<S> {
    recording.validate(&model.graph)?;
    if pred.times.len() != recording.times.len()
        || pred
            .times
            .iter()
            .zip(&recording.times)
            .any(|(a, b)| (a - b).abs() > 1e-9)
    {
        return Err("recording times must match the simulation save grid".into());
    }
    let mut sum = S::constant(0.0);
    let mut weight = 0.0;
    for trace in &recording.traces {
        let i = model.graph.neuron(&trace.neuron)?;
        let w = trace.provenance.id_confidence;
        for (t, value) in trace.values.iter().enumerate() {
            if let Some(value) = value {
                let error = pred.fluorescence[t][i] - S::constant(*value);
                sum = sum + S::constant(w) * error * error;
                weight += w;
            }
        }
    }
    if weight == 0.0 {
        return Err("loss has no observed samples with positive confidence".into());
    }
    Ok(sum / S::constant(weight))
}
pub fn gradient(
    model: &Model,
    params: &Parameters<f64>,
    cfg: &Config,
    recording: &Recording,
    indices: &[usize],
) -> Result<(f64, Vec<f64>)> {
    let value = loss(model, &simulate(model, params, cfg)?, recording)?;
    let mut grad = Vec::with_capacity(indices.len());
    for &index in indices {
        if index >= params.raw.len() {
            return Err("gradient index out of bounds".into());
        }
        let p = Parameters {
            raw: params
                .raw
                .iter()
                .enumerate()
                .map(|(i, &value)| Dual {
                    value,
                    tangent: if i == index { 1.0 } else { 0.0 },
                })
                .collect(),
        };
        grad.push(loss(model, &simulate(model, &p, cfg)?, recording)?.tangent);
    }
    Ok((value, grad))
}
/// Adam on an explicitly selected subset. Callers own train/held-out splits.
pub fn adam(
    model: &Model,
    params: &mut Parameters<f64>,
    cfg: &Config,
    recording: &Recording,
    indices: &[usize],
    steps: usize,
    lr: f64,
) -> Result<Vec<f64>> {
    if !lr.is_finite()
        || lr <= 0.0
        || indices
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != indices.len()
    {
        return Err("invalid learning rate or repeated fit index".into());
    }
    let mut m = vec![0.0; indices.len()];
    let mut v = m.clone();
    let mut history = Vec::new();
    for step in 1..=steps {
        let (loss, grad) = gradient(model, params, cfg, recording, indices)?;
        history.push(loss);
        if !loss.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return Err("nonfinite loss or gradient".into());
        }
        for (j, &i) in indices.iter().enumerate() {
            let g = grad[j].clamp(-10.0, 10.0);
            m[j] = 0.9 * m[j] + 0.1 * g;
            v[j] = 0.999 * v[j] + 0.001 * g * g;
            params.raw[i] -= lr * (m[j] / (1.0 - 0.9f64.powf(step as f64)))
                / ((v[j] / (1.0 - 0.999f64.powf(step as f64))).sqrt() + 1e-8);
        }
    }
    history.push(loss(model, &simulate(model, params, cfg)?, recording)?);
    Ok(history)
}
