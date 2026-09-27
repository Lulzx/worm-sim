//! History-only state inference for the entire Level 0 network.
//! Exact discrete Euler state adjoint; no perturbation protocols in this fitter.
use crate::{
    Result,
    data::Recording,
    math::Scalar,
    model::{Inputs, Model, Parameters, Prepared},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Readout {
    pub offset: Vec<f64>,
    pub gain: Vec<f64>,
}
impl Readout {
    pub fn identity(n: usize) -> Self {
        Self {
            offset: vec![0.0; n],
            gain: vec![1.0; n],
        }
    }
    pub(crate) fn validate(&self, n: usize) -> Result<()> {
        if self.offset.len() != n
            || self.gain.len() != n
            || self.offset.iter().any(|x| !x.is_finite())
            || self.gain.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err("invalid affine fluorescence readout".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceMethod {
    #[default]
    Shooting,
    BlockEkf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceConfig {
    #[serde(default)]
    pub method: InferenceMethod,
    #[serde(default)]
    pub filter: crate::state_filter::FilterConfig,
    pub dt: f64,
    pub iterations: usize,
    pub learning_rate: f64,
    pub prior_weight: f64,
}
impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            method: InferenceMethod::Shooting,
            filter: Default::default(),
            dt: 0.005,
            iterations: 30,
            learning_rate: 0.02,
            prior_weight: 1e-3,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferredState {
    /// Complete V/C/gate state at the start of the observed prefix.
    pub initial_state: Vec<f64>,
    /// Complete V/C/gate state at the forecast origin, including unobserved neurons.
    pub forecast_state: Vec<f64>,
    pub history_objective: Vec<f64>,
    pub observed_neurons: usize,
    pub latent_neurons: usize,
    pub history_samples: usize,
    /// Affine-readout predictions at observed history frames. For filtering these
    /// are posterior reconstructions, not an unforced replay of initial_state.
    #[serde(default)]
    pub history_predictions: Vec<Vec<f64>>,
    #[serde(default)]
    pub filter_diagnostics: Option<crate::state_filter::FilterDiagnostics>,
}
struct Tape {
    states: Vec<Vec<f64>>,
    steps: Vec<f64>,
    samples: Vec<usize>,
}
fn rollout(
    model: &Model,
    p: &Prepared<f64>,
    initial: &[f64],
    times: &[f64],
    dt: f64,
) -> Result<Tape> {
    if !dt.is_finite()
        || dt <= 0.0
        || times.len() < 2
        || times[0] != 0.0
        || times.windows(2).any(|w| !w[1].is_finite() || w[1] <= w[0])
        || times[times.len() - 1] / dt > 1e6
    {
        return Err("invalid inference time grid or step size".into());
    }
    if initial.len() != model.state_len() || initial.iter().any(|v| !v.is_finite()) {
        return Err("invalid initial state".into());
    }
    let mut states = vec![initial.to_vec()];
    let mut steps = vec![];
    let mut samples = vec![0];
    let mut rhs = vec![0.0; model.state_len()];
    let mut release = vec![0.0; model.n()];
    let input = Inputs::new(model.n());
    let mut t = 0.0;
    for &target in &times[1..] {
        while t < target {
            let next = (t + dt).min(target);
            let h = next - t;
            if h <= 0.0 {
                return Err("inference step cannot advance".into());
            }
            let previous = states.last().unwrap();
            model.rhs(p, previous, &input, &mut release, &mut rhs);
            let state: Vec<_> = previous.iter().zip(&rhs).map(|(y, d)| y + h * d).collect();
            if state.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite inference rollout; reduce integration dt".into());
            }
            states.push(state);
            steps.push(h);
            t = next;
        }
        samples.push(states.len() - 1);
    }
    Ok(Tape {
        states,
        steps,
        samples,
    })
}
/// J(state)^T * cotangent for the unforced, unablated Level 0 RHS.
fn state_vjp(model: &Model, p: &Prepared<f64>, y: &[f64], adj: &[f64], out: &mut [f64]) {
    let n = model.n();
    out.fill(0.0);
    for i in 0..n {
        let r = ((y[i] - p.threshold[i]) * p.slope[i]).sigmoid();
        let dr = p.slope[i] * r * (1.0 - r);
        out[i] = -p.inv_tau[i] * adj[i]
            + dr * (p.inv_calcium_tau[i] * adj[n + i]
                + (1.0 - y[2 * n + i]) * p.inv_synapse_tau * adj[2 * n + i]);
        out[n + i] = -p.inv_calcium_tau[i] * adj[n + i];
        out[2 * n + i] = -(r + 1.0) * p.inv_synapse_tau * adj[2 * n + i];
    }
    for e in 0..model.pre.len() {
        let a = model.pre[e] as usize;
        let b = model.post[e] as usize;
        let z = adj[b] * p.inv_tau[b] * p.weight[e];
        out[b] -= z * y[2 * n + a];
        out[2 * n + a] += z * (p.reversal[e] - y[b]);
    }
    for e in 0..model.gap_a.len() {
        let a = model.gap_a[e] as usize;
        let b = model.gap_b[e] as usize;
        let z = p.gap[e] * (adj[a] * p.inv_tau[a] - adj[b] * p.inv_tau[b]);
        out[a] -= z;
        out[b] += z;
    }
}
/// Accumulate h * (d RHS / d raw parameters)^T * adjoint.
fn parameter_vjp(
    model: &Model,
    raw: &Parameters<f64>,
    p: &Prepared<f64>,
    y: &[f64],
    adj: &[f64],
    h: f64,
    g: &mut [f64],
) {
    let n = model.n();
    let m = model.pre.len();
    // Reconstruct the current before division by voltage tau. This avoids
    // dividing by a parameter derivative and matches the forward arithmetic.
    let mut current: Vec<f64> = (0..n).map(|i| -(y[i] - p.rest[i])).collect();
    for e in 0..m {
        let a = model.pre[e] as usize;
        let b = model.post[e] as usize;
        let canonical = model.parameter_edge[e];
        let gate = y[2 * n + a];
        let delta = p.reversal[e] - y[b];
        current[b] += p.weight[e] * gate * delta;
        let z = h * adj[b] * p.inv_tau[b] * gate;
        g[6 * n + canonical] += z * delta * model.counts[e] * raw.raw[6 * n + canonical].sigmoid();
        let q = raw.raw[6 * n + m + canonical].sigmoid();
        g[6 * n + m + canonical] += z * p.weight[e] * 2.0 * q * (1.0 - q);
    }
    for e in 0..model.gap_a.len() {
        let a = model.gap_a[e] as usize;
        let b = model.gap_b[e] as usize;
        let delta = y[b] - y[a];
        current[a] += p.gap[e] * delta;
        current[b] -= p.gap[e] * delta;
        let index = 6 * n + 2 * m + e;
        g[index] += h
            * delta
            * (adj[a] * p.inv_tau[a] - adj[b] * p.inv_tau[b])
            * model.gap_sizes[e]
            * raw.raw[index].sigmoid();
    }
    let synapse = model.parameter_count() - 1;
    for i in 0..n {
        let r = ((y[i] - p.threshold[i]) * p.slope[i]).sigmoid();
        let gate = y[2 * n + i];
        g[i] -= h * adj[i] * current[i] * p.inv_tau[i] * p.inv_tau[i] * raw.raw[i].sigmoid();
        g[n + i] += h * adj[i] * p.inv_tau[i];
        let release = h
            * r
            * (1.0 - r)
            * (adj[n + i] * p.inv_calcium_tau[i]
                + adj[2 * n + i] * (1.0 - gate) * p.inv_synapse_tau);
        g[2 * n + i] -= release * p.slope[i];
        g[3 * n + i] += release * (y[i] - p.threshold[i]) * raw.raw[3 * n + i].sigmoid();
        g[4 * n + i] -= h
            * adj[n + i]
            * (r - y[n + i])
            * p.inv_calcium_tau[i]
            * p.inv_calcium_tau[i]
            * raw.raw[4 * n + i].sigmoid();
        g[synapse] -= h
            * adj[2 * n + i]
            * (r * (1.0 - gate) - gate)
            * p.inv_synapse_tau
            * p.inv_synapse_tau
            * raw.raw[synapse].sigmoid();
    }
}
/// Objective derivatives with initial state and prior held fixed for parameter
/// differentiation. This is the conditional parameter gradient, not an implicit
/// derivative through the initial-state optimizer.
#[derive(Debug)]
pub struct ObjectiveGradient {
    pub value: f64,
    pub initial: Vec<f64>,
    pub parameters: Vec<f64>,
    pub readout_offset: Vec<f64>,
    pub readout_log_gain: Vec<f64>,
    pub terminal: Vec<f64>,
}
pub fn objective_gradient(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    readout: &Readout,
    initial: &[f64],
    prior: &[f64],
    cfg: &InferenceConfig,
) -> Result<(f64, Vec<f64>, Vec<f64>)> {
    let g = objective_impl(
        model, params, recording, readout, initial, prior, cfg, false,
    )?;
    Ok((g.value, g.initial, g.terminal))
}
pub fn parameter_gradient(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    readout: &Readout,
    initial: &[f64],
    dt: f64,
) -> Result<ObjectiveGradient> {
    let cfg = InferenceConfig {
        dt,
        prior_weight: 0.0,
        ..Default::default()
    };
    objective_impl(
        model, params, recording, readout, initial, initial, &cfg, true,
    )
}
#[allow(clippy::too_many_arguments)]
fn objective_impl(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    readout: &Readout,
    initial: &[f64],
    prior: &[f64],
    cfg: &InferenceConfig,
    parameter_derivatives: bool,
) -> Result<ObjectiveGradient> {
    recording.validate(&model.graph)?;
    readout.validate(model.n())?;
    if prior.len() != model.state_len()
        || prior.iter().any(|v| !v.is_finite())
        || !cfg.prior_weight.is_finite()
        || cfg.prior_weight < 0.0
    {
        return Err("invalid state prior".into());
    }
    let p = model.prepare(params)?;
    let tape = rollout(model, &p, initial, &recording.times, cfg.dt)?;
    let n = model.n();
    let weight: f64 = recording
        .traces
        .iter()
        .map(|t| {
            t.values.iter().filter(|v| v.is_some()).count() as f64 * t.provenance.id_confidence
        })
        .sum();
    if weight <= 0.0 {
        return Err("history has no positive-confidence observations".into());
    }
    // Observation gradients occupy only sample times, not every integration step.
    let mut injections = vec![vec![0.0; n]; tape.samples.len()];
    let mut loss = 0.0;
    let mut param_gradient = if parameter_derivatives {
        vec![0.0; model.parameter_count()]
    } else {
        vec![]
    };
    let mut offset_gradient = vec![0.0; n];
    let mut gain_gradient = vec![0.0; n];
    for trace in &recording.traces {
        let i = model.graph.neuron(&trace.neuron)?;
        let gain = readout.gain[i] * p.calcium_scale[i];
        let w = trace.provenance.id_confidence / weight;
        for (t, value) in trace.values.iter().enumerate() {
            if let Some(value) = value {
                let error = readout.offset[i] + gain * tape.states[tape.samples[t]][n + i] - value;
                loss += w * error * error;
                injections[t][i] += 2.0 * w * gain * error;
                if parameter_derivatives {
                    let calcium = tape.states[tape.samples[t]][n + i];
                    param_gradient[5 * n + i] += 2.0
                        * w
                        * error
                        * readout.gain[i]
                        * calcium
                        * params.raw[5 * n + i].sigmoid();
                    offset_gradient[i] += 2.0 * w * error;
                    gain_gradient[i] += 2.0 * w * error * gain * calcium;
                }
            }
        }
    }
    let mut adj = vec![0.0; model.state_len()];
    let mut vjp = adj.clone();
    let mut sample = tape.samples.len();
    for step in (0..tape.states.len()).rev() {
        if sample > 0 && tape.samples[sample - 1] == step {
            sample -= 1;
            for i in 0..n {
                adj[n + i] += injections[sample][i];
            }
        }
        if step > 0 {
            if parameter_derivatives {
                parameter_vjp(
                    model,
                    params,
                    &p,
                    &tape.states[step - 1],
                    &adj,
                    tape.steps[step - 1],
                    &mut param_gradient,
                );
            }
            state_vjp(model, &p, &tape.states[step - 1], &adj, &mut vjp);
            for i in 0..adj.len() {
                adj[i] += tape.steps[step - 1] * vjp[i];
            }
        }
    }
    for i in 0..initial.len() {
        let d = initial[i] - prior[i];
        loss += cfg.prior_weight * d * d / initial.len() as f64;
        adj[i] += 2.0 * cfg.prior_weight * d / initial.len() as f64;
    }
    if !loss.is_finite() || adj.iter().any(|g| !g.is_finite()) {
        return Err("nonfinite inference objective/gradient".into());
    }
    if param_gradient
        .iter()
        .chain(&offset_gradient)
        .chain(&gain_gradient)
        .any(|g| !g.is_finite())
    {
        return Err("nonfinite parameter/readout gradient".into());
    }
    Ok(ObjectiveGradient {
        value: loss,
        initial: adj,
        parameters: param_gradient,
        readout_offset: offset_gradient,
        readout_log_gain: gain_gradient,
        terminal: tape.states.last().unwrap().clone(),
    })
}
/// Optimize a full-network initial condition using only samples <= origin.
/// This is a point estimate with a prior, not proof of hidden-state identifiability.
pub fn infer(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    origin: f64,
    readout: &Readout,
    cfg: &InferenceConfig,
) -> Result<InferredState> {
    recording.validate(&model.graph)?;
    readout.validate(model.n())?;
    if !origin.is_finite() || !cfg.dt.is_finite() || cfg.dt <= 0.0 {
        return Err("invalid inference origin/step".into());
    }
    let end = recording
        .times
        .iter()
        .position(|t| (*t - origin).abs() < 1e-9)
        .ok_or("origin must be on recording grid")?;
    if end == 0 {
        return Err("state inference requires at least two history samples".into());
    }
    let mut history = recording.clone();
    let start = history.times[0];
    history.times.truncate(end + 1);
    for t in &mut history.times {
        *t -= start;
    }
    for trace in &mut history.traces {
        trace.values.truncate(end + 1);
    }
    for values in history.behavior.values_mut() {
        values.truncate(end + 1);
    }
    if cfg.method == InferenceMethod::BlockEkf {
        return crate::state_filter::infer(model, params, &history, readout, cfg.dt, &cfg.filter);
    }
    if !cfg.learning_rate.is_finite() || cfg.learning_rate <= 0.0 || cfg.iterations > 10000 {
        return Err("invalid shooting-inference configuration".into());
    }
    let observed = history
        .traces
        .iter()
        .filter(|t| t.provenance.id_confidence > 0.0 && t.values.iter().any(Option::is_some))
        .count();
    let prior = model.initial(&model.prepare(params)?);
    let mut state = prior.clone();
    let mut m = vec![0.0; state.len()];
    let mut v = m.clone();
    let mut objectives = vec![];
    let (mut value, mut grad, mut terminal) =
        objective_gradient(model, params, &history, readout, &state, &prior, cfg)?;
    objectives.push(value);
    for iteration in 1..=cfg.iterations {
        let mut direction = vec![0.0; state.len()];
        for i in 0..state.len() {
            m[i] = 0.9 * m[i] + 0.1 * grad[i];
            v[i] = 0.999 * v[i] + 0.001 * grad[i] * grad[i];
            direction[i] = (m[i] / (1.0 - 0.9f64.powi(iteration as i32)))
                / ((v[i] / (1.0 - 0.999f64.powi(iteration as i32))).sqrt() + 1e-8);
        }
        let mut accepted = None;
        // Backtracking prevents an optimizer step from increasing prefix loss.
        for attempt in 0..12 {
            let rate = cfg.learning_rate * 0.5f64.powi(attempt);
            let candidate: Vec<_> = state
                .iter()
                .zip(&direction)
                .enumerate()
                .map(|(i, (x, d))| {
                    if i < model.n() {
                        x - rate * d
                    } else {
                        (x - rate * d).clamp(1e-8, 1.0 - 1e-8)
                    }
                })
                .collect();
            if let Ok((next, g, end)) =
                objective_gradient(model, params, &history, readout, &candidate, &prior, cfg)
                && next < value
            {
                accepted = Some((candidate, next, g, end));
                break;
            }
        }
        let Some((next_state, next_value, next_grad, next_terminal)) = accepted else {
            break;
        };
        state = next_state;
        value = next_value;
        grad = next_grad;
        terminal = next_terminal;
        objectives.push(value);
    }
    let prepared = model.prepare(params)?;
    let tape = rollout(model, &prepared, &state, &history.times, cfg.dt)?;
    let history_predictions = tape
        .samples
        .iter()
        .map(|&t| {
            (0..model.n())
                .map(|i| {
                    readout.offset[i]
                        + readout.gain[i]
                            * (prepared.calcium_scale[i] * tape.states[t][model.n() + i])
                })
                .collect()
        })
        .collect();
    Ok(InferredState {
        initial_state: state,
        forecast_state: terminal,
        history_objective: objectives,
        observed_neurons: observed,
        latent_neurons: model.n() - observed,
        history_samples: end + 1,
        history_predictions,
        filter_diagnostics: None,
    })
}
