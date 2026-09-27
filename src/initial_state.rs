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
    intervals: Vec<usize>,
}
pub(crate) fn validate_currents(
    currents: Option<&[Vec<f64>]>,
    frames: usize,
    neurons: usize,
) -> Result<()> {
    if currents.is_some_and(|rows| {
        rows.len() != frames
            || rows
                .iter()
                .any(|r| r.len() != neurons || r.iter().any(|v| !v.is_finite()))
    }) {
        return Err("invalid piecewise current shape/values".into());
    }
    Ok(())
}
fn rollout(
    model: &Model,
    p: &Prepared<f64>,
    initial: &[f64],
    times: &[f64],
    dt: f64,
    currents: Option<&[Vec<f64>]>,
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
    validate_currents(currents, times.len(), model.n())?;
    let mut intervals = vec![];
    let mut states = vec![initial.to_vec()];
    let mut steps = vec![];
    let mut samples = vec![0];
    let mut rhs = vec![0.0; model.state_len()];
    let mut release = vec![0.0; model.n()];
    let mut input = Inputs::new(model.n());
    let mut t = 0.0;
    for (interval, &target) in times[1..].iter().enumerate() {
        if let Some(rows) = currents {
            input.current.copy_from_slice(&rows[interval]);
        }
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
            intervals.push(interval);
            t = next;
        }
        samples.push(states.len() - 1);
    }
    Ok(Tape {
        states,
        steps,
        samples,
        intervals,
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
#[allow(clippy::too_many_arguments)]
fn parameter_vjp(
    model: &Model,
    raw: &Parameters<f64>,
    p: &Prepared<f64>,
    y: &[f64],
    adj: &[f64],
    h: f64,
    g: &mut [f64],
    input_current: Option<&[f64]>,
) {
    let n = model.n();
    let m = model.pre.len();
    // Reconstruct the current before division by voltage tau. This avoids
    // dividing by a parameter derivative and matches the forward arithmetic.
    let mut current: Vec<f64> = (0..n).map(|i| -(y[i] - p.rest[i])).collect();
    if let Some(input) = input_current {
        for (i, v) in current.iter_mut().enumerate() {
            *v += input[i];
        }
    }
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
    /// Derivatives with respect to held current rows; last row is unused.
    pub currents: Vec<Vec<f64>>,
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
        model, params, recording, readout, initial, prior, cfg, false, None, None,
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
        model, params, recording, readout, initial, initial, &cfg, true, None, None,
    )
}
pub fn parameter_gradient_with_currents(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    readout: &Readout,
    initial: &[f64],
    dt: f64,
    currents: &[Vec<f64>],
) -> Result<ObjectiveGradient> {
    let cfg = InferenceConfig {
        dt,
        prior_weight: 0.0,
        ..Default::default()
    };
    objective_impl(
        model,
        params,
        recording,
        readout,
        initial,
        initial,
        &cfg,
        true,
        Some(currents),
        None,
    )
}
/// Affine readout at sample boundaries; each current row drives its following interval.
pub fn forecast_with_currents(
    model: &Model,
    params: &Parameters<f64>,
    initial: &[f64],
    times: &[f64],
    readout: &Readout,
    dt: f64,
    currents: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>> {
    readout.validate(model.n())?;
    let p = model.prepare(params)?;
    let tape = rollout(model, &p, initial, times, dt, Some(currents))?;
    Ok(tape
        .samples
        .iter()
        .map(|&s| {
            (0..model.n())
                .map(|i| {
                    readout.offset[i]
                        + readout.gain[i] * (p.calcium_scale[i] * tape.states[s][model.n() + i])
                })
                .collect()
        })
        .collect())
}
/// Training loss for an atlas response expressed relative to initial calcium.
/// The affine offset is retained; use zero offsets for a zero initial response.
/// Differentiates the subtracted initial calcium as well as the trajectory.
pub fn response_gradient_with_currents(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    readout: &Readout,
    initial: &[f64],
    dt: f64,
    currents: &[Vec<f64>],
) -> Result<ObjectiveGradient> {
    let cfg = InferenceConfig {
        dt,
        prior_weight: 0.0,
        ..Default::default()
    };
    objective_impl(
        model,
        params,
        recording,
        readout,
        initial,
        initial,
        &cfg,
        true,
        Some(currents),
        Some(0),
    )
}
/// Relative fluorescence response, not a calibrated conversion of optical power
/// to membrane current. No observed response is used to construct predictions.
pub fn response_with_currents(
    model: &Model,
    params: &Parameters<f64>,
    initial: &[f64],
    times: &[f64],
    readout: &Readout,
    dt: f64,
    currents: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>> {
    let mut values = forecast_with_currents(model, params, initial, times, readout, dt, currents)?;
    let p = model.prepare(params)?;
    for row in &mut values {
        for i in 0..model.n() {
            row[i] -= readout.gain[i] * p.calcium_scale[i] * initial[model.n() + i];
        }
    }
    Ok(values)
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
    currents: Option<&[Vec<f64>]>,
    baseline_sample: Option<usize>,
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
    let tape = rollout(model, &p, initial, &recording.times, cfg.dt, currents)?;
    if baseline_sample.is_some_and(|i| i >= tape.samples.len()) {
        return Err("invalid response baseline sample".into());
    }
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
                let calcium = tape.states[tape.samples[t]][n + i]
                    - baseline_sample.map_or(0.0, |b| tape.states[tape.samples[b]][n + i]);
                let error = readout.offset[i] + gain * calcium - value;
                loss += w * error * error;
                injections[t][i] += 2.0 * w * gain * error;
                if let Some(b) = baseline_sample {
                    injections[b][i] -= 2.0 * w * gain * error;
                }
                if parameter_derivatives {
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
    let mut current_gradient = if currents.is_some() {
        vec![vec![0.0; n]; recording.times.len()]
    } else {
        vec![]
    };
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
            let interval = tape.intervals[step - 1];
            if currents.is_some() {
                for i in 0..n {
                    current_gradient[interval][i] += tape.steps[step - 1] * adj[i] * p.inv_tau[i];
                }
            }
            if parameter_derivatives {
                parameter_vjp(
                    model,
                    params,
                    &p,
                    &tape.states[step - 1],
                    &adj,
                    tape.steps[step - 1],
                    &mut param_gradient,
                    currents.map(|rows| rows[interval].as_slice()),
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
        .chain(current_gradient.iter().flatten())
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
        currents: current_gradient,
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
    infer_impl(model, params, recording, origin, readout, cfg, None)
}
pub fn infer_with_currents(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    origin: f64,
    readout: &Readout,
    cfg: &InferenceConfig,
    currents: &[Vec<f64>],
) -> Result<InferredState> {
    infer_impl(
        model,
        params,
        recording,
        origin,
        readout,
        cfg,
        Some(currents),
    )
}
fn infer_impl(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    origin: f64,
    readout: &Readout,
    cfg: &InferenceConfig,
    currents: Option<&[Vec<f64>]>,
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
    if currents.is_some_and(|r| r.len() != recording.times.len()) {
        return Err("current grid differs from recording".into());
    }
    let history_currents = currents.map(|r| &r[..=end]);
    validate_currents(history_currents, history.times.len(), model.n())?;
    if cfg.method == InferenceMethod::BlockEkf {
        return crate::state_filter::infer(
            model,
            params,
            &history,
            readout,
            cfg.dt,
            &cfg.filter,
            history_currents,
        );
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
    let objective = |state: &[f64]| -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let g = objective_impl(
            model,
            params,
            &history,
            readout,
            state,
            &prior,
            cfg,
            false,
            history_currents,
            None,
        )?;
        Ok((g.value, g.initial, g.terminal))
    };
    let (mut value, mut grad, mut terminal) = objective(&state)?;
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
            if let Ok((next, g, end)) = objective(&candidate)
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
    let tape = rollout(
        model,
        &prepared,
        &state,
        &history.times,
        cfg.dt,
        history_currents,
    )?;
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

fn preparation_grid(
    times: &[f64],
    currents: &[Vec<f64>],
    n: usize,
    seconds: f64,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    if !seconds.is_finite() || seconds <= 0. || times.len() < 2 || times[0] != 0. {
        return Err("invalid response preparation period/grid".into());
    }
    validate_currents(Some(currents), times.len(), n)?;
    let mut extended = vec![0.];
    extended.extend(times.iter().map(|t| t + seconds));
    let mut drive = vec![vec![0.; n]];
    drive.extend_from_slice(currents);
    Ok((extended, drive))
}
/// Unforced preparation followed by a response relative to the prepared calcium.
/// A finite preparation period approximates equilibration; callers must check it.
#[allow(clippy::too_many_arguments)]
pub fn prepared_response_with_currents(
    model: &Model,
    params: &Parameters<f64>,
    seed: &[f64],
    times: &[f64],
    readout: &Readout,
    dt: f64,
    currents: &[Vec<f64>],
    preparation_seconds: f64,
) -> Result<Vec<Vec<f64>>> {
    if preparation_seconds == 0. {
        return response_with_currents(model, params, seed, times, readout, dt, currents);
    }
    let (extended, drive) = preparation_grid(times, currents, model.n(), preparation_seconds)?;
    readout.validate(model.n())?;
    let p = model.prepare(params)?;
    let tape = rollout(model, &p, seed, &extended, dt, Some(&drive))?;
    let baseline = &tape.states[tape.samples[1]];
    Ok(tape.samples[1..]
        .iter()
        .map(|&s| {
            (0..model.n())
                .map(|i| {
                    readout.offset[i]
                        + readout.gain[i]
                            * p.calcium_scale[i]
                            * (tape.states[s][model.n() + i] - baseline[model.n() + i])
                })
                .collect()
        })
        .collect())
}
/// Exact discrete gradients include the entire unforced prefix and its calcium
/// baseline. `initial` is the derivative with respect to the preparation seed;
/// returned current rows correspond only to the original response grid.
#[allow(clippy::too_many_arguments)]
pub fn prepared_response_gradient_with_currents(
    model: &Model,
    params: &Parameters<f64>,
    recording: &Recording,
    readout: &Readout,
    seed: &[f64],
    dt: f64,
    currents: &[Vec<f64>],
    preparation_seconds: f64,
) -> Result<ObjectiveGradient> {
    if preparation_seconds == 0. {
        return response_gradient_with_currents(
            model, params, recording, readout, seed, dt, currents,
        );
    }
    recording.validate(&model.graph)?;
    let (times, drive) =
        preparation_grid(&recording.times, currents, model.n(), preparation_seconds)?;
    let mut extended = recording.clone();
    extended.times = times;
    // No measured data or behavior are supplied during preparation.
    extended.behavior.clear();
    for trace in &mut extended.traces {
        trace.values.insert(0, None);
    }
    let cfg = InferenceConfig {
        dt,
        prior_weight: 0.,
        ..Default::default()
    };
    let mut result = objective_impl(
        model,
        params,
        &extended,
        readout,
        seed,
        seed,
        &cfg,
        true,
        Some(&drive),
        Some(1),
    )?;
    result.currents.remove(0);
    Ok(result)
}
/// Prepared state and its residual unforced derivative for stationarity checks.
pub fn prepared_state(
    model: &Model,
    params: &Parameters<f64>,
    seed: &[f64],
    dt: f64,
    seconds: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("invalid preparation step size".into());
    }
    let p = model.prepare(params)?;
    let state = if seconds == 0. {
        if seed.len() != model.state_len() || seed.iter().any(|v| !v.is_finite()) {
            return Err("invalid preparation seed".into());
        }
        seed.to_vec()
    } else {
        if !seconds.is_finite() || seconds < 0. {
            return Err("invalid preparation duration".into());
        }
        rollout(model, &p, seed, &[0., seconds], dt, None)?
            .states
            .pop()
            .ok_or("empty preparation")?
    };
    let mut derivative = vec![0.; model.state_len()];
    let mut scratch = vec![0.; model.n()];
    model.rhs(
        &p,
        &state,
        &Inputs::new(model.n()),
        &mut scratch,
        &mut derivative,
    );
    if derivative.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite preparation derivative".into());
    }
    Ok((state, derivative))
}

/// Apply an arbitrary differentiable loss to prepared baseline-relative outputs.
/// The callback returns its scalar value and derivatives with respect to every
/// [response frame][neuron] fluorescence value. This is a single forward/adjoint
/// pass, including preparation and baseline derivatives; no detached pseudo-targets.
#[allow(clippy::too_many_arguments)]
pub fn prepared_response_objective_gradient(
    model: &Model,
    params: &Parameters<f64>,
    readout: &Readout,
    seed: &[f64],
    times: &[f64],
    dt: f64,
    currents: &[Vec<f64>],
    preparation_seconds: f64,
    objective: impl FnOnce(&[Vec<f64>]) -> Result<(f64, Vec<Vec<f64>>)>,
) -> Result<ObjectiveGradient> {
    let n = model.n();
    readout.validate(n)?;
    let (extended, drive, baseline) = if preparation_seconds == 0. {
        validate_currents(Some(currents), times.len(), n)?;
        (times.to_vec(), currents.to_vec(), 0)
    } else {
        let (t, u) = preparation_grid(times, currents, n, preparation_seconds)?;
        (t, u, 1)
    };
    let p = model.prepare(params)?;
    let tape = rollout(model, &p, seed, &extended, dt, Some(&drive))?;
    let base = &tape.states[tape.samples[baseline]];
    let outputs: Vec<Vec<_>> = tape.samples[baseline..]
        .iter()
        .map(|&s| {
            (0..n)
                .map(|i| {
                    readout.offset[i]
                        + readout.gain[i]
                            * p.calcium_scale[i]
                            * (tape.states[s][n + i] - base[n + i])
                })
                .collect()
        })
        .collect();
    let (value, cotangent) = objective(&outputs)?;
    if !value.is_finite()
        || cotangent.len() != times.len()
        || cotangent
            .iter()
            .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
    {
        return Err("invalid response loss or fluorescence derivatives".into());
    }
    let mut injections = vec![vec![0.; n]; tape.samples.len()];
    let mut param_gradient = vec![0.; model.parameter_count()];
    let mut offset_gradient = vec![0.; n];
    let mut gain_gradient = vec![0.; n];
    for (t, row) in cotangent.iter().enumerate() {
        let sample = t + baseline;
        for i in 0..n {
            let calcium = tape.states[tape.samples[sample]][n + i] - base[n + i];
            let gain = readout.gain[i] * p.calcium_scale[i];
            injections[sample][i] += row[i] * gain;
            injections[baseline][i] -= row[i] * gain;
            param_gradient[5 * n + i] +=
                row[i] * readout.gain[i] * calcium * params.raw[5 * n + i].sigmoid();
            offset_gradient[i] += row[i];
            gain_gradient[i] += row[i] * gain * calcium;
        }
    }
    let mut current_gradient = vec![vec![0.; n]; extended.len()];
    let mut adj = vec![0.; model.state_len()];
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
            let interval = tape.intervals[step - 1];
            let h = tape.steps[step - 1];
            for i in 0..n {
                current_gradient[interval][i] += h * adj[i] * p.inv_tau[i];
            }
            parameter_vjp(
                model,
                params,
                &p,
                &tape.states[step - 1],
                &adj,
                h,
                &mut param_gradient,
                Some(&drive[interval]),
            );
            state_vjp(model, &p, &tape.states[step - 1], &adj, &mut vjp);
            for i in 0..adj.len() {
                adj[i] += h * vjp[i];
            }
        }
    }
    if baseline == 1 {
        current_gradient.remove(0);
    }
    if adj
        .iter()
        .chain(&param_gradient)
        .chain(&offset_gradient)
        .chain(&gain_gradient)
        .chain(current_gradient.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err("nonfinite response objective gradient".into());
    }
    Ok(ObjectiveGradient {
        value,
        initial: adj,
        parameters: param_gradient,
        readout_offset: offset_gradient,
        readout_log_gain: gain_gradient,
        terminal: tape.states.last().unwrap().clone(),
        currents: current_gradient,
    })
}
