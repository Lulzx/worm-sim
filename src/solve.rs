//! Fixed-step reference integrators; event boundaries split steps exactly.
use crate::{
    Result,
    math::Scalar,
    model::{Inputs, Model, Parameters},
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Euler,
    Rk4,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Stimulate {
        neuron: String,
        start: f64,
        end: f64,
        amplitude: f64,
    },
    /// Piecewise-linear current at absolute simulation times, zero outside support.
    CurrentWaveform {
        neuron: String,
        times: Vec<f64>,
        amplitudes: Vec<f64>,
    },
    /// Piecewise-linear nonnegative conductance with a fixed reversal potential.
    ConductanceWaveform {
        neuron: String,
        times: Vec<f64>,
        conductances: Vec<f64>,
        reversal: f64,
    },
    /// Prescribed piecewise-linear voltage; other state variables still evolve.
    VoltageClamp {
        neuron: String,
        times: Vec<f64>,
        voltages: Vec<f64>,
    },
    Silence {
        neuron: String,
        start: f64,
        end: f64,
    },
    Ablate {
        neuron: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub duration: f64,
    pub dt: f64,
    pub save_dt: f64,
    pub method: Method,
    #[serde(default)]
    pub events: Vec<Event>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            duration: 1.0,
            dt: 0.001,
            save_dt: 0.01,
            method: Method::Rk4,
            events: vec![],
        }
    }
}
#[derive(Debug)]
pub struct Trajectory<S> {
    pub times: Vec<f64>,
    pub voltage: Vec<Vec<S>>,
    pub fluorescence: Vec<Vec<S>>,
    /// Full voltage/calcium/gate state at the final time, for causal continuation.
    pub final_state: Vec<S>,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if [self.duration, self.dt, self.save_dt]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err("duration, dt and save_dt must be finite and positive".into());
        }
        if self.duration / self.dt > 1e9 || self.duration / self.save_dt > 1e7 {
            return Err("requested run exceeds reference solver size limit".into());
        }
        Ok(())
    }
}
struct Clamp<'a> {
    neuron: usize,
    times: &'a [f64],
    voltages: &'a [f64],
}
impl Clamp<'_> {
    fn active(&self, t: f64) -> bool {
        t >= self.times[0] && t < self.times[self.times.len() - 1]
    }
    fn value(&self, t: f64) -> f64 {
        let k = self
            .times
            .partition_point(|v| *v <= t)
            .saturating_sub(1)
            .min(self.times.len() - 2);
        let f = ((t - self.times[k]) / (self.times[k + 1] - self.times[k])).clamp(0., 1.);
        (1. - f) * self.voltages[k] + f * self.voltages[k + 1]
    }
}
fn project_clamps<S: Scalar>(clamps: &[Clamp<'_>], anchor: f64, time: f64, state: &mut [S]) {
    for c in clamps.iter().filter(|c| c.active(anchor)) {
        state[c.neuron] = S::constant(c.value(time));
    }
}
fn clamp_derivatives<S: Scalar>(clamps: &[Clamp<'_>], anchor: f64, derivative: &mut [S]) {
    for c in clamps.iter().filter(|c| c.active(anchor)) {
        derivative[c.neuron] = S::constant(0.);
    }
}
fn validate_waveform(times: &[f64], values: &[f64], duration: f64) -> Result<()> {
    if times.len() < 2
        || times.len() != values.len()
        || times.iter().chain(values).any(|v| !v.is_finite())
        || times[0] < 0.
        || times[times.len() - 1] > duration
        || times.windows(2).any(|w| w[1] <= w[0])
    {
        return Err(
            "waveform needs matching finite values and strictly increasing times within the run"
                .into(),
        );
    }
    Ok(())
}
/// Output is allocated only on the save grid. RK stages reuse one workspace.
pub fn simulate<S: Scalar>(
    model: &Model,
    params: &Parameters<S>,
    cfg: &Config,
) -> Result<Trajectory<S>> {
    simulate_from_state(model, params, cfg, None)
}
/// Continue from an explicit differentiable state, or use model equilibrium defaults.
pub fn simulate_from_state<S: Scalar>(
    model: &Model,
    params: &Parameters<S>,
    cfg: &Config,
    initial: Option<&[S]>,
) -> Result<Trajectory<S>> {
    cfg.validate()?;
    let p = model.prepare(params)?;
    let n = model.n();
    let len = model.state_len();
    let mut boundaries = vec![0.0, cfg.duration];
    let mut events = Vec::new();
    let mut waveforms = Vec::new();
    let mut clamps = Vec::new();
    for event in &cfg.events {
        if let Event::VoltageClamp {
            neuron,
            times,
            voltages,
        } = event
        {
            validate_waveform(times, voltages, cfg.duration)?;
            clamps.push(Clamp {
                neuron: model.graph.neuron(neuron)?,
                times,
                voltages,
            });
            boundaries.extend(times);
            continue;
        }
        let waveform = match event {
            Event::CurrentWaveform {
                neuron,
                times,
                amplitudes,
            } => Some((neuron, times, amplitudes, None)),
            Event::ConductanceWaveform {
                neuron,
                times,
                conductances,
                reversal,
            } => {
                if !reversal.is_finite()
                    || conductances
                        .iter()
                        .any(|g| *g < 0. || !(g * reversal).is_finite())
                {
                    return Err(
                        "conductance must be nonnegative with finite reversal and drive".into(),
                    );
                }
                Some((neuron, times, conductances, Some(*reversal)))
            }
            _ => None,
        };
        if let Some((neuron, times, amplitudes, reversal)) = waveform {
            validate_waveform(times, amplitudes, cfg.duration)?;
            waveforms.push((model.graph.neuron(neuron)?, times, amplitudes, reversal));
            boundaries.extend(times);
            continue;
        }
        let (name, start, end, amplitude, kind) = match event {
            Event::Stimulate {
                neuron,
                start,
                end,
                amplitude,
            } => (neuron, *start, *end, *amplitude, 0),
            Event::Silence { neuron, start, end } => (neuron, *start, *end, 0.0, 1),
            Event::Ablate { neuron } => (neuron, 0.0, cfg.duration, 0.0, 2),
            Event::CurrentWaveform { .. }
            | Event::ConductanceWaveform { .. }
            | Event::VoltageClamp { .. } => unreachable!(),
        };
        if !start.is_finite()
            || !end.is_finite()
            || !amplitude.is_finite()
            || start < 0.0
            || end <= start
            || end > cfg.duration
        {
            return Err("invalid event interval or amplitude".into());
        }
        events.push((model.graph.neuron(name)?, start, end, amplitude, kind));
        boundaries.extend([start, end]);
    }
    for (i, c) in clamps.iter().enumerate() {
        if events.iter().any(|e| e.0 == c.neuron && e.4 == 2) {
            return Err("cannot clamp an ablated neuron".into());
        }
        if clamps[..i].iter().any(|b| {
            b.neuron == c.neuron
                && b.times[0] < c.times[c.times.len() - 1]
                && c.times[0] < b.times[b.times.len() - 1]
        }) {
            return Err("overlapping voltage clamps on one neuron".into());
        }
    }
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup();
    let mut y = match initial {
        Some(state) => {
            if state.len() != len || state.iter().any(|v| !v.value().is_finite()) {
                return Err("initial state has wrong size or nonfinite values".into());
            }
            state.to_vec()
        }
        None => model.initial(&p),
    };
    let mut temp = y.clone();
    let mut k1 = vec![S::constant(0.0); len];
    let mut k2 = k1.clone();
    let mut k3 = k1.clone();
    let mut k4 = k1.clone();
    let mut release = vec![S::constant(0.0); n];
    let mut input = Inputs::new(n);
    let mut constant_current = vec![0.; n];
    let mut result = Trajectory {
        times: vec![],
        voltage: vec![],
        fluorescence: vec![],
        final_state: vec![],
    };
    let save = |t: f64, y: &[S], out: &mut Trajectory<S>| {
        out.times.push(t);
        out.voltage.push(y[..n].to_vec());
        out.fluorescence
            .push((0..n).map(|i| y[n + i] * p.calcium_scale[i]).collect());
    };
    project_clamps(&clamps, 0., 0., &mut y);
    save(0.0, &y, &mut result);
    let mut t = 0.0;
    let mut sample = 1usize;
    let mut boundary = 1usize;
    while t < cfg.duration {
        while boundary < boundaries.len() && boundaries[boundary] <= t {
            boundary += 1;
        }
        let next_save = (sample as f64 * cfg.save_dt).min(cfg.duration);
        let target = (t + cfg.dt)
            .min(next_save)
            .min(*boundaries.get(boundary).unwrap_or(&cfg.duration))
            .min(cfg.duration);
        let h = target - t;
        if h <= 0.0 {
            return Err("step cannot advance time".into());
        }
        input.current.fill(0.0);
        input.silenced.fill(false);
        input.ablated.fill(false);
        // A stage ending at an event uses the left-hand forcing. The next step
        // uses the right-hand forcing, avoiding RK4 leakage across discontinuities.
        for &(i, start, end, amplitude, kind) in &events {
            if t >= start && t < end {
                match kind {
                    0 => input.current[i] += amplitude,
                    1 => input.silenced[i] = true,
                    _ => input.ablated[i] = true,
                }
            }
        }
        constant_current.copy_from_slice(&input.current);
        // Select segments using the step's left endpoint. At support boundaries
        // the final stage therefore uses the left limit, as rectangular pulses do.
        let set_current = |stage_time: f64, input: &mut Inputs| {
            input.current.copy_from_slice(&constant_current);
            input.conductance.fill(0.);
            input.conductance_drive.fill(0.);
            for &(i, times, amplitudes, reversal) in &waveforms {
                if t >= times[0] && t < times[times.len() - 1] {
                    let k = times.partition_point(|v| *v <= t) - 1;
                    let fraction =
                        ((stage_time - times[k]) / (times[k + 1] - times[k])).clamp(0., 1.);
                    let amplitude = (1. - fraction) * amplitudes[k] + fraction * amplitudes[k + 1];
                    if let Some(reversal) = reversal {
                        input.conductance[i] += amplitude;
                        input.conductance_drive[i] += amplitude * reversal;
                    } else {
                        input.current[i] += amplitude;
                    }
                }
            }
        };
        set_current(t, &mut input);
        model.rhs(&p, &y, &input, &mut release, &mut k1);
        clamp_derivatives(&clamps, t, &mut k1);
        match cfg.method {
            Method::Euler => {
                for i in 0..len {
                    y[i] = y[i] + S::constant(h) * k1[i];
                }
            }
            Method::Rk4 => {
                for i in 0..len {
                    temp[i] = y[i] + S::constant(h * 0.5) * k1[i];
                }
                set_current(t + h * 0.5, &mut input);
                project_clamps(&clamps, t, t + h * 0.5, &mut temp);
                model.rhs(&p, &temp, &input, &mut release, &mut k2);
                clamp_derivatives(&clamps, t, &mut k2);
                for i in 0..len {
                    temp[i] = y[i] + S::constant(h * 0.5) * k2[i];
                }
                project_clamps(&clamps, t, t + h * 0.5, &mut temp);
                model.rhs(&p, &temp, &input, &mut release, &mut k3);
                clamp_derivatives(&clamps, t, &mut k3);
                for i in 0..len {
                    temp[i] = y[i] + S::constant(h) * k3[i];
                }
                set_current(target, &mut input);
                project_clamps(&clamps, t, target, &mut temp);
                model.rhs(&p, &temp, &input, &mut release, &mut k4);
                clamp_derivatives(&clamps, t, &mut k4);
                for i in 0..len {
                    y[i] = y[i]
                        + S::constant(h / 6.0)
                            * (k1[i] + S::constant(2.0) * (k2[i] + k3[i]) + k4[i]);
                }
            }
        }
        // Complete the outgoing clamp at its endpoint, then apply any newly
        // starting clamp. Samples are right-continuous at clamp onsets.
        project_clamps(&clamps, t, target, &mut y);
        project_clamps(&clamps, target, target, &mut y);
        if y.iter().any(|v| !v.value().is_finite()) {
            return Err(format!("nonfinite state at t={target}; reduce dt"));
        }
        t = target;
        if t == next_save {
            save(t, &y, &mut result);
            sample += 1;
        }
    }
    result.final_state = y;
    Ok(result)
}
