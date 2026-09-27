//! Approximate extended Kalman history filter: full coupled mean dynamics, with
//! independent 3x3 voltage/calcium/gate covariance blocks per neuron. Cross-neuron
//! covariances are omitted; this is not a full-network EKF or calibrated posterior.
use crate::{
    Result,
    data::Recording,
    initial_state::{InferredState, Readout},
    math::Scalar,
    model::{Inputs, Model, Parameters, Prepared},
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterConfig {
    /// Variances in normalized V, calcium, gate coordinates.
    pub initial_variance: [f64; 3],
    /// Additive process variance rates per simulated second.
    pub process_variance_rate: [f64; 3],
    /// Observation variance after converting fluorescence into calcium units.
    pub observation_variance: f64,
}
impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            initial_variance: [0.25, 0.04, 0.04],
            process_variance_rate: [0.05, 0.005, 0.005],
            observation_variance: 0.0025,
        }
    }
}
impl FilterConfig {
    fn validate(&self) -> Result<()> {
        if self
            .initial_variance
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || self
                .process_variance_rate
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || !self.observation_variance.is_finite()
            || self.observation_variance <= 0.0
        {
            return Err("invalid state-filter variances".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterDiagnostics {
    pub observation_updates: usize,
    pub projected_updates: usize,
    pub prior_innovation_mse: f64,
    pub posterior_reconstruction_mse: f64,
    /// Per-neuron [V,C,gate] posterior covariance at the origin. Approximate blocks
    /// only; cross-neuron covariance and effects of mean projection are absent.
    pub forecast_covariance_blocks: Vec<[f64; 9]>,
}
fn local_jacobians(model: &Model, p: &Prepared<f64>, y: &[f64], out: &mut [[f64; 9]]) {
    let n = model.n();
    for i in 0..n {
        let r = ((y[i] - p.threshold[i]) * p.slope[i]).sigmoid();
        let dr = p.slope[i] * r * (1.0 - r);
        out[i] = [
            -p.inv_tau[i],
            0.0,
            0.0,
            dr * p.inv_calcium_tau[i],
            -p.inv_calcium_tau[i],
            0.0,
            dr * (1.0 - y[2 * n + i]) * p.inv_synapse_tau,
            0.0,
            -(r + 1.0) * p.inv_synapse_tau,
        ];
    }
    for e in 0..model.pre.len() {
        let a = model.pre[e] as usize;
        let b = model.post[e] as usize;
        out[b][0] -= p.inv_tau[b] * p.weight[e] * y[2 * n + a];
        if a == b {
            out[b][2] += p.inv_tau[b] * p.weight[e] * (p.reversal[e] - y[b]);
        }
    }
    for e in 0..model.gap_a.len() {
        let a = model.gap_a[e] as usize;
        let b = model.gap_b[e] as usize;
        out[a][0] -= p.inv_tau[a] * p.gap[e];
        out[b][0] -= p.inv_tau[b] * p.gap[e];
    }
}
// Off-neuron derivatives only affect the receiving neuron's voltage. Combine
// chemical and gap paths for each ordered pair before propagating variance, so
// their within-source V/gate covariance cross term is retained.
type Influence = (usize, usize, [f64; 3]);
fn influences(model: &Model, p: &Prepared<f64>) -> Vec<Influence> {
    let mut map = std::collections::BTreeMap::<(usize, usize), [f64; 3]>::new();
    for e in 0..model.pre.len() {
        let a = model.pre[e] as usize;
        let b = model.post[e] as usize;
        if a == b {
            continue;
        }
        let v = map.entry((b, a)).or_default();
        v[1] += p.weight[e] * p.reversal[e] * p.inv_tau[b];
        v[2] += p.weight[e] * p.inv_tau[b];
    }
    for e in 0..model.gap_a.len() {
        let a = model.gap_a[e] as usize;
        let b = model.gap_b[e] as usize;
        map.entry((a, b)).or_default()[0] += p.gap[e] * p.inv_tau[a];
        map.entry((b, a)).or_default()[0] += p.gap[e] * p.inv_tau[b];
    }
    map.into_iter()
        .map(|((target, source), v)| (target, source, v))
        .collect()
}
fn cross_variances(edges: &[Influence], y: &[f64], cov: &[[f64; 9]], out: &mut [f64]) {
    out.fill(0.0);
    for &(target, source, [a, b, c]) in edges {
        let z = b - c * y[target];
        let p = &cov[source];
        out[target] += a * a * p[0] + a * z * (p[2] + p[6]) + z * z * p[8];
    }
}
fn transform_covariance(f: &[f64; 9], p: &[f64; 9]) -> [f64; 9] {
    let mut fp = [0.0; 9];
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                fp[i * 3 + j] += f[i * 3 + k] * p[k * 3 + j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i * 3 + j] += fp[i * 3 + k] * f[j * 3 + k];
            }
        }
    }
    out
}
fn symmetrize(p: &mut [f64; 9]) {
    for i in 0..3 {
        for j in 0..i {
            let v = 0.5 * (p[i * 3 + j] + p[j * 3 + i]);
            p[i * 3 + j] = v;
            p[j * 3 + i] = v;
        }
    }
}
/// The caller supplies a history-only recording with times rebased to zero.
pub(crate) fn infer(
    model: &Model,
    params: &Parameters<f64>,
    history: &Recording,
    readout: &Readout,
    dt: f64,
    cfg: &FilterConfig,
    currents: Option<&[Vec<f64>]>,
) -> Result<InferredState> {
    cfg.validate()?;
    crate::initial_state::validate_currents(currents, history.times.len(), model.n())?;
    history.validate(&model.graph)?;
    readout.validate(model.n())?;
    if !dt.is_finite()
        || dt <= 0.0
        || history.times.len() < 2
        || history.times[0] != 0.0
        || history.times.last().unwrap() / dt > 1e6
    {
        return Err("invalid state-filter grid/step".into());
    }
    let p = model.prepare(params)?;
    let n = model.n();
    let mut y = model.initial(&p);
    let mut initial = y.clone();
    let mut cov = vec![[0.0; 9]; n];
    for c in &mut cov {
        for i in 0..3 {
            c[i * 3 + i] = cfg.initial_variance[i];
        }
    }
    let mut jac = vec![[0.0; 9]; n];
    let edges = influences(model, &p);
    let mut cross = vec![0.0; n];
    let mut dy = vec![0.0; model.state_len()];
    let mut release = vec![0.0; n];
    let mut input = Inputs::new(n);
    let mut traces = vec![];
    for trace in &history.traces {
        if trace.provenance.id_confidence > 0.0 {
            traces.push((model.graph.neuron(&trace.neuron)?, trace));
        }
    }
    let observed = traces
        .iter()
        .filter(|(_, t)| t.values.iter().any(Option::is_some))
        .count();
    let mut predictions = vec![];
    let mut before = 0.0;
    let mut after = 0.0;
    let mut weight = 0.0;
    let mut updates = 0;
    let mut projected = 0;
    let mut time = 0.0;
    for (frame, &target) in history.times.iter().enumerate() {
        if frame > 0
            && let Some(rows) = currents
        {
            input.current.copy_from_slice(&rows[frame - 1]);
        }
        while time < target {
            let next = (time + dt).min(target);
            let h = next - time;
            if h <= 0.0 {
                return Err("filter step cannot advance".into());
            }
            model.rhs(&p, &y, &input, &mut release, &mut dy);
            local_jacobians(model, &p, &y, &mut jac);
            cross_variances(&edges, &y, &cov, &mut cross);
            for i in 0..n {
                let mut f = jac[i];
                for v in &mut f {
                    *v *= h;
                }
                for j in 0..3 {
                    f[j * 3 + j] += 1.0;
                }
                let mut next = transform_covariance(&f, &cov[i]);
                next[0] += h * h * cross[i];
                for j in 0..3 {
                    next[j * 3 + j] += h * cfg.process_variance_rate[j];
                }
                symmetrize(&mut next);
                cov[i] = next;
            }
            for (v, d) in y.iter_mut().zip(&dy) {
                *v += h * d;
            }
            time = next;
            if y.iter().any(|v| !v.is_finite()) || cov.iter().flatten().any(|v| !v.is_finite()) {
                return Err("nonfinite filter prediction; reduce dt".into());
            }
        }
        for &(i, trace) in &traces {
            let Some(value) = trace.values[frame] else {
                continue;
            };
            let gain = readout.gain[i] * p.calcium_scale[i];
            let measured = (value - readout.offset[i]) / gain;
            let residual = measured - y[n + i];
            let w = trace.provenance.id_confidence;
            let r = cfg.observation_variance / w;
            let variance = cov[i][4] + r;
            if !variance.is_finite() || variance <= 0.0 {
                return Err("invalid filter innovation variance".into());
            }
            before += w * (gain * residual).powi(2);
            weight += w;
            updates += 1;
            let k = [
                cov[i][1] / variance,
                cov[i][4] / variance,
                cov[i][7] / variance,
            ];
            for j in 0..3 {
                y[j * n + i] += k[j] * residual;
            }
            let mut clipped = false;
            for j in 1..3 {
                let old = y[j * n + i];
                y[j * n + i] = old.clamp(1e-8, 1.0 - 1e-8);
                clipped |= old != y[j * n + i];
            }
            if clipped {
                projected += 1;
            }
            // Joseph form preserves positive semidefiniteness under roundoff.
            let mut transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            for j in 0..3 {
                transform[j * 3 + 1] -= k[j];
            }
            let mut next = transform_covariance(&transform, &cov[i]);
            for a in 0..3 {
                for b in 0..3 {
                    next[a * 3 + b] += r * k[a] * k[b];
                }
            }
            symmetrize(&mut next);
            cov[i] = next;
            after += w * (value - readout.offset[i] - gain * y[n + i]).powi(2);
        }
        if frame == 0 {
            initial.clone_from(&y);
        }
        let prediction: Vec<_> = (0..n)
            .map(|i| readout.offset[i] + readout.gain[i] * (p.calcium_scale[i] * y[n + i]))
            .collect();
        if prediction.iter().any(|v| !v.is_finite())
            || y.iter().any(|v| !v.is_finite())
            || cov.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("nonfinite filter posterior".into());
        }
        predictions.push(prediction);
    }
    if weight <= 0.0 || !before.is_finite() || !after.is_finite() {
        return Err("filter history has no valid observations or overflowing error".into());
    }
    Ok(InferredState {
        initial_state: initial,
        forecast_state: y,
        history_objective: vec![],
        observed_neurons: observed,
        latent_neurons: n - observed,
        history_samples: history.times.len(),
        history_predictions: predictions,
        filter_diagnostics: Some(FilterDiagnostics {
            observation_updates: updates,
            projected_updates: projected,
            prior_innovation_mse: before / weight,
            posterior_reconstruction_mse: after / weight,
            forecast_covariance_blocks: cov,
        }),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_blocks_match_finite_differences_with_chemical_self_edges_and_gaps() {
        let mut graph = crate::fixtures::synthetic(4, 3, 2);
        let mut edge = graph.chemical[0].clone();
        edge.post = edge.pre.clone();
        graph.chemical.push(edge);
        let model = Model::new(graph.compile().unwrap()).unwrap();
        let p = model.prepare(&model.defaults()).unwrap();
        let n = model.n();
        let mut y = model.initial(&p);
        for (i, v) in y.iter_mut().enumerate() {
            *v += 0.02 * (i as f64).sin();
        }
        let mut jac = vec![[0.0; 9]; n];
        local_jacobians(&model, &p, &y, &mut jac);
        let input = Inputs::new(n);
        let mut release = vec![0.0; n];
        for i in 0..n {
            for column in 0..3 {
                let eps = 1e-6;
                let mut a = y.clone();
                let mut b = y.clone();
                a[column * n + i] += eps;
                b[column * n + i] -= eps;
                let mut fa = vec![0.0; y.len()];
                let mut fb = fa.clone();
                model.rhs(&p, &a, &input, &mut release, &mut fa);
                model.rhs(&p, &b, &input, &mut release, &mut fb);
                for row in 0..3 {
                    assert!(
                        (jac[i][row * 3 + column]
                            - (fa[row * n + i] - fb[row * n + i]) / (2.0 * eps))
                            .abs()
                            < 1e-7
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod covariance_tests {
    use super::*;
    #[test]
    fn projected_covariance_matches_full_network_jacobian_prediction() {
        let model = Model::new(crate::fixtures::synthetic(3, 2, 1).compile().unwrap()).unwrap();
        let p = model.prepare(&model.defaults()).unwrap();
        let n = model.n();
        let d = 3 * n;
        let h = 0.003;
        let y = model.initial(&p);
        let cov = vec![[0.3, 0.02, 0.01, 0.02, 0.1, 0.015, 0.01, 0.015, 0.05]; n];
        let mut jac = vec![[0.0; 9]; n];
        local_jacobians(&model, &p, &y, &mut jac);
        let mut cross = vec![0.0; n];
        cross_variances(&influences(&model, &p), &y, &cov, &mut cross);
        let mut f = vec![0.0; d * d];
        let mut release = vec![0.0; n];
        let input = Inputs::new(n);
        for column in 0..d {
            let eps = 1e-6;
            let mut a = y.clone();
            let mut b = y.clone();
            a[column] += eps;
            b[column] -= eps;
            let mut fa = vec![0.0; d];
            let mut fb = fa.clone();
            model.rhs(&p, &a, &input, &mut release, &mut fa);
            model.rhs(&p, &b, &input, &mut release, &mut fb);
            for row in 0..d {
                f[row * d + column] =
                    h * (fa[row] - fb[row]) / (2.0 * eps) + if row == column { 1.0 } else { 0.0 };
            }
        }
        for i in 0..n {
            let mut local = jac[i];
            for v in &mut local {
                *v *= h;
            }
            for j in 0..3 {
                local[j * 3 + j] += 1.0;
            }
            let mut actual = transform_covariance(&local, &cov[i]);
            actual[0] += h * h * cross[i];
            for a in 0..3 {
                for b in 0..3 {
                    let mut expected = 0.0;
                    for source in 0..n {
                        for x in 0..3 {
                            for z in 0..3 {
                                expected += f[(a * n + i) * d + x * n + source]
                                    * cov[source][x * 3 + z]
                                    * f[(b * n + i) * d + z * n + source];
                            }
                        }
                    }
                    assert!((actual[a * 3 + b] - expected).abs() < 1e-9);
                }
            }
        }
    }
}
