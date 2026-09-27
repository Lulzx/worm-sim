use std::collections::BTreeMap;
use wormsim::{
    data::{Provenance, Recording, Trace},
    fixtures,
    initial_state::{self, InferenceConfig, Readout},
    math::inverse_softplus,
    model::Model,
    solve::{self, Config, Method},
};
fn fixture() -> (
    Model,
    wormsim::model::Parameters<f64>,
    Recording,
    Vec<f64>,
    Readout,
    Config,
) {
    let model = Model::new(fixtures::synthetic(4, 3, 2).compile().unwrap()).unwrap();
    let mut p = model.defaults();
    for i in 0..model.n() {
        p.raw[i] = inverse_softplus(2.0);
        p.raw[4 * model.n() + i] = inverse_softplus(1.5);
    }
    p.raw[model.parameter_count() - 1] = inverse_softplus(0.5);
    let mut initial = model.initial(&model.prepare(&p).unwrap());
    initial[0] += 0.7;
    initial[1] -= 0.2;
    initial[4] = 0.65;
    initial[6] = 0.3;
    initial[8] = 0.6;
    let cfg = Config {
        duration: 2.0,
        dt: 0.02,
        save_dt: 0.2,
        method: Method::Euler,
        events: vec![],
    };
    let truth = solve::simulate_from_state(&model, &p, &cfg, Some(&initial)).unwrap();
    let readout = Readout {
        offset: vec![-0.5; 4],
        gain: vec![1.7, 0.8, 2.0, 1.0],
    };
    let traces = [0, 2]
        .into_iter()
        .map(|i| Trace {
            neuron: model.graph.names[i].clone(),
            values: truth
                .fluorescence
                .iter()
                .enumerate()
                .map(|(t, v)| {
                    if t == 3 && i == 2 {
                        None
                    } else {
                        Some(readout.offset[i] + readout.gain[i] * v[i])
                    }
                })
                .collect(),
            provenance: Provenance {
                dataset: "synthetic".into(),
                version: "1".into(),
                id_confidence: if i == 0 { 1.0 } else { 0.6 },
            },
        })
        .collect();
    let recording = Recording {
        dataset: "synthetic".into(),
        animal_id: "a".into(),
        condition: "state-inference".into(),
        times: truth.times,
        traces,
        behavior: BTreeMap::new(),
    };
    (model, p, recording, initial, readout, cfg)
}
#[test]
fn all_initial_state_adjoint_components_match_central_differences() {
    let (model, p, recording, _, readout, _) = fixture();
    let prior = model.initial(&model.prepare(&p).unwrap());
    let mut start = prior.clone();
    start[1] += 0.12;
    let cfg = InferenceConfig {
        dt: 0.02,
        prior_weight: 0.03,
        ..Default::default()
    };
    let (_, gradient, _) =
        initial_state::objective_gradient(&model, &p, &recording, &readout, &start, &prior, &cfg)
            .unwrap();
    for i in 0..start.len() {
        let eps = 1e-5;
        let mut plus = start.clone();
        plus[i] += eps;
        let mut minus = start.clone();
        minus[i] -= eps;
        let a = initial_state::objective_gradient(
            &model, &p, &recording, &readout, &plus, &prior, &cfg,
        )
        .unwrap()
        .0;
        let b = initial_state::objective_gradient(
            &model, &p, &recording, &readout, &minus, &prior, &cfg,
        )
        .unwrap()
        .0;
        assert!(
            (gradient[i] - (a - b) / (2.0 * eps)).abs() < 1e-8,
            "component {i}: {} vs {}",
            gradient[i],
            (a - b) / (2.0 * eps)
        );
    }
}
#[test]
fn inference_reduces_prefix_loss_is_causal_and_carries_full_state() {
    let (model, p, mut recording, _, readout, _) = fixture();
    let cfg = InferenceConfig {
        dt: 0.02,
        iterations: 80,
        learning_rate: 0.04,
        prior_weight: 1e-5,
        ..Default::default()
    };
    let inferred = initial_state::infer(&model, &p, &recording, 1.0, &readout, &cfg).unwrap();
    assert_eq!(inferred.observed_neurons, 2);
    assert_eq!(inferred.latent_neurons, 2);
    assert_eq!(inferred.forecast_state.len(), 12);
    assert_eq!(inferred.history_samples, 6);
    assert!(inferred.history_objective.last().unwrap() < &(inferred.history_objective[0] * 0.03));
    assert!(inferred.history_objective.windows(2).all(|p| p[1] < p[0]));
    for trace in &mut recording.traces {
        for i in 6..trace.values.len() {
            trace.values[i] = Some(123456.0);
        }
    }
    let other = initial_state::infer(&model, &p, &recording, 1.0, &readout, &cfg).unwrap();
    assert_eq!(inferred.initial_state, other.initial_state);
    assert_eq!(inferred.forecast_state, other.forecast_state);
    let forecast = solve::simulate_from_state(
        &model,
        &p,
        &Config {
            duration: 1.0,
            dt: 0.02,
            save_dt: 0.2,
            method: Method::Euler,
            events: vec![],
        },
        Some(&inferred.forecast_state),
    )
    .unwrap();
    assert_eq!(forecast.voltage[0], inferred.forecast_state[..model.n()]);
}
#[test]
fn explicit_default_state_matches_original_solver_and_invalid_states_fail() {
    let (model, p, _, _, _, cfg) = fixture();
    let initial = model.initial(&model.prepare(&p).unwrap());
    let a = solve::simulate(&model, &p, &cfg).unwrap();
    let b = solve::simulate_from_state(&model, &p, &cfg, Some(&initial)).unwrap();
    assert_eq!(a.fluorescence, b.fluorescence);
    assert_eq!(a.final_state, b.final_state);
    assert!(solve::simulate_from_state(&model, &p, &cfg, Some(&initial[..2])).is_err());
    let mut invalid = initial;
    invalid[0] = f64::NAN;
    assert!(solve::simulate_from_state(&model, &p, &cfg, Some(&invalid)).is_err());
}

#[test]
fn conditional_parameter_and_readout_gradients_match_finite_differences() {
    let (model, p, recording, initial, readout, _) = fixture();
    let cfg = InferenceConfig {
        dt: 0.02,
        prior_weight: 0.0,
        ..Default::default()
    };
    let g = initial_state::parameter_gradient(&model, &p, &recording, &readout, &initial, 0.02)
        .unwrap();
    let value = |p: &wormsim::model::Parameters<f64>, r: &Readout| {
        initial_state::objective_gradient(&model, p, &recording, r, &initial, &initial, &cfg)
            .unwrap()
            .0
    };
    // Move the targets away from the exact generating model before checking;
    // otherwise every first derivative would be zero at the synthetic truth.
    let mut observed = recording.clone();
    for trace in &mut observed.traces {
        for x in trace.values.iter_mut().flatten() {
            *x += 0.15;
        }
    }
    let g2 =
        initial_state::parameter_gradient(&model, &p, &observed, &readout, &initial, 0.02).unwrap();
    assert!(g.parameters.iter().all(|v| v.abs() < 1e-12));
    for i in 0..p.raw.len() {
        let eps = 1e-5;
        let mut a = p.clone();
        let mut b = p.clone();
        a.raw[i] += eps;
        b.raw[i] -= eps;
        let va = initial_state::objective_gradient(
            &model, &a, &observed, &readout, &initial, &initial, &cfg,
        )
        .unwrap()
        .0;
        let vb = initial_state::objective_gradient(
            &model, &b, &observed, &readout, &initial, &initial, &cfg,
        )
        .unwrap()
        .0;
        assert!(
            (g2.parameters[i] - (va - vb) / (2.0 * eps)).abs() < 1e-8,
            "parameter {i}: {} vs {}",
            g2.parameters[i],
            (va - vb) / (2.0 * eps)
        );
    }
    assert!(value(&p, &readout) < 1e-20);
    for i in 0..model.n() {
        for log_gain in [false, true] {
            let eps: f64 = 1e-5;
            let mut a = readout.clone();
            let mut b = readout.clone();
            if log_gain {
                a.gain[i] *= eps.exp();
                b.gain[i] *= (-eps).exp();
            } else {
                a.offset[i] += eps;
                b.offset[i] -= eps;
            }
            let va = initial_state::objective_gradient(
                &model, &p, &observed, &a, &initial, &initial, &cfg,
            )
            .unwrap()
            .0;
            let vb = initial_state::objective_gradient(
                &model, &p, &observed, &b, &initial, &initial, &cfg,
            )
            .unwrap()
            .0;
            let derivative = if log_gain {
                g2.readout_log_gain[i]
            } else {
                g2.readout_offset[i]
            };
            assert!((derivative - (va - vb) / (2.0 * eps)).abs() < 1e-8);
        }
    }
}

#[test]
fn driven_state_parameter_and_current_adjoints_match_finite_differences() {
    let (model, mut p, recording, mut initial, readout, cfg) = fixture();
    let mut currents: Vec<Vec<f64>> = (0..recording.times.len())
        .map(|t| {
            (0..model.n())
                .map(|i| 0.4 * (t as f64 * 0.3 + i as f64).sin())
                .collect()
        })
        .collect();
    let g = initial_state::parameter_gradient_with_currents(
        &model, &p, &recording, &readout, &initial, cfg.dt, &currents,
    )
    .unwrap();
    let loss = |p: &wormsim::model::Parameters<f64>, initial: &[f64], u: &[Vec<f64>]| {
        let ys = initial_state::forecast_with_currents(
            &model,
            p,
            initial,
            &recording.times,
            &readout,
            cfg.dt,
            u,
        )
        .unwrap();
        let (mut sum, mut weight) = (0.0, 0.0);
        for tr in &recording.traces {
            let i = model.graph.neuron(&tr.neuron).unwrap();
            for (t, y) in tr.values.iter().enumerate() {
                if let Some(y) = y {
                    let w = tr.provenance.id_confidence;
                    sum += w * (ys[t][i] - y).powi(2);
                    weight += w;
                }
            }
        }
        sum / weight
    };
    assert!((g.value - loss(&p, &initial, &currents)).abs() < 1e-14);
    let eps = 1e-6;
    for i in 0..p.raw.len() {
        let old = p.raw[i];
        p.raw[i] = old + eps;
        let a = loss(&p, &initial, &currents);
        p.raw[i] = old - eps;
        let b = loss(&p, &initial, &currents);
        p.raw[i] = old;
        assert!(
            ((a - b) / (2.0 * eps) - g.parameters[i]).abs() < 1e-7,
            "parameter {i}"
        );
    }
    for i in 0..initial.len() {
        let old = initial[i];
        initial[i] = old + eps;
        let a = loss(&p, &initial, &currents);
        initial[i] = old - eps;
        let b = loss(&p, &initial, &currents);
        initial[i] = old;
        assert!(
            ((a - b) / (2.0 * eps) - g.initial[i]).abs() < 1e-7,
            "initial {i}"
        );
    }
    for t in 0..currents.len() {
        for i in 0..model.n() {
            let old = currents[t][i];
            currents[t][i] = old + eps;
            let a = loss(&p, &initial, &currents);
            currents[t][i] = old - eps;
            let b = loss(&p, &initial, &currents);
            currents[t][i] = old;
            assert!(
                ((a - b) / (2.0 * eps) - g.currents[t][i]).abs() < 1e-7,
                "current {t} {i}"
            );
        }
    }
    assert!(g.currents.last().unwrap().iter().all(|g| *g == 0.0));
}
#[test]
fn driven_forecast_matches_independent_event_solver_and_inference_ignores_future_currents() {
    let (model, p, mut recording, initial, readout, mut cfg) = fixture();
    let currents: Vec<Vec<f64>> = (0..recording.times.len())
        .map(|t| {
            (0..model.n())
                .map(|i| 0.4 * (t as f64 * 0.3 + i as f64).sin())
                .collect()
        })
        .collect();
    for (t, pair) in recording.times.windows(2).enumerate() {
        for (i, amplitude) in currents[t].iter().enumerate() {
            cfg.events.push(solve::Event::Stimulate {
                neuron: model.graph.names[i].clone(),
                start: pair[0],
                end: pair[1],
                amplitude: *amplitude,
            });
        }
    }
    let truth = solve::simulate_from_state(&model, &p, &cfg, Some(&initial)).unwrap();
    let pred = initial_state::forecast_with_currents(
        &model,
        &p,
        &initial,
        &recording.times,
        &readout,
        cfg.dt,
        &currents,
    )
    .unwrap();
    for (actual, expected) in pred.iter().zip(&truth.fluorescence) {
        for i in 0..model.n() {
            assert!((actual[i] - readout.offset[i] - readout.gain[i] * expected[i]).abs() < 1e-12);
        }
    }
    let prior = model.initial(&model.prepare(&p).unwrap());
    for tr in &mut recording.traces {
        tr.values.fill(None);
        let i = model.graph.neuron(&tr.neuron).unwrap();
        tr.values[0] = Some(
            readout.offset[i]
                + readout.gain[i]
                    * model.prepare(&p).unwrap().calcium_scale[i]
                    * prior[model.n() + i],
        );
    }
    let origin = recording.times[5];
    cfg.duration = origin;
    cfg.events
        .retain(|e| matches!(e,solve::Event::Stimulate{end,..} if *end<=origin));
    let expected = solve::simulate_from_state(&model, &p, &cfg, Some(&prior)).unwrap();
    for method in [
        initial_state::InferenceMethod::Shooting,
        initial_state::InferenceMethod::BlockEkf,
    ] {
        let options = InferenceConfig {
            method,
            dt: cfg.dt,
            iterations: 0,
            ..Default::default()
        };
        let inferred = initial_state::infer_with_currents(
            &model, &p, &recording, origin, &readout, &options, &currents,
        )
        .unwrap();
        for (a, b) in inferred.forecast_state.iter().zip(&expected.final_state) {
            assert!((a - b).abs() < 1e-12);
        }
        let mut changed = currents.clone();
        for r in &mut changed[6..] {
            r.fill(f64::NAN);
        }
        assert_eq!(
            inferred.forecast_state,
            initial_state::infer_with_currents(
                &model, &p, &recording, origin, &readout, &options, &changed
            )
            .unwrap()
            .forecast_state
        );
    }
}

#[test]
fn relative_response_adjoint_matches_parameter_state_gain_and_current_differences() {
    check_response_gradient(0.0);
}
#[test]
fn prepared_response_adjoint_differentiates_seed_prefix_and_calcium_baseline() {
    check_response_gradient(0.4);
}

fn check_response_gradient(preparation: f64) {
    let (model, params, recording, initial, mut readout, config) = fixture();
    readout.offset.fill(0.0);
    let n = model.n();
    let mut currents = vec![vec![0.; n]; recording.times.len()];
    currents[0][1] = 0.7;
    currents[2][2] = -0.3;
    let gradient = initial_state::prepared_response_gradient_with_currents(
        &model,
        &params,
        &recording,
        &readout,
        &initial,
        config.dt,
        &currents,
        preparation,
    )
    .unwrap();
    let objective =
        |p: &wormsim::model::Parameters<f64>, y: &[f64], r: &Readout, u: &[Vec<f64>]| {
            let prediction = initial_state::prepared_response_with_currents(
                &model,
                p,
                y,
                &recording.times,
                r,
                config.dt,
                u,
                preparation,
            )
            .unwrap();
            let mut error = 0.;
            let mut weight = 0.;
            for trace in &recording.traces {
                let i = model.graph.neuron(&trace.neuron).unwrap();
                for (t, value) in trace.values.iter().enumerate() {
                    if let Some(v) = value {
                        error += trace.provenance.id_confidence * (prediction[t][i] - v).powi(2);
                        weight += trace.provenance.id_confidence;
                    }
                }
            }
            error / weight
        };
    assert!((objective(&params, &initial, &readout, &currents) - gradient.value).abs() < 1e-12);
    let eps = 1e-5;
    let compare = |a: f64, b: f64| assert!((a - b).abs() < 2e-7, "{a} != {b}");
    for i in 0..params.raw.len() {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus.raw[i] += eps;
        minus.raw[i] -= eps;
        compare(
            (objective(&plus, &initial, &readout, &currents)
                - objective(&minus, &initial, &readout, &currents))
                / (2. * eps),
            gradient.parameters[i],
        );
    }
    for i in 0..initial.len() {
        let mut plus = initial.clone();
        let mut minus = initial.clone();
        plus[i] += eps;
        minus[i] -= eps;
        compare(
            (objective(&params, &plus, &readout, &currents)
                - objective(&params, &minus, &readout, &currents))
                / (2. * eps),
            gradient.initial[i],
        );
    }
    for i in 0..n {
        let mut plus = readout.clone();
        let mut minus = readout.clone();
        plus.gain[i] *= eps.exp();
        minus.gain[i] *= (-eps).exp();
        compare(
            (objective(&params, &initial, &plus, &currents)
                - objective(&params, &initial, &minus, &currents))
                / (2. * eps),
            gradient.readout_log_gain[i],
        );
        plus = readout.clone();
        minus = readout.clone();
        plus.offset[i] += eps;
        minus.offset[i] -= eps;
        compare(
            (objective(&params, &initial, &plus, &currents)
                - objective(&params, &initial, &minus, &currents))
                / (2. * eps),
            gradient.readout_offset[i],
        );
    }
    for t in 0..currents.len() {
        let mut plus = currents.clone();
        let mut minus = currents.clone();
        plus[t][1] += eps;
        minus[t][1] -= eps;
        compare(
            (objective(&params, &initial, &readout, &plus)
                - objective(&params, &initial, &readout, &minus))
                / (2. * eps),
            gradient.currents[t][1],
        );
    }
    let response = initial_state::prepared_response_with_currents(
        &model,
        &params,
        &initial,
        &recording.times,
        &readout,
        config.dt,
        &currents,
        preparation,
    )
    .unwrap();
    assert!(response[0].iter().all(|v| v.abs() < 1e-12));
}

#[test]
fn prepared_state_matches_event_solver_and_zero_duration_preserves_old_responses() {
    let (model, params, recording, initial, readout, config) = fixture();
    let expected = solve::simulate_from_state(
        &model,
        &params,
        &Config {
            duration: 0.4,
            save_dt: 0.4,
            ..config.clone()
        },
        Some(&initial),
    )
    .unwrap();
    let (state, derivative) =
        initial_state::prepared_state(&model, &params, &initial, config.dt, 0.4).unwrap();
    for (a, b) in state.iter().zip(expected.final_state) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(derivative.iter().all(|v| v.is_finite()));
    let currents = vec![vec![0.1; model.n()]; recording.times.len()];
    let old = initial_state::response_with_currents(
        &model,
        &params,
        &initial,
        &recording.times,
        &readout,
        config.dt,
        &currents,
    )
    .unwrap();
    let compatible = initial_state::prepared_response_with_currents(
        &model,
        &params,
        &initial,
        &recording.times,
        &readout,
        config.dt,
        &currents,
        0.,
    )
    .unwrap();
    assert_eq!(old, compatible);
    let prepared = initial_state::prepared_response_with_currents(
        &model,
        &params,
        &initial,
        &recording.times,
        &readout,
        config.dt,
        &currents,
        0.4,
    )
    .unwrap();
    let staged = initial_state::response_with_currents(
        &model,
        &params,
        &state,
        &recording.times,
        &readout,
        config.dt,
        &currents,
    )
    .unwrap();
    for (a, b) in prepared.iter().flatten().zip(staged.iter().flatten()) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(initial_state::prepared_state(&model, &params, &initial, 0., 0.).is_err());
    assert!(
        initial_state::prepared_response_with_currents(
            &model,
            &params,
            &initial,
            &recording.times,
            &readout,
            config.dt,
            &currents,
            -1.
        )
        .is_err()
    );
}
