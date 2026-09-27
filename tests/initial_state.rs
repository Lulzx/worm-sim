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
