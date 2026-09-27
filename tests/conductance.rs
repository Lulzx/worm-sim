use wormsim::{
    fixtures,
    math::{Dual, Scalar, inverse_softplus},
    model::{Model, Parameters},
    solve::*,
};
fn fixture() -> (Model, Parameters<f64>) {
    let m = Model::new(fixtures::synthetic(1, 0, 0).compile().unwrap()).unwrap();
    let mut p = m.defaults();
    p.raw[0] = inverse_softplus(0.1);
    p.raw[1] = 0.;
    (m, p)
}
fn event(g: Vec<f64>, e: f64) -> Event {
    Event::ConductanceWaveform {
        neuron: "N000".into(),
        times: vec![0.037, 0.173],
        conductances: g,
        reversal: e,
    }
}
#[test]
fn overlapping_baths_match_passive_analytic_solution() {
    let (m, p) = fixture();
    let cfg = Config {
        duration: 0.25,
        dt: 0.0005,
        save_dt: 0.25,
        events: vec![event(vec![2., 2.], 0.7), event(vec![1., 1.], -0.5)],
        ..Config::default()
    };
    let tau = p.raw[0].softplus() + 1e-9;
    let expected = (2. * 0.7 - 0.5) / 4.
        * (1. - (-4. * (0.173 - 0.037) / tau).exp())
        * (-(0.25 - 0.173) / tau).exp();
    let out = simulate(&m, &p, &cfg).unwrap();
    assert!((out.voltage[1][0] - expected).abs() < 1e-9);
}
#[test]
fn conductance_ramp_state_derivative_matches_integrated_shunt() {
    let (m, p) = fixture();
    let dual = Parameters {
        raw: p.raw.iter().map(|&x| Dual::constant(x)).collect(),
    };
    let cfg = Config {
        duration: 0.25,
        dt: 0.001,
        save_dt: 0.25,
        events: vec![event(vec![1., 3.], 0.7)],
        ..Config::default()
    };
    let initial = vec![
        Dual {
            value: 0.2,
            tangent: 1.,
        },
        Dual::constant(0.),
        Dual::constant(0.),
    ];
    let result = simulate_from_state(&m, &dual, &cfg, Some(&initial)).unwrap();
    let tau = p.raw[0].softplus() + 1e-9;
    let expected = (-(0.25 + 2. * (0.173 - 0.037)) / tau).exp();
    assert!((result.voltage[1][0].tangent - expected).abs() < 1e-9);
    assert!(result.voltage[1][0].tangent < (-0.25 / tau).exp());
}
#[test]
fn conductance_respects_ablation_and_rejects_invalid_parameters() {
    let (m, p) = fixture();
    let mut cfg = Config {
        duration: 0.25,
        events: vec![
            event(vec![1., 1.], 1.),
            Event::Ablate {
                neuron: "N000".into(),
            },
        ],
        ..Config::default()
    };
    let out = simulate(&m, &p, &cfg).unwrap();
    assert!(out.voltage.iter().all(|v| v[0] == 0.));
    for event in [
        event(vec![-1., 1.], 1.),
        event(vec![1., 1.], f64::NAN),
        event(vec![1., f64::INFINITY], 1.),
        event(vec![2., 2.], f64::MAX),
    ] {
        cfg.events = vec![event];
        assert!(simulate(&m, &p, &cfg).is_err());
    }
}
