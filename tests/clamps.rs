use wormsim::{
    fixtures,
    math::{Dual, Scalar, inverse_softplus},
    model::{Model, Parameters},
    solve::*,
};
fn clamp(start: f64, end: f64, values: Vec<f64>) -> Event {
    Event::VoltageClamp {
        neuron: "N000".into(),
        times: vec![start, end],
        voltages: values,
    }
}
fn fixture() -> (Model, Parameters<f64>) {
    let m = Model::new(fixtures::synthetic(1, 0, 0).compile().unwrap()).unwrap();
    let mut p = m.defaults();
    p.raw[0] = inverse_softplus(0.1);
    p.raw[1] = 0.;
    (m, p)
}
#[test]
fn clamp_start_ramp_and_release_have_exact_semantics() {
    let (m, p) = fixture();
    for method in [Method::Euler, Method::Rk4] {
        let cfg = Config {
            duration: 0.25,
            dt: 0.0001,
            save_dt: 0.01,
            method,
            events: vec![clamp(0.037, 0.173, vec![0.8, -0.2])],
        };
        let out = simulate_from_state(&m, &p, &cfg, Some(&[0., 0., 0.])).unwrap();
        let tau = p.raw[0].softplus() + 1e-9;
        for (&t, v) in out.times.iter().zip(&out.voltage) {
            let expected = if t < 0.037 {
                0.
            } else if t <= 0.173 {
                0.8 - (t - 0.037) / (0.173 - 0.037)
            } else {
                -0.2 * (-(t - 0.173) / tau).exp()
            };
            let tolerance = match method {
                Method::Euler => 4e-5,
                Method::Rk4 => 1e-12,
            };
            assert!(
                (v[0] - expected).abs() < tolerance,
                "t={t}: {} vs {expected}",
                v[0]
            );
        }
    }
}
#[test]
fn adjacent_clamps_are_right_continuous_and_order_independent() {
    let (m, p) = fixture();
    let mut cfg = Config {
        duration: 0.2,
        dt: 0.007,
        save_dt: 0.1,
        events: vec![
            clamp(0., 0.1, vec![0.2, 0.4]),
            clamp(0.1, 0.2, vec![-0.3, -0.3]),
        ],
        ..Config::default()
    };
    let first = simulate(&m, &p, &cfg).unwrap();
    assert_eq!(first.voltage, vec![vec![0.2], vec![-0.3], vec![-0.3]]);
    cfg.events.reverse();
    assert_eq!(first.voltage, simulate(&m, &p, &cfg).unwrap().voltage);
}
#[test]
fn calcium_and_its_gradient_follow_prescribed_voltage() {
    let (m, p) = fixture();
    let prepared = m.prepare(&p).unwrap();
    let mut dual = Parameters {
        raw: p.raw.iter().map(|&x| Dual::constant(x)).collect(),
    };
    dual.raw[2].tangent = 1.; // Threshold, while voltage is externally fixed.
    let cfg = Config {
        duration: 0.2,
        dt: 0.0002,
        save_dt: 0.2,
        events: vec![
            clamp(0., 0.2, vec![0.7, 0.7]),
            Event::Stimulate {
                neuron: "N000".into(),
                start: 0.,
                end: 0.2,
                amplitude: 100.,
            },
        ],
        ..Config::default()
    };
    let out = simulate_from_state(
        &m,
        &dual,
        &cfg,
        Some(&[
            Dual {
                value: 2.,
                tangent: 1.,
            },
            Dual::constant(0.),
            Dual::constant(0.),
        ]),
    )
    .unwrap();
    let release = ((0.7 - prepared.threshold[0]) * prepared.slope[0]).sigmoid();
    let factor = (1. - (-0.2 * prepared.inv_calcium_tau[0]).exp()) * prepared.calcium_scale[0];
    assert!((out.fluorescence[1][0].value - release * factor).abs() < 1e-12);
    assert!(
        (out.fluorescence[1][0].tangent + prepared.slope[0] * release * (1. - release) * factor)
            .abs()
            < 1e-12
    );
    assert_eq!(out.voltage[0][0].tangent, 0.);
    assert_eq!(out.voltage[1][0].tangent, 0.);
}
#[test]
fn clamp_drives_unclamped_gap_neighbor() {
    let m = Model::new(fixtures::synthetic(2, 0, 1).compile().unwrap()).unwrap();
    let mut p = m.defaults();
    p.raw[2] = 0.;
    p.raw[3] = 0.;
    let prepared = m.prepare(&p).unwrap();
    for start in [0., 1.] {
        let cfg = Config {
            duration: 0.2,
            dt: 0.0002,
            save_dt: 0.2,
            events: vec![clamp(0., 0.2, vec![start, 1.])],
            ..Config::default()
        };
        let out = simulate_from_state(&m, &p, &cfg, Some(&[0.; 6])).unwrap();
        let g = prepared.gap[0];
        let rate = (1. + g) * prepared.inv_tau[1];
        let rise = 1. - (-0.2 * rate).exp();
        let slope = (1. - start) / 0.2;
        let expected = g / (1. + g) * (start * rise + slope * (0.2 - rise / rate));
        assert!((out.voltage[1][1] - expected).abs() < 1e-12);
    }
}
#[test]
fn conflicting_clamps_are_rejected() {
    let (m, p) = fixture();
    for events in [
        vec![clamp(0., 0.2, vec![1., 1.]), clamp(0.1, 0.3, vec![0., 0.])],
        vec![
            clamp(0., 0.2, vec![1., 1.]),
            Event::Ablate {
                neuron: "N000".into(),
            },
        ],
        vec![clamp(0., 0.2, vec![f64::NAN, 0.])],
        vec![clamp(0., 0.2, vec![0.])],
    ] {
        let cfg = Config {
            events,
            ..Config::default()
        };
        assert!(simulate(&m, &p, &cfg).is_err());
    }
}
