use wormsim::{
    fixtures,
    math::{Scalar, inverse_softplus},
    model::Model,
    solve::*,
};

fn fixture() -> (Model, wormsim::model::Parameters<f64>) {
    let model = Model::new(fixtures::synthetic(1, 0, 0).compile().unwrap()).unwrap();
    let mut p = model.defaults();
    p.raw[0] = inverse_softplus(0.1);
    p.raw[1] = 0.;
    (model, p)
}
fn waveform(times: Vec<f64>, amplitudes: Vec<f64>) -> Event {
    Event::CurrentWaveform {
        neuron: "N000".into(),
        times,
        amplitudes,
    }
}

#[test]
fn ramp_with_off_grid_knots_matches_exact_passive_solution() {
    let (m, p) = fixture();
    let start = 0.037;
    let end = 0.173;
    let cfg = Config {
        duration: 0.25,
        dt: 0.003,
        save_dt: 0.25,
        events: vec![waveform(vec![start, end], vec![0., 2.])],
        ..Config::default()
    };
    let out = simulate(&m, &p, &cfg).unwrap();
    let tau = p.raw[0].softplus() + 1e-9;
    let length = end - start;
    let expected = 2. / length
        * (length - tau + tau * (-length / tau).exp())
        * (-(cfg.duration - end) / tau).exp();
    assert!((out.voltage[1][0] - expected).abs() < 1e-8);
    // Signed waveforms and simultaneous rectangular events add in the RHS.
    let mut both = cfg.clone();
    both.events.push(waveform(vec![start, end], vec![0., -2.]));
    both.events.push(Event::Stimulate {
        neuron: "N000".into(),
        start: 0.,
        end: 0.25,
        amplitude: 0.5,
    });
    let out = simulate(&m, &p, &both).unwrap();
    assert!((out.voltage[1][0] - 0.5 * (1. - (-0.25 / tau).exp())).abs() < 1e-8);
}

#[test]
fn constant_waveform_equals_pulse_including_discontinuous_endpoints() {
    let (m, p) = fixture();
    for method in [Method::Euler, Method::Rk4] {
        let mut cfg = Config {
            duration: 0.3,
            dt: 0.007,
            save_dt: 0.03,
            method,
            events: vec![waveform(vec![0.031, 0.177], vec![1.5, 1.5])],
        };
        let wave = simulate(&m, &p, &cfg).unwrap();
        cfg.events = vec![Event::Stimulate {
            neuron: "N000".into(),
            start: 0.031,
            end: 0.177,
            amplitude: 1.5,
        }];
        let pulse = simulate(&m, &p, &cfg).unwrap();
        for (a, b) in wave
            .voltage
            .iter()
            .flatten()
            .zip(pulse.voltage.iter().flatten())
        {
            assert!((a - b).abs() < 1e-14);
        }
    }
}

#[test]
fn waveform_validation_and_protocol_roundtrip() {
    let (m, p) = fixture();
    for (times, values) in [
        (vec![], vec![]),
        (vec![0.], vec![1.]),
        (vec![0., 0.2], vec![1.]),
        (vec![0.2, 0.2], vec![1., 2.]),
        (vec![0.2, 0.1], vec![1., 2.]),
        (vec![-0.1, 0.2], vec![1., 2.]),
        (vec![0., 2.], vec![1., 2.]),
        (vec![0., f64::NAN], vec![1., 2.]),
        (vec![0., 0.2], vec![1., f64::INFINITY]),
    ] {
        let cfg = Config {
            events: vec![waveform(times, values)],
            ..Config::default()
        };
        assert!(simulate(&m, &p, &cfg).is_err());
    }
    let cfg = Config {
        events: vec![waveform(vec![0., 0.1, 0.2], vec![0., 1., 0.])],
        ..Config::default()
    };
    let decoded: Config = serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
    assert_eq!(
        simulate(&m, &p, &cfg).unwrap().voltage,
        simulate(&m, &p, &decoded).unwrap().voltage
    );
}

#[test]
fn waveform_preserves_parameter_derivatives() {
    use wormsim::{math::Dual, model::Parameters};
    let (m, p) = fixture();
    let cfg = Config {
        duration: 0.3,
        dt: 0.003,
        save_dt: 0.3,
        events: vec![waveform(vec![0.013, 0.071, 0.211], vec![-0.2, 1., 0.3])],
        ..Config::default()
    };
    let dual = Parameters {
        raw: p
            .raw
            .iter()
            .enumerate()
            .map(|(i, &value)| Dual {
                value,
                tangent: if i == 0 { 1. } else { 0. },
            })
            .collect(),
    };
    let gradient = simulate(&m, &dual, &cfg).unwrap().voltage[1][0].tangent;
    let mut plus = p.clone();
    let mut minus = p.clone();
    plus.raw[0] += 1e-5;
    minus.raw[0] -= 1e-5;
    let finite_difference = (simulate(&m, &plus, &cfg).unwrap().voltage[1][0]
        - simulate(&m, &minus, &cfg).unwrap().voltage[1][0])
        / 2e-5;
    assert!(gradient.abs() > 1e-4);
    assert!((gradient - finite_difference).abs() < 1e-8);
}
