use wormsim::{
    bench::connectome_lds::{ConnectomeLds, StimulusSequence},
    fixtures,
};
#[test]
fn shared_input_em_matches_independent_full_normal_system() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/connectome_lds.json")).unwrap();
    let model: ConnectomeLds = serde_json::from_value(fixture["model"].clone()).unwrap();
    let expected: ConnectomeLds = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let sequences: Vec<StimulusSequence> =
        serde_json::from_value(fixture["sequences"].clone()).unwrap();
    assert!(sequences.iter().all(|s| s.target != 2));
    let graph = fixtures::synthetic(3, 1, 0).compile().unwrap();
    model.validate_for_graph(&graph).unwrap();
    let (actual, report) = model
        .em_step(
            &sequences,
            fixture["ridge"].as_f64().unwrap(),
            fixture["cap"].as_f64().unwrap(),
        )
        .unwrap();
    assert!(
        (report.preceding_negative_log_likelihood
            - fixture["preceding_negative_log_likelihood"]
                .as_f64()
                .unwrap())
        .abs()
            < 1e-10
    );
    assert!(
        (report.unprojected_transition_norm
            - fixture["unprojected_transition_norm"].as_f64().unwrap())
        .abs()
            < 1e-10
    );
    assert_eq!(
        report.observations,
        fixture["observations"].as_u64().unwrap() as usize
    );
    assert!(report.unprojected_transition_norm > fixture["cap"].as_f64().unwrap());
    for (a, b) in [
        (&actual.kernel, &expected.kernel),
        (&actual.gaussian.transition, &expected.gaussian.transition),
        (&actual.gaussian.process_cov, &expected.gaussian.process_cov),
        (&actual.gaussian.initial_cov, &expected.gaussian.initial_cov),
        (&actual.gaussian.noise, &expected.gaussian.noise),
    ] {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-10, "{x} != {y}");
        }
    }
    // Duplicating every sequence preserves the normalized moment update and
    // forces reuse within each pattern group, including different targets.
    let repeated: Vec<_> = sequences
        .iter()
        .cycle()
        .take(sequences.len() * 3)
        .cloned()
        .collect();
    let (repeated_model, repeated_report) = model
        .em_step(
            &repeated,
            fixture["ridge"].as_f64().unwrap(),
            fixture["cap"].as_f64().unwrap(),
        )
        .unwrap();
    assert_eq!(repeated_report.observations, 3 * report.observations);
    assert!(
        (repeated_report.preceding_negative_log_likelihood
            - 3.0 * report.preceding_negative_log_likelihood)
            .abs()
            < 1e-10
    );
    for (a, b) in [
        (&actual.kernel, &repeated_model.kernel),
        (
            &actual.gaussian.transition,
            &repeated_model.gaussian.transition,
        ),
        (
            &actual.gaussian.process_cov,
            &repeated_model.gaussian.process_cov,
        ),
        (
            &actual.gaussian.initial_cov,
            &repeated_model.gaussian.initial_cov,
        ),
        (&actual.gaussian.noise, &repeated_model.gaussian.noise),
    ] {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-10);
        }
    }
    actual.validate_for_graph(&graph).unwrap();
    let (uncapped, _) = model
        .em_step(&sequences, fixture["ridge"].as_f64().unwrap(), 0.99)
        .unwrap();
    for (name, values) in [
        ("unprojected_transition", &uncapped.gaussian.transition),
        ("unprojected_kernel", &uncapped.kernel),
    ] {
        let expected: Vec<f64> = serde_json::from_value(fixture[name].clone()).unwrap();
        for (a, b) in values.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10);
        }
    }
    let impulse = actual.impulse(2, 4).unwrap();
    assert_eq!(impulse[1][2], actual.kernel[0]);
    assert!(impulse[1][2].abs() > 1e-5);
    assert_eq!(actual.inputs(2, 4).unwrap()[3], vec![0.; 3]);
}
#[test]
fn support_and_unidentifiable_kernel_lags_are_rejected() {
    let graph = fixtures::synthetic(3, 1, 0).compile().unwrap();
    let mut model = ConnectomeLds::new(&graph, 4, 0.5).unwrap();
    let seq = StimulusSequence {
        target: 0,
        observations: vec![vec![(0, 0., 1.)]; 4],
    };
    assert!(
        model
            .em_step(&[seq], 0.001, 0.99)
            .unwrap_err()
            .contains("unidentifiable")
    );
    model.gaussian.transition[1] = 0.2;
    assert!(model.validate_for_graph(&graph).is_err());
    assert!(ConnectomeLds::new(&graph, 0, 0.5).is_err());
}
