use wormsim::bench::optimization::LearningRateSchedule;

#[test]
fn schedule_endpoints_single_step_and_legacy_rate_are_explicit() {
    let schedule = LearningRateSchedule::Cosine {
        minimum_fraction: 0.2,
    };
    for (update, expected) in [(1, 0.01), (2, 0.006), (3, 0.002)] {
        assert!((schedule.rate(0.01, update, 3).unwrap() - expected).abs() < 1e-15);
        assert_eq!(
            LearningRateSchedule::Constant {}
                .rate(0.01, update, 3)
                .unwrap(),
            0.01
        );
    }
    assert_eq!(schedule.rate(0.01, 1, 1).unwrap(), 0.01);
    let rates: Vec<_> = (1..=100)
        .map(|i| schedule.rate(0.01, i, 100).unwrap())
        .collect();
    assert!(rates.windows(2).all(|r| r[0] >= r[1]));
    assert_eq!(
        LearningRateSchedule::Cosine {
            minimum_fraction: 0.
        }
        .rate(0.01, 3, 3)
        .unwrap(),
        0.
    );
    for value in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(
            LearningRateSchedule::Cosine {
                minimum_fraction: value
            }
            .validate()
            .is_err()
        );
    }
    for (base, update, total) in [
        (0., 1, 2),
        (-1., 1, 2),
        (f64::NAN, 1, 2),
        (0.1, 0, 2),
        (0.1, 3, 2),
        (0.1, 1, 0),
    ] {
        assert!(schedule.rate(base, update, total).is_err());
    }
    let parsed: LearningRateSchedule =
        serde_json::from_str(r#"{"kind":"cosine","minimum_fraction":0.2}"#).unwrap();
    assert_eq!(
        parsed.rate(0.01, 2, 3).unwrap(),
        schedule.rate(0.01, 2, 3).unwrap()
    );
    assert!(
        serde_json::from_str::<LearningRateSchedule>(r#"{"kind":"constant","ignored":2}"#).is_err()
    );
}
