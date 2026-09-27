use wormsim::bench::lds::{GaussianLds, Observation};

#[test]
fn reused_covariances_match_full_smoothing_with_changed_data_and_inputs() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lds_controlled.json")).unwrap();
    let g: GaussianLds = serde_json::from_value(f["gaussian"].clone()).unwrap();
    let mut sequence: Vec<Vec<Observation>> =
        serde_json::from_value(f["sequence"].clone()).unwrap();
    let mut inputs: Vec<Vec<f64>> = serde_json::from_value(f["inputs"].clone()).unwrap();
    // Also exercise missing frames and repeated scalar updates in a fixed order.
    sequence.push(vec![]);
    sequence.push(vec![(1, 0.3, 0.4), (0, -0.2, 0.7), (1, 0.1, 0.9)]);
    inputs.extend([vec![0.2, -0.1], vec![0.0, 0.0]]);
    let plan = g.prepare_smoother(&sequence).unwrap();
    for repeat in 0..4 {
        for (_, y, _) in sequence.iter_mut().flatten() {
            *y = *y * -0.6 + repeat as f64 * 0.17;
        }
        for u in inputs.iter_mut().flatten() {
            *u = *u * 0.4 - 0.2;
        }
        let full = g.smooth_with_inputs(&sequence, &inputs).unwrap();
        let reused = plan.smooth(&sequence, &inputs).unwrap();
        assert_eq!(full.observations, reused.observations);
        assert!((full.negative_log_likelihood - reused.negative_log_likelihood).abs() < 1e-12);
        for (a, b) in full
            .means
            .iter()
            .flatten()
            .zip(reused.means.iter().flatten())
        {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(full.covariances, reused.covariances);
        assert_eq!(full.lag_covariances, reused.lag_covariances);
    }
    let mut changed = sequence.clone();
    changed[0][0].2 = 0.3;
    assert!(plan.smooth(&changed, &inputs).is_err());
    changed = sequence.clone();
    changed[0].reverse();
    assert!(plan.smooth(&changed, &inputs).is_err());
    changed = sequence.clone();
    changed[0][0].1 = f64::NAN;
    assert!(plan.smooth(&changed, &inputs).is_err());
    assert!(plan.smooth(&sequence[..1], &inputs[..1]).is_err());
    assert!(plan.smooth(&sequence, &inputs[..1]).is_err());
    assert!(g.prepare_smoother(&[]).is_err());
    assert!(g.prepare_smoother(&[vec![(0, 0.0, 0.0)]]).is_err());
}
