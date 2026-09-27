use wormsim::baseline::{self, Bundle, operator::Operator};
fn fixture() -> Bundle {
    serde_json::from_str(include_str!("fixtures/linear_baseline.json")).unwrap()
}
#[test]
fn independent_numpy_fixture_matches_full_protocol() {
    let bundle = fixture();
    let report = baseline::evaluate(&bundle).unwrap();
    assert!(report.all_parity_passed);
    assert!(report.models[0].max_stams_error < 1e-13);
    assert!(report.models[0].max_correlation_error < 1e-13);
    assert_eq!(report.models[0].stams_test.pairs, 5);
}
#[test]
fn first_frame_contains_direct_and_filtered_input_without_transition() {
    let bundle = fixture();
    let p = baseline::predict(&bundle.models[0]).unwrap();
    // C00 * H00 + D00 = 1.1*1 + 0.1; W must not act on H before t=0.
    assert!((p.probes[0].values[0] - 1.2).abs() < 1e-14);
    // At t=1: C00*(W00*H00 + H_lag1,00); D has no second lag.
    assert!((p.probes[0].values[1] - 0.88).abs() < 1e-14);
}
#[test]
fn compact_operators_match_dense_batch_products() {
    let compact = [
        Operator::Zero { rows: 2, cols: 2 },
        Operator::Identity { size: 2 },
        Operator::LaggedDiagonal {
            size: 2,
            lags: 2,
            values: vec![1., 2., 3., 4.],
        },
        Operator::Csr {
            rows: 2,
            cols: 3,
            offsets: vec![0, 2, 3],
            columns: vec![0, 2, 1],
            values: vec![1., -2., 3.],
        },
    ];
    for op in compact {
        op.validate().unwrap();
        let (rows, cols) = op.shape();
        let right: Vec<_> = (0..cols * 3).map(|i| i as f64 * 0.1).collect();
        let mut actual = vec![0.; rows * 3];
        op.multiply(&right, 3, &mut actual).unwrap();
        let dense = op.dense();
        for r in 0..rows {
            for b in 0..3 {
                let expected = (0..cols)
                    .map(|c| dense[r * cols + c] * right[c * 3 + b])
                    .sum::<f64>();
                assert!((actual[r * 3 + b] - expected).abs() < 1e-14);
            }
        }
    }
}
#[test]
fn malformed_operators_and_ordering_rejected() {
    for op in [
        Operator::Csr {
            rows: 2,
            cols: 2,
            offsets: vec![0, 2, 1],
            columns: vec![0],
            values: vec![1.],
        },
        Operator::Csr {
            rows: 1,
            cols: 2,
            offsets: vec![0, 2],
            columns: vec![1, 1],
            values: vec![1., 1.],
        },
        Operator::LaggedDiagonal {
            size: 2,
            lags: 2,
            values: vec![1.],
        },
    ] {
        assert!(op.validate().is_err());
    }
    let mut b = fixture();
    b.models[0].neurons[1] = "A".into();
    assert!(b.validate().is_err());
    let mut b = fixture();
    b.models[0].sample_rate = 10.0;
    assert!(b.validate().is_err());
}
#[test]
fn baseline_archive_roundtrip_and_corruption() {
    let b = fixture();
    let bytes = baseline::pack(&b).unwrap();
    let roundtrip = baseline::unpack(&bytes).unwrap();
    assert!(baseline::evaluate(&roundtrip).unwrap().all_parity_passed);
    for n in [0, 3, 43, bytes.len() - 1] {
        assert!(baseline::unpack(&bytes[..n]).is_err());
    }
    let mut bad = bytes;
    bad[12] ^= 1;
    assert!(baseline::unpack(&bad).is_err());
}
#[test]
fn scoring_masks_missing_and_diagonal_without_imputation() {
    let predicted = [999., 1., 2., 3., -999., 4., 5., 6., 42.];
    let observed = [
        Some(0.),
        Some(2.),
        None,
        Some(6.),
        Some(0.),
        Some(8.),
        Some(10.),
        Some(12.),
        Some(0.),
    ];
    let s = baseline::score(&predicted, &observed, 3).unwrap();
    assert_eq!(s.pairs, 5);
    assert!((s.correlation - 1.0).abs() < 1e-14);
    assert!(baseline::score(&[1.; 9], &observed, 3).is_err());
    assert!(baseline::score(&predicted, &[None; 9], 3).is_err());
}
#[test]
fn reference_mismatch_is_reported_as_failure() {
    let mut b = fixture();
    b.models[0].reference.stams[0] += 0.1;
    assert!(!baseline::evaluate(&b).unwrap().all_parity_passed);
}
