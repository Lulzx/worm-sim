use std::collections::BTreeMap;
use wormsim::{
    bench::{
        self, Axis, Dataset, Partition, Split, Trial,
        lds::{self, FitConfig, GaussianLds},
    },
    data::{Provenance, Recording, Trace},
    fixtures,
};
#[test]
fn kalman_rts_matches_independent_two_state_gaussian_conditioning() {
    let g = GaussianLds {
        dim: 1,
        outputs: 1,
        transition: vec![0.8],
        observation: vec![1.0],
        input_dim: 0,
        input_weights: vec![],
        process_cov: vec![0.36],
        noise: vec![0.25],
        initial_cov: vec![1.0],
    };
    let p = g
        .smooth(&[vec![(0, 1.0, 1.0)], vec![(0, -0.5, 0.5)]])
        .unwrap();
    // Joint prior covariance [[1,.8],[.8,1]], observation variances .25,.5.
    // Direct 2x2 precision inversion gives posterior covariance
    // [[43,20],[20,61]]/247 and mean [152,19]/247.
    assert!((p.means[0][0] - 152.0 / 247.0).abs() < 1e-12);
    assert!((p.means[1][0] - 19.0 / 247.0).abs() < 1e-12);
    assert!((p.covariances[0][0] - 43.0 / 247.0).abs() < 1e-12);
    assert!((p.covariances[1][0] - 61.0 / 247.0).abs() < 1e-12);
    assert!((p.lag_covariances[0][0] - 20.0 / 247.0).abs() < 1e-12);
    let missing = g.smooth(&[vec![(0, 1.0, 1.0)], vec![]]).unwrap();
    assert!((missing.means[1][0] - 0.64).abs() < 1e-12);
    assert!((missing.covariances[1][0] - 0.488).abs() < 1e-12);
    assert_eq!(missing.observations, 1);
    assert!(g.smooth(&[vec![(0, f64::NAN, 1.0)]]).is_err());
}
#[test]
fn masked_em_improves_training_likelihood_and_enforces_contraction() {
    let sequences: Vec<_> = (0..6)
        .map(|s| {
            (0..80)
                .map(|t| {
                    if t % 11 == 4 {
                        vec![]
                    } else {
                        vec![(0, ((t as f64) * 0.08 + s as f64).sin(), 1.0)]
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let mut g = GaussianLds {
        dim: 1,
        outputs: 1,
        transition: vec![0.7],
        observation: vec![1.0],
        input_dim: 0,
        input_weights: vec![],
        process_cov: vec![0.2],
        noise: vec![0.2],
        initial_cov: vec![1.0],
    };
    let nll = |g: &GaussianLds| {
        sequences
            .iter()
            .map(|s| g.smooth(s).unwrap().negative_log_likelihood)
            .sum::<f64>()
    };
    let first = nll(&g);
    for _ in 0..5 {
        g.em_step(&sequences, 0.995, 0.0).unwrap();
        assert!(g.transition_norm_bound().unwrap() <= 0.995 + 1e-10);
    }
    assert!(nll(&g) < first - 10.0);
    // A nonnormal 2x2 matrix can have eigenvalues <1 yet large transient growth.
    let nonnormal = GaussianLds {
        dim: 2,
        outputs: 1,
        transition: vec![0.5, 3.0, 0.0, 0.5],
        observation: vec![1.0, 0.0],
        input_dim: 0,
        input_weights: vec![],
        process_cov: vec![1.0, 0.0, 0.0, 1.0],
        noise: vec![0.1],
        initial_cov: vec![1.0, 0.0, 0.0, 1.0],
    };
    assert!(nonnormal.transition_norm_bound().unwrap() > 3.0);
}
fn data() -> (wormsim::data::IndexedGraph, Dataset) {
    let graph = fixtures::synthetic(3, 1, 0).compile().unwrap();
    let trials = (0..8)
        .map(|animal| {
            let values: Vec<Vec<Option<f64>>> = (0..3)
                .map(|i| {
                    (0..81)
                        .map(|t| {
                            let phase = animal as f64 * 0.8 + t as f64 * 0.04;
                            let r = 0.98f64.powi(t);
                            if i == 1 && t % 7 == 2 {
                                None
                            } else {
                                Some(
                                    r * match i {
                                        0 => phase.cos(),
                                        1 => phase.sin(),
                                        _ => 0.5 * (phase.cos() + phase.sin()),
                                    },
                                )
                            }
                        })
                        .collect()
                })
                .collect();
            Trial {
                id: format!("trial-{animal}"),
                stimulated_neuron: None,
                forecast_origin: Some(10.0),
                response_labels: BTreeMap::new(),
                recording: Recording {
                    dataset: "synthetic LDS".into(),
                    animal_id: format!("animal-{animal}"),
                    condition: "rotating latent".into(),
                    times: (0..81).map(|t| t as f64 * 0.5).collect(),
                    traces: (0..3)
                        .map(|i| Trace {
                            neuron: graph.names[i].clone(),
                            values: values[i].clone(),
                            provenance: Provenance {
                                dataset: "synthetic".into(),
                                version: "1".into(),
                                id_confidence: if i == 1 { 0.6 } else { 1.0 },
                            },
                        })
                        .collect(),
                    behavior: BTreeMap::new(),
                },
            }
        })
        .collect();
    let d = Dataset {
        schema_version: 1,
        name: "latent fit".into(),
        graph_hash: graph.hash.clone(),
        source: "synthetic numerical check".into(),
        trials,
    };
    (graph, d)
}
#[test]
fn latent_fitting_and_history_filtering_never_read_test_futures() {
    let (graph, mut data) = data();
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let cfg = FitConfig {
        ranks: vec![1, 2],
        iterations: 3,
        ..Default::default()
    };
    let (model, report) =
        lds::fit_select(&data, &graph, &split, cfg.clone(), |_, _| Ok(())).unwrap();
    assert_eq!(report.candidates.len(), 8);
    assert_eq!(model.training_trials, split.train);
    let before = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    let mut legacy = serde_json::to_value(&model).unwrap();
    legacy.as_object_mut().unwrap().remove("behavior");
    legacy["gaussian"]
        .as_object_mut()
        .unwrap()
        .remove("input_dim");
    legacy["gaussian"]
        .as_object_mut()
        .unwrap()
        .remove("input_weights");
    let legacy: lds::LatentModel = serde_json::from_value(legacy).unwrap();
    let old = legacy
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    for (a, b) in before.trials.iter().zip(old.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
    let score = bench::evaluate(&data, &graph, &split, &before, Partition::Test).unwrap();
    assert!(score.forecast_horizons[0].macro_neuron_r2.unwrap() > 0.0);
    for trial in &mut data.trials {
        if split.test.contains(&trial.id) {
            for trace in &mut trial.recording.traces {
                for v in &mut trace.values[21..] {
                    *v = Some(12345.0);
                }
            }
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let (other, other_report) =
        lds::fit_select(&data, &graph, &updated, cfg, |_, _| Ok(())).unwrap();
    assert_eq!(model.gaussian.transition, other.gaussian.transition);
    assert_eq!(model.gaussian.observation, other.gaussian.observation);
    assert_eq!(model.mean, other.mean);
    assert_eq!(model.scale, other.scale);
    assert_eq!(report.selected_iteration, other_report.selected_iteration);
    assert_eq!(report.selected_rank, other_report.selected_rank);
    let after = other
        .predict(&data, &graph, &updated, Partition::Test)
        .unwrap();
    for (a, b) in before.trials.iter().zip(after.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
    let mut invalid = other;
    invalid.gaussian.transition[0] = 10.0;
    assert!(
        invalid
            .predict(&data, &graph, &updated, Partition::Test)
            .is_err()
    );
}

#[test]
fn multivariate_rts_matches_independent_numpy_joint_gaussian_fixture() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lds_gaussian.json")).unwrap();
    let gaussian: GaussianLds = serde_json::from_value(fixture["gaussian"].clone()).unwrap();
    let sequence: Vec<Vec<lds::Observation>> =
        serde_json::from_value(fixture["sequence"].clone()).unwrap();
    let posterior = gaussian.smooth(&sequence).unwrap();
    let means: Vec<Vec<f64>> = serde_json::from_value(fixture["means"].clone()).unwrap();
    let cov: Vec<Vec<f64>> = serde_json::from_value(fixture["covariances"].clone()).unwrap();
    let lag: Vec<f64> = serde_json::from_value(fixture["lag"].clone()).unwrap();
    for (a, b) in posterior.means.iter().flatten().zip(means.iter().flatten()) {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in posterior
        .covariances
        .iter()
        .flatten()
        .zip(cov.iter().flatten())
    {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in posterior.lag_covariances[0].iter().zip(lag) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(
        (posterior.negative_log_likelihood - fixture["negative_log_likelihood"].as_f64().unwrap())
            .abs()
            < 1e-12
    );
}

#[test]
fn weighted_observation_noise_update_uses_observation_count() {
    let mut g = GaussianLds {
        dim: 1,
        outputs: 1,
        transition: vec![0.8],
        observation: vec![1.0],
        input_dim: 0,
        input_weights: vec![],
        process_cov: vec![0.36],
        noise: vec![0.25],
        initial_cov: vec![1.0],
    };
    let means = [152.0 / 247.0, 19.0 / 247.0];
    let variances = [43.0 / 247.0, 61.0 / 247.0];
    let y = [1.0, -0.5];
    let weight = [1.0, 0.5];
    let moment: f64 = (0..2)
        .map(|i| weight[i] * (variances[i] + means[i] * means[i]))
        .sum();
    let cross: f64 = (0..2).map(|i| weight[i] * y[i] * means[i]).sum();
    let c = cross / moment;
    let r = (0..2)
        .map(|i| {
            weight[i]
                * (y[i] * y[i] - 2.0 * y[i] * c * means[i]
                    + c * c * (variances[i] + means[i] * means[i]))
        })
        .sum::<f64>()
        / 2.0;
    g.em_step(
        &[vec![vec![(0, 1.0, 1.0)], vec![(0, -0.5, 0.5)]]],
        0.995,
        0.0,
    )
    .unwrap();
    assert!((g.observation[0] - c).abs() < 1e-12);
    assert!((g.noise[0] - r).abs() < 1e-12);
}

#[test]
fn controlled_smoother_and_projected_em_match_joint_gaussian_oracle() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lds_controlled.json")).unwrap();
    let mut g: GaussianLds = serde_json::from_value(f["gaussian"].clone()).unwrap();
    let seq: Vec<Vec<lds::Observation>> = serde_json::from_value(f["sequence"].clone()).unwrap();
    let inputs: Vec<Vec<f64>> = serde_json::from_value(f["inputs"].clone()).unwrap();
    let p = g.smooth_with_inputs(&seq, &inputs).unwrap();
    for (actual, name) in [
        (&p.means, "means"),
        (&p.covariances, "covariances"),
        (&p.lag_covariances, "lag_covariances"),
    ] {
        let expected: Vec<Vec<f64>> = serde_json::from_value(f[name].clone()).unwrap();
        for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert!((a - b).abs() < 1e-11, "{name}: {a} vs {b}");
        }
    }
    let nll = f["negative_log_likelihood"].as_f64().unwrap();
    assert!((p.negative_log_likelihood - nll).abs() < 1e-11);
    assert!(g.smooth(&seq).is_err());
    let mut changed = inputs.clone();
    changed.last_mut().unwrap().fill(1234.0);
    assert_eq!(g.smooth_with_inputs(&seq, &changed).unwrap().means, p.means);
    let cap = f["cap"].as_f64().unwrap();
    assert!(f["unprojected_transition_norm"].as_f64().unwrap() > cap);
    let value = g
        .em_step_with_inputs(&[seq], &[inputs], cap, f["ridge"].as_f64().unwrap())
        .unwrap();
    assert!((value - nll / 6.0).abs() < 1e-11);
    let expected: GaussianLds = serde_json::from_value(f["em"].clone()).unwrap();
    for (actual, expected) in [
        (&g.transition, &expected.transition),
        (&g.input_weights, &expected.input_weights),
        (&g.observation, &expected.observation),
        (&g.process_cov, &expected.process_cov),
        (&g.noise, &expected.noise),
        (&g.initial_cov, &expected.initial_cov),
    ] {
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10, "EM: {a} vs {b}");
        }
    }
    assert!(g.transition_norm_bound().unwrap() <= cap + 1e-12);
}
#[test]
fn controlled_fit_and_prediction_exclude_test_neural_and_behavior_futures() {
    let (graph, mut data) = data();
    for (i, t) in data.trials.iter_mut().enumerate() {
        t.recording.behavior.insert(
            "velocity".into(),
            (0..81)
                .map(|j| {
                    if j % 17 == 2 {
                        None
                    } else {
                        Some((j as f64 * 0.04 + i as f64 * 0.8).sin())
                    }
                })
                .collect(),
        );
    }
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let cfg = FitConfig {
        ranks: vec![1, 2],
        iterations: 2,
        behavior_channels: vec!["velocity".into()],
        ..Default::default()
    };
    let (model, report) =
        lds::fit_select(&data, &graph, &split, cfg.clone(), |_, _| Ok(())).unwrap();
    assert_eq!(report.candidates.len(), 6);
    assert_eq!(model.gaussian.input_dim, 2);
    let before = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    for t in &mut data.trials {
        if split.test.contains(&t.id) {
            for trace in &mut t.recording.traces {
                trace.values[21..].fill(Some(1234.0));
            }
            t.recording.behavior.get_mut("velocity").unwrap()[21..].fill(Some(-1234.0));
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let (other, other_report) =
        lds::fit_select(&data, &graph, &updated, cfg, |_, _| Ok(())).unwrap();
    assert_eq!(model.gaussian.transition, other.gaussian.transition);
    assert_eq!(model.gaussian.input_weights, other.gaussian.input_weights);
    assert_eq!(model.gaussian.observation, other.gaussian.observation);
    assert_eq!(
        model.behavior.as_ref().unwrap().channels,
        other.behavior.as_ref().unwrap().channels
    );
    assert_eq!(report.selected_rank, other_report.selected_rank);
    assert_eq!(report.selected_iteration, other_report.selected_iteration);
    let after = other
        .predict(&data, &graph, &updated, Partition::Test)
        .unwrap();
    for (a, b) in before.trials.iter().zip(after.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
    let mut invalid = other;
    invalid.gaussian.input_weights.pop();
    assert!(
        invalid
            .predict(&data, &graph, &updated, Partition::Test)
            .is_err()
    );
}
