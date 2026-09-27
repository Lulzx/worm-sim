use std::collections::BTreeMap;
use wormsim::{
    bench::{
        self, Axis, Dataset, Partition, Split, Trial,
        gru::{self, FitConfig, Network},
    },
    data::{Provenance, Recording, Trace},
    fixtures,
};
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
fn every_weight_gradient_includes_missing_history_and_free_feedback() {
    let mut net = Network::initialize(2, 3, 17).unwrap();
    let seq = vec![
        vec![(0, 0.7, 0.6)],
        vec![(1, -0.4, 1.0)],
        vec![],
        vec![(0, 0.8, 0.8), (1, -0.6, 1.0)],
        vec![(0, 0.1, 0.6)],
    ];
    let (loss, grad) = net.loss_gradient(&seq, 1).unwrap();
    let independent_loss = |net: &Network| {
        let out = net.predict(&seq, 1).unwrap();
        let mut sum = 0.0;
        let mut weight = 0.0;
        for t in 2..seq.len() {
            for &(i, y, w) in &seq[t] {
                sum += w * (out[t][i] - y).powi(2);
                weight += w;
            }
        }
        sum / weight
    };
    assert!((loss - independent_loss(&net)).abs() < 1e-14);
    for (i, &g) in grad.iter().enumerate() {
        let old = net.weights[i];
        let eps = 1e-6;
        net.weights[i] = old + eps;
        let plus = independent_loss(&net);
        net.weights[i] = old - eps;
        let minus = independent_loss(&net);
        net.weights[i] = old;
        let fd = (plus - minus) / (2.0 * eps);
        assert!((fd - g).abs() < 1e-7, "weight {i}: {fd} vs {g}");
    }
    let mut changed = seq.clone();
    changed[2] = vec![(1, 9999.0, 0.2)];
    changed[3] = vec![];
    assert_eq!(
        net.predict(&seq, 1).unwrap(),
        net.predict(&changed, 1).unwrap()
    );
    let initial = loss;
    for _ in 0..300 {
        let (_, g) = net.loss_gradient(&seq, 1).unwrap();
        for (v, d) in net.weights.iter_mut().zip(g) {
            *v -= 0.05 * d;
        }
    }
    assert!(independent_loss(&net) < 0.3 * initial);
}
#[test]
fn scalar_gru_matches_explicit_reset_before_recurrence() {
    let net = Network {
        outputs: 1,
        hidden: 1,
        weights: vec![
            0.2, -0.1, 0.3, 0.05, -0.4, 0.2, 0.1, -0.2, 0.5, 0.3, -0.6, 0.1, 0.7, -0.1,
        ],
    };
    let seq = vec![vec![(0, 0.8, 0.5)], vec![], vec![(0, 100.0, 1.0)], vec![]];
    let out = net.predict(&seq, 0).unwrap();
    let mut h = 0.0f64;
    let mut y = -0.1;
    assert_eq!(out[0][0], y);
    for t in 0..3 {
        let (x, m) = if t == 0 { (0.8, 0.5) } else { (y, 0.0) };
        let r = 1.0 / (1.0 + (-0.2 * x + 0.1 * m - 0.3 * h - 0.05).exp());
        let z = 1.0 / (1.0 + (0.4 * x - 0.2 * m - 0.1 * h + 0.2).exp());
        let c = (0.5 * x + 0.3 * m - 0.6 * r * h + 0.1).tanh();
        h = z * h + (1.0 - z) * c;
        y = 0.7 * h - 0.1;
        assert!((out[t + 1][0] - y).abs() < 1e-14);
    }
    let mut bad = net.clone();
    bad.weights.pop();
    assert!(bad.predict(&seq, 0).is_err());
    assert!(
        net.predict(&[vec![(0, 0.0, 1.0), (0, 0.1, 1.0)], vec![]], 0)
            .is_err()
    );
}
#[test]
fn fitting_uses_training_animals_and_forecasts_ignore_test_futures() {
    let (graph, mut data) = data();
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let cfg = FitConfig {
        hidden: 3,
        epochs: 3,
        ..Default::default()
    };
    let (model, report) =
        gru::fit_select(&data, &graph, &split, cfg.clone(), |_, _| Ok(())).unwrap();
    assert_eq!(report.candidates.len(), 4);
    assert_eq!(report.candidates[1].updates, split.train.len());
    assert_eq!(model.free_parameters(), model.network.weights.len() + 6);
    let before = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    bench::evaluate(&data, &graph, &split, &before, Partition::Test).unwrap();
    for t in &mut data.trials {
        if split.test.contains(&t.id) {
            for tr in &mut t.recording.traces {
                for y in &mut tr.values[21..] {
                    *y = Some(12345.0);
                }
            }
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let (other, other_report) =
        gru::fit_select(&data, &graph, &updated, cfg, |_, _| Ok(())).unwrap();
    assert_eq!(model.network.weights, other.network.weights);
    assert_eq!(model.mean, other.mean);
    assert_eq!(model.scale, other.scale);
    assert_eq!(report.selected_epoch, other_report.selected_epoch);
    for (a, b) in report.candidates.iter().zip(&other_report.candidates) {
        assert_eq!(a.validation_horizon_r2, b.validation_horizon_r2);
    }
    let after = other
        .predict(&data, &graph, &updated, Partition::Test)
        .unwrap();
    for (a, b) in before.trials.iter().zip(&after.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
    let mut bad = other;
    bad.sample_dt = 0.25;
    assert!(
        bad.predict(&data, &graph, &updated, Partition::Test)
            .is_err()
    );
}
