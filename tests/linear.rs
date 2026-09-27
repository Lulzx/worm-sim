use std::collections::BTreeMap;
use wormsim::{
    bench::{self, Axis, Dataset, Partition, Split, Trial, linear::training_statistics},
    data::{Provenance, Recording, Trace},
    fixtures,
};
fn data() -> (wormsim::data::IndexedGraph, Dataset) {
    let graph = fixtures::synthetic(2, 1, 0).compile().unwrap();
    let trials = (0..8)
        .map(|animal| {
            let mut x = animal as f64 * 0.2 + 0.5;
            let mut y = animal as f64 * 0.13 - 0.8;
            let mut a = vec![Some(x)];
            let mut b = vec![Some(y)];
            for _ in 0..80 {
                let u = 0.7 * x + 0.2 * y + 0.1;
                let v = -0.1 * x + 0.8 * y - 0.05;
                x = u;
                y = v;
                a.push(Some(x));
                b.push(Some(y));
            }
            Trial {
                id: format!("trial-{animal}"),
                stimulated_neuron: None,
                forecast_origin: Some(10.0),
                response_labels: BTreeMap::new(),
                recording: Recording {
                    dataset: "known-linear-system".into(),
                    animal_id: format!("animal-{animal}"),
                    condition: "synthetic".into(),
                    times: (0..81).map(|t| t as f64 * 0.5).collect(),
                    traces: vec![
                        Trace {
                            neuron: "N000".into(),
                            values: a,
                            provenance: Provenance {
                                dataset: "synthetic".into(),
                                version: "1".into(),
                                id_confidence: 1.0,
                            },
                        },
                        Trace {
                            neuron: "N001".into(),
                            values: b,
                            provenance: Provenance {
                                dataset: "synthetic".into(),
                                version: "1".into(),
                                id_confidence: 0.7,
                            },
                        },
                    ],
                    behavior: BTreeMap::new(),
                },
            }
        })
        .collect();
    let data = Dataset {
        schema_version: 1,
        name: "known-linear-system".into(),
        graph_hash: graph.hash.clone(),
        source: "synthetic numerical test".into(),
        trials,
    };
    (graph, data)
}
#[test]
fn dense_linear_fit_recovers_known_dynamics_and_never_reads_future_targets() {
    let (graph, mut data) = data();
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 1, 2).unwrap();
    let model = training_statistics(&data, &graph, &split)
        .unwrap()
        .fit(1e-10)
        .unwrap();
    let prediction = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    let scores = bench::evaluate(&data, &graph, &split, &prediction, Partition::Test).unwrap();
    assert!(scores.pooled_trace_scores.mse.unwrap() < 1e-12);
    assert_eq!(model.free_parameters(), 10);
    for trial in &mut data.trials {
        if !split.train.contains(&trial.id) {
            for trace in &mut trial.recording.traces {
                for (t, v) in trace.values.iter_mut().enumerate() {
                    if t > 20 {
                        *v = Some(1e4 + t as f64);
                    }
                }
            }
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 1, 2).unwrap();
    let other = training_statistics(&data, &graph, &updated)
        .unwrap()
        .fit(1e-10)
        .unwrap();
    assert_eq!(model.mean, other.mean);
    assert_eq!(model.scale, other.scale);
    assert_eq!(model.coefficients, other.coefficients);
    let other_prediction = other
        .predict(&data, &graph, &updated, Partition::Test)
        .unwrap();
    for (a, b) in prediction.trials.iter().zip(other_prediction.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
}
#[test]
fn missing_targets_match_independent_two_by_two_normal_equations() {
    let (graph, mut data) = data();
    for trial in &mut data.trials {
        trial.recording.traces.truncate(1);
        trial.recording.traces[0].values[2] = None;
        trial.recording.traces[0].values[5] = None;
    }
    let split = Split::generate(&data, &graph, Axis::Animal, 23, 1, 2).unwrap();
    let ridge = 0.1;
    let model = training_statistics(&data, &graph, &split)
        .unwrap()
        .fit(ridge)
        .unwrap();
    let mut a = 0.0;
    let mut b = 0.0;
    let mut c = 0.0;
    let mut d = 0.0;
    let mut count = 0.0;
    for trial in &data.trials {
        if split.train.contains(&trial.id) {
            let values = &trial.recording.traces[0].values;
            for pair in values.windows(2) {
                if let Some(target) = pair[1] {
                    let x = pair[0].map_or(0.0, |x| (x - model.mean[0]) / model.scale[0]);
                    let y = (target - model.mean[0]) / model.scale[0];
                    a += x * x;
                    b += x;
                    c += x * y;
                    d += y;
                    count += 1.0;
                }
            }
        }
    }
    a = a / count + ridge;
    b /= count;
    c /= count;
    d /= count;
    let intercept = 1.0 + ridge;
    let determinant = a * intercept - b * b;
    assert!((model.coefficients[0] - (intercept * c - b * d) / determinant).abs() < 1e-12);
    assert!((model.coefficients[1] - (a * d - b * c) / determinant).abs() < 1e-12);
    assert!(
        training_statistics(&data, &graph, &split)
            .unwrap()
            .fit(0.0)
            .is_err()
    );
}

#[test]
fn controls_are_causal_and_bootstrap_is_by_animal() {
    use bench::controls::{Control, predict};
    let (graph, mut data) = data();
    // Keep cross-animal variance well conditioned at every forecast horizon.
    for (animal, trial) in data.trials.iter_mut().enumerate() {
        for trace in &mut trial.recording.traces {
            for (i, value) in trace.values.iter_mut().enumerate() {
                *value = Some(animal as f64 + (i as f64 * 0.2).sin());
            }
        }
    }

    let split = Split::generate(&data, &graph, Axis::Animal, 42, 1, 2).unwrap();
    let controls = [
        Control::HistoryMean,
        Control::HalfBlend,
        Control::TrainingMean,
        Control::Autoregressive,
    ];
    let original: Vec<_> = controls
        .iter()
        .map(|&c| predict(&data, &graph, &split, Partition::Test, c).unwrap())
        .collect();
    let report = bench::evaluate(&data, &graph, &split, &original[0], Partition::Test).unwrap();
    let bootstrap = report.animal_bootstrap.unwrap();
    assert_eq!(bootstrap.animals, 2);
    assert_eq!(bootstrap.replicates, 2000);
    for (h, uncertainty) in report.forecast_horizons.iter().zip(&bootstrap.horizons) {
        assert!(
            (h.macro_neuron_r2.unwrap() - uncertainty.confidence_weighted.point.unwrap()).abs()
                < 1e-9
        );
        assert_eq!(uncertainty.per_animal.len(), 2);
        assert!(uncertainty.confidence_weighted.defined_replicates > 0);
        assert!(uncertainty.confidence_weighted.defined_replicates <= 2000);
        // For two animals, draws are AA, AB/BA, BB: all interval endpoints
        // must equal an extreme of the full score and the two animal scores.
        let mut possible: Vec<_> = uncertainty
            .per_animal
            .iter()
            .filter_map(|a| a.macro_neuron_r2)
            .collect();
        possible.push(h.macro_neuron_r2.unwrap());
        possible.sort_by(f64::total_cmp);
        assert!((uncertainty.confidence_weighted.lower_95.unwrap() - possible[0]).abs() < 1e-8);
        assert!(
            (uncertainty.confidence_weighted.upper_95.unwrap() - *possible.last().unwrap()).abs()
                < 1e-8
        );
    }
    for trial in &mut data.trials {
        if !split.train.contains(&trial.id) {
            for trace in &mut trial.recording.traces {
                for (i, value) in trace.values.iter_mut().enumerate() {
                    if i > 20 {
                        *value = Some(9999.0);
                    }
                }
            }
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 1, 2).unwrap();
    for (control, before) in controls.into_iter().zip(original) {
        let after = predict(&data, &graph, &updated, Partition::Test, control).unwrap();
        for (a, b) in before.trials.iter().zip(after.trials) {
            assert_eq!(a.fluorescence, b.fluorescence);
        }
    }
}

#[test]
fn merged_moments_equal_explicit_cluster_duplication() {
    use bench::metrics::Moments;
    let mut a = Moments::default();
    let mut b = Moments::default();
    let mut expected = Moments::default();
    let cluster_a = [(1e8 + 1.0, 1e8 + 0.8, 0.5), (1e8 + 3.0, 1e8 + 2.5, 1.0)];
    let cluster_b = [(1e8 + 4.0, 1e8 + 4.2, 0.7)];
    for (y, p, w) in cluster_a {
        a.push(y, p, w).unwrap();
    }
    for (y, p, w) in cluster_b {
        b.push(y, p, w).unwrap();
    }
    let mut merged = a.clone();
    merged.merge(&b);
    merged.merge(&a);
    for (y, p, w) in cluster_a.into_iter().chain(cluster_b).chain(cluster_a) {
        expected.push(y, p, w).unwrap();
    }
    assert_eq!(merged.scores().samples, expected.scores().samples);
    assert!((merged.scores().r2.unwrap() - expected.scores().r2.unwrap()).abs() < 1e-8);
    assert!(
        (merged.scores().correlation.unwrap() - expected.scores().correlation.unwrap()).abs()
            < 1e-8
    );
}
