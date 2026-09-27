use std::collections::BTreeMap;
use wormsim::{
    bench::{self, Axis, Dataset, Partition, Split, Trial, atlas},
    data::{Provenance, Recording, Trace},
    fixtures,
};
fn fixture() -> (wormsim::data::IndexedGraph, Dataset, Split, atlas::Evidence) {
    let graph = fixtures::synthetic(3, 1, 0).compile().unwrap();
    let data = Dataset {
        schema_version: 1,
        name: "synthetic pair classification".into(),
        graph_hash: graph.hash.clone(),
        source: "synthetic".into(),
        trials: (0..6)
            .map(|k| Trial {
                id: format!("trial{k}"),
                stimulated_neuron: Some(graph.names[k % 3].clone()),
                forecast_origin: None,
                response_labels: BTreeMap::new(),
                recording: Recording {
                    dataset: "synthetic".into(),
                    animal_id: format!("animal{k}"),
                    condition: "stim".into(),
                    times: vec![0., 1.],
                    behavior: BTreeMap::new(),
                    traces: graph
                        .names
                        .iter()
                        .map(|n| Trace {
                            neuron: n.clone(),
                            values: vec![Some(0.), Some(1.)],
                            provenance: Provenance {
                                dataset: "synthetic".into(),
                                version: "1".into(),
                                id_confidence: 1.,
                            },
                        })
                        .collect(),
                },
            })
            .collect(),
    };
    let split = Split::generate(&data, &graph, Axis::StimulatedNeuron, 42, 1, 1).unwrap();
    let mut pairs = vec![];
    for (i, s) in graph.names.iter().enumerate() {
        for (j, r) in graph.names.iter().enumerate() {
            if i != j {
                pairs.push(atlas::Pair {
                    stimulated: s.clone(),
                    responding: r.clone(),
                    q: if j == (i + 1) % 3 { 0.01 } else { 0.2 },
                    equivalence_q: Some(0.01),
                    observations: 2,
                });
            }
        }
    }
    let evidence = atlas::Evidence {
        schema_version: 1,
        dataset_hash: data.content_hash().unwrap(),
        graph_hash: graph.hash.clone(),
        source_sha256: "0".repeat(64),
        source_version: "synthetic".into(),
        equivalence_threshold: 1.2,
        detection_q_threshold: 0.05,
        pairs,
    };
    (graph, data, split, evidence)
}
#[test]
fn pair_scoring_is_once_per_pair_and_checks_coverage_and_training_lineage() {
    let (graph, data, split, evidence) = fixture();
    let pairs = evidence
        .partition_pairs(&data, &split, Partition::Test)
        .unwrap();
    let mut pred = atlas::Predictions {
        evidence_hash: evidence.content_hash().unwrap(),
        split_hash: split.content_hash().unwrap(),
        model: "synthetic oracle".into(),
        free_parameters: 0,
        source_commit: "test".into(),
        training_trials: split.train.clone(),
        selection_trials: split.validation.clone(),
        pairs: pairs
            .iter()
            .map(|p| atlas::Prediction {
                stimulated: p.stimulated.clone(),
                responding: p.responding.clone(),
                score: if p.q < 0.05 { 2. } else { 0. },
            })
            .collect(),
    };
    let r = atlas::evaluate(&evidence, &data, &graph, &split, &pred, Partition::Test).unwrap();
    assert_eq!(r.pairs, 2);
    assert_eq!(r.detected, 1);
    assert_eq!(r.not_detected, 1);
    assert_eq!(r.detected_and_equivalent, 1);
    assert_eq!(
        serde_json::to_value(r.auroc).unwrap(),
        serde_json::to_value(bench::metrics::auroc(&[(true, 2., 1.), (false, 0., 1.)]).unwrap())
            .unwrap()
    );
    pred.training_trials.push(split.test[0].clone());
    assert!(atlas::evaluate(&evidence, &data, &graph, &split, &pred, Partition::Test).is_err());
    pred.training_trials = split.train.clone();
    pred.pairs.push(pred.pairs[0].clone());
    assert!(atlas::evaluate(&evidence, &data, &graph, &split, &pred, Partition::Test).is_err());
    pred.pairs.pop();
    pred.pairs.pop();
    assert!(atlas::evaluate(&evidence, &data, &graph, &split, &pred, Partition::Test).is_err());
    let mut bad = evidence.clone();
    bad.pairs[0].q = f64::NAN;
    assert!(bad.validate(&data, &graph).is_err());
    let mut bad = evidence.clone();
    bad.pairs[0].responding = bad.pairs[0].stimulated.clone();
    assert!(bad.validate(&data, &graph).is_err());
}
#[cfg(feature = "hdf5")]
#[test]
fn native_hdf5_pair_import_preserves_direction_missing_q_and_source_hash() {
    use hdf5::types::FixedAscii;
    use sha2::{Digest, Sha256};
    let (graph, data, _, _) = fixture();
    let path = std::env::temp_dir().join(format!("wormsim-atlas-{}.h5", std::process::id()));
    {
        let f = hdf5::File::create(&path).unwrap();
        let ids: Vec<_> = graph
            .names
            .iter()
            .map(|n| FixedAscii::<5>::from_ascii(n).unwrap())
            .collect();
        f.new_dataset_builder()
            .with_data(&ids)
            .create("neuron_ids")
            .unwrap();
        f.new_attr::<FixedAscii<19>>()
            .create("time_compiled")
            .unwrap()
            .write_scalar(&FixedAscii::<19>::from_ascii("2023-06-28_19-52-14").unwrap())
            .unwrap();
        f.create_group("wt").unwrap();
        for (name, values) in [
            (
                "wt/q",
                vec![0.1, 0.01, f64::NAN, 0.2, 0.1, 0.3, 0.4, 0.5, 0.1],
            ),
            ("wt/q_eq", vec![f64::NAN; 9]),
            ("wt/occ1", vec![3.; 9]),
        ] {
            f.new_dataset::<f64>()
                .shape([3, 3])
                .create(name)
                .unwrap()
                .write_raw(&values)
                .unwrap();
        }
        f.new_dataset::<f64>()
            .create("wt/q_eq_th")
            .unwrap()
            .write_scalar(&1.2)
            .unwrap();
    }
    let digest = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
    let evidence = atlas::import_hdf5(&path, &data, &graph, &digest).unwrap();
    assert_eq!(evidence.pairs.len(), 5);
    let p = evidence
        .pairs
        .iter()
        .find(|p| p.stimulated == "N001" && p.responding == "N000")
        .unwrap();
    assert_eq!(p.q, 0.01);
    assert_eq!(p.equivalence_q, None);
    assert_eq!(p.observations, 3);
    assert!(atlas::import_hdf5(&path, &data, &graph, &"0".repeat(64)).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn atlas_bootstrap_retains_target_clusters_and_reports_undefined_draws() {
    use bench::atlas_uncertainty;
    let (_, _, _, mut evidence) = fixture();
    evidence.pairs = vec![
        atlas::Pair {
            stimulated: "A".into(),
            responding: "X".into(),
            q: 0.01,
            equivalence_q: None,
            observations: 1,
        },
        atlas::Pair {
            stimulated: "B".into(),
            responding: "X".into(),
            q: 0.5,
            equivalence_q: None,
            observations: 1,
        },
    ];
    let mut predictions = atlas::Predictions {
        evidence_hash: evidence.content_hash().unwrap(),
        split_hash: "test".into(),
        model: "synthetic".into(),
        free_parameters: 0,
        source_commit: "test".into(),
        training_trials: vec![],
        selection_trials: vec![],
        pairs: vec![
            atlas::Prediction {
                stimulated: "A".into(),
                responding: "X".into(),
                score: 1.,
            },
            atlas::Prediction {
                stimulated: "B".into(),
                responding: "X".into(),
                score: 0.,
            },
        ],
    };
    let report = atlas_uncertainty::pairs(&evidence, &predictions, 42, 2000).unwrap();
    assert_eq!(report.clusters, 2);
    assert_eq!(report.auroc.point, Some(1.));
    assert_eq!(report.auroc.lower_95, Some(1.));
    assert_eq!(report.auroc.upper_95, Some(1.));
    assert!(report.auroc.defined_replicates > 800 && report.auroc.defined_replicates < 1200);
    let mut reverse = predictions.clone();
    reverse.pairs[0].score = 0.0;
    reverse.pairs[1].score = 1.0;
    let difference =
        atlas_uncertainty::pair_difference(&evidence, &predictions, &reverse, 42, 2000).unwrap();
    assert_eq!(difference.auroc.point, Some(1.0));
    assert_eq!(difference.auroc.lower_95, Some(1.0));
    assert_eq!(
        difference.auroc.defined_replicates,
        report.auroc.defined_replicates
    );
    let swapped =
        atlas_uncertainty::pair_difference(&evidence, &reverse, &predictions, 42, 2000).unwrap();
    assert_eq!(swapped.auroc.upper_95, Some(-1.0));
    let same = atlas_uncertainty::pair_difference(&evidence, &predictions, &predictions, 42, 2000)
        .unwrap();
    assert_eq!(same.auroc.lower_95, Some(0.0));
    predictions.pairs[0].score = 0.;
    let tied = atlas_uncertainty::pairs(&evidence, &predictions, 42, 2000).unwrap();
    assert_eq!(tied.auroc.point, Some(0.5));
    assert_eq!(tied.auroc.lower_95, Some(0.5));
    assert_eq!(
        tied.auroc.defined_replicates,
        report.auroc.defined_replicates
    );
    predictions.pairs.push(predictions.pairs[0].clone());
    assert!(atlas_uncertainty::pairs(&evidence, &predictions, 42, 2000).is_err());
}

#[test]
fn atlas_trace_intervals_keep_recordings_and_targets_distinct() {
    let (graph, data, split, _) = fixture();
    let prediction = bench::Predictions {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash().unwrap(),
        model: "synthetic oracle".into(),
        free_parameters: 0,
        training_trials: vec![],
        selection_trials: vec![],
        source_commit: "test".into(),
        seed: 42,
        trials: data
            .trials
            .iter()
            .filter(|t| split.test.contains(&t.id))
            .map(|t| bench::PredictedTrial {
                id: t.id.clone(),
                times: t.recording.times.clone(),
                fluorescence: t
                    .recording
                    .traces
                    .iter()
                    .map(|r| (r.neuron.clone(), vec![0., 1.]))
                    .collect(),
                response_scores: BTreeMap::new(),
            })
            .collect(),
    };
    let report = bench::evaluate(&data, &graph, &split, &prediction, Partition::Test).unwrap();
    let mut shifted = prediction.clone();
    for trial in &mut shifted.trials {
        for values in trial.fluorescence.values_mut() {
            for v in values {
                *v += 2.0;
            }
        }
    }
    let shifted_report = bench::evaluate(&data, &graph, &split, &shifted, Partition::Test).unwrap();
    let difference =
        bench::atlas_uncertainty::trace_difference(&data, &report, &shifted_report, 42, 100)
            .unwrap();
    for group in difference {
        assert_eq!(group.pooled_mse.point, Some(-4.0));
        assert_eq!(group.pooled_mse.upper_95, Some(-4.0));
        assert!(group.macro_trace_correlation.point.unwrap().abs() < 1e-12);
    }
    let swapped =
        bench::atlas_uncertainty::trace_difference(&data, &shifted_report, &report, 42, 100)
            .unwrap();
    assert_eq!(swapped[0].pooled_mse.lower_95, Some(4.0));
    let intervals = bench::atlas_uncertainty::traces(&data, &report, 42, 100).unwrap();
    assert_eq!(intervals[0].clusters, 1);
    assert_eq!(intervals[1].clusters, 2);
    for group in intervals {
        assert_eq!(group.pooled_mse.point, Some(0.));
        assert_eq!(group.pooled_mse.upper_95, Some(0.));
        assert!((group.macro_trace_correlation.lower_95.unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(group.macro_trace_correlation.defined_replicates, 100);
    }
}

#[test]
fn response_aggregation_preserves_original_weighted_loss_and_gradient() {
    let (graph, mut data, mut split, _) = fixture();
    for (k, trial) in data.trials.iter_mut().enumerate() {
        for (i, trace) in trial.recording.traces.iter_mut().enumerate() {
            trace.provenance.id_confidence = 0.2 + 0.1 * i as f64;
            trace.values = vec![Some(k as f64 * 0.3 + i as f64), Some(0.2 - k as f64 * 0.1)];
        }
    }
    split.dataset_hash = data.content_hash().unwrap();
    let groups = bench::atlas_training::aggregate(&data, &graph, &split).unwrap();
    assert_eq!(groups.len(), 1);
    let group = &groups[0];
    let mut raw_loss = 0.;
    let mut raw_gradient = 0.;
    let mut raw_weight = 0.;
    for t in data.trials.iter().filter(|t| split.train.contains(&t.id)) {
        for trace in &t.recording.traces {
            for y in &trace.values {
                let w = trace.provenance.id_confidence;
                let error = 0.7 - y.unwrap();
                raw_loss += w * error * error;
                raw_gradient += 2. * w * error;
                raw_weight += w;
            }
        }
    }
    let mut loss = 0.;
    let mut gradient = 0.;
    let mut weight = 0.;
    for trace in &group.recording.traces {
        for y in &trace.values {
            let w = trace.provenance.id_confidence;
            let error = 0.7 - y.unwrap();
            loss += w * error * error;
            gradient += 2. * w * error;
            weight += w;
        }
    }
    assert!((raw_weight - group.sample_weight).abs() < 1e-12);
    assert!((raw_loss / raw_weight - loss / weight - group.irreducible_mse).abs() < 1e-12);
    assert!((raw_gradient / raw_weight - gradient / weight).abs() < 1e-12);
    assert_eq!(
        group
            .training_trials
            .iter()
            .collect::<std::collections::BTreeSet<_>>(),
        split.train.iter().collect()
    );
    let training = data
        .trials
        .iter_mut()
        .find(|t| split.train.contains(&t.id))
        .unwrap();
    training.recording.traces[0].values[0] = None;
    split.dataset_hash = data.content_hash().unwrap();
    assert!(
        bench::atlas_training::aggregate(&data, &graph, &split)
            .unwrap_err()
            .contains("complete")
    );
}

#[test]
fn generic_response_ranking_is_label_independent_and_rejects_inconsistent_trials() {
    let (graph, data, split, mut evidence) = fixture();
    let mut predictions = bench::Predictions {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash().unwrap(),
        model: "synthetic fixed response".into(),
        free_parameters: 0,
        training_trials: vec![],
        selection_trials: vec![],
        source_commit: "test".into(),
        seed: 42,
        trials: data
            .trials
            .iter()
            .filter(|t| split.test.contains(&t.id))
            .map(|t| bench::PredictedTrial {
                id: t.id.clone(),
                times: t.recording.times.clone(),
                fluorescence: t
                    .recording
                    .traces
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (r.neuron.clone(), vec![-0.2, 0.5 + i as f64]))
                    .collect(),
                response_scores: BTreeMap::new(),
            })
            .collect(),
    };
    let first = atlas::rank_responses(
        &evidence,
        &data,
        &graph,
        &split,
        &predictions,
        Partition::Test,
    )
    .unwrap();
    for pair in &first.pairs {
        let i = graph.neuron(&pair.responding).unwrap();
        assert!((pair.score - (0.7 + i as f64)).abs() < 1e-12);
    }
    for pair in &mut evidence.pairs {
        pair.q = 1. - pair.q;
    }
    let changed = atlas::rank_responses(
        &evidence,
        &data,
        &graph,
        &split,
        &predictions,
        Partition::Test,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(first.pairs).unwrap(),
        serde_json::to_value(changed.pairs).unwrap()
    );
    predictions.trials[0]
        .fluorescence
        .values_mut()
        .next()
        .unwrap()[1] += 1.;
    assert!(
        atlas::rank_responses(
            &evidence,
            &data,
            &graph,
            &split,
            &predictions,
            Partition::Test
        )
        .unwrap_err()
        .contains("inconsistent")
    );
    predictions.trials.pop();
    assert!(
        atlas::rank_responses(
            &evidence,
            &data,
            &graph,
            &split,
            &predictions,
            Partition::Test
        )
        .is_err()
    );
}

#[test]
fn classification_training_labels_exclude_held_out_targets_and_trial_multiplicity() {
    use wormsim::bench::atlas_classification::training_labels;
    let (graph, data, split, mut evidence) = fixture();
    let before = training_labels(&evidence, &data, &graph, &split).unwrap();
    assert_eq!(before.pairs, 2);
    assert_eq!(before.detected, 1);
    assert_eq!(before.by_target.len(), 1);
    for pair in &mut evidence.pairs {
        if !before
            .by_target
            .contains_key(&graph.neuron(&pair.stimulated).unwrap())
        {
            pair.q = if pair.q < 0.05 { 1. } else { 0. };
        }
    }
    let after = training_labels(&evidence, &data, &graph, &split).unwrap();
    assert_ne!(before.evidence_hash, after.evidence_hash);
    assert_eq!(before.by_target, after.by_target);
    assert_eq!(before.pairs, after.pairs);
    assert_eq!(before.detected, after.detected);
}

#[test]
fn classification_loss_gradients_match_finite_differences_and_reject_invalid_inputs() {
    use wormsim::bench::atlas_classification::Classifier;
    let classifier = Classifier {
        bias: -2.,
        raw_slope: 0.3,
        area_scale: 0.01,
        epsilon: 0.001,
    };
    let response = vec![vec![0., 0., 0.], vec![0.2, -0.3, 0.], vec![-0.1, 0.05, 0.]];
    let labels = [(0, true), (1, false), (2, true)];
    let gradient = classifier.loss(&response, &labels, 0.5).unwrap();
    let eps = 1e-6;
    let compare = |a: f64, b: f64| assert!((a - b).abs() < 1e-7, "{a} != {b}");
    for t in 0..response.len() {
        for i in 0..3 {
            let mut plus = response.clone();
            let mut minus = response.clone();
            plus[t][i] += eps;
            minus[t][i] -= eps;
            compare(
                (classifier.loss(&plus, &labels, 0.5).unwrap().value
                    - classifier.loss(&minus, &labels, 0.5).unwrap().value)
                    / (2. * eps),
                gradient.fluorescence[t][i],
            );
        }
    }
    for slope in [false, true] {
        let mut plus = classifier.clone();
        let mut minus = classifier.clone();
        if slope {
            plus.raw_slope += eps;
            minus.raw_slope -= eps;
        } else {
            plus.bias += eps;
            minus.bias -= eps;
        }
        compare(
            (plus.loss(&response, &labels, 0.5).unwrap().value
                - minus.loss(&response, &labels, 0.5).unwrap().value)
                / (2. * eps),
            if slope {
                gradient.raw_slope_gradient
            } else {
                gradient.bias_gradient
            },
        );
    }
    let empty = classifier.loss(&response, &[], 0.5).unwrap();
    assert_eq!(empty.value, 0.);
    assert!(empty.fluorescence.iter().flatten().all(|v| *v == 0.));
    assert!(
        classifier
            .loss(&response, &[(0, true), (0, false)], 0.5)
            .is_err()
    );
    assert!(classifier.loss(&response, &[(3, true)], 0.5).is_err());
    assert!(classifier.loss(&response, &labels, 0.).is_err());
    assert!(
        classifier
            .loss(&vec![vec![f64::NAN; 3]; 2], &labels, 0.5)
            .is_err()
    );
    let mut extreme = classifier.clone();
    extreme.bias = 1000.;
    assert!(
        extreme
            .loss(&response, &labels, 0.5)
            .unwrap()
            .value
            .is_finite()
    );
    extreme.bias = -1000.;
    assert!(
        extreme
            .loss(&response, &labels, 0.5)
            .unwrap()
            .value
            .is_finite()
    );
}
