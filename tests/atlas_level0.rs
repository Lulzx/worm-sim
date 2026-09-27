use std::collections::BTreeMap;
use wormsim::{
    bench::{
        Axis, Dataset, Partition, Split, Trial,
        atlas_level0::{self, FitConfig},
    },
    data::{Provenance, Recording, Trace},
    fixtures,
};
fn fixture() -> (wormsim::data::IndexedGraph, Dataset, Split) {
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
    (graph, data, split)
}
#[test]
fn fit_and_impulses_exclude_held_out_fluorescence() {
    let (graph, mut data, split) = fixture();
    let config = FitConfig {
        classification: None,
        epochs: 2,
        dt: 0.02,
        preparation_seconds: 0.4,
        learning_rate: 0.01,
        prior_strength: 0.01,
        sign_prior_strength: 0.01,
        kernel_prior_strength: 0.01,
        sharing: Default::default(),
        kernel_lags: 1,
    };
    let (model, candidates) =
        atlas_level0::fit_select(&data, &graph, &split, config.clone(), |_, _| Ok(())).unwrap();
    assert_eq!(candidates.len(), 3);
    let mut legacy = serde_json::to_value(&model).unwrap();
    legacy["config"]
        .as_object_mut()
        .unwrap()
        .remove("preparation_seconds");
    let legacy: atlas_level0::AtlasModel = serde_json::from_value(legacy).unwrap();
    assert_eq!(legacy.config.preparation_seconds, 0.0);

    let best = candidates
        .iter()
        .min_by(|a, b| a.validation_mse.total_cmp(&b.validation_mse))
        .unwrap();
    assert_eq!(model.epoch, best.epoch);
    let before = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    for t in &mut data.trials {
        if split.test.contains(&t.id) {
            for trace in &mut t.recording.traces {
                trace.values.fill(Some(999.));
            }
        }
    }
    let mut changed_split = split.clone();
    changed_split.dataset_hash = data.content_hash().unwrap();
    let (changed, changed_candidates) =
        atlas_level0::fit_select(&data, &graph, &changed_split, config, |_, _| Ok(())).unwrap();
    assert_eq!(
        serde_json::to_value(&model.parameters).unwrap(),
        serde_json::to_value(&changed.parameters).unwrap()
    );
    assert_eq!(model.epoch, changed.epoch);
    assert_eq!(model.kernel_raw, changed.kernel_raw);
    assert_eq!(model.initial, changed.initial);
    for (a, b) in candidates.iter().zip(&changed_candidates) {
        assert_eq!(a.validation_mse, b.validation_mse);
    }
    let after = changed
        .predict(&data, &graph, &changed_split, Partition::Test)
        .unwrap();
    assert_eq!(
        serde_json::to_value(before.trials).unwrap(),
        serde_json::to_value(after.trials).unwrap()
    );
    let mut bad = changed;
    bad.training_trials.push(changed_split.test[0].clone());
    assert!(
        bad.predict(&data, &graph, &changed_split, Partition::Test)
            .is_err()
    );
}

#[test]
fn joint_fit_uses_training_labels_and_preserves_mse_selection() {
    use wormsim::bench::{
        atlas, atlas_classification::Classifier, atlas_level0::ClassificationConfig,
    };
    let (graph, data, split) = fixture();
    let mut evidence = atlas::Evidence {
        schema_version: 1,
        dataset_hash: data.content_hash().unwrap(),
        graph_hash: graph.hash.clone(),
        source_sha256: "0".repeat(64),
        source_version: "synthetic".into(),
        equivalence_threshold: 1.2,
        detection_q_threshold: 0.05,
        pairs: graph
            .names
            .iter()
            .enumerate()
            .flat_map(|(i, s)| {
                graph
                    .names
                    .iter()
                    .enumerate()
                    .filter(move |(j, _)| *j != i)
                    .map(move |(j, r)| atlas::Pair {
                        stimulated: s.clone(),
                        responding: r.clone(),
                        q: if j == (i + 1) % 3 { 0.01 } else { 0.2 },
                        equivalence_q: None,
                        observations: 2,
                    })
            })
            .collect(),
    };
    let config = FitConfig {
        epochs: 2,
        dt: 0.02,
        preparation_seconds: 0.4,
        learning_rate: 0.01,
        prior_strength: 0.01,
        sign_prior_strength: 0.01,
        kernel_prior_strength: 0.01,
        sharing: Default::default(),
        kernel_lags: 1,
        classification: Some(ClassificationConfig {
            weight: 0.1,
            area_scale: 0.01,
            epsilon: 1e-6,
        }),
    };
    let mut checkpoints = vec![];
    let (selected, reports) = atlas_level0::fit_select_with_evidence(
        &data,
        &graph,
        &split,
        config.clone(),
        Some(&evidence),
        |model, _| {
            checkpoints.push(model.clone());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        selected.epoch,
        reports
            .iter()
            .min_by(|a, b| a.validation_mse.total_cmp(&b.validation_mse))
            .unwrap()
            .epoch
    );
    let initial: &Classifier = checkpoints[0].classifier.as_ref().unwrap();
    let last = checkpoints[2].classifier.as_ref().unwrap();
    assert_ne!(initial.bias, last.bias);
    assert!(reports[0].preceding_training_classification_bce.is_none());
    // Independent forward evaluation confirms the reported MSE retains original
    // trial weights, and pair BCE counts repeated stimulation trials only once.
    let pred = checkpoints[0]
        .predict(&data, &graph, &split, Partition::Train)
        .unwrap();
    let score = wormsim::bench::evaluate(&data, &graph, &split, &pred, Partition::Train).unwrap();
    assert!(
        (score.pooled_trace_scores.mse.unwrap() - reports[1].preceding_training_mse.unwrap()).abs()
            < 1e-12
    );
    let train_labels =
        wormsim::bench::atlas_classification::training_labels(&evidence, &data, &graph, &split)
            .unwrap();
    let response: Vec<Vec<_>> = (0..2)
        .map(|t| {
            graph
                .names
                .iter()
                .map(|n| pred.trials[0].fluorescence[n][t])
                .collect()
        })
        .collect();
    let expected = initial
        .loss(
            &response,
            train_labels.by_target.values().next().unwrap(),
            1.,
        )
        .unwrap()
        .value
        / train_labels.pairs as f64;
    assert!((expected - reports[1].preceding_training_classification_bce.unwrap()).abs() < 1e-12);
    for pair in &mut evidence.pairs {
        if !train_labels
            .by_target
            .contains_key(&graph.neuron(&pair.stimulated).unwrap())
        {
            pair.q = 1. - pair.q;
        }
    }
    let mut changed = vec![];
    let (other, _) = atlas_level0::fit_select_with_evidence(
        &data,
        &graph,
        &split,
        config.clone(),
        Some(&evidence),
        |model, _| {
            changed.push(model.clone());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(selected.epoch, other.epoch);
    for (a, b) in checkpoints.iter().zip(&changed) {
        assert_eq!(
            serde_json::to_value(&a.parameters).unwrap(),
            serde_json::to_value(&b.parameters).unwrap()
        );
        assert_eq!(a.kernel_raw, b.kernel_raw);
        assert_eq!(
            serde_json::to_value(&a.classifier).unwrap(),
            serde_json::to_value(&b.classifier).unwrap()
        );
        assert_ne!(
            a.classification_evidence_hash,
            b.classification_evidence_hash
        );
    }
    assert!(
        atlas_level0::fit_select(&data, &graph, &split, config.clone(), |_, _| Ok(())).is_err()
    );
    let mut mse_only = config;
    mse_only.classification = None;
    let mut mse_checkpoints = vec![];
    atlas_level0::fit_select(&data, &graph, &split, mse_only, |m, _| {
        mse_checkpoints.push(m.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(
        checkpoints[0].free_parameters(),
        mse_checkpoints[0].free_parameters() + 2
    );
    assert_ne!(
        serde_json::to_value(&checkpoints[1].parameters).unwrap(),
        serde_json::to_value(&mse_checkpoints[1].parameters).unwrap()
    );
    let mut broken = selected;
    broken.classification_evidence_hash = None;
    assert!(
        broken
            .predict(&data, &graph, &split, Partition::Test)
            .is_err()
    );
}
