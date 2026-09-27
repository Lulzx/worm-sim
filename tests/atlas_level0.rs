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
        sign_initialization: None,
        optimizer: Default::default(),
        learning_rate_schedule: Default::default(),
        correlation: None,
        observation_gain: None,
        classification: None,
        molecular_sign_priors: None,
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
        sign_initialization: None,
        optimizer: Default::default(),
        learning_rate_schedule: Default::default(),
        correlation: Some(atlas_level0::CorrelationConfig {
            weight: 0.02,
            epsilon: 0.01,
        }),
        observation_gain: Some(atlas_level0::ObservationGainConfig {
            initial_gain: 2.,
            prior_strength: 0.01,
        }),
        molecular_sign_priors: None,
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
    assert_ne!(
        checkpoints[0].observation_log_gain,
        checkpoints[2].observation_log_gain
    );
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
    let group = wormsim::bench::atlas_training::aggregate(&data, &graph, &split).unwrap();
    let shape =
        wormsim::bench::atlas_correlation::loss(&group[0].recording, &graph, &response, 0.01)
            .unwrap();
    assert!(
        (shape.value / shape.pairs as f64
            - reports[1].preceding_training_pair_correlation_loss.unwrap())
        .abs()
            < 1e-12
    );
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
        assert_eq!(a.observation_log_gain, b.observation_log_gain);
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

#[test]
fn molecular_prior_fit_preserves_graph_and_excludes_test_fluorescence() {
    use wormsim::{math::Scalar, molecular::SignPriors};
    let (graph, mut data, split) = fixture();
    let graph_before = serde_json::to_value(&graph.graph).unwrap();
    let priors = SignPriors {
        graph_hash: graph.hash.clone(),
        evidence_hash: "1".repeat(64),
        catalog_hash: "2".repeat(64),
        expression_hash: "3".repeat(64),
        mapping_hash: "4".repeat(64),
        confidence: 0.75,
        excitatory_edges: vec![0],
        inhibitory_edges: vec![1],
    };
    let config = FitConfig {
        sign_initialization: Some(atlas_level0::SignInitialization {
            seed: 42,
            reversal_magnitude: 0.5,
        }),
        optimizer: Default::default(),
        learning_rate_schedule: Default::default(),
        correlation: Some(atlas_level0::CorrelationConfig {
            weight: 0.02,
            epsilon: 0.01,
        }),
        observation_gain: Some(atlas_level0::ObservationGainConfig {
            initial_gain: 3.,
            prior_strength: 0.01,
        }),
        molecular_sign_priors: Some(priors),
        classification: None,
        epochs: 2,
        dt: 0.02,
        preparation_seconds: 0.4,
        learning_rate: 0.01,
        kernel_lags: 1,
        prior_strength: 0.01,
        sign_prior_strength: 0.01,
        kernel_prior_strength: 0.01,
        sharing: Default::default(),
    };
    let mut checkpoints = vec![];
    let (selected, _) = atlas_level0::fit_select(&data, &graph, &split, config.clone(), |m, _| {
        checkpoints.push(m.clone());
        Ok(())
    })
    .unwrap();
    let network = wormsim::model::Model::new(graph.clone()).unwrap();
    let initial = checkpoints[0].parameters.expand(&network).unwrap();
    let start = 6 * network.n() + network.pre.len();
    for (i, p) in [0.75, 0.25, 0.5].iter().enumerate() {
        let group =
            &checkpoints[0].parameters.groups[checkpoints[0].parameters.raw_to_group[start + i]];
        assert!((group.prior_mean.sigmoid() - p).abs() < 1e-12);
        assert!(((2. * initial.raw[start + i].sigmoid() - 1.).abs() - 0.5).abs() < 1e-12);
    }
    assert_eq!(serde_json::to_value(&graph.graph).unwrap(), graph_before);
    assert_eq!(selected.dataset_hash, split.dataset_hash);
    for trial in &mut data.trials {
        if split.test.contains(&trial.id) {
            for t in &mut trial.recording.traces {
                t.values.fill(Some(999.));
            }
        }
    }
    let mut changed_split = split.clone();
    changed_split.dataset_hash = data.content_hash().unwrap();
    let mut changed = vec![];
    let (other, _) = atlas_level0::fit_select(&data, &graph, &changed_split, config, |m, _| {
        changed.push(m.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(selected.epoch, other.epoch);
    for (a, b) in checkpoints.iter().zip(&changed) {
        assert_eq!(
            serde_json::to_value(&a.parameters).unwrap(),
            serde_json::to_value(&b.parameters).unwrap()
        );
        assert_eq!(a.kernel_raw, b.kernel_raw);
        assert_eq!(a.observation_log_gain, b.observation_log_gain);
    }
    let mut invalid = other;
    invalid
        .config
        .molecular_sign_priors
        .as_mut()
        .unwrap()
        .graph_hash = "0".repeat(64);
    assert!(
        invalid
            .predict(&data, &graph, &changed_split, Partition::Test)
            .is_err()
    );
}

#[test]
fn global_gain_first_update_matches_finite_difference_and_legacy_is_identity() {
    let (graph, data, split) = fixture();
    let config: FitConfig = serde_json::from_value(serde_json::json!({
        "epochs":1,"dt":0.02,"preparation_seconds":0.4,"learning_rate":0.01,
        "kernel_lags":1,"prior_strength":0.01,"sign_prior_strength":0.01,
        "kernel_prior_strength":0.01,"sharing":wormsim::parameters::Sharing::default(),
        "observation_gain":{"initial_gain":3.,"prior_strength":0.1}
    }))
    .unwrap();
    let mut checkpoints = vec![];
    atlas_level0::fit_select(&data, &graph, &split, config, |m, _| {
        checkpoints.push(m.clone());
        Ok(())
    })
    .unwrap();
    let start = &checkpoints[0];
    let score = |model: &atlas_level0::AtlasModel| {
        let pred = model
            .predict(&data, &graph, &split, Partition::Train)
            .unwrap();
        wormsim::bench::evaluate(&data, &graph, &split, &pred, Partition::Train)
            .unwrap()
            .pooled_trace_scores
            .mse
            .unwrap()
    };
    let eps = 1e-5;
    let mut plus = start.clone();
    let mut minus = start.clone();
    *plus.observation_log_gain.as_mut().unwrap() += eps;
    *minus.observation_log_gain.as_mut().unwrap() -= eps;
    let derivative = (score(&plus) - score(&minus)) / (2. * eps);
    assert!(derivative.abs() > 1e-6);
    // At initialization the log-gain shrinkage gradient is zero. First Adam
    // moment bias correction gives this exact scalar update.
    let expected = 3.0_f64.ln() - 0.01 * derivative / (derivative.abs() + 1e-8);
    assert!((checkpoints[1].observation_log_gain.unwrap() - expected).abs() < 1e-9);

    let mut serialized = serde_json::to_value(start).unwrap();
    serialized
        .as_object_mut()
        .unwrap()
        .remove("observation_log_gain");
    serialized["config"]
        .as_object_mut()
        .unwrap()
        .remove("observation_gain");
    let legacy: atlas_level0::AtlasModel = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        legacy.readout(graph.names.len()).unwrap().gain,
        vec![1.; graph.names.len()]
    );
    assert_eq!(start.free_parameters(), legacy.free_parameters() + 1);
    let a = start
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    let b = legacy
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    for (a, b) in a.trials.iter().zip(&b.trials) {
        for (name, values) in &a.fluorescence {
            for (x, y) in values.iter().zip(&b.fluorescence[name]) {
                assert!((x - 3. * y).abs() < 1e-12);
            }
        }
    }
    for bad_gain in [None, Some(f64::NAN), Some(1000.), Some(-1000.)] {
        let mut bad = start.clone();
        bad.observation_log_gain = bad_gain;
        assert!(bad.predict(&data, &graph, &split, Partition::Test).is_err());
    }
    let mut bad = legacy;
    bad.observation_log_gain = Some(0.);
    assert!(bad.predict(&data, &graph, &split, Partition::Test).is_err());
}

#[test]
fn cosine_schedule_is_applied_and_zero_final_rate_preserves_parameters() {
    let (graph, data, split) = fixture();
    let mut config: FitConfig = serde_json::from_value(serde_json::json!({
        "epochs":2,"dt":0.02,"preparation_seconds":0.4,"learning_rate":0.01,
        "learning_rate_schedule":{"kind":"cosine","minimum_fraction":0.},
        "kernel_lags":1,"prior_strength":0.01,"sign_prior_strength":0.01,
        "kernel_prior_strength":0.01,"sharing":wormsim::parameters::Sharing::default(),
        "observation_gain":{"initial_gain":3.,"prior_strength":0.1}
    }))
    .unwrap();
    let mut checkpoints = vec![];
    let (_, reports) = atlas_level0::fit_select(&data, &graph, &split, config.clone(), |m, _| {
        checkpoints.push(m.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(
        reports
            .iter()
            .map(|r| r.applied_learning_rate)
            .collect::<Vec<_>>(),
        vec![None, Some(0.01), Some(0.)]
    );
    assert_eq!(
        serde_json::to_value(&checkpoints[1].parameters).unwrap(),
        serde_json::to_value(&checkpoints[2].parameters).unwrap()
    );
    assert_eq!(checkpoints[1].kernel_raw, checkpoints[2].kernel_raw);
    assert_eq!(
        checkpoints[1].observation_log_gain,
        checkpoints[2].observation_log_gain
    );
    config.learning_rate_schedule = Default::default();
    let mut constant = vec![];
    atlas_level0::fit_select(&data, &graph, &split, config, |m, _| {
        constant.push(m.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(checkpoints[1].kernel_raw, constant[1].kernel_raw);
    assert_ne!(checkpoints[2].kernel_raw, constant[2].kernel_raw);
}

#[test]
fn adamw_fit_decays_only_trainable_raw_coordinates() {
    use wormsim::bench::optimization::Optimizer;
    let (graph, data, split) = fixture();
    let mut config: FitConfig = serde_json::from_value(serde_json::json!({
        "epochs":1,"dt":0.02,"preparation_seconds":0.4,"learning_rate":0.01,
        "kernel_lags":1,"prior_strength":0.01,"sign_prior_strength":0.01,
        "kernel_prior_strength":0.01,"sharing":wormsim::parameters::Sharing::default(),
        "observation_gain":{"initial_gain":3.,"prior_strength":0.1}
    }))
    .unwrap();
    let mut baseline = vec![];
    atlas_level0::fit_select(&data, &graph, &split, config.clone(), |m, _| {
        baseline.push(m.clone());
        Ok(())
    })
    .unwrap();
    config.optimizer = Optimizer::AdamW { weight_decay: 0.2 };
    let mut decayed = vec![];
    atlas_level0::fit_select(&data, &graph, &split, config, |m, _| {
        decayed.push(m.clone());
        Ok(())
    })
    .unwrap();
    for ((start, adam), adamw) in baseline[0]
        .parameters
        .groups
        .iter()
        .zip(&baseline[1].parameters.groups)
        .zip(&decayed[1].parameters.groups)
    {
        let expected = if start.trainable {
            adam.value - 0.002 * start.value
        } else {
            start.value
        };
        assert!((adamw.value - expected).abs() < 1e-12);
    }
    assert!(
        (decayed[1].kernel_raw[0]
            - (baseline[1].kernel_raw[0] - 0.002 * baseline[0].kernel_raw[0]))
            .abs()
            < 1e-12
    );
    assert!(
        (decayed[1].observation_log_gain.unwrap()
            - (baseline[1].observation_log_gain.unwrap()
                - 0.002 * baseline[0].observation_log_gain.unwrap()))
        .abs()
            < 1e-12
    );
    assert_eq!(decayed[1].free_parameters(), baseline[1].free_parameters());
}
