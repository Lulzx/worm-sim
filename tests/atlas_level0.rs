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
