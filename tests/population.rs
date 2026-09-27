use std::collections::BTreeMap;
use wormsim::{
    bench::{
        self, Axis, Dataset, Partition, Split, Trial,
        population::{self, FitConfig},
    },
    data::{Provenance, Recording, Trace},
    fixtures,
    initial_state::InferenceConfig,
};
fn fixture() -> (wormsim::data::IndexedGraph, Dataset) {
    let graph = fixtures::synthetic(2, 1, 0).compile().unwrap();
    let trials = (0..8)
        .map(|animal| Trial {
            id: format!("trial-{animal}"),
            stimulated_neuron: None,
            forecast_origin: Some(10.0),
            response_labels: BTreeMap::new(),
            recording: Recording {
                dataset: "synthetic".into(),
                animal_id: format!("animal-{animal}"),
                condition: "fit".into(),
                times: (0..81).map(|t| t as f64 * 0.5).collect(),
                traces: (0..2)
                    .map(|i| Trace {
                        neuron: graph.names[i].clone(),
                        values: (0..81)
                            .map(|t| Some(0.1 * animal as f64 + (t as f64 * 0.1 + i as f64).sin()))
                            .collect(),
                        provenance: Provenance {
                            dataset: "synthetic".into(),
                            version: "1".into(),
                            id_confidence: 1.0,
                        },
                    })
                    .collect(),
                behavior: BTreeMap::new(),
            },
        })
        .collect();
    let data = Dataset {
        schema_version: 1,
        name: "population test".into(),
        graph_hash: graph.hash.clone(),
        source: "synthetic".into(),
        trials,
    };
    (graph, data)
}
#[test]
fn population_fit_uses_all_training_windows_and_excludes_test_targets() {
    let (graph, mut data) = fixture();
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let config = FitConfig {
        epochs: 1,
        inference: InferenceConfig {
            dt: 0.05,
            iterations: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut snapshots = vec![];
    let (model, report) = population::fit(&data, &graph, &split, config.clone(), |m, r| {
        snapshots.push((m.clone(), r.clone()));
        Ok(())
    })
    .unwrap();
    assert_eq!(snapshots.len(), 2);
    assert_eq!(report.epochs[1].updates, split.train.len());
    assert_eq!(model.training_trials, split.train);
    assert_eq!(model.selection_trials, split.validation);
    assert!(
        snapshots[0]
            .0
            .parameters
            .groups
            .iter()
            .zip(&snapshots[1].0.parameters.groups)
            .any(|(a, b)| (a.value - b.value).abs() > 1e-8)
    );
    let before = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    let scored = bench::evaluate(&data, &graph, &split, &before, Partition::Test).unwrap();
    assert_eq!(scored.forecast_horizons.len(), 3);
    for trial in &mut data.trials {
        if split.test.contains(&trial.id) {
            for trace in &mut trial.recording.traces {
                for x in &mut trace.values[21..] {
                    *x = Some(50000.0);
                }
            }
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let (other, other_report) =
        population::fit(&data, &graph, &updated, config, |_, _| Ok(())).unwrap();
    assert_eq!(model.selected_epoch, other.selected_epoch);
    assert_eq!(model.readout.offset, other.readout.offset);
    assert_eq!(model.readout.gain, other.readout.gain);
    for (a, b) in model.parameters.groups.iter().zip(&other.parameters.groups) {
        assert_eq!(a.value, b.value);
    }
    assert_eq!(
        report.epochs[1].validation_horizon_r2,
        other_report.epochs[1].validation_horizon_r2
    );
    let after = other
        .predict(&data, &graph, &updated, Partition::Test)
        .unwrap();
    for (a, b) in before.trials.iter().zip(after.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
}

#[test]
fn population_filter_forecast_carries_assimilated_origin_not_unforced_replay() {
    use wormsim::{
        initial_state::{self, InferenceMethod},
        model::Model,
    };
    let (graph, mut data) = fixture();
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let config = FitConfig {
        epochs: 1,
        inference: InferenceConfig {
            method: InferenceMethod::BlockEkf,
            dt: 0.05,
            ..Default::default()
        },
        ..Default::default()
    };
    let (model, _) = population::fit(&data, &graph, &split, config.clone(), |_, _| Ok(())).unwrap();
    let prediction = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    let neural = Model::new(graph.clone()).unwrap();
    let parameters = model.parameters.expand(&neural).unwrap();
    let prepared = neural.prepare(&parameters).unwrap();
    for trial in data.trials.iter().filter(|t| split.test.contains(&t.id)) {
        let inferred = initial_state::infer(
            &neural,
            &parameters,
            &trial.recording,
            10.0,
            &model.readout,
            &model.config.inference,
        )
        .unwrap();
        let pred = prediction.trials.iter().find(|p| p.id == trial.id).unwrap();
        for trace in &trial.recording.traces {
            let i = graph.neuron(&trace.neuron).unwrap();
            let expected = model.readout.offset[i]
                + model.readout.gain[i]
                    * prepared.calcium_scale[i]
                    * inferred.forecast_state[neural.n() + i];
            assert!((pred.fluorescence[&trace.neuron][20] - expected).abs() < 1e-12);
        }
    }
    for trial in &mut data.trials {
        if split.test.contains(&trial.id) {
            for trace in &mut trial.recording.traces {
                for value in &mut trace.values[21..] {
                    *value = Some(9999.0);
                }
            }
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let (same, _) = population::fit(&data, &graph, &updated, config, |_, _| Ok(())).unwrap();
    assert_eq!(same.selected_epoch, model.selected_epoch);
    assert_eq!(same.readout.offset, model.readout.offset);
    assert_eq!(same.readout.gain, model.readout.gain);
    for (a, b) in same.parameters.groups.iter().zip(&model.parameters.groups) {
        assert_eq!(a.value, b.value);
    }
    let after = same
        .predict(&data, &graph, &updated, Partition::Test)
        .unwrap();
    for (a, b) in prediction.trials.iter().zip(after.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
}

#[test]
fn driven_population_refit_excludes_test_futures_and_learns_tied_inputs() {
    use wormsim::initial_state::InferenceMethod;
    let (graph, mut data) = fixture();
    for trial in &mut data.trials {
        trial.recording.behavior.insert(
            "velocity".into(),
            trial
                .recording
                .times
                .iter()
                .map(|t| Some((t * 0.13).cos()))
                .collect(),
        );
    }
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let config = FitConfig {
        epochs: 1,
        behavior_channels: vec!["velocity".into()],
        inference: InferenceConfig {
            method: InferenceMethod::BlockEkf,
            dt: 0.05,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut snapshots = vec![];
    let (model, report) = population::fit(&data, &graph, &split, config.clone(), |m, _| {
        snapshots.push(m.clone());
        Ok(())
    })
    .unwrap();
    assert!(
        snapshots[0]
            .input_weights
            .as_ref()
            .unwrap()
            .weights
            .iter()
            .all(|w| *w == 0.0)
    );
    assert!(
        snapshots[1]
            .input_weights
            .as_ref()
            .unwrap()
            .weights
            .iter()
            .any(|w| w.abs() > 1e-8)
    );
    assert_eq!(report.behavior_forecast_parameters, 4);
    assert_eq!(
        report.input_parameters,
        model.input_weights.as_ref().unwrap().weights.len()
    );
    let before = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    for trial in &mut data.trials {
        if split.test.contains(&trial.id) {
            for trace in &mut trial.recording.traces {
                trace.values[21..].fill(Some(12345.0));
            }
            trial.recording.behavior.get_mut("velocity").unwrap()[21..].fill(Some(-98765.0));
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let mut others = vec![];
    let (other, other_report) = population::fit(&data, &graph, &updated, config, |m, _| {
        others.push(m.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(model.selected_epoch, other.selected_epoch);
    for (a, b) in snapshots.iter().zip(&others) {
        assert_eq!(a.readout.offset, b.readout.offset);
        assert_eq!(a.readout.gain, b.readout.gain);
        assert_eq!(
            serde_json::to_value(&a.input_weights).unwrap(),
            serde_json::to_value(&b.input_weights).unwrap()
        );
        for (x, y) in a.parameters.groups.iter().zip(&b.parameters.groups) {
            assert_eq!(x.value, y.value);
        }
        assert_eq!(
            serde_json::to_value(&a.behavior.as_ref().unwrap().channels).unwrap(),
            serde_json::to_value(&b.behavior.as_ref().unwrap().channels).unwrap()
        );
    }
    for (a, b) in report.epochs.iter().zip(&other_report.epochs) {
        assert_eq!(a.validation_horizon_r2, b.validation_horizon_r2);
        assert_eq!(a.training_conditional_mse, b.training_conditional_mse);
    }
    let after = other
        .predict(&data, &graph, &updated, Partition::Test)
        .unwrap();
    for (a, b) in before.trials.iter().zip(after.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
}

#[test]
fn legacy_population_artifact_defaults_to_no_behavior() {
    let (graph, data) = fixture();
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 2, 2).unwrap();
    let config = FitConfig {
        epochs: 1,
        inference: InferenceConfig {
            dt: 0.05,
            iterations: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let (model, _) = population::fit(&data, &graph, &split, config, |_, _| Ok(())).unwrap();
    let mut legacy = serde_json::to_value(&model).unwrap();
    let object = legacy.as_object_mut().unwrap();
    object.remove("behavior");
    object.remove("input_weights");
    let cfg = object.get_mut("config").unwrap().as_object_mut().unwrap();
    cfg.remove("behavior_channels");
    cfg.remove("input_prior_strength");
    let restored: population::PopulationModel = serde_json::from_value(legacy).unwrap();
    assert!(restored.behavior.is_none());
    let a = model
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    let b = restored
        .predict(&data, &graph, &split, Partition::Test)
        .unwrap();
    for (a, b) in a.trials.iter().zip(b.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
}
