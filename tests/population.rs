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
