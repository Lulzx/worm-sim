use std::collections::BTreeMap;
use wormsim::{
    bench::{
        self, Axis, Dataset, Partition, PredictedTrial, Predictions, Split, Trial,
        metrics::{self, Moments},
    },
    data::{IndexedGraph, Provenance, Recording, Trace},
    fixtures,
};
fn fixture() -> (IndexedGraph, Dataset) {
    let graph = fixtures::synthetic(6, 1, 1).compile().unwrap();
    let trials = (0..6)
        .flat_map(|i| (0..2).map(move |repeat| (i, repeat)))
        .map(|(i, repeat)| {
            let offset = (i + repeat) as f64;
            Trial {
                id: format!("trial-{i}-{repeat}"),
                stimulated_neuron: Some(format!("N{i:03}")),
                forecast_origin: Some(0.0),
                recording: Recording {
                    behavior: Default::default(),
                    dataset: "synthetic-benchmark-contract-test".into(),
                    animal_id: format!("animal-{i}"),
                    condition: "synthetic".into(),
                    times: vec![0.0, 1.0, 10.0, 30.0],
                    traces: (0..2)
                        .map(|n| Trace {
                            neuron: format!("N{n:03}"),
                            values: vec![
                                Some(offset),
                                Some(offset + 1.0),
                                None,
                                Some(offset + 3.0),
                            ],
                            provenance: Provenance {
                                dataset: "synthetic".into(),
                                version: "1".into(),
                                id_confidence: if n == 0 { 1.0 } else { 0.5 },
                            },
                        })
                        .collect(),
                },
                response_labels: BTreeMap::from([("N000".into(), true), ("N001".into(), false)]),
            }
        })
        .collect();
    let data = Dataset {
        schema_version: 1,
        name: "synthetic-contract-fixture".into(),
        graph_hash: graph.hash.clone(),
        source: "synthetic; no biological validation".into(),
        trials,
    };
    (graph, data)
}
fn predictions(data: &Dataset, split: &Split) -> Predictions {
    Predictions {
        schema_version: 1,
        dataset_hash: data.content_hash().unwrap(),
        split_hash: split.content_hash().unwrap(),
        model: "oracle for metric tests only".into(),
        free_parameters: 0,
        training_trials: split.train.clone(),
        selection_trials: split.validation.clone(),
        source_commit: "test-fixture".into(),
        seed: 0,
        trials: data
            .trials
            .iter()
            .filter(|t| split.test.contains(&t.id))
            .map(|t| PredictedTrial {
                id: t.id.clone(),
                times: t.recording.times.clone(),
                fluorescence: t
                    .recording
                    .traces
                    .iter()
                    .map(|r| {
                        (
                            r.neuron.clone(),
                            r.values.iter().map(|v| v.unwrap_or(123.0)).collect(),
                        )
                    })
                    .collect(),
                response_scores: BTreeMap::from([("N000".into(), 0.9), ("N001".into(), 0.1)]),
            })
            .collect(),
    }
}
#[test]
fn weighted_metrics_match_independent_hand_calculations() {
    let mut m = Moments::default();
    m.push(0.0, 1.0, 1.0).unwrap();
    m.push(2.0, 1.0, 3.0).unwrap();
    let s = m.scores();
    assert_eq!(s.samples, 2);
    assert_eq!(s.mse, Some(1.0));
    assert_eq!(s.correlation, None);
    assert!((s.r2.unwrap() + 1.0 / 3.0).abs() < 1e-14);
    m.push(99.0, -42.0, 0.0).unwrap();
    assert_eq!(m.scores().samples, 2);
    let auc = metrics::auroc(&[(true, 0.5, 2.0), (false, 0.5, 1.0), (false, 0.0, 3.0)]).unwrap();
    assert_eq!(auc.value, Some(0.875));
    assert_eq!(metrics::auroc(&[(true, 1.0, 1.0)]).unwrap().value, None);
    assert_eq!(
        metrics::auroc(&[(true, 0.0, 1.0), (false, -0.0, 1.0)])
            .unwrap()
            .value,
        Some(0.5)
    );
    assert!(m.push(1.0, f64::NAN, 1.0).is_err());
    assert!(metrics::auroc(&[(true, 0.0, -1.0)]).is_err());
}
#[test]
fn moments_preserve_small_variance_at_large_offsets() {
    let mut m = Moments::default();
    for x in [1e12, 1e12 + 1.0, 1e12 + 2.0] {
        m.push(x, x + 1.0, 1.0).unwrap();
    }
    assert!((m.scores().correlation.unwrap() - 1.0).abs() < 1e-14);
    assert_eq!(m.scores().r2, Some(-0.5));
}
#[test]
fn splits_are_content_bound_order_independent_and_group_disjoint() {
    let (g, mut data) = fixture();
    for axis in [Axis::Animal, Axis::StimulatedNeuron] {
        let split = Split::generate(&data, &g, axis, 42, 1, 2).unwrap();
        assert_eq!(split.train.len(), 6);
        assert_eq!(split.validation.len(), 2);
        assert_eq!(split.test.len(), 4);
        let original_hash = data.content_hash().unwrap();
        data.trials.reverse();
        for trial in &mut data.trials {
            trial.recording.traces.reverse();
        }
        assert_eq!(data.content_hash().unwrap(), original_hash);
        let other = Split::generate(&data, &g, axis, 42, 1, 2).unwrap();
        assert_eq!(split.content_hash().unwrap(), other.content_hash().unwrap());
        let mut leaked = split.clone();
        std::mem::swap(&mut leaked.train[0], &mut leaked.test[0]);
        assert!(leaked.validate(&data, &g).unwrap_err().contains("leakage"));
        let mut duplicate = split.clone();
        duplicate.test.push(duplicate.test[0].clone());
        assert!(duplicate.validate(&data, &g).is_err());
        let mut changed = data.clone();
        changed.trials[0].recording.traces[0].values[0] = Some(55.0);
        assert!(split.validate(&changed, &g).is_err());
    }
}
#[test]
fn scoring_masks_missing_samples_and_evaluates_exact_horizons() {
    let (g, data) = fixture();
    let split = Split::generate(&data, &g, Axis::Animal, 9, 1, 2).unwrap();
    let pred = predictions(&data, &split);
    let report = bench::evaluate(&data, &g, &split, &pred, Partition::Test).unwrap();
    assert_eq!(report.trials, 4);
    assert_eq!(report.missing_samples, 8);
    assert_eq!(report.pooled_trace_scores.mse, Some(0.0));
    assert_eq!(report.macro_trace_correlation, Some(1.0));
    assert_eq!(report.response_auroc.value, Some(1.0));
    for h in [&report.forecast_horizons[0], &report.forecast_horizons[2]] {
        assert_eq!(h.macro_neuron_r2, Some(1.0));
        assert_eq!(h.defined_neuron_r2, 2);
        assert_eq!(h.pooled.samples, 8);
        assert_eq!(h.unavailable_grid_or_missing_samples, 0);
    }
    let missing = &report.forecast_horizons[1];
    assert_eq!(missing.macro_neuron_r2, None);
    assert_eq!(missing.unavailable_grid_or_missing_samples, 8);
    // Predictions before/at forecast origin must never improve held-out scores.
    let mut initial_wrong = pred.clone();
    for t in &mut initial_wrong.trials {
        for v in t.fluorescence.values_mut() {
            v[0] = 1e6;
        }
    }
    let other = bench::evaluate(&data, &g, &split, &initial_wrong, Partition::Test).unwrap();
    assert_eq!(
        other.pooled_trace_scores.mse,
        report.pooled_trace_scores.mse
    );
}
#[test]
fn scoring_rejects_leakage_missing_predictions_and_wrong_time_grids() {
    let (g, data) = fixture();
    let split = Split::generate(&data, &g, Axis::StimulatedNeuron, 1, 1, 2).unwrap();
    let pred = predictions(&data, &split);
    let score = |p: &Predictions| bench::evaluate(&data, &g, &split, p, Partition::Test);
    assert!(score(&pred).is_ok());
    let mut bad = pred.clone();
    bad.training_trials.push(split.test[0].clone());
    assert!(score(&bad).is_err());
    let mut bad = pred.clone();
    bad.selection_trials.push(split.train[0].clone());
    assert!(score(&bad).is_err());
    let mut bad = pred.clone();
    bad.trials.pop();
    assert!(score(&bad).is_err());
    let mut bad = pred.clone();
    bad.trials[0].times[1] += 0.1;
    assert!(score(&bad).is_err());
    let mut bad = pred.clone();
    bad.trials[0].fluorescence.remove("N000");
    assert!(score(&bad).is_err());
    let mut bad = pred.clone();
    bad.trials[0].response_scores.remove("N001");
    bad.trials[0].response_scores.insert("N002".into(), 0.1);
    assert!(score(&bad).is_err());
    let mut bad = pred.clone();
    bad.trials[0].fluorescence.get_mut("N000").unwrap()[2] = f64::NAN;
    assert!(score(&bad).is_err());
}
#[test]
fn unavailable_and_zero_confidence_observations_remain_undefined() {
    let (g, mut data) = fixture();
    for trial in &mut data.trials {
        trial.recording.traces[0].provenance.id_confidence = 0.0;
        trial.recording.traces[1].values.fill(Some(3.0));
    }
    let split = Split::generate(&data, &g, Axis::StimulatedNeuron, 10, 0, 1).unwrap();
    let pred = predictions(&data, &split);
    let report = bench::evaluate(&data, &g, &split, &pred, Partition::Test).unwrap();
    assert_eq!(report.zero_confidence_traces, 2);
    assert_eq!(report.macro_trace_correlation, None);
    assert_eq!(report.response_auroc.value, None);
    assert_eq!(report.pooled_trace_scores.mse, Some(0.0));
    assert_eq!(report.pooled_trace_scores.r2, None);
}

#[test]
fn cli_split_and_score_round_trip() {
    let (graph, data) = fixture();
    let dir = std::env::temp_dir().join(format!(
        "wormsim-bench-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let graph_path = dir.join("graph.wsc");
    let data_path = dir.join("data.json");
    let split_path = dir.join("split.json");
    let pred_path = dir.join("pred.json");
    let report_path = dir.join("report.json");
    std::fs::write(&graph_path, wormsim::codec::encode(&graph).unwrap()).unwrap();
    std::fs::write(&data_path, serde_json::to_vec(&data).unwrap()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_wormsim"))
        .arg("bench-split")
        .arg(&graph_path)
        .arg(&data_path)
        .args(["neuron", "42", "1", "2"])
        .arg(&split_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let split: Split = serde_json::from_slice(&std::fs::read(split_path).unwrap()).unwrap();
    std::fs::write(
        &pred_path,
        serde_json::to_vec(&predictions(&data, &split)).unwrap(),
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_wormsim"))
        .arg("bench-score")
        .arg(&graph_path)
        .arg(&data_path)
        .arg(dir.join("split.json"))
        .arg(&pred_path)
        .arg("test")
        .arg(&report_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: bench::Report =
        serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report.response_auroc.value, Some(1.0));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn persistence_reads_only_observed_history() {
    let (graph, mut data) = fixture();
    for trial in &mut data.trials {
        trial.response_labels.clear();
    }
    let split = Split::generate(&data, &graph, Axis::Animal, 42, 1, 2).unwrap();
    let predictions = bench::persistence(&data, &graph, &split, Partition::Test).unwrap();
    assert_eq!(predictions.free_parameters, 0);
    assert!(predictions.training_trials.is_empty());
    let report = bench::evaluate(&data, &graph, &split, &predictions, Partition::Test).unwrap();
    assert_eq!(report.forecast_horizons[0].pooled.mse, Some(1.0));
    assert_eq!(report.forecast_horizons[2].pooled.mse, Some(9.0));
    // Change all future outcomes, rebuild only the content binding, and prove
    // that predictions themselves are unchanged.
    for trial in &mut data.trials {
        for trace in &mut trial.recording.traces {
            trace.values[1..].fill(Some(-999.0));
        }
    }
    let updated = Split::generate(&data, &graph, Axis::Animal, 42, 1, 2).unwrap();
    let other = bench::persistence(&data, &graph, &updated, Partition::Test).unwrap();
    for (a, b) in predictions.trials.iter().zip(other.trials) {
        assert_eq!(a.fluorescence, b.fluorescence);
    }
}
