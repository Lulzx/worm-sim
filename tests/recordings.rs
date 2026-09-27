use wormsim::recordings::{WindowConfig, interpolation, resample};
#[test]
fn resampling_has_no_extrapolation_or_gap_bridging() {
    let times = [0.0, 0.5, 1.0, 5.0, 5.5];
    let plan = interpolation(&times, &[-0.1, 0.0, 0.25, 1.0, 3.0, 5.25, 6.0], 1.0).unwrap();
    let values = resample(&[0.0, 1.0, 2.0, 10.0, 11.0], &plan).unwrap();
    assert_eq!(
        values,
        vec![
            None,
            Some(0.0),
            Some(0.5),
            Some(2.0),
            None,
            Some(10.5),
            None
        ]
    );
    let missing = resample(&[0.0, f64::NAN, 2.0, 10.0, 11.0], &plan).unwrap();
    assert_eq!(missing[1], Some(0.0));
    assert_eq!(missing[2], None);
    assert!(interpolation(&[0.0, 0.0], &[0.0], 1.0).is_err());
    let bad = WindowConfig {
        sample_dt: 0.3,
        ..WindowConfig::default()
    };
    assert!(bad.validate().is_err());
}

#[cfg(feature = "hdf5")]
#[test]
fn native_hdf5_import_checks_orientation_roi_confidence_and_integrity() {
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, fs};
    use wormsim::{fixtures, recordings::wormwideweb};
    let root = std::env::temp_dir().join(format!(
        "wormsim-hdf5-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let graph = fixtures::synthetic(3, 1, 1).compile().unwrap();
    let mut animals = vec![];
    let mut all_labels = BTreeMap::new();
    for (id, transpose) in [("animal-a", false), ("animal-b", true)] {
        let name = format!("{id}.h5");
        let path = root.join(&name);
        let file = hdf5::File::create(&path).unwrap();
        file.create_group("timing").unwrap();
        file.create_group("gcamp").unwrap();
        file.create_group("behavior").unwrap();
        let dt = if transpose { 0.73 } else { 1.0 };
        let times: Vec<_> = (0..101).map(|i| i as f64 * dt).collect();
        file.new_dataset::<f64>()
            .shape([101])
            .create("timing/timestamp_confocal")
            .unwrap()
            .write_raw(&times)
            .unwrap();
        let (shape, raw) = if transpose {
            (
                [3, 101],
                (0..3)
                    .flat_map(|n| (0..101).map(move |t| 1000.0 * n as f64 + t as f64 * dt))
                    .collect::<Vec<_>>(),
            )
        } else {
            (
                [101, 3],
                (0..101)
                    .flat_map(|t| (0..3).map(move |n| 1000.0 * n as f64 + t as f64 * dt))
                    .collect::<Vec<_>>(),
            )
        };
        file.new_dataset::<f64>()
            .shape(shape)
            .create("gcamp/trace_array")
            .unwrap()
            .write_raw(&raw)
            .unwrap();
        file.new_dataset::<f64>()
            .shape([101])
            .create("behavior/velocity")
            .unwrap()
            .write_raw(&times)
            .unwrap();
        file.close().unwrap();
        let checksum = format!("{:x}", Sha256::digest(fs::read(&path).unwrap()));
        animals.push(serde_json::json!({"file":name,"animal_id":id,"sha256":checksum}));
        all_labels.insert(id,serde_json::json!({"idx_neuron-label":{"1":{"label":"N000","confidence":5},"2":{"label":"N001","confidence":4},"3":{"label":"N002","confidence":2}}}));
    }
    let labels = root.join("labels.json");
    let receipt = root.join("receipt.json");
    fs::write(
        &labels,
        serde_json::to_vec(&serde_json::json!({"data":all_labels})).unwrap(),
    )
    .unwrap();
    fs::write(
        &receipt,
        serde_json::to_vec(
            &serde_json::json!({"zenodo_record":"synthetic-test","animals":animals}),
        )
        .unwrap(),
    )
    .unwrap();
    let (data, report) =
        wormwideweb::import(&root, &labels, &receipt, &graph, &WindowConfig::default()).unwrap();
    assert_eq!(data.trials.len(), 3);
    assert_eq!(report.labels.iter().filter(|l| l.accepted).count(), 4);
    for trial in &data.trials {
        assert_eq!(trial.recording.traces.len(), 2);
        let a = &trial.recording.traces[0];
        let b = &trial.recording.traces[1];
        assert_eq!(a.neuron, "N000");
        assert_eq!(b.neuron, "N001");
        assert_eq!(b.provenance.id_confidence, 0.8);
        assert_eq!(b.values[21].unwrap() - a.values[21].unwrap(), 1000.0);
        assert_eq!(trial.recording.behavior["velocity"][21], a.values[21]);
    }
    let irregular = data
        .trials
        .iter()
        .find(|t| t.recording.animal_id == "animal-b")
        .unwrap();
    assert_eq!(irregular.recording.traces[0].values[20], Some(14.0 * 0.73));
    // Label/behavior modifications alter the benchmark identity.
    let mut changed = data.clone();
    changed.trials[0]
        .recording
        .behavior
        .get_mut("velocity")
        .unwrap()[0] = Some(-5.0);
    assert_ne!(
        changed.content_hash().unwrap(),
        data.content_hash().unwrap()
    );
    fs::write(root.join("animal-a.h5"), b"corruption").unwrap();
    assert!(
        wormwideweb::import(&root, &labels, &receipt, &graph, &WindowConfig::default())
            .unwrap_err()
            .contains("checksum")
    );
    fs::remove_dir_all(root).unwrap();
}
