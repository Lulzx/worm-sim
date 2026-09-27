use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
use wormsim::{
    bench::{Axis, Split},
    fixtures,
    recordings::randi::{self, Manifest, SourceFile},
};
struct Source {
    root: PathBuf,
    manifest: Manifest,
}
impl Source {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "wormsim-randi-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let mut source = Self {
            root,
            manifest: Manifest {
                schema_version: 1,
                source: "synthetic atlas export".into(),
                url: "synthetic".into(),
                archive_sha256: "synthetic".into(),
                archive_bytes: 0,
                files: vec![],
            },
        };
        source.set("labels.txt", "N000\nN001\nN002\nN002\nAWCON\n\n");
        source.set("ds_name.txt", "/source/recording-1/\n");
        source.set(
            "t.txt",
            &(0..201).map(|i| format!("{i}\n")).collect::<String>(),
        );
        source.set("stim_volume_i.txt", "20\n60\n70\n120\n160\n");
        source.set("stim_neurons.txt", "0\n1\n-1\n2\n1\n");
        let mut matrix = String::new();
        for t in 0..201 {
            let value = if (20..40).contains(&t) { 15 } else { 10 };
            let missing = if t == 21 { "nan" } else { "10" };
            matrix.push_str(&format!("{value} {missing} 10 10 10 10\n"));
        }
        source.set("gcamp.txt", &matrix);
        source
    }
    fn set(&mut self, suffix: &str, value: &str) {
        let path = format!("0_{suffix}");
        fs::write(self.root.join(&path), value).unwrap();
        self.manifest.files.retain(|e| e.path != path);
        self.manifest.files.push(SourceFile {
            path,
            bytes: value.len(),
            sha256: format!("{:x}", Sha256::digest(value.as_bytes())),
        });
        fs::write(
            self.root.join("manifest.json"),
            serde_json::to_vec(&self.manifest).unwrap(),
        )
        .unwrap();
    }
    fn import(&self) -> wormsim::Result<(wormsim::bench::Dataset, randi::Report)> {
        randi::import(
            &self.root,
            &self.root.join("manifest.json"),
            &fixtures::synthetic(3, 1, 0).compile().unwrap(),
            randi::Config::default(),
        )
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}
#[test]
fn native_trials_preserve_timing_missingness_and_anatomical_ambiguity() {
    let source = Source::new();
    let (data, report) = source.import().unwrap();
    assert_eq!(report.source_events, 5);
    assert_eq!(data.trials.len(), 2);
    assert_eq!(report.excluded_events["overlapping_source_windows"], 2);
    assert_eq!(report.excluded_events["unknown_or_ambiguous_stimulus"], 1);
    assert_eq!(report.excluded_labels["N002"], 2);
    let trial = &data.trials[0];
    assert_eq!(
        trial.recording.times,
        (0..20).map(|i| i as f64).collect::<Vec<_>>()
    );
    assert_eq!(trial.recording.traces[0].values, vec![Some(0.5); 20]);
    assert_eq!(trial.recording.traces[1].values[1], None);
    assert!(trial.response_labels.is_empty());
    assert_eq!(trial.stimulated_neuron.as_deref(), Some("N000"));
    assert_eq!(report.events[0].source_frames, [10, 40]);
    assert!(
        report
            .events
            .windows(2)
            .all(|p| p[0].source_frames[1] <= p[1].source_frames[0])
    );
    let graph = fixtures::synthetic(3, 1, 0).compile().unwrap();
    let split = Split::generate(&data, &graph, Axis::StimulatedNeuron, 42, 0, 1).unwrap();
    split.validate(&data, &graph).unwrap();
    assert_eq!(split.train.len(), 1);
    assert_eq!(split.test.len(), 1);
}
#[test]
fn malformed_sources_fail_and_nonpositive_baselines_are_excluded() {
    let mut source = Source::new();
    fs::write(source.root.join("0_labels.txt"), "tampered").unwrap();
    assert!(source.import().unwrap_err().contains("hash mismatch"));
    source.set("labels.txt", "N000\nN001\nN002\nN002\nAWCON\n\n");
    source.set("stim_volume_i.txt", "20\n20\n70\n120\n160\n");
    assert!(source.import().is_err());
    source.set("stim_volume_i.txt", "20\n60\n70\n120\n160\n");
    source.set(
        "gcamp.txt",
        &(0..201).map(|_| "0 10 10 10 10 10\n").collect::<String>(),
    );
    let (data, report) = source.import().unwrap();
    assert_eq!(report.excluded_trace_windows["nonpositive_baseline"], 2);
    assert!(data.trials.iter().all(|t| t.recording.traces.len() == 1));
    source.set(
        "gcamp.txt",
        &(0..201).map(|_| "inf 10 10 10 10 10\n").collect::<String>(),
    );
    assert!(source.import().is_err());
}
