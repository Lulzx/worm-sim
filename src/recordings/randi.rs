//! Native ingestion of the pinned Randi wild-type text export.
//! No anatomical guesses, inferred response labels, or invented pulse calibration.
use crate::{
    Result,
    bench::{Dataset, Trial},
    data::{IndexedGraph, Provenance, Recording, Trace},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub baseline_seconds: f64,
    pub response_seconds: f64,
    pub minimum_baseline_fraction: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            baseline_seconds: 10.0,
            response_seconds: 20.0,
            minimum_baseline_fraction: 0.8,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFile {
    pub path: String,
    pub bytes: usize,
    pub sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub source: String,
    pub url: String,
    pub archive_sha256: String,
    pub archive_bytes: usize,
    pub files: Vec<SourceFile>,
}
#[derive(Debug, Serialize)]
pub struct EventReceipt {
    pub trial: String,
    pub source_recording: String,
    pub event_index: usize,
    pub source_stimulation_frame: usize,
    pub source_stimulation_seconds: f64,
    /// Half-open source-frame range, including the baseline used for normalization.
    pub source_frames: [usize; 2],
    pub observed_traces: usize,
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub source_commit: String,
    pub manifest_sha256: String,
    pub dataset_hash: String,
    pub config: Config,
    pub source_recordings: usize,
    pub source_events: usize,
    pub trailing_blank_labels: BTreeMap<usize, usize>,
    pub excluded_events: BTreeMap<String, usize>,
    pub excluded_trace_windows: BTreeMap<String, usize>,
    pub excluded_labels: BTreeMap<String, usize>,
    pub events: Vec<EventReceipt>,
    pub limitations: Vec<String>,
}
fn count(map: &mut BTreeMap<String, usize>, key: &str) {
    *map.entry(key.into()).or_default() += 1;
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn numbers<T: std::str::FromStr>(text: &str) -> Result<Vec<T>> {
    text.split_whitespace()
        .map(|s| s.parse().map_err(|_| format!("invalid numeric token: {s}")))
        .collect()
}
fn verified_text(root: &Path, entry: &SourceFile) -> Result<String> {
    let path = Path::new(&entry.path);
    if path.components().count() != 1
        || !matches!(
            path.components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err("source path must be a plain filename".into());
    }
    let bytes = fs::read(root.join(path)).map_err(|e| e.to_string())?;
    if bytes.len() != entry.bytes || sha(&bytes) != entry.sha256 {
        return Err(format!("source size/hash mismatch: {}", entry.path));
    }
    String::from_utf8(bytes).map_err(|e| e.to_string())
}
/// Emit post-stimulus dF/F trials. Baselines and response samples are never shared
/// between retained trials, even if the neighboring event has an unknown identity.
pub fn import(
    root: &Path,
    manifest_path: &Path,
    graph: &IndexedGraph,
    config: Config,
) -> Result<(Dataset, Report)> {
    if [
        config.baseline_seconds,
        config.response_seconds,
        config.minimum_baseline_fraction,
    ]
    .iter()
    .any(|v| !v.is_finite() || *v <= 0.0)
        || config.minimum_baseline_fraction > 1.0
    {
        return Err("invalid atlas window configuration".into());
    }
    let manifest_bytes = fs::read(manifest_path).map_err(|e| e.to_string())?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    if manifest.schema_version != 1 || manifest.source.is_empty() || manifest.files.is_empty() {
        return Err("invalid atlas manifest".into());
    }
    let manifest_hash = sha(&manifest_bytes);
    let mut files = BTreeMap::new();
    let mut records = BTreeSet::new();
    const SUFFIXES: [&str; 6] = [
        "gcamp.txt",
        "t.txt",
        "labels.txt",
        "stim_volume_i.txt",
        "stim_neurons.txt",
        "ds_name.txt",
    ];
    for entry in &manifest.files {
        if files.insert(entry.path.clone(), entry).is_some() {
            return Err("duplicate manifest file".into());
        }
        let (id, suffix) = entry
            .path
            .split_once('_')
            .ok_or("invalid source filename")?;
        let id: usize = id.parse().map_err(|_| "invalid source recording index")?;
        if !SUFFIXES.contains(&suffix) {
            return Err("unrecognized source file suffix".into());
        }
        records.insert(id);
    }
    if files.len() != records.len() * SUFFIXES.len() {
        return Err("incomplete source recording files".into());
    }
    let mut report = Report {
        schema_version: 1, source_commit: option_env!("WORMSIM_COMMIT").unwrap_or("unversioned").into(),
        manifest_sha256: manifest_hash.clone(), dataset_hash: String::new(), config: config.clone(),
        source_recordings: records.len(), source_events: 0,
        trailing_blank_labels: BTreeMap::new(), excluded_events: BTreeMap::new(),
        excluded_trace_windows: BTreeMap::new(), excluded_labels: BTreeMap::new(), events: vec![],
        limitations: vec![
            "Processed fluorescence export: upstream spike removal, smoothing and photobleaching correction; not raw images or prospectively processed signals.".into(),
            "Source provides recording IDs, not a verified recording-to-animal map; animal_id stores recording identity. Do not interpret a recording bootstrap as an animal bootstrap.".into(),
            "No published response/nonresponse labels in this export; response_labels are empty and AUROC cannot be evaluated.".into(),
            "Stimulation frame is supplied, but per-event pulse duration, optical power, and membrane-current calibration are absent.".into(),
            "Exact anatomical names only; ambiguous classes, unknown labels, AWCON/OFF and duplicate identities are excluded rather than mapped to arbitrary neurons.".into(),
            "Unit identity confidence is an explicit weighting convention, not a calibrated identity probability.".into(),
            "Nonoverlapping source samples prevent direct duplication across neuron partitions; shared recordings and residual responses to previous stimuli remain possible dependencies.".into(),
        ],
    };
    let mut data = Dataset {
        schema_version: 1,
        name: "Randi 2023 wild-type stimulation trials".into(),
        graph_hash: graph.hash.clone(),
        source: format!(
            "{}; manifest SHA256 {}; post-stimulus dF/F using preceding observed mean; no response labels; configuration {}",
            manifest.source,
            manifest_hash,
            serde_json::to_string(&config).map_err(|e| e.to_string())?
        ),
        trials: vec![],
    };
    let mut recording_names = BTreeSet::new();
    for record in records {
        let read = |suffix: &str| -> Result<String> {
            verified_text(
                root,
                files
                    .get(&format!("{record}_{suffix}"))
                    .ok_or("missing source recording file")?,
            )
        };
        let labels_text = read("labels.txt")?;
        let mut labels: Vec<_> = labels_text.lines().map(str::trim).collect();
        let matrix_text = read("gcamp.txt")?;
        let columns = matrix_text
            .lines()
            .next()
            .ok_or("empty fluorescence matrix")?
            .split_whitespace()
            .count();
        if columns == 0 || labels.len() < columns || labels[columns..].iter().any(|s| !s.is_empty())
        {
            return Err(format!(
                "fluorescence/label dimension mismatch in recording {record}"
            ));
        }
        if labels.len() > columns {
            report
                .trailing_blank_labels
                .insert(record, labels.len() - columns);
            labels.truncate(columns);
        }
        let mut multiplicity = BTreeMap::new();
        for label in &labels {
            *multiplicity.entry(*label).or_insert(0usize) += 1;
        }
        let accepted: Vec<_> = labels
            .iter()
            .map(|label| {
                if graph.names.iter().any(|name| name == label) && multiplicity[label] == 1 {
                    Some((*label).to_string())
                } else {
                    count(
                        &mut report.excluded_labels,
                        if label.is_empty() { "<blank>" } else { label },
                    );
                    None
                }
            })
            .collect();
        let times: Vec<f64> = numbers(&read("t.txt")?)?;
        let events: Vec<usize> = numbers(&read("stim_volume_i.txt")?)?;
        let neurons: Vec<i64> = numbers(&read("stim_neurons.txt")?)?;
        if times.len() < 2
            || times.iter().any(|v| !v.is_finite() || *v < 0.0)
            || times.windows(2).any(|v| v[1] <= v[0])
            || events.len() != neurons.len()
            || events.iter().any(|i| *i >= times.len())
            || events.windows(2).any(|v| v[1] <= v[0])
            || neurons.iter().any(|i| *i >= labels.len() as i64)
        {
            return Err(format!(
                "invalid times/events/dimensions in recording {record}"
            ));
        }
        let mut matrix = Vec::with_capacity(times.len() * labels.len());
        let mut rows = 0;
        for line in matrix_text.lines() {
            let row: Vec<f64> = numbers(line)?;
            if row.len() != labels.len() || row.iter().any(|v| v.is_infinite()) {
                return Err(format!("invalid fluorescence row in recording {record}"));
            }
            matrix.extend(row);
            rows += 1;
        }
        if rows != times.len() {
            return Err("fluorescence/timestamp length mismatch".into());
        }
        let source_name = read("ds_name.txt")?;
        let name = Path::new(source_name.trim().trim_end_matches('/'))
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("missing recording name")?;
        if !recording_names.insert(name.to_string()) {
            return Err("duplicate recording name".into());
        }
        let recording_id = format!("recording:{name}");
        report.source_events += events.len();
        for (event, &frame) in events.iter().enumerate() {
            let onset = times[frame];
            let start = onset - config.baseline_seconds;
            let end = onset + config.response_seconds;
            if start < times[0] || end > times[times.len() - 1] {
                count(&mut report.excluded_events, "incomplete_window");
                continue;
            }
            if (event > 0 && times[events[event - 1]] + config.response_seconds > start + 1e-9)
                || (event + 1 < events.len()
                    && end > times[events[event + 1]] - config.baseline_seconds + 1e-9)
            {
                count(&mut report.excluded_events, "overlapping_source_windows");
                continue;
            }
            let neuron = neurons[event];
            let Some(stimulus) = usize::try_from(neuron)
                .ok()
                .and_then(|i| accepted.get(i))
                .and_then(Option::as_ref)
            else {
                count(&mut report.excluded_events, "unknown_or_ambiguous_stimulus");
                continue;
            };
            let first = times.partition_point(|t| *t < start);
            let stop = times.partition_point(|t| *t < end);
            if first == frame || stop - frame < 2 {
                return Err("insufficient samples in requested window".into());
            }
            let mut traces = vec![];
            for (column, label) in accepted.iter().enumerate() {
                let Some(label) = label else {
                    continue;
                };
                let baseline: Vec<_> = (first..frame)
                    .map(|i| matrix[i * labels.len() + column])
                    .filter(|v| v.is_finite())
                    .collect();
                if baseline.len() as f64 / ((frame - first) as f64)
                    < config.minimum_baseline_fraction
                {
                    count(&mut report.excluded_trace_windows, "missing_baseline");
                    continue;
                }
                let mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
                if !mean.is_finite() || mean <= 1e-12 {
                    count(&mut report.excluded_trace_windows, "nonpositive_baseline");
                    continue;
                }
                let values: Vec<_> = (frame..stop)
                    .map(|i| {
                        let v = (matrix[i * labels.len() + column] - mean) / mean;
                        v.is_finite().then_some(v)
                    })
                    .collect();
                if values.iter().flatten().count() < 2 {
                    count(&mut report.excluded_trace_windows, "missing_response");
                    continue;
                }
                traces.push(Trace {
                    neuron: label.clone(),
                    values,
                    provenance: Provenance {
                        dataset: "Randi2023-processed-wildtype".into(),
                        version: manifest.archive_sha256.clone(),
                        id_confidence: 1.0,
                    },
                });
            }
            if traces.is_empty() {
                count(&mut report.excluded_events, "no_usable_traces");
                continue;
            }
            let id = format!("randi-{record:03}-event-{event:03}");
            report.events.push(EventReceipt {
                trial: id.clone(),
                source_recording: recording_id.clone(),
                event_index: event,
                source_stimulation_frame: frame,
                source_stimulation_seconds: onset,
                source_frames: [first, stop],
                observed_traces: traces.len(),
            });
            data.trials.push(Trial {
                id,
                stimulated_neuron: Some(stimulus.clone()),
                forecast_origin: None,
                response_labels: BTreeMap::new(),
                recording: Recording {
                    dataset: "Randi2023-processed-wildtype".into(),
                    animal_id: recording_id.clone(),
                    condition: "wildtype; stimulation at t=0; pulse duration/power unspecified"
                        .into(),
                    times: times[frame..stop].iter().map(|t| t - onset).collect(),
                    traces,
                    behavior: BTreeMap::new(),
                },
            });
        }
    }
    data.validate(graph)?;
    report.dataset_hash = data.content_hash()?;
    Ok((data, report))
}
