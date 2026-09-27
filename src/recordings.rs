//! Explicit recording resampling and optional native HDF5 ingestion.
use crate::Result;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowConfig {
    pub sample_dt: f64,
    pub history_seconds: f64,
    pub forecast_seconds: f64,
    pub stride_seconds: f64,
    pub maximum_source_gap: f64,
    /// Ordinal upstream NeuroPAL rating, not a calibrated probability.
    pub minimum_label_rating: f64,
}
impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            sample_dt: 0.5,
            history_seconds: 10.0,
            forecast_seconds: 30.0,
            stride_seconds: 40.0,
            maximum_source_gap: 1.0,
            minimum_label_rating: 3.0,
        }
    }
}
impl WindowConfig {
    pub fn validate(&self) -> Result<()> {
        if [
            self.sample_dt,
            self.history_seconds,
            self.forecast_seconds,
            self.stride_seconds,
            self.maximum_source_gap,
            self.minimum_label_rating,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
            || self.minimum_label_rating > 5.0
            || self.forecast_seconds < 30.0
        {
            return Err("invalid window configuration; forecast must cover 30 s".into());
        }
        for seconds in [self.history_seconds, self.forecast_seconds, 1.0, 10.0, 30.0] {
            let ratio = seconds / self.sample_dt;
            if (ratio - ratio.round()).abs() > 1e-8 {
                return Err("history, horizon and 1/10/30 s must lie on the sample grid".into());
            }
        }
        if (self.history_seconds + self.forecast_seconds) / self.sample_dt > 1e6 {
            return Err("window grid exceeds size limit".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Interpolation {
    pub left: usize,
    pub right: usize,
    pub fraction: f64,
}
pub fn interpolation(
    times: &[f64],
    requested: &[f64],
    maximum_gap: f64,
) -> Result<Vec<Option<Interpolation>>> {
    if times.len() < 2
        || times.iter().any(|t| !t.is_finite())
        || times.windows(2).any(|p| p[1] <= p[0])
        || !maximum_gap.is_finite()
        || maximum_gap <= 0.0
        || requested.iter().any(|t| !t.is_finite())
    {
        return Err("invalid source timestamps or interpolation request".into());
    }
    Ok(requested
        .iter()
        .map(|&t| {
            let right = times.partition_point(|&x| x < t);
            if right < times.len() && times[right] == t {
                return Some(Interpolation {
                    left: right,
                    right,
                    fraction: 0.0,
                });
            }
            if right == 0 || right == times.len() || times[right] - times[right - 1] > maximum_gap {
                return None;
            }
            Some(Interpolation {
                left: right - 1,
                right,
                fraction: (t - times[right - 1]) / (times[right] - times[right - 1]),
            })
        })
        .collect())
}
pub fn resample(values: &[f64], plan: &[Option<Interpolation>]) -> Result<Vec<Option<f64>>> {
    plan.iter()
        .map(|entry| {
            let Some(p) = entry else {
                return Ok(None);
            };
            let a = *values
                .get(p.left)
                .ok_or("interpolation index out of range")?;
            let b = *values
                .get(p.right)
                .ok_or("interpolation index out of range")?;
            if !a.is_finite() || !b.is_finite() {
                return Ok(None);
            }
            let value = (1.0 - p.fraction) * a + p.fraction * b;
            Ok(value.is_finite().then_some(value))
        })
        .collect()
}

#[cfg(feature = "hdf5")]
pub mod wormwideweb {
    use super::*;
    use crate::{
        bench::{Dataset, Trial},
        data::{IndexedGraph, Provenance, Recording, Trace},
    };
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, fs, path::Path};
    #[derive(Debug, Deserialize)]
    struct Labels {
        data: BTreeMap<String, AnimalLabels>,
    }
    #[derive(Debug, Deserialize)]
    struct AnimalLabels {
        #[serde(rename = "idx_neuron-label")]
        neurons: BTreeMap<String, Label>,
    }
    #[derive(Debug, Deserialize)]
    struct Label {
        label: String,
        confidence: f64,
    }
    #[derive(Debug, Deserialize)]
    struct Receipt {
        zenodo_record: String,
        animals: Vec<SourceAnimal>,
    }
    #[derive(Debug, Deserialize)]
    struct SourceAnimal {
        file: String,
        animal_id: String,
        sha256: String,
    }
    #[derive(Debug, Serialize)]
    pub struct LabelDecision {
        pub animal: String,
        pub source_index: usize,
        pub label: String,
        pub rating: f64,
        pub accepted: bool,
        pub reason: String,
    }
    #[derive(Debug, Serialize)]
    pub struct AnimalReport {
        pub animal: String,
        pub source_sha256: String,
        pub source_frames: usize,
        pub source_neurons: usize,
        pub accepted_neurons: usize,
        pub windows: usize,
        pub windows_skipped_for_gaps_or_no_labels: usize,
    }
    #[derive(Debug, Serialize)]
    pub struct ImportReport {
        pub schema_version: u32,
        pub graph_hash: String,
        pub dataset_hash: String,
        pub labels_sha256: String,
        pub config: WindowConfig,
        pub animals: Vec<AnimalReport>,
        pub labels: Vec<LabelDecision>,
        pub confidence_mapping: String,
        pub preprocessing: String,
    }
    fn read(file: &hdf5::File, path: &str) -> Result<(Vec<usize>, Vec<f64>)> {
        let dataset = file.dataset(path).map_err(|e| format!("{path}: {e}"))?;
        if dataset.size() > 100_000_000 {
            return Err(format!("oversized HDF5 dataset: {path}"));
        }
        Ok((
            dataset.shape(),
            dataset
                .read_raw::<f64>()
                .map_err(|e| format!("{path}: {e}"))?,
        ))
    }
    pub fn import(
        directory: &Path,
        labels_path: &Path,
        receipt_path: &Path,
        graph: &IndexedGraph,
        config: &WindowConfig,
    ) -> Result<(Dataset, ImportReport)> {
        config.validate()?;
        let label_bytes = fs::read(labels_path).map_err(|e| e.to_string())?;
        let labels: Labels = serde_json::from_slice(&label_bytes).map_err(|e| e.to_string())?;
        let receipt: Receipt =
            serde_json::from_slice(&fs::read(receipt_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let label_hash = format!("{:x}", Sha256::digest(&label_bytes));
        let preprocessing = "Published gcamp/trace_array unchanged in scale; forecast origin aligned to a real source frame; linear interpolation to a fixed relative grid using no post-origin sample for observed history; no extrapolation, interpolation across gaps, z-scoring, smoothing or fitting; exact canonical labels only; unknown/ambiguous/duplicate labels excluded and logged.";
        let mut data = Dataset {
            schema_version: 1,
            name: "atanas-kim-2023-labeled-baseline".into(),
            graph_hash: graph.hash.clone(),
            source: format!(
                "https://doi.org/10.1016/j.cell.2023.07.035; Zenodo {}; labels SHA256 {}; {}; window config {}",
                receipt.zenodo_record,
                label_hash,
                preprocessing,
                serde_json::to_string(config).map_err(|e| e.to_string())?
            ),
            trials: vec![],
        };
        let mut report=ImportReport {schema_version:1,graph_hash:graph.hash.clone(),dataset_hash:String::new(),labels_sha256:label_hash,config:config.clone(),animals:vec![],labels:vec![],confidence_mapping:"Upstream ordinal confidence / 5 used as an explicit heuristic weight, not a calibrated identity probability.".into(),preprocessing:preprocessing.into()};
        for animal in receipt.animals {
            // Receipt paths are filenames, never arbitrary filesystem paths.
            if Path::new(&animal.file).file_name().and_then(|n| n.to_str())
                != Some(animal.file.as_str())
            {
                return Err("receipt contains a nonlocal filename".into());
            }
            let path = directory.join(&animal.file);
            let mut input = fs::File::open(&path).map_err(|e| e.to_string())?;
            let mut digest = Sha256::new();
            std::io::copy(&mut input, &mut digest).map_err(|e| e.to_string())?;
            if format!("{:x}", digest.finalize()) != animal.sha256 {
                return Err(format!("HDF5 checksum mismatch: {}", animal.animal_id));
            }
            let file = hdf5::File::open(&path).map_err(|e| e.to_string())?;
            let (_, times) = read(&file, "timing/timestamp_confocal")?;
            let (shape, raw) = read(&file, "gcamp/trace_array")?;
            if shape.len() != 2 || (shape[0] == times.len()) == (shape[1] == times.len()) {
                return Err("trace shape must have one unambiguous time axis".into());
            }
            let time_first = shape[0] == times.len();
            let neurons = if time_first { shape[1] } else { shape[0] };
            // Validate even recordings too short to yield a complete window.
            interpolation(&times, &[], config.maximum_source_gap)?;
            let animal_labels = &labels
                .data
                .get(&animal.animal_id)
                .ok_or("missing animal labels")?
                .neurons;
            let mut multiplicity = BTreeMap::new();
            for label in animal_labels.values() {
                *multiplicity.entry(&label.label).or_insert(0) += 1;
            }
            let mut selected = vec![];
            for (index, label) in animal_labels {
                let source_index = index.parse::<usize>().map_err(|_| "invalid ROI index")?;
                if source_index == 0
                    || source_index > neurons
                    || !label.confidence.is_finite()
                    || !(0.0..=5.0).contains(&label.confidence)
                {
                    return Err("label index/rating outside expected source range".into());
                }
                let reason = if graph.neuron(&label.label).is_err() {
                    "not an exact canonical neuron"
                } else if multiplicity[&label.label] > 1 {
                    "duplicate canonical identity"
                } else if label.confidence < config.minimum_label_rating {
                    "below ordinal rating threshold"
                } else {
                    "accepted"
                };
                report.labels.push(LabelDecision {
                    animal: animal.animal_id.clone(),
                    source_index,
                    label: label.label.clone(),
                    rating: label.confidence,
                    accepted: reason == "accepted",
                    reason: reason.into(),
                });
                if reason == "accepted" {
                    let roi = source_index - 1;
                    let values: Vec<_> = (0..times.len())
                        .map(|t| {
                            raw[if time_first {
                                t * neurons + roi
                            } else {
                                roi * times.len() + t
                            }]
                        })
                        .collect();
                    selected.push((label.label.clone(), label.confidence / 5.0, values));
                }
            }
            selected.sort_by(|a, b| a.0.cmp(&b.0));
            let mut behavior = BTreeMap::new();
            for name in ["velocity", "head_angle", "angular_velocity", "pumping"] {
                let path = format!("behavior/{name}");
                if file.link_exists(&path) {
                    let (_, values) = read(&file, &path)?;
                    if values.len() != times.len() {
                        return Err(format!(
                            "behavior channel {name} does not match confocal grid"
                        ));
                    }
                    behavior.insert(name.to_string(), values);
                }
            }
            let duration = config.history_seconds + config.forecast_seconds;
            let frames = (duration / config.sample_dt).round() as usize + 1;
            let relative: Vec<_> = (0..frames).map(|i| i as f64 * config.sample_dt).collect();
            let mut windows = 0;
            let mut skipped = 0;
            let mut window = 0;
            loop {
                let desired_origin =
                    times[0] + window as f64 * config.stride_seconds + config.history_seconds;
                let origin_index = times.partition_point(|&t| t < desired_origin);
                if origin_index == times.len() {
                    break;
                }
                let origin = times[origin_index];
                let start = origin - config.history_seconds;
                if origin + config.forecast_seconds > *times.last().unwrap() {
                    break;
                }
                // Align the observed/forecast boundary to an actual source frame.
                // Otherwise interpolation of the last history point could read
                // a frame after the nominal forecast origin.
                let requested: Vec<_> = relative
                    .iter()
                    .map(|t| origin + (*t - config.history_seconds))
                    .collect();
                let plan = interpolation(&times, &requested, config.maximum_source_gap)?;
                // A window spanning a recording discontinuity is not a forecast.
                let source_gap = times.windows(2).any(|pair| {
                    pair[0] < start + duration
                        && pair[1] > start
                        && pair[1] - pair[0] > config.maximum_source_gap
                });
                if plan.iter().any(Option::is_none) || source_gap || selected.is_empty() {
                    skipped += 1;
                    window += 1;
                    continue;
                }
                let traces = selected
                    .iter()
                    .map(|(name, confidence, values)| {
                        Ok(Trace {
                            neuron: name.clone(),
                            values: resample(values, &plan)?,
                            provenance: Provenance {
                                dataset: "atanas-kim-2023".into(),
                                version: animal.sha256.clone(),
                                id_confidence: *confidence,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let behavior = behavior
                    .iter()
                    .map(|(name, values)| Ok((name.clone(), resample(values, &plan)?)))
                    .collect::<Result<_>>()?;
                data.trials.push(Trial {
                    id: format!("{}-window-{window:04}", animal.animal_id),
                    stimulated_neuron: None,
                    forecast_origin: Some(config.history_seconds),
                    recording: Recording {
                        dataset: "atanas-kim-2023".into(),
                        animal_id: animal.animal_id.clone(),
                        condition: format!("baseline; source start={start:.9} s"),
                        times: relative.clone(),
                        traces,
                        behavior,
                    },
                    response_labels: BTreeMap::new(),
                });
                windows += 1;
                window += 1;
            }
            report.animals.push(AnimalReport {
                animal: animal.animal_id,
                source_sha256: animal.sha256,
                source_frames: times.len(),
                source_neurons: neurons,
                accepted_neurons: selected.len(),
                windows,
                windows_skipped_for_gaps_or_no_labels: skipped,
            });
        }
        data.validate(graph)?;
        report.dataset_hash = data.content_hash()?;
        Ok((data, report))
    }
}

pub mod randi;
