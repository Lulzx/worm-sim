//! Pair-level atlas classification evidence, separate from individual response traces.
use super::{Axis, Dataset, Partition, Split, declared_subset, metrics};
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pair {
    pub stimulated: String,
    pub responding: String,
    pub q: f64,
    pub equivalence_q: Option<f64>,
    pub observations: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub graph_hash: String,
    pub source_sha256: String,
    pub source_version: String,
    pub equivalence_threshold: f64,
    pub detection_q_threshold: f64,
    pub pairs: Vec<Pair>,
}
impl Evidence {
    pub fn content_hash(&self) -> Result<String> {
        super::hash(self)
    }
    pub fn validate(&self, data: &Dataset, graph: &IndexedGraph) -> Result<()> {
        data.validate(graph)?;
        if self.schema_version != 1
            || self.dataset_hash != data.content_hash()?
            || self.graph_hash != graph.hash
            || self.source_sha256.len() != 64
            || self.source_version.is_empty()
            || !self.detection_q_threshold.is_finite()
            || !(0.0..1.0).contains(&self.detection_q_threshold)
            || self.detection_q_threshold == 0.0
            || !self.equivalence_threshold.is_finite()
            || self.equivalence_threshold <= 0.0
            || self.pairs.is_empty()
        {
            return Err("invalid atlas pair evidence metadata".into());
        }
        let observed = observed_pairs(data)?;
        let mut seen = BTreeSet::new();
        for p in &self.pairs {
            if p.observations == 0
                || !p.q.is_finite()
                || !(0.0..=1.0).contains(&p.q)
                || p.equivalence_q
                    .is_some_and(|q| !q.is_finite() || !(0.0..=1.0).contains(&q))
                || !observed.contains(&(p.stimulated.clone(), p.responding.clone()))
                || !seen.insert((&p.stimulated, &p.responding))
            {
                return Err("invalid, duplicate or unobserved atlas pair".into());
            }
        }
        Ok(())
    }
    pub fn partition_pairs<'a>(
        &'a self,
        data: &Dataset,
        split: &Split,
        partition: Partition,
    ) -> Result<Vec<&'a Pair>> {
        if split.axis != Axis::StimulatedNeuron {
            return Err("atlas classification requires a stimulated-neuron split".into());
        }
        let targets: BTreeSet<_> = data
            .trials
            .iter()
            .filter(|t| split.ids(partition).contains(&t.id))
            .filter_map(|t| t.stimulated_neuron.as_ref())
            .collect();
        Ok(self
            .pairs
            .iter()
            .filter(|p| targets.contains(&p.stimulated))
            .collect())
    }
}
fn observed_pairs(data: &Dataset) -> Result<BTreeSet<(String, String)>> {
    let mut pairs = BTreeSet::new();
    for trial in &data.trials {
        let stim = trial
            .stimulated_neuron
            .as_ref()
            .ok_or("atlas trial lacks stimulated identity")?;
        for trace in &trial.recording.traces {
            if &trace.neuron != stim {
                pairs.insert((stim.clone(), trace.neuron.clone()));
            }
        }
    }
    Ok(pairs)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prediction {
    pub stimulated: String,
    pub responding: String,
    /// Arbitrary finite ranking score, not necessarily a calibrated probability.
    pub score: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Predictions {
    pub evidence_hash: String,
    pub split_hash: String,
    pub model: String,
    pub free_parameters: usize,
    pub source_commit: String,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
    pub pairs: Vec<Prediction>,
}
/// Fixed absolute-response-area ranking from a complete common trace prediction.
/// Requires identical uniform grids and identical per-pair areas across trials;
/// no response labels are used to choose a ranking function or fit coefficients.
pub fn rank_responses(
    evidence: &Evidence,
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    prediction: &super::Predictions,
    partition: Partition,
) -> Result<Predictions> {
    // Validates complete coverage, lineage, finite values and recording grids.
    super::evaluate(data, graph, split, prediction, partition)?;
    evidence.validate(data, graph)?;
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let grid = &prediction.trials.first().ok_or("empty partition")?.times;
    if grid.len() < 2 {
        return Err("pair ranking requires at least two frames".into());
    }
    let dt = grid[1] - grid[0];
    if !dt.is_finite() || dt <= 0.0 || grid.windows(2).any(|p| (p[1] - p[0] - dt).abs() > 1e-10) {
        return Err("pair ranking requires a uniform grid".into());
    }
    let mut scores = BTreeMap::new();
    for trial in &prediction.trials {
        if &trial.times != grid {
            return Err("pair ranking requires the same response grid for all trials".into());
        }
        let target = indexed[&trial.id]
            .stimulated_neuron
            .as_ref()
            .ok_or("missing stimulus")?;
        for (neuron, values) in &trial.fluorescence {
            let value = values.iter().map(|v| v.abs()).sum::<f64>() * dt;
            if !value.is_finite() {
                return Err("nonfinite response area".into());
            }
            if let Some(previous) = scores.insert((target.clone(), neuron.clone()), value)
                && previous != value
            {
                return Err("inconsistent impulse scores across trials".into());
            }
        }
    }
    let pairs = evidence
        .partition_pairs(data, split, partition)?
        .iter()
        .map(|p| {
            Ok(Prediction {
                stimulated: p.stimulated.clone(),
                responding: p.responding.clone(),
                score: *scores
                    .get(&(p.stimulated.clone(), p.responding.clone()))
                    .ok_or("missing pair score")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Predictions {
        evidence_hash: evidence.content_hash()?,
        split_hash: prediction.split_hash.clone(),
        model: prediction.model.clone(),
        free_parameters: prediction.free_parameters,
        source_commit: prediction.source_commit.clone(),
        training_trials: prediction.training_trials.clone(),
        selection_trials: prediction.selection_trials.clone(),
        pairs,
    })
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub evidence_hash: String,
    pub split_hash: String,
    pub model: String,
    pub free_parameters: usize,
    pub prediction_source_commit: String,
    pub partition: Partition,
    pub pairs: usize,
    pub detected: usize,
    pub not_detected: usize,
    pub detected_and_equivalent: usize,
    pub auroc: metrics::Auroc,
    pub interpretation: String,
}
pub fn evaluate(
    evidence: &Evidence,
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    prediction: &Predictions,
    partition: Partition,
) -> Result<Report> {
    split.validate(data, graph)?;
    evidence.validate(data, graph)?;
    if prediction.evidence_hash != evidence.content_hash()?
        || prediction.split_hash != split.content_hash()?
        || prediction.model.is_empty()
        || prediction.source_commit.is_empty()
        || !declared_subset(&prediction.training_trials, &split.train)
        || !declared_subset(&prediction.selection_trials, &split.validation)
    {
        return Err("invalid atlas prediction lineage".into());
    }
    let expected = evidence.partition_pairs(data, split, partition)?;
    let mut scores = BTreeMap::new();
    for p in &prediction.pairs {
        if !p.score.is_finite()
            || scores
                .insert((&p.stimulated, &p.responding), p.score)
                .is_some()
        {
            return Err("invalid or duplicate pair score".into());
        }
    }
    if scores.len() != expected.len() {
        return Err("missing or surplus atlas pair predictions".into());
    }
    let mut observations = vec![];
    let mut both = 0;
    for p in expected {
        let score = *scores
            .get(&(&p.stimulated, &p.responding))
            .ok_or("missing atlas pair prediction")?;
        let label = p.q < evidence.detection_q_threshold;
        both += usize::from(
            label
                && p.equivalence_q
                    .is_some_and(|q| q < evidence.detection_q_threshold),
        );
        observations.push((label, score, 1.0));
    }
    let detected = observations.iter().filter(|p| p.0).count();
    Ok(Report { evidence_hash:prediction.evidence_hash.clone(),split_hash:prediction.split_hash.clone(),model:prediction.model.clone(),free_parameters:prediction.free_parameters,prediction_source_commit:prediction.source_commit.clone(),partition,pairs:observations.len(),detected,not_detected:observations.len()-detected,detected_and_equivalent:both,auroc:metrics::auroc(&observations)?,interpretation:"One unit-weighted score per ordered non-self pair. Positive means published detection q below the declared threshold; negative means not detected, not proven absence. Missing q is excluded. Equivalence evidence is retained separately. Lineage declarations are checked but do not prove training information access.".into() })
}

#[cfg(feature = "hdf5")]
pub fn import_hdf5(
    path: &std::path::Path,
    data: &Dataset,
    graph: &IndexedGraph,
    expected_sha256: &str,
) -> Result<Evidence> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != expected_sha256 {
        return Err("atlas source hash mismatch".into());
    }
    let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
    let ids: Vec<_> = file
        .dataset("neuron_ids")
        .and_then(|d| d.read_raw::<hdf5::types::FixedAscii<5>>())
        .map_err(|e| e.to_string())?
        .iter()
        .map(|s| s.as_str().to_string())
        .collect();
    if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err("duplicate atlas identity".into());
    }
    let n = ids.len();
    let matrix = |name: &str| -> Result<Vec<f64>> {
        let d = file.dataset(name).map_err(|e| e.to_string())?;
        if d.shape() != [n, n] {
            return Err("invalid atlas matrix shape".into());
        }
        d.read_raw::<f64>().map_err(|e| e.to_string())
    };
    let q = matrix("wt/q")?;
    let eq = matrix("wt/q_eq")?;
    let counts = matrix("wt/occ1")?;
    let observed = observed_pairs(data)?;
    let mut pairs = vec![];
    // Published matrices use responding neuron as row, stimulated neuron as column.
    for (i, responding) in ids.iter().enumerate() {
        for (j, stimulated) in ids.iter().enumerate() {
            let at = i * n + j;
            if !observed.contains(&(stimulated.clone(), responding.clone())) || q[at].is_nan() {
                continue;
            }
            if !counts[at].is_finite()
                || counts[at] <= 0.0
                || counts[at].fract() != 0.0
                || counts[at] > usize::MAX as f64
            {
                return Err("invalid atlas observation count".into());
            }
            pairs.push(Pair {
                stimulated: stimulated.clone(),
                responding: responding.clone(),
                q: q[at],
                equivalence_q: (!eq[at].is_nan()).then_some(eq[at]),
                observations: counts[at] as usize,
            });
        }
    }
    pairs.sort_by(|a, b| (&a.stimulated, &a.responding).cmp(&(&b.stimulated, &b.responding)));
    let evidence = Evidence {
        schema_version: 1,
        dataset_hash: data.content_hash()?,
        graph_hash: graph.hash.clone(),
        source_sha256: digest,
        source_version: file
            .attr("time_compiled")
            .and_then(|a| a.read_scalar::<hdf5::types::FixedAscii<19>>())
            .map_err(|e| e.to_string())?
            .as_str()
            .into(),
        equivalence_threshold: file
            .dataset("wt/q_eq_th")
            .and_then(|d| d.read_scalar::<f64>())
            .map_err(|e| e.to_string())?,
        detection_q_threshold: 0.05,
        pairs,
    };
    evidence.validate(data, graph)?;
    Ok(evidence)
}
