//! Observation-free prediction plans for external atlas dynamics.
use super::{Dataset, Partition, Split, atlas_level0::AtlasModel};
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialPlan {
    pub id: String,
    pub target: usize,
    pub times: Vec<f64>,
    pub neurons: Vec<String>,
    pub response_neurons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionPlan {
    pub schema_version: u32,
    pub graph_hash: String,
    pub dataset_hash: String,
    pub split_hash: String,
    pub partition: Partition,
    pub names: Vec<String>,
    pub chemical_topology: Vec<(usize, usize, f64)>,
    pub gap_topology: Vec<(usize, usize, f64)>,
    pub training_trials: Vec<String>,
    pub selection_trials: Vec<String>,
    pub sample_dt: f64,
    pub trials: Vec<TrialPlan>,
}

impl PredictionPlan {
    pub fn new(
        model: &AtlasModel,
        data: &Dataset,
        graph: &IndexedGraph,
        split: &Split,
        partition: Partition,
    ) -> Result<Self> {
        model.validate(data, graph, split)?;
        if matches!(partition, Partition::Train) || split.ids(partition).is_empty() {
            return Err("external prediction plans require nonempty validation or test".into());
        }
        let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
        let trials = split
            .ids(partition)
            .iter()
            .map(|id| {
                let t = indexed[id];
                let times = &t.recording.times;
                super::atlas_level0::check_grid(times, model.sample_dt)?;
                if t.forecast_origin.is_some() {
                    return Err("atlas prediction plans cannot have forecast origins".into());
                }
                Ok(TrialPlan {
                    id: id.clone(),
                    target: graph
                        .neuron(t.stimulated_neuron.as_deref().ok_or("missing stimulus")?)?,
                    times: times.clone(),
                    neurons: t
                        .recording
                        .traces
                        .iter()
                        .map(|t| t.neuron.clone())
                        .collect(),
                    response_neurons: t.response_labels.keys().cloned().collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            schema_version: 1,
            graph_hash: graph.hash.clone(),
            dataset_hash: split.dataset_hash.clone(),
            split_hash: split.content_hash()?,
            partition,
            names: graph.names.clone(),
            chemical_topology: graph.chemical.iter().map(|e| (e.0, e.1, e.2)).collect(),
            gap_topology: graph.gaps.clone(),
            training_trials: model.training_trials.clone(),
            selection_trials: model.selection_trials.clone(),
            sample_dt: model.sample_dt,
            trials,
        })
    }
}

/// This envelope deliberately cannot be loaded as a native AtlasModel: native
/// simulation does not implement these extensions. Rust validates lineage and
/// scores external predictions; it does not interpret the extension equations.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub format: String,
    pub schema_version: u32,
    pub base_model: AtlasModel,
    pub configuration: serde_json::Value,
    pub extension_parameters: serde_json::Value,
}
impl Checkpoint {
    pub fn validate(&self, data: &Dataset, graph: &IndexedGraph, split: &Split) -> Result<()> {
        if self.format != "wormsim-jax-atlas"
            || self.schema_version != 1
            || !self.configuration.is_object()
            || !self.extension_parameters.is_object()
        {
            return Err("invalid external atlas checkpoint envelope".into());
        }
        self.base_model.validate(data, graph, split)?;
        Ok(())
    }
}
