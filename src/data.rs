//! Canonical graph interchange and deterministic, content-addressed indexing.
use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub dataset: String,
    pub version: String,
    pub id_confidence: f64,
}
impl Provenance {
    fn validate(&self) -> Result<()> {
        if self.dataset.trim().is_empty()
            || self.version.trim().is_empty()
            || !probability(self.id_confidence)
        {
            return Err("provenance requires dataset, version and confidence in [0,1]".into());
        }
        Ok(())
    }
}
fn probability(x: f64) -> bool {
    x.is_finite() && (0.0..=1.0).contains(&x)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronType {
    Unknown,
    Sensory,
    Inter,
    Motor,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Unknown,
    Left,
    Right,
    Unpaired,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Neuron {
    pub id: String,
    pub class: String,
    pub side: Side,
    pub kind: NeuronType,
    #[serde(default)]
    pub neurotransmitters: Vec<String>,
    #[serde(default)]
    pub receptors: Vec<String>,
    #[serde(default)]
    pub channels: Vec<String>,
    #[serde(default)]
    pub peptides_released: Vec<String>,
    #[serde(default)]
    pub peptide_receptors: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChemicalEdge {
    pub pre: String,
    pub post: String,
    pub synapse_count: f64,
    pub sign_prior: f64,
    pub provenance: Vec<Provenance>,
    #[serde(default)]
    pub receptor_candidates: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GapEdge {
    pub a: String,
    pub b: String,
    pub size: f64,
    pub provenance: Vec<Provenance>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub schema_version: u32,
    pub neurons: Vec<Neuron>,
    pub chemical: Vec<ChemicalEdge>,
    pub gaps: Vec<GapEdge>,
}
#[derive(Clone, Debug)]
pub struct IndexedGraph {
    pub graph: Graph,
    pub hash: String,
    pub names: Vec<String>,
    pub chemical: Vec<(usize, usize, f64, f64)>,
    pub gaps: Vec<(usize, usize, f64)>,
}
impl Graph {
    pub fn compile(mut self) -> Result<IndexedGraph> {
        if self.schema_version != 1 || self.neurons.is_empty() {
            return Err("expected schema_version 1 and nonempty neurons".into());
        }
        self.neurons.sort_by(|a, b| a.id.cmp(&b.id));
        let mut ids = BTreeMap::new();
        for (i, n) in self.neurons.iter().enumerate() {
            if n.id.is_empty()
                || n.class.is_empty()
                || n.id.trim() != n.id
                || n.id.to_uppercase() != n.id
                || ids.insert(n.id.clone(), i).is_some()
            {
                return Err(format!(
                    "invalid or duplicate canonical neuron ID: {}",
                    n.id
                ));
            }
        }
        let index = |name: &str| {
            ids.get(name)
                .copied()
                .ok_or_else(|| format!("unknown neuron: {name}"))
        };
        self.chemical
            .sort_by(|a, b| (&a.pre, &a.post).cmp(&(&b.pre, &b.post)));
        let mut seen = BTreeSet::new();
        let mut chemical = Vec::new();
        for e in &self.chemical {
            if !e.synapse_count.is_finite()
                || e.synapse_count <= 0.0
                || !probability(e.sign_prior)
                || !seen.insert((&e.pre, &e.post))
            {
                return Err("invalid or duplicate chemical edge".into());
            }
            validate_sources(&e.provenance)?;
            chemical.push((
                index(&e.pre)?,
                index(&e.post)?,
                e.synapse_count,
                e.sign_prior,
            ));
        }
        for e in &mut self.gaps {
            if e.a > e.b {
                std::mem::swap(&mut e.a, &mut e.b);
            }
        }
        self.gaps.sort_by(|a, b| (&a.a, &a.b).cmp(&(&b.a, &b.b)));
        let mut seen = BTreeSet::new();
        let mut gaps = Vec::new();
        for e in &self.gaps {
            if e.a == e.b || !e.size.is_finite() || e.size <= 0.0 || !seen.insert((&e.a, &e.b)) {
                return Err("invalid or duplicate gap edge".into());
            }
            validate_sources(&e.provenance)?;
            gaps.push((index(&e.a)?, index(&e.b)?, e.size));
        }
        let hash = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&self).map_err(|e| e.to_string())?)
        );
        Ok(IndexedGraph {
            names: self.neurons.iter().map(|n| n.id.clone()).collect(),
            graph: self,
            hash,
            chemical,
            gaps,
        })
    }
}
fn validate_sources(sources: &[Provenance]) -> Result<()> {
    if sources.is_empty() {
        return Err("edge provenance is required".into());
    }
    for p in sources {
        p.validate()?;
    }
    Ok(())
}
impl IndexedGraph {
    pub fn neuron(&self, name: &str) -> Result<usize> {
        self.names
            .binary_search_by(|n| n.as_str().cmp(name))
            .map_err(|_| format!("unknown neuron: {name}; supply a canonical neuron ID"))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NameMapping {
    pub source: String,
    pub canonical: String,
}
/// Reconcile explicitly declared aliases; never guess whether a class means L or R.
pub fn reconcile(
    names: &[String],
    aliases: &BTreeMap<String, String>,
    graph: &IndexedGraph,
) -> Result<Vec<NameMapping>> {
    names
        .iter()
        .map(|name| {
            let canonical = aliases.get(name).unwrap_or(name);
            graph.neuron(canonical)?;
            Ok(NameMapping {
                source: name.clone(),
                canonical: canonical.clone(),
            })
        })
        .collect()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub neuron: String,
    pub values: Vec<Option<f64>>,
    pub provenance: Provenance,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recording {
    pub dataset: String,
    pub animal_id: String,
    pub condition: String,
    pub times: Vec<f64>,
    pub traces: Vec<Trace>,
    #[serde(default)]
    pub behavior: BTreeMap<String, Vec<Option<f64>>>,
}
impl Recording {
    pub fn validate(&self, graph: &IndexedGraph) -> Result<()> {
        if self.dataset.is_empty()
            || self.animal_id.is_empty()
            || self.times.is_empty()
            || self.times.iter().any(|t| !t.is_finite() || *t < 0.0)
            || self.times.windows(2).any(|t| t[1] <= t[0])
        {
            return Err("invalid recording metadata or times".into());
        }
        if self.behavior.iter().any(|(name, values)| {
            name.is_empty()
                || values.len() != self.times.len()
                || values.iter().flatten().any(|value| !value.is_finite())
        }) {
            return Err("invalid behavior channel name, length or sample".into());
        }
        let mut seen = BTreeSet::new();
        for trace in &self.traces {
            graph.neuron(&trace.neuron)?;
            trace.provenance.validate()?;
            if trace.values.len() != self.times.len()
                || !seen.insert(&trace.neuron)
                || trace.values.iter().flatten().any(|v| !v.is_finite())
            {
                return Err("invalid trace length, duplicate neuron or nonfinite sample".into());
            }
        }
        Ok(())
    }
}
