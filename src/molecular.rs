//! Source-separated molecular evidence. Expression supports qualitative priors,
//! not calibrated probabilities or physiological proof of a synapse's sign.
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub fn content_hash(value: &impl Serialize) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).map_err(|e| e.to_string())?)
    ))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub url: String,
    pub sha256: String,
    pub version: String,
    pub license: String,
}
impl Source {
    fn validate(&self) -> Result<()> {
        if !self.url.starts_with("https://")
            || self.version.trim().is_empty()
            || self.license.trim().is_empty()
            || self.sha256.len() != 64
            || !self.sha256.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err("invalid molecular source provenance".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Polarity {
    Excitatory,
    Inhibitory,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transmitters {
    pub neuron: String,
    pub source_neuron: String,
    pub dominant: Option<String>,
    pub alternative: Option<String>,
    pub source_row: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receptor {
    pub gene: String,
    pub transmitter: String,
    pub polarity: Polarity,
    pub source_cell: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Catalog {
    pub schema_version: u32,
    pub source: Source,
    pub transmitters: Vec<Transmitters>,
    pub receptors: Vec<Receptor>,
}
impl Catalog {
    pub fn validate(&self) -> Result<()> {
        self.source.validate()?;
        let valid_nt = |s: &str| ["Glu", "ACh", "GABA"].contains(&s);
        let mut cells = BTreeSet::new();
        let mut receptors = BTreeSet::new();
        if self.schema_version != 1 || self.transmitters.is_empty() || self.receptors.is_empty() {
            return Err("empty or unsupported molecular catalog".into());
        }
        for t in &self.transmitters {
            if t.neuron.is_empty()
                || t.source_neuron.is_empty()
                || t.source_row < 2
                || !cells.insert(&t.neuron)
                || t.dominant
                    .iter()
                    .chain(&t.alternative)
                    .any(|s| !valid_nt(s))
            {
                return Err("invalid transmitter catalog row".into());
            }
        }
        for r in &self.receptors {
            if r.gene.trim().is_empty()
                || r.source_cell.is_empty()
                || !valid_nt(&r.transmitter)
                || !receptors.insert((
                    &r.gene,
                    &r.transmitter,
                    matches!(r.polarity, Polarity::Excitatory),
                ))
            {
                return Err("invalid receptor catalog row".into());
            }
        }
        for nt in self
            .transmitters
            .iter()
            .flat_map(|t| t.dominant.iter().chain(&t.alternative))
        {
            for polarity in [Polarity::Excitatory, Polarity::Inhibitory] {
                if !self
                    .receptors
                    .iter()
                    .any(|r| &r.transmitter == nt && r.polarity == polarity)
                {
                    return Err("transmitter lacks a complete receptor-polarity catalog".into());
                }
            }
        }
        Ok(())
    }
    pub fn genes(&self) -> BTreeSet<String> {
        self.receptors.iter().map(|r| r.gene.clone()).collect()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gene {
    pub name: String,
    pub wormbase_id: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Expression {
    pub schema_version: u32,
    pub source: Source,
    /// Source threshold category, 1 (liberal) to 4 (stringent), not a TPM cutoff.
    pub threshold: u8,
    pub classes: Vec<String>,
    pub genes: Vec<Gene>,
    /// Row-major [source class][present requested gene], preserving source f32.
    pub tpm: Vec<f32>,
    /// Requested names absent from the source's gene-name column, never zero-filled.
    pub missing_genes: Vec<String>,
}
impl Expression {
    pub fn validate(&self) -> Result<()> {
        self.source.validate()?;
        let unique = |v: &[String]| {
            !v.iter().any(|s| s.trim().is_empty())
                && v.iter().collect::<BTreeSet<_>>().len() == v.len()
        };
        let names: Vec<_> = self.genes.iter().map(|g| g.name.clone()).collect();
        let ids: Vec<_> = self.genes.iter().map(|g| g.wormbase_id.clone()).collect();
        if self.schema_version != 1
            || !(1..=4).contains(&self.threshold)
            || self.classes.is_empty()
            || !unique(&self.classes)
            || !unique(&names)
            || !unique(&ids)
            || !unique(&self.missing_genes)
            || self.missing_genes.iter().any(|g| names.contains(g))
            || self.tpm.len()
                != self
                    .classes
                    .len()
                    .checked_mul(self.genes.len())
                    .ok_or("expression size overflow")?
            || self.tpm.iter().any(|v| !v.is_finite() || *v < 0.)
        {
            return Err("invalid expression table".into());
        }
        Ok(())
    }
}
#[cfg(feature = "hdf5")]
pub fn import_cengen(
    path: &std::path::Path,
    requested: &BTreeSet<String>,
    threshold: u8,
    source: Source,
) -> Result<Expression> {
    source.validate()?;
    if !(1..=4).contains(&threshold)
        || requested.is_empty()
        || requested.iter().any(|s| s.trim().is_empty())
    {
        return Err("invalid expression gene request or threshold".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(bytes)) != source.sha256 {
        return Err("expression source hash mismatch".into());
    }
    let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
    let strings = |name: &str| -> Result<Vec<String>> {
        let d = file.dataset(name).map_err(|e| e.to_string())?;
        if d.ndim() != 1 {
            return Err("expression identifiers must be vectors".into());
        }
        d.read_raw::<hdf5::types::VarLenUnicode>()
            .map(|v| v.iter().map(|s| s.as_str().to_owned()).collect())
            .map_err(|e| e.to_string())
    };
    let classes = strings("neuron_ids")?;
    let names = strings(&format!("gene_names_th{threshold}"))?;
    let ids = strings(&format!("gene_wbids_th{threshold}"))?;
    let dataset = file
        .dataset(&format!("tpm_th{threshold}"))
        .map_err(|e| e.to_string())?;
    if !dataset.dtype().map_err(|e| e.to_string())?.is::<f32>()
        || dataset.shape() != [classes.len(), names.len()]
        || ids.len() != names.len()
        || names.iter().any(|n| n.trim().is_empty())
        || names.iter().collect::<BTreeSet<_>>().len() != names.len()
        || ids.iter().any(|n| n.trim().is_empty())
        || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
    {
        return Err("invalid source expression identifiers or matrix dimensions".into());
    }
    let values = dataset.read_raw::<f32>().map_err(|e| e.to_string())?;
    if values.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err("invalid source TPM".into());
    }
    let indices: BTreeMap<_, _> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let mut genes = vec![];
    let mut columns = vec![];
    let mut missing_genes = vec![];
    for name in requested {
        if let Some(&i) = indices.get(name.as_str()) {
            genes.push(Gene {
                name: name.clone(),
                wormbase_id: ids[i].clone(),
            });
            columns.push(i);
        } else {
            missing_genes.push(name.clone());
        }
    }
    let mut tpm = Vec::with_capacity(classes.len() * columns.len());
    for row in 0..classes.len() {
        for &col in &columns {
            tpm.push(values[row * names.len() + col]);
        }
    }
    let out = Expression {
        schema_version: 1,
        source,
        threshold,
        classes,
        genes,
        tpm,
        missing_genes,
    };
    out.validate()?;
    Ok(out)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellMapping {
    pub schema_version: u32,
    pub graph_hash: String,
    pub provenance: String,
    pub cell_to_class: BTreeMap<String, String>,
    /// Explicit reasons for cells without a transferable expression class.
    pub unmapped: BTreeMap<String, String>,
}
impl CellMapping {
    pub fn validate(&self, graph: &IndexedGraph, expression: &Expression) -> Result<()> {
        if self.schema_version != 1
            || self.graph_hash != graph.hash
            || self.provenance.trim().is_empty()
            || self
                .cell_to_class
                .keys()
                .any(|n| self.unmapped.contains_key(n))
            || self
                .cell_to_class
                .keys()
                .chain(self.unmapped.keys())
                .collect::<BTreeSet<_>>()
                != graph.names.iter().collect::<BTreeSet<_>>()
            || self
                .cell_to_class
                .values()
                .any(|c| !expression.classes.contains(c))
            || self.unmapped.values().any(|r| r.trim().is_empty())
        {
            return Err("invalid or incomplete explicit expression cell mapping".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceState {
    Excitatory,
    Inhibitory,
    Conflicting,
    IncompleteReceptors,
    NoDetectedReceptor,
    NoTransmitterEvidence,
    UnmappedPostsynapticClass,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeEvidence {
    pub pre: String,
    pub post: String,
    pub transmitters: Vec<String>,
    pub expressed_excitatory: Vec<String>,
    pub expressed_inhibitory: Vec<String>,
    pub missing_receptor_genes: Vec<String>,
    pub state: EvidenceState,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    pub schema_version: u32,
    pub graph_hash: String,
    pub catalog_hash: String,
    pub expression_hash: String,
    pub mapping_hash: String,
    pub rule: String,
    pub edges: Vec<EdgeEvidence>,
}
impl Evidence {
    /// Only complete, unopposed directional evidence receives a non-neutral prior.
    /// Confidence is a modeling choice, not inferred or calibrated by this method.
    pub fn probabilities(&self, confidence: f64) -> Result<Vec<f64>> {
        if !confidence.is_finite() || confidence <= 0.5 || confidence >= 1. {
            return Err("directional prior confidence must lie strictly between 0.5 and 1".into());
        }
        Ok(self
            .edges
            .iter()
            .map(|e| match e.state {
                EvidenceState::Excitatory => confidence,
                EvidenceState::Inhibitory => 1. - confidence,
                _ => 0.5,
            })
            .collect())
    }
}
/// Infer evidence for existing directed chemical edges only. Uses dominant and
/// alternative transmitters and retains unknown receptor coverage explicitly.
pub fn infer(
    graph: &IndexedGraph,
    catalog: &Catalog,
    expression: &Expression,
    mapping: &CellMapping,
) -> Result<Evidence> {
    catalog.validate()?;
    expression.validate()?;
    mapping.validate(graph, expression)?;
    let requested = catalog.genes();
    if expression
        .genes
        .iter()
        .map(|g| g.name.clone())
        .chain(expression.missing_genes.iter().cloned())
        .collect::<BTreeSet<_>>()
        != requested
    {
        return Err("expression coverage differs from receptor catalog".into());
    }
    let transmitters: BTreeMap<_, _> = catalog
        .transmitters
        .iter()
        .map(|t| (t.neuron.as_str(), t))
        .collect();
    if graph
        .names
        .iter()
        .any(|n| !transmitters.contains_key(n.as_str()))
    {
        return Err("missing graph neuron in transmitter catalog".into());
    }
    let columns: BTreeMap<_, _> = expression
        .genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.name.as_str(), i))
        .collect();
    let rows: BTreeMap<_, _> = expression
        .classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();
    let mut edges = vec![];
    for edge in &graph.graph.chemical {
        let t = transmitters[edge.pre.as_str()];
        let nt: BTreeSet<_> = t.dominant.iter().chain(&t.alternative).cloned().collect();
        let mut positive = BTreeSet::new();
        let mut negative = BTreeSet::new();
        let mut missing = BTreeSet::new();
        let state = if nt.is_empty() {
            EvidenceState::NoTransmitterEvidence
        } else if let Some(class) = mapping.cell_to_class.get(&edge.post) {
            let row = rows[class.as_str()];
            for receptor in catalog
                .receptors
                .iter()
                .filter(|r| nt.contains(&r.transmitter))
            {
                if let Some(&col) = columns.get(receptor.gene.as_str()) {
                    if expression.tpm[row * expression.genes.len() + col] > 0. {
                        match receptor.polarity {
                            Polarity::Excitatory => {
                                positive.insert(receptor.gene.clone());
                            }
                            Polarity::Inhibitory => {
                                negative.insert(receptor.gene.clone());
                            }
                        }
                    }
                } else {
                    missing.insert(receptor.gene.clone());
                }
            }
            if !positive.is_empty() && !negative.is_empty() {
                EvidenceState::Conflicting
            } else if !missing.is_empty() {
                EvidenceState::IncompleteReceptors
            } else if !positive.is_empty() {
                EvidenceState::Excitatory
            } else if !negative.is_empty() {
                EvidenceState::Inhibitory
            } else {
                EvidenceState::NoDetectedReceptor
            }
        } else {
            EvidenceState::UnmappedPostsynapticClass
        };
        edges.push(EdgeEvidence {
            pre: edge.pre.clone(),
            post: edge.post.clone(),
            transmitters: nt.into_iter().collect(),
            expressed_excitatory: positive.into_iter().collect(),
            expressed_inhibitory: negative.into_iter().collect(),
            missing_receptor_genes: missing.into_iter().collect(),
            state,
        });
    }
    Ok(Evidence { schema_version:1, graph_hash:graph.hash.clone(), catalog_hash:content_hash(catalog)?, expression_hash:content_hash(expression)?, mapping_hash:content_hash(mapping)?, rule:"Dominant plus alternative transmitter; source-threshold TPM > 0 ionotropic receptor evidence. Both polarities => conflict. Any missing candidate => incomplete unless conflict is already demonstrated. No anatomical edges added; no AWC ON/OFF side assignment.".into(), edges })
}

/// Compact projection of an audited evidence artifact into a fitting prior.
/// Edge indices refer to the canonical chemical-edge order of graph_hash.
/// All edges not listed are neutral; full uncertainty reasons remain in evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignPriors {
    pub graph_hash: String,
    pub evidence_hash: String,
    pub catalog_hash: String,
    pub expression_hash: String,
    pub mapping_hash: String,
    /// Declared modeling confidence, not an estimated biological probability.
    pub confidence: f64,
    pub excitatory_edges: Vec<usize>,
    pub inhibitory_edges: Vec<usize>,
}
impl SignPriors {
    pub fn from_evidence(
        evidence: &Evidence,
        graph: &IndexedGraph,
        confidence: f64,
    ) -> Result<Self> {
        if evidence.schema_version != 1
            || evidence.graph_hash != graph.hash
            || evidence.edges.len() != graph.chemical.len()
            || evidence.rule.trim().is_empty()
        {
            return Err("molecular evidence graph or schema mismatch".into());
        }
        let mut excitatory_edges = vec![];
        let mut inhibitory_edges = vec![];
        for (i, (edge, anatomy)) in evidence.edges.iter().zip(&graph.graph.chemical).enumerate() {
            let pos = !edge.expressed_excitatory.is_empty();
            let neg = !edge.expressed_inhibitory.is_empty();
            let missing = !edge.missing_receptor_genes.is_empty();
            let nt = !edge.transmitters.is_empty();
            let coherent = match edge.state {
                EvidenceState::Excitatory => nt && pos && !neg && !missing,
                EvidenceState::Inhibitory => nt && !pos && neg && !missing,
                EvidenceState::Conflicting => nt && pos && neg,
                EvidenceState::IncompleteReceptors => nt && missing && !(pos && neg),
                EvidenceState::NoDetectedReceptor | EvidenceState::UnmappedPostsynapticClass => {
                    nt && !pos && !neg && !missing
                }
                EvidenceState::NoTransmitterEvidence => !nt && !pos && !neg && !missing,
            };
            let invalid_list = [
                &edge.transmitters,
                &edge.expressed_excitatory,
                &edge.expressed_inhibitory,
                &edge.missing_receptor_genes,
            ]
            .iter()
            .any(|v| v.iter().any(|s| s.trim().is_empty()) || v.windows(2).any(|p| p[0] >= p[1]));
            if edge.pre != anatomy.pre
                || edge.post != anatomy.post
                || !coherent
                || invalid_list
                || edge.missing_receptor_genes.iter().any(|g| {
                    edge.expressed_excitatory.contains(g) || edge.expressed_inhibitory.contains(g)
                })
            {
                return Err("inconsistent molecular evidence or chemical-edge ordering".into());
            }
            match edge.state {
                EvidenceState::Excitatory => excitatory_edges.push(i),
                EvidenceState::Inhibitory => inhibitory_edges.push(i),
                _ => {}
            }
        }
        let out = Self {
            graph_hash: graph.hash.clone(),
            evidence_hash: content_hash(evidence)?,
            catalog_hash: evidence.catalog_hash.clone(),
            expression_hash: evidence.expression_hash.clone(),
            mapping_hash: evidence.mapping_hash.clone(),
            confidence,
            excitatory_edges,
            inhibitory_edges,
        };
        out.probabilities(graph)?;
        Ok(out)
    }
    pub fn probabilities(&self, graph: &IndexedGraph) -> Result<Vec<f64>> {
        let valid_hash = |s: &str| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit());
        if self.graph_hash != graph.hash
            || ![
                &self.evidence_hash,
                &self.catalog_hash,
                &self.expression_hash,
                &self.mapping_hash,
            ]
            .iter()
            .all(|s| valid_hash(s))
            || !self.confidence.is_finite()
            || self.confidence <= 0.5
            || self.confidence >= 1.
            || [&self.excitatory_edges, &self.inhibitory_edges]
                .iter()
                .any(|v| {
                    v.iter().any(|&i| i >= graph.chemical.len())
                        || v.windows(2).any(|p| p[0] >= p[1])
                })
            || self
                .excitatory_edges
                .iter()
                .any(|i| self.inhibitory_edges.binary_search(i).is_ok())
        {
            return Err("invalid molecular sign-prior projection".into());
        }
        let mut out = vec![0.5; graph.chemical.len()];
        for &i in &self.excitatory_edges {
            out[i] = self.confidence;
        }
        for &i in &self.inhibitory_edges {
            out[i] = 1. - self.confidence;
        }
        Ok(out)
    }
}
