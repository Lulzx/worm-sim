//! c302's Source,Target,Weight,Type CSV importer. Neuron membership must be
//! supplied explicitly: muscle/glia names cannot silently become neurons.
use crate::{Result, data::*};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Serialize)]
pub struct GapConflict {
    pub a: String,
    pub b: String,
    pub forward: f64,
    pub reverse: f64,
    pub resolved: f64,
}
#[derive(Debug, Serialize)]
pub struct ImportReport {
    pub source_sha256: String,
    pub source_rows: usize,
    pub excluded_unmapped_rows: usize,
    pub name_mappings: Vec<NameMapping>,
    pub excluded_self_gap_rows: usize,
    pub excluded_names: BTreeSet<String>,
    pub chemical_edges: usize,
    pub gap_edges: usize,
    pub gap_conflicts: Vec<GapConflict>,
    pub gap_policy: String,
    pub assumptions: Vec<String>,
}
pub fn c302_csv(
    bytes: &[u8],
    names: &[String],
    version: &str,
    mean_mirrors: bool,
) -> Result<(IndexedGraph, ImportReport)> {
    let ids: BTreeSet<_> = names.iter().cloned().collect();
    if ids.len() != names.len() || ids.is_empty() {
        return Err("neuron manifest must be nonempty and unique".into());
    }
    let mut report=ImportReport {source_sha256:format!("{:x}",Sha256::digest(bytes)),source_rows:0,excluded_unmapped_rows:0,name_mappings:vec![],excluded_self_gap_rows:0,excluded_names:BTreeSet::new(),chemical_edges:0,gap_edges:0,gap_conflicts:vec![],gap_policy:if mean_mirrors {"mean of mirrored directed totals"}else{"require equal mirrors"}.into(),assumptions:vec!["Only IDs in the manifest are neurons; explicit numbered-neuron zero-padding aliases are logged; other endpoints are excluded and reported.".into(),"Neuron class, side, and type are unannotated; no class tying is inferred.".into(),"Sign prior 0.5 is unknown, not a measured transmitter assignment.".into(),"ID confidence 1 means exact canonical-name match, not certainty about anatomical measurements.".into()]};
    // Derive only the documented numbered-neuron spelling variant from the
    // explicit manifest. No bilateral/class expansion or heuristic case folding.
    let mut aliases = BTreeMap::new();
    for name in names {
        let split = name
            .find(|c: char| c.is_ascii_digit())
            .unwrap_or(name.len());
        let (prefix, digits) = name.split_at(split);
        if ["AS", "DA", "DB", "DD", "VA", "VB", "VC", "VD"].contains(&prefix)
            && let Ok(number) = digits.parse::<u32>()
        {
            let source = format!("{prefix}{number:02}");
            if source != *name {
                aliases.insert(source, name.clone());
            }
        }
    }
    let mut observed_mappings = BTreeMap::new();
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(bytes);
    if reader
        .headers()
        .map_err(|e| e.to_string())?
        .iter()
        .collect::<Vec<_>>()
        != ["Source", "Target", "Weight", "Type"]
    {
        return Err("expected c302 Source,Target,Weight,Type columns".into());
    }
    let mut chemical = BTreeMap::<(String, String), f64>::new();
    let mut gaps = BTreeMap::<(String, String), (Option<f64>, Option<f64>)>::new();
    for row in reader.records() {
        let row = row.map_err(|e| e.to_string())?;
        report.source_rows += 1;
        let source_a = &row[0];
        let source_b = &row[1];
        let a = aliases
            .get(source_a)
            .map(String::as_str)
            .unwrap_or(source_a);
        let b = aliases
            .get(source_b)
            .map(String::as_str)
            .unwrap_or(source_b);
        for (source, canonical) in [(source_a, a), (source_b, b)] {
            if source != canonical {
                observed_mappings.insert(source.to_owned(), canonical.to_owned());
            }
        }
        let weight: f64 = row[2].parse().map_err(|_| "invalid edge weight")?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err("edge weight must be finite and positive".into());
        }
        if &row[3] != "chemical" && &row[3] != "electrical" {
            return Err(format!("unsupported edge type: {}", &row[3]));
        }
        if !ids.contains(a) || !ids.contains(b) {
            report.excluded_unmapped_rows += 1;
            for name in [a, b] {
                if !ids.contains(name) {
                    report.excluded_names.insert(name.into());
                }
            }
            continue;
        }
        if &row[3] == "chemical" {
            *chemical.entry((a.into(), b.into())).or_default() += weight;
        } else {
            if a == b {
                report.excluded_self_gap_rows += 1;
                continue;
            }
            let forward = a < b;
            let key = if forward {
                (a.into(), b.into())
            } else {
                (b.into(), a.into())
            };
            let pair = gaps.entry(key).or_default();
            let slot = if forward { &mut pair.0 } else { &mut pair.1 };
            *slot = Some(slot.unwrap_or(0.0) + weight);
        }
    }
    let provenance = vec![Provenance {
        dataset: "openworm/c302/herm_full_edgelist".into(),
        version: format!("{version};sha256:{}", report.source_sha256),
        id_confidence: 1.0,
    }];
    let chemical: Vec<_> = chemical
        .into_iter()
        .map(|((pre, post), synapse_count)| ChemicalEdge {
            pre,
            post,
            synapse_count,
            sign_prior: 0.5,
            provenance: provenance.clone(),
            receptor_candidates: vec![],
        })
        .collect();
    let mut gap_edges = Vec::new();
    for ((a, b), (forward, reverse)) in gaps {
        let size = match (forward, reverse) {
            (Some(x), Some(y)) if x != y => {
                if !mean_mirrors {
                    return Err(format!(
                        "conflicting gap mirrors {a}/{b}: {x} versus {y}; enable mean_mirrors to reconcile explicitly"
                    ));
                }
                let resolved = 0.5 * x + 0.5 * y;
                report.gap_conflicts.push(GapConflict {
                    a: a.clone(),
                    b: b.clone(),
                    forward: x,
                    reverse: y,
                    resolved,
                });
                resolved
            }
            (Some(x), _) | (_, Some(x)) => x,
            _ => unreachable!(),
        };
        gap_edges.push(GapEdge {
            a,
            b,
            size,
            provenance: provenance.clone(),
        });
    }
    report.name_mappings = observed_mappings
        .into_iter()
        .map(|(source, canonical)| NameMapping { source, canonical })
        .collect();
    report.chemical_edges = chemical.len();
    report.gap_edges = gap_edges.len();
    let neurons = names
        .iter()
        .map(|id| Neuron {
            id: id.clone(),
            class: "unannotated".into(),
            side: Side::Unknown,
            kind: NeuronType::Unknown,
            neurotransmitters: vec![],
            receptors: vec![],
            channels: vec![],
            peptides_released: vec![],
            peptide_receptors: vec![],
        })
        .collect();
    Ok((
        Graph {
            schema_version: 1,
            neurons,
            chemical,
            gaps: gap_edges,
        }
        .compile()?,
        report,
    ))
}
