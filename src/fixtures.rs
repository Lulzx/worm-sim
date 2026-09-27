//! Synthetic fixtures only: these are not biological connectomes.
use crate::data::*;
pub fn synthetic(n: usize, chemical_per_neuron: usize, gaps_per_neuron: usize) -> Graph {
    let names: Vec<_> = (0..n).map(|i| format!("N{i:03}")).collect();
    let provenance = vec![Provenance {
        dataset: "synthetic".into(),
        version: "1".into(),
        id_confidence: 1.0,
    }];
    let neurons = names
        .iter()
        .map(|id| Neuron {
            id: id.clone(),
            class: id.clone(),
            side: Side::Unpaired,
            kind: NeuronType::Inter,
            neurotransmitters: vec![],
            receptors: vec![],
            channels: vec![],
            peptides_released: vec![],
            peptide_receptors: vec![],
        })
        .collect();
    let mut chemical = Vec::new();
    let mut gaps = Vec::new();
    let mut pairs = std::collections::BTreeSet::new();
    for a in 0..n {
        for d in 1..=chemical_per_neuron.min(n.saturating_sub(1)) {
            let b = (a + d) % n;
            chemical.push(ChemicalEdge {
                pre: names[a].clone(),
                post: names[b].clone(),
                synapse_count: (1 + (a + d) % 5) as f64,
                sign_prior: if a % 4 == 0 { 0.1 } else { 0.9 },
                provenance: provenance.clone(),
                receptor_candidates: vec![],
            });
        }
        for d in 1..=gaps_per_neuron.min(n.saturating_sub(1)) {
            let b = (a + d) % n;
            let pair = (a.min(b), a.max(b));
            if pairs.insert(pair) {
                gaps.push(GapEdge {
                    a: names[pair.0].clone(),
                    b: names[pair.1].clone(),
                    size: 1.0,
                    provenance: provenance.clone(),
                });
            }
        }
    }
    Graph {
        schema_version: 1,
        neurons,
        chemical,
        gaps,
    }
}
