//! Named parameter sharing with explicit annotation and prior boundaries.
use crate::{
    Result,
    math::{Scalar, inverse_softplus},
    model::{Model, Parameters},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sharing {
    /// Explicit full or partial neuron -> class assignments. Unassigned neurons
    /// use annotated graph classes, then exact matching L/R suffix pairs, then self.
    #[serde(default)]
    pub classes: BTreeMap<String, String>,
    pub allow_suffix_pairs: bool,
    pub provenance: String,
}
impl Default for Sharing {
    fn default() -> Self {
        Self{classes:BTreeMap::new(),allow_suffix_pairs:true,provenance:"Graph annotations when available; matching terminal L/R names otherwise. Suffix pairing is an explicit modeling assumption, not a biological annotation.".into()}
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Group {
    pub name: String,
    pub value: f64,
    pub prior_mean: f64,
    pub trainable: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TiedParameters {
    pub groups: Vec<Group>,
    pub raw_to_group: Vec<usize>,
    pub sharing: Sharing,
}
impl TiedParameters {
    pub fn new(model: &Model, initial: &Parameters<f64>, sharing: Sharing) -> Result<Self> {
        model.prepare(initial)?;
        if sharing.provenance.trim().is_empty()
            || sharing.classes.values().any(|c| c.trim().is_empty())
        {
            return Err("sharing requires named classes and provenance".into());
        }
        for name in sharing.classes.keys() {
            model.graph.neuron(name)?;
        }
        let names: BTreeSet<_> = model.graph.names.iter().map(String::as_str).collect();
        let classes: Vec<_> = model
            .graph
            .graph
            .neurons
            .iter()
            .map(|neuron| {
                if let Some(class) = sharing.classes.get(&neuron.id) {
                    return format!("explicit:{class}");
                }
                if !["unannotated", "unknown"].contains(&neuron.class.as_str()) {
                    return format!("annotated:{}", neuron.class);
                }
                if sharing.allow_suffix_pairs
                    && let Some(stem) = neuron
                        .id
                        .strip_suffix('L')
                        .or_else(|| neuron.id.strip_suffix('R'))
                    && names.contains(format!("{stem}L").as_str())
                    && names.contains(format!("{stem}R").as_str())
                {
                    return format!("suffix-pair:{stem}");
                }
                format!("neuron:{}", neuron.id)
            })
            .collect();
        let mut keys = vec![];
        let n = model.n();
        for kind in [
            "tau",
            "rest",
            "threshold",
            "slope",
            "calcium_tau",
            "calcium_scale",
        ] {
            for c in &classes {
                keys.push(format!("{kind}/{c}"));
            }
        }
        for kind in ["chemical_strength", "chemical_sign"] {
            for &(a, b, _, _) in &model.graph.chemical {
                keys.push(format!("{kind}/{}->{}", classes[a], classes[b]));
            }
        }
        for &(a, b, _) in &model.graph.gaps {
            let (x, y) = if classes[a] <= classes[b] {
                (&classes[a], &classes[b])
            } else {
                (&classes[b], &classes[a])
            };
            keys.push(format!("gap_strength/{x}--{y}"));
        }
        keys.push("synapse_tau/global".into());
        let mut grouped: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for (key, &value) in keys.iter().zip(&initial.raw) {
            let e = grouped.entry(key.clone()).or_default();
            e.0 += value;
            e.1 += 1;
        }
        let groups: Vec<_> = grouped
            .iter()
            .map(|(name, (sum, count))| Group {
                name: name.clone(),
                value: sum / *count as f64,
                prior_mean: sum / *count as f64,
                trainable: !name.starts_with("calcium_scale/"),
            })
            .collect();
        let ids: BTreeMap<_, _> = groups
            .iter()
            .enumerate()
            .map(|(i, g)| (g.name.as_str(), i))
            .collect();
        let raw_to_group = keys.iter().map(|key| ids[key.as_str()]).collect();
        assert_eq!(
            keys.len(),
            6 * n + 2 * model.pre.len() + model.gap_a.len() + 1
        );
        Ok(Self {
            groups,
            raw_to_group,
            sharing,
        })
    }
    pub fn expand(&self, model: &Model) -> Result<Parameters<f64>> {
        if self.raw_to_group.len() != model.parameter_count()
            || self.groups.is_empty()
            || self
                .groups
                .iter()
                .any(|g| !g.value.is_finite() || !g.prior_mean.is_finite() || g.name.is_empty())
            || self.raw_to_group.iter().any(|&g| g >= self.groups.len())
        {
            return Err("invalid tied parameter mapping".into());
        }
        Ok(Parameters {
            raw: self
                .raw_to_group
                .iter()
                .map(|&i| self.groups[i].value)
                .collect(),
        })
    }
    pub fn reduce_gradient(&self, raw: &[f64]) -> Result<Vec<f64>> {
        if raw.len() != self.raw_to_group.len()
            || raw.iter().any(|v| !v.is_finite())
            || self.raw_to_group.iter().any(|&g| g >= self.groups.len())
        {
            return Err("invalid raw parameter gradient".into());
        }
        let mut out = vec![0.0; self.groups.len()];
        for (&g, &d) in self.raw_to_group.iter().zip(raw) {
            if self.groups[g].trainable {
                out[g] += d;
            }
        }
        Ok(out)
    }
    pub fn free_parameters(&self) -> usize {
        self.groups.iter().filter(|g| g.trainable).count()
    }
    /// Mean raw-coordinate shrinkage plus mean Bernoulli cross entropy of
    /// relaxed signs against graph priors. Unknown 0.5 stays explicitly neutral.
    pub fn prior(
        &self,
        model: &Model,
        strength: f64,
        sign_strength: f64,
    ) -> Result<(f64, Vec<f64>)> {
        if !strength.is_finite()
            || strength < 0.0
            || !sign_strength.is_finite()
            || sign_strength < 0.0
        {
            return Err("invalid prior strengths".into());
        }
        let raw = self.expand(model)?;
        let mut grad = vec![0.0; self.groups.len()];
        let mut loss = 0.0;
        let count = self.free_parameters().max(1) as f64;
        for (i, g) in self.groups.iter().enumerate() {
            if g.trainable {
                let d = g.value - g.prior_mean;
                loss += strength * d * d / count;
                grad[i] += 2.0 * strength * d / count;
            }
        }
        let n = model.n();
        let m = model.pre.len();
        for (edge, &(_, _, _, prior)) in model.graph.chemical.iter().enumerate() {
            let index = 6 * n + m + edge;
            let q = raw.raw[index];
            let scale = sign_strength / m as f64;
            loss += scale * (q.softplus() - prior * q);
            grad[self.raw_to_group[index]] += scale * (q.sigmoid() - prior);
        }
        Ok((loss, grad))
    }
}
/// Declared population-fit initialization, centered to the affine training readout.
/// Time constants are starting assumptions, not biological estimates.
pub fn forecast_defaults(model: &Model) -> Parameters<f64> {
    let mut p = model.defaults();
    let n = model.n();
    for i in 0..n {
        p.raw[i] = inverse_softplus(2.0);
        p.raw[n + i] = 0.0;
        p.raw[2 * n + i] = 0.0;
        p.raw[4 * n + i] = inverse_softplus(2.0);
    }
    let last = p.raw.len() - 1;
    p.raw[last] = inverse_softplus(0.2);
    p
}
