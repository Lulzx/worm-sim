//! Named parameter sharing with explicit annotation and prior boundaries.
use crate::{
    Result,
    math::{Scalar, inverse_softplus},
    model::{Model, Parameters},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
    /// Initialize each tied sign at logit(mean edge probability), the optimum
    /// of its edge-uniform Bernoulli prior. Preserve ties and all other groups.
    pub fn initialize_sign_priors(&mut self, model: &Model, probabilities: &[f64]) -> Result<()> {
        validate_sign_probabilities(model, probabilities)?;
        if probabilities.iter().any(|&p| p <= 0. || p >= 1.) {
            return Err("sign initialization requires strictly interior probabilities".into());
        }
        self.expand(model)?;
        let start = 6 * model.n() + model.pre.len();
        let mut sums = BTreeMap::<usize, (f64, usize)>::new();
        for (i, &p) in probabilities.iter().enumerate() {
            let group = self.raw_to_group[start + i];
            if !self.groups[group].name.starts_with("chemical_sign/")
                || !self.groups[group].trainable
            {
                return Err("sign-prior initialization requires trainable sign groups".into());
            }
            let entry = sums.entry(group).or_default();
            entry.0 += p;
            entry.1 += 1;
        }
        for (group, (sum, count)) in sums {
            let p = sum / count as f64;
            let value = (p / (1. - p)).ln();
            self.groups[group].value = value;
            self.groups[group].prior_mean = value;
        }
        Ok(())
    }

    /// Draw one initial polarity per tied group, without changing prior centers.
    /// A name-keyed SHA-256 stream makes draws independent of traversal order.
    pub fn initialize_sign_restart(
        &mut self,
        model: &Model,
        probabilities: &[f64],
        seed: u64,
        reversal_magnitude: f64,
    ) -> Result<()> {
        validate_sign_probabilities(model, probabilities)?;
        if !reversal_magnitude.is_finite() || reversal_magnitude <= 0. || reversal_magnitude >= 1. {
            return Err("initial reversal magnitude must be strictly between zero and one".into());
        }
        self.expand(model)?;
        let start = 6 * model.n() + model.pre.len();
        let mut sums = BTreeMap::<usize, (f64, usize)>::new();
        for (i, p) in probabilities.iter().enumerate() {
            let group = self.raw_to_group[start + i];
            if !self.groups[group].trainable
                || !self.groups[group].name.starts_with("chemical_sign/")
            {
                return Err("sign restart requires trainable chemical sign groups".into());
            }
            let entry = sums.entry(group).or_default();
            entry.0 += p;
            entry.1 += 1;
        }
        let magnitude = ((1. + reversal_magnitude) / (1. - reversal_magnitude)).ln();
        for (index, (sum, count)) in sums {
            let group = &mut self.groups[index];
            let mut hash = Sha256::new();
            hash.update(b"wormsim-sign-init-v1\0");
            hash.update(seed.to_le_bytes());
            hash.update(group.name.as_bytes());
            let bytes = hash.finalize();
            let bits = u64::from_le_bytes(bytes[..8].try_into().map_err(|_| "invalid sign hash")?);
            let uniform = (bits >> 11) as f64 / (1_u64 << 53) as f64;
            group.value = if uniform < sum / count as f64 {
                magnitude
            } else {
                -magnitude
            };
        }
        Ok(())
    }
    /// Mean raw-coordinate shrinkage plus mean Bernoulli cross entropy of
    /// relaxed signs against graph priors. Unknown 0.5 stays explicitly neutral.
    pub fn prior(
        &self,
        model: &Model,
        strength: f64,
        sign_strength: f64,
    ) -> Result<(f64, Vec<f64>)> {
        self.prior_with_sign_probabilities(model, strength, sign_strength, None)
    }
    /// Explicit source overlay replaces graph sign priors without changing anatomy.
    pub fn prior_with_sign_probabilities(
        &self,
        model: &Model,
        strength: f64,
        sign_strength: f64,
        probabilities: Option<&[f64]>,
    ) -> Result<(f64, Vec<f64>)> {
        if let Some(p) = probabilities {
            validate_sign_probabilities(model, p)?;
        }
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
            let prior = probabilities.map_or(prior, |p| p[edge]);
            let q = raw.raw[index];
            let scale = sign_strength / m as f64;
            loss += scale * (q.softplus() - prior * q);
            grad[self.raw_to_group[index]] += scale * (q.sigmoid() - prior);
        }
        Ok((loss, grad))
    }
}
fn validate_sign_probabilities(model: &Model, probabilities: &[f64]) -> Result<()> {
    if probabilities.len() != model.pre.len()
        || probabilities
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("invalid chemical sign prior probabilities".into());
    }
    Ok(())
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

/// Signed current weights tied by the same neuron groups as leak-rest parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TiedInputs {
    pub groups: Vec<String>,
    pub neuron_to_group: Vec<usize>,
    pub columns: usize,
    pub weights: Vec<f64>,
}
impl TiedInputs {
    pub fn new(model: &Model, tied: &TiedParameters, columns: usize) -> Result<Self> {
        tied.expand(model)?;
        if columns == 0 || columns > 64 {
            return Err("invalid input width".into());
        }
        let names: Vec<_> = (0..model.n())
            .map(|i| tied.groups[tied.raw_to_group[model.n() + i]].name.clone())
            .collect();
        let groups: Vec<_> = names
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let neuron_to_group = names
            .iter()
            .map(|name| groups.binary_search(name).unwrap())
            .collect();
        let weights = vec![0.0; groups.len() * columns];
        Ok(Self {
            groups,
            neuron_to_group,
            columns,
            weights,
        })
    }
    pub fn validate(&self, model: &Model, tied: &TiedParameters, columns: usize) -> Result<()> {
        let expected = Self::new(model, tied, columns)?;
        if self.groups != expected.groups
            || self.neuron_to_group != expected.neuron_to_group
            || self.columns != columns
            || self.weights.len() != expected.weights.len()
            || self.weights.iter().any(|v| !v.is_finite())
        {
            return Err("invalid tied current weights".into());
        }
        Ok(())
    }
    fn check(&self, features: &[Vec<f64>]) -> Result<()> {
        if self.columns == 0
            || self.columns > 64
            || self.weights.len() != self.groups.len() * self.columns
            || self.weights.iter().any(|v| !v.is_finite())
            || self.neuron_to_group.iter().any(|&g| g >= self.groups.len())
            || features
                .iter()
                .any(|r| r.len() != self.columns || r.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid current projection dimensions/values".into());
        }
        Ok(())
    }
    pub fn currents(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        self.check(features)?;
        let rows: Vec<Vec<f64>> = features
            .iter()
            .map(|u| {
                self.neuron_to_group
                    .iter()
                    .map(|&g| {
                        self.weights[g * self.columns..(g + 1) * self.columns]
                            .iter()
                            .zip(u)
                            .map(|(w, u)| w * u)
                            .sum()
                    })
                    .collect()
            })
            .collect();
        if rows.iter().flatten().any(|v| !v.is_finite()) {
            return Err("current projection overflow".into());
        }
        Ok(rows)
    }
    pub fn reduce_gradient(
        &self,
        features: &[Vec<f64>],
        current_gradients: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        self.check(features)?;
        if features.len() != current_gradients.len()
            || current_gradients
                .iter()
                .any(|r| r.len() != self.neuron_to_group.len() || r.iter().any(|v| !v.is_finite()))
        {
            return Err("current gradient dimensions/values".into());
        }
        let mut out = vec![0.0; self.weights.len()];
        for (u, g) in features.iter().zip(current_gradients) {
            for (i, &row) in self.neuron_to_group.iter().enumerate() {
                for j in 0..self.columns {
                    out[row * self.columns + j] += g[i] * u[j];
                }
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err("current gradient overflow".into());
        }
        Ok(out)
    }
}
