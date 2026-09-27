//! Training-only ordered-pair labels and a differentiable response-detection loss.
use super::{Dataset, Partition, Split, atlas};
use crate::{Result, data::IndexedGraph, math::Scalar};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Classifier {
    pub bias: f64,
    /// Positive slope softplus(raw_slope). No sign inversion is permitted.
    pub raw_slope: f64,
    /// Fixed area unit for log1p; not a measured detection threshold.
    pub area_scale: f64,
    /// Smooth absolute value sqrt(f²+epsilon²)-epsilon.
    pub epsilon: f64,
}
#[derive(Clone, Debug)]
pub struct TrainingLabels {
    pub evidence_hash: String,
    pub by_target: BTreeMap<usize, Vec<(usize, bool)>>,
    pub pairs: usize,
    pub detected: usize,
}
pub fn training_labels(
    evidence: &atlas::Evidence,
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
) -> Result<TrainingLabels> {
    split.validate(data, graph)?;
    evidence.validate(data, graph)?;
    let pairs = evidence.partition_pairs(data, split, Partition::Train)?;
    if pairs.is_empty() {
        return Err("no training atlas labels".into());
    }
    let mut by_target: BTreeMap<usize, Vec<_>> = BTreeMap::new();
    let mut detected = 0;
    for pair in &pairs {
        let label = pair.q < evidence.detection_q_threshold;
        detected += usize::from(label);
        by_target
            .entry(graph.neuron(&pair.stimulated)?)
            .or_default()
            .push((graph.neuron(&pair.responding)?, label));
    }
    for values in by_target.values_mut() {
        values.sort_by_key(|p| p.0);
    }
    Ok(TrainingLabels {
        evidence_hash: evidence.content_hash()?,
        by_target,
        pairs: pairs.len(),
        detected,
    })
}
#[derive(Debug)]
pub struct Loss {
    /// Sum over labels, not mean. Caller normalizes once by all training pairs.
    pub value: f64,
    pub fluorescence: Vec<Vec<f64>>,
    pub bias_gradient: f64,
    pub raw_slope_gradient: f64,
}
impl Classifier {
    pub fn validate(&self) -> Result<()> {
        if !self.bias.is_finite()
            || !self.raw_slope.is_finite()
            || !self.area_scale.is_finite()
            || self.area_scale <= 0.
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.
        {
            return Err("invalid atlas classifier".into());
        }
        Ok(())
    }
    /// Stable Bernoulli cross entropy on a smooth absolute-response area.
    /// Every published ordered pair occurs once, regardless of trial count.
    pub fn loss(&self, response: &[Vec<f64>], labels: &[(usize, bool)], dt: f64) -> Result<Loss> {
        self.validate()?;
        let n = response.first().ok_or("empty response")?.len();
        if n == 0
            || response.len() < 2
            || !dt.is_finite()
            || dt <= 0.
            || response
                .iter()
                .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
            || labels.iter().any(|p| p.0 >= n)
            || labels.iter().map(|p| p.0).collect::<BTreeSet<_>>().len() != labels.len()
        {
            return Err("invalid response classification inputs".into());
        }
        let mut out = Loss {
            value: 0.,
            fluorescence: vec![vec![0.; n]; response.len()],
            bias_gradient: 0.,
            raw_slope_gradient: 0.,
        };
        let slope = self.raw_slope.softplus();
        for &(i, label) in labels {
            // Rationalized form avoids cancellation of sqrt(x²+eps²)-eps near zero.
            let area = dt
                * response
                    .iter()
                    .map(|r| {
                        let h = r[i].hypot(self.epsilon);
                        r[i] * (r[i] / (h + self.epsilon))
                    })
                    .sum::<f64>();
            let feature = (area / self.area_scale).ln_1p();
            let logit = self.bias + slope * feature;
            let y = f64::from(label);
            out.value += if label {
                (-logit).softplus()
            } else {
                logit.softplus()
            };
            let residual = logit.sigmoid() - y;
            out.bias_gradient += residual;
            out.raw_slope_gradient += residual * feature * self.raw_slope.sigmoid();
            let multiplier = residual * slope * dt / (self.area_scale + area);
            for (row, gradient) in response.iter().zip(&mut out.fluorescence) {
                gradient[i] += multiplier * row[i] / row[i].hypot(self.epsilon);
            }
        }
        if !out.value.is_finite()
            || !out.bias_gradient.is_finite()
            || !out.raw_slope_gradient.is_finite()
            || out.fluorescence.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("nonfinite response classification loss".into());
        }
        Ok(out)
    }
}
