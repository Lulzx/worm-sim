//! Confidence-weighted streaming moments and exact tie-aware response AUROC.
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default)]
pub struct Moments {
    count: usize,
    weight: f64,
    mean_target: f64,
    mean_prediction: f64,
    target_ss: f64,
    prediction_ss: f64,
    cross: f64,
    error_ss: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scores {
    pub samples: usize,
    pub weight: f64,
    pub mse: Option<f64>,
    pub correlation: Option<f64>,
    pub r2: Option<f64>,
}
impl Moments {
    pub fn push(&mut self, target: f64, prediction: f64, weight: f64) -> Result<()> {
        if !target.is_finite() || !prediction.is_finite() || !weight.is_finite() || weight < 0.0 {
            return Err("metrics require finite values and nonnegative weights".into());
        }
        if weight == 0.0 {
            return Ok(());
        }
        let total = self.weight + weight;
        let dt = target - self.mean_target;
        let dp = prediction - self.mean_prediction;
        let ratio = weight / total;
        self.mean_target += ratio * dt;
        self.mean_prediction += ratio * dp;
        self.target_ss += weight * dt * (target - self.mean_target);
        self.prediction_ss += weight * dp * (prediction - self.mean_prediction);
        self.cross += weight * dt * (prediction - self.mean_prediction);
        self.error_ss += weight * (target - prediction).powi(2);
        self.weight = total;
        self.count += 1;
        if [
            self.weight,
            self.target_ss,
            self.prediction_ss,
            self.cross,
            self.error_ss,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err("metric accumulation overflow".into());
        }
        Ok(())
    }
    /// Merge independent blocks, including repeated blocks in a cluster bootstrap.
    pub fn merge(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        }
        if self.weight == 0.0 {
            *self = other.clone();
            return;
        }
        let total = self.weight + other.weight;
        let dt = other.mean_target - self.mean_target;
        let dp = other.mean_prediction - self.mean_prediction;
        let bridge = self.weight * (other.weight / total);
        self.target_ss += other.target_ss + dt * dt * bridge;
        self.prediction_ss += other.prediction_ss + dp * dp * bridge;
        self.cross += other.cross + dt * dp * bridge;
        self.error_ss += other.error_ss;
        self.mean_target += dt * (other.weight / total);
        self.mean_prediction += dp * (other.weight / total);
        self.weight = total;
        self.count += other.count;
    }
    pub fn scores(&self) -> Scores {
        Scores {
            samples: self.count,
            weight: self.weight,
            mse: (self.weight > 0.0).then(|| self.error_ss / self.weight),
            correlation: (self.target_ss > 0.0 && self.prediction_ss > 0.0).then(|| {
                ((self.cross / self.target_ss.sqrt()) / self.prediction_ss.sqrt()).clamp(-1.0, 1.0)
            }),
            r2: (self.target_ss > 0.0)
                .then(|| 1.0 - self.error_ss / self.target_ss)
                .filter(|v| v.is_finite()),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Auroc {
    pub value: Option<f64>,
    pub positive_pairs: usize,
    pub negative_pairs: usize,
    pub positive_weight: f64,
    pub negative_weight: f64,
}
/// Score is any finite ranking score; equal scores receive half credit.
pub fn auroc(pairs: &[(bool, f64, f64)]) -> Result<Auroc> {
    if pairs
        .iter()
        .any(|(_, score, w)| !score.is_finite() || !w.is_finite() || *w < 0.0)
    {
        return Err("AUROC requires finite scores and nonnegative weights".into());
    }
    let mut sorted: Vec<_> = pairs.iter().copied().filter(|x| x.2 > 0.0).collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut result = Auroc {
        value: None,
        positive_pairs: 0,
        negative_pairs: 0,
        positive_weight: 0.0,
        negative_weight: 0.0,
    };
    let mut negative_before = 0.0;
    let mut wins = 0.0;
    let mut start = 0;
    while start < sorted.len() {
        let mut end = start + 1;
        while end < sorted.len() && sorted[end].1 == sorted[start].1 {
            end += 1;
        }
        let mut positive = 0.0;
        let mut negative = 0.0;
        for &(label, _, w) in &sorted[start..end] {
            if label {
                positive += w;
                result.positive_pairs += 1;
            } else {
                negative += w;
                result.negative_pairs += 1;
            }
        }
        wins += positive * (negative_before + 0.5 * negative);
        negative_before += negative;
        result.positive_weight += positive;
        result.negative_weight += negative;
        start = end;
    }
    let denominator = result.positive_weight * result.negative_weight;
    if !denominator.is_finite() || !wins.is_finite() {
        return Err("AUROC accumulation overflow".into());
    }
    if denominator > 0.0 {
        result.value = Some(wins / denominator);
    }
    Ok(result)
}
