//! Shared, training-fitted behavior covariates. No actual post-origin behavior is read.
use super::{Axis, Dataset, Split, Trial};
use crate::{Result, data::IndexedGraph};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    pub name: String,
    pub mean: f64,
    pub scale: f64,
    /// AR(1) in standardized coordinates, fitted to adjacent training pairs.
    pub slope: f64,
    pub intercept: f64,
    pub observations: usize,
    pub adjacent_pairs: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehaviorModel {
    pub schema_version: u32,
    pub dataset_hash: String,
    pub split_hash: String,
    pub graph_hash: String,
    pub source_commit: String,
    pub training_trials: Vec<String>,
    pub sample_dt: f64,
    pub channels: Vec<Channel>,
}
impl BehaviorModel {
    pub fn free_parameters(&self) -> usize {
        4 * self.channels.len()
    }
    pub fn input_dim(&self) -> usize {
        2 * self.channels.len()
    }
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.channels.is_empty()
            || self.channels.len() > 32
            || !self.sample_dt.is_finite()
            || self.sample_dt <= 0.0
            || self.source_commit.trim().is_empty()
            || self.channels.windows(2).any(|p| p[0].name >= p[1].name)
            || self.channels.iter().any(|c| {
                c.name.is_empty()
                    || c.observations < 2
                    || c.adjacent_pairs < 2
                    || [c.mean, c.scale, c.slope, c.intercept]
                        .iter()
                        .any(|x| !x.is_finite())
                    || c.scale <= 0.0
                    || c.slope.abs() > 0.995
            })
        {
            return Err("invalid behavior forecast artifact".into());
        }
        Ok(())
    }
    pub fn validate_lineage(
        &self,
        data: &Dataset,
        graph: &IndexedGraph,
        split: &Split,
    ) -> Result<()> {
        self.validate()?;
        split.validate(data, graph)?;
        if split.axis != Axis::Animal
            || self.dataset_hash != split.dataset_hash
            || self.split_hash != split.content_hash()?
            || self.graph_hash != graph.hash
            || self.training_trials != split.train
        {
            return Err("behavior forecast lineage mismatch".into());
        }
        Ok(())
    }
    pub fn content_hash(&self) -> Result<String> {
        super::hash(self)
    }
    /// Frame t contains behavior available/forecast at t and an observed-value mask.
    /// Neural consumers use u[t] for the transition t -> t+1. Missing history and
    /// the future propagate AR estimates; completely absent channels start at mean.
    pub fn inputs(&self, trial: &Trial) -> Result<Vec<Vec<f64>>> {
        self.validate()?;
        let times = &trial.recording.times;
        let origin = trial
            .forecast_origin
            .ok_or("behavior input needs forecast origin")?;
        if times.len() < 2
            || times.iter().any(|t| !t.is_finite())
            || !times.contains(&origin)
            || times
                .windows(2)
                .any(|p| (p[1] - p[0] - self.sample_dt).abs() > 1e-8)
        {
            return Err("behavior input grid/origin mismatch".into());
        }
        let b = self.channels.len();
        let mut out = vec![vec![0.0; 2 * b]; times.len()];
        for (i, c) in self.channels.iter().enumerate() {
            let observed = trial.recording.behavior.get(&c.name);
            if observed.is_some_and(|v| v.len() != times.len()) {
                return Err("behavior input length mismatch".into());
            }
            let mut state = 0.0;
            for (t, &time) in times.iter().enumerate() {
                if t > 0 {
                    state = c.intercept + c.slope * state;
                }
                if time <= origin
                    && let Some(Some(value)) = observed.map(|v| v[t])
                {
                    if !value.is_finite() {
                        return Err("nonfinite observed behavior history".into());
                    }
                    state = (value - c.mean) / c.scale;
                    out[t][b + i] = 1.0;
                }
                if !state.is_finite() {
                    return Err("nonfinite behavior forecast".into());
                }
                out[t][i] = state;
            }
        }
        Ok(out)
    }
}
#[derive(Default)]
struct Stats {
    n: usize,
    mean: f64,
    m2: f64,
    pairs: usize,
    mx: f64,
    my: f64,
    xx: f64,
    xy: f64,
}
impl Stats {
    fn value(&mut self, y: f64) {
        self.n += 1;
        let d = y - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (y - self.mean);
    }
    fn pair(&mut self, x: f64, y: f64) {
        self.pairs += 1;
        let dx = x - self.mx;
        let dy = y - self.my;
        self.mx += dx / self.pairs as f64;
        self.my += dy / self.pairs as f64;
        self.xx += dx * (x - self.mx);
        self.xy += dx * (y - self.my);
    }
}
pub fn fit(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    names: &[String],
) -> Result<BehaviorModel> {
    split.validate(data, graph)?;
    if split.axis != Axis::Animal
        || names.is_empty()
        || names.len() > 32
        || names.iter().any(|s| s.trim().is_empty())
        || names.iter().collect::<BTreeSet<_>>().len() != names.len()
    {
        return Err("invalid behavior training selection".into());
    }
    let mut names = names.to_vec();
    names.sort();
    let mut trials: Vec<_> = data
        .trials
        .iter()
        .filter(|t| split.train.contains(&t.id))
        .collect();
    trials.sort_by(|a, b| a.id.cmp(&b.id));
    let mut stats: Vec<_> = names.iter().map(|_| Stats::default()).collect();
    let mut dt: Option<f64> = None;
    for trial in trials {
        for p in trial.recording.times.windows(2) {
            let step = p[1] - p[0];
            if dt.is_some_and(|d| (d - step).abs() > 1e-8) {
                return Err("behavior fit needs uniform time grid".into());
            }
            dt = Some(step);
        }
        for (i, name) in names.iter().enumerate() {
            let Some(values) = trial.recording.behavior.get(name) else {
                continue;
            };
            for &v in values.iter().flatten() {
                stats[i].value(v);
            }
            for pair in values.windows(2) {
                if let (Some(x), Some(y)) = (pair[0], pair[1]) {
                    stats[i].pair(x, y);
                }
            }
        }
    }
    let channels = names
        .into_iter()
        .zip(stats)
        .map(|(name, s)| {
            let scale = if s.m2 > 1e-16 {
                (s.m2 / s.n as f64).sqrt()
            } else {
                1.0
            };
            let slope = if s.xx > 1e-16 {
                (s.xy / s.xx).clamp(-0.995, 0.995)
            } else {
                0.0
            };
            Channel {
                name,
                mean: s.mean,
                scale,
                slope,
                intercept: (s.my - s.mean - slope * (s.mx - s.mean)) / scale,
                observations: s.n,
                adjacent_pairs: s.pairs,
            }
        })
        .collect();
    let model = BehaviorModel {
        schema_version: 1,
        dataset_hash: split.dataset_hash.clone(),
        split_hash: split.content_hash()?,
        graph_hash: graph.hash.clone(),
        source_commit: option_env!("WORMSIM_COMMIT")
            .unwrap_or("unversioned")
            .into(),
        training_trials: split.train.clone(),
        sample_dt: dt.ok_or("no behavior training intervals")?,
        channels,
    };
    model.validate()?;
    Ok(model)
}
