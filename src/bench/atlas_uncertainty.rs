//! Separate target-identity and recording cluster bootstraps for Task 1.
//! These are marginal resampling analyses, not a crossed-cluster independence claim.
use super::{Dataset, Report, atlas, metrics, uncertainty::Interval};
use crate::Result;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct TraceBootstrap {
    pub grouping: String,
    pub clusters: usize,
    pub seed: u64,
    pub replicates: usize,
    pub pooled_mse: Interval,
    pub macro_trace_correlation: Interval,
}
#[derive(Debug, Serialize)]
pub struct PairBootstrap {
    pub grouping: String,
    pub clusters: usize,
    pub seed: u64,
    pub replicates: usize,
    pub auroc: Interval,
}
fn interval(point: Option<f64>, mut values: Vec<f64>) -> Interval {
    values.sort_by(f64::total_cmp);
    let q = |p: f64| {
        if values.is_empty() {
            None
        } else {
            let x = p * (values.len() - 1) as f64;
            Some(
                values[x.floor() as usize]
                    + (values[x.ceil() as usize] - values[x.floor() as usize]) * x.fract(),
            )
        }
    };
    Interval {
        point,
        lower_95: q(0.025),
        upper_95: q(0.975),
        defined_replicates: values.len(),
    }
}
fn draws(n: usize, seed: u64, replicates: usize) -> Result<Vec<Vec<usize>>> {
    if n == 0 || replicates == 0 || replicates > 100_000 {
        return Err("invalid atlas bootstrap size".into());
    }
    let mut state = seed;
    let bound = n as u64;
    let threshold = bound.wrapping_neg() % bound;
    let mut index = || loop {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        if z >= threshold {
            break (z % bound) as usize;
        }
    };
    Ok((0..replicates)
        .map(|_| (0..n).map(|_| index()).collect())
        .collect())
}
/// Input must be a common-scorer report for this dataset. Report identity is checked.
pub fn traces(
    data: &Dataset,
    report: &Report,
    seed: u64,
    replicates: usize,
) -> Result<Vec<TraceBootstrap>> {
    if report.dataset_hash != data.content_hash()? || report.axis != super::Axis::StimulatedNeuron {
        return Err("atlas trace report identity mismatch".into());
    }
    let trials: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let mut output = vec![];
    for by_target in [true, false] {
        // SSE, sample weight, weighted correlation, defined correlation weight.
        let mut groups: BTreeMap<String, [f64; 4]> = BTreeMap::new();
        for trace in &report.traces {
            let trial = trials.get(&trace.trial).ok_or("unknown report trial")?;
            let key = if by_target {
                trial.stimulated_neuron.as_ref().ok_or("missing stimulus")?
            } else {
                &trial.recording.animal_id
            };
            let g = groups.entry(key.clone()).or_default();
            if let Some(mse) = trace.scores.mse {
                g[0] += mse * trace.scores.weight;
                g[1] += trace.scores.weight;
            }
            if let Some(c) = trace.scores.correlation {
                g[2] += c * trace.confidence;
                g[3] += trace.confidence;
            }
        }
        let blocks: Vec<_> = groups.values().collect();
        let mut mse = vec![];
        let mut correlation = vec![];
        for draw in draws(blocks.len(), seed, replicates)? {
            let mut sum = [0.; 4];
            for i in draw {
                for (a, b) in sum.iter_mut().zip(blocks[i]) {
                    *a += b;
                }
            }
            if sum[1] > 0. {
                mse.push(sum[0] / sum[1]);
            }
            if sum[3] > 0. {
                correlation.push(sum[2] / sum[3]);
            }
        }
        output.push(TraceBootstrap {
            grouping: if by_target {
                "stimulated_neuron"
            } else {
                "recording_id_not_verified_animal"
            }
            .into(),
            clusters: blocks.len(),
            seed,
            replicates,
            pooled_mse: interval(report.pooled_trace_scores.mse, mse),
            macro_trace_correlation: interval(report.macro_trace_correlation, correlation),
        });
    }
    Ok(output)
}
/// Call after atlas::evaluate has validated coverage and lineage. Duplicated
/// target clusters multiply pair weights, retaining every responding pair.
pub fn pairs(
    evidence: &atlas::Evidence,
    predictions: &atlas::Predictions,
    seed: u64,
    replicates: usize,
) -> Result<PairBootstrap> {
    if predictions.evidence_hash != evidence.content_hash()? {
        return Err("atlas evidence mismatch".into());
    }
    let evidence_pairs: BTreeMap<_, _> = evidence
        .pairs
        .iter()
        .map(|p| ((&p.stimulated, &p.responding), p))
        .collect();
    let mut groups: BTreeMap<String, Vec<(bool, f64, f64)>> = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    for prediction in &predictions.pairs {
        let key = (&prediction.stimulated, &prediction.responding);
        if !seen.insert(key) {
            return Err("duplicate atlas prediction".into());
        }
        let pair = evidence_pairs.get(&key).ok_or("unknown atlas pair")?;
        groups
            .entry(prediction.stimulated.clone())
            .or_default()
            .push((
                pair.q < evidence.detection_q_threshold,
                prediction.score,
                1.,
            ));
    }
    let blocks: Vec<_> = groups.values().collect();
    let all: Vec<_> = blocks.iter().flat_map(|b| b.iter().copied()).collect();
    let point = metrics::auroc(&all)?.value;
    let mut values = vec![];
    for draw in draws(blocks.len(), seed, replicates)? {
        let mut counts = vec![0usize; blocks.len()];
        for i in draw {
            counts[i] += 1;
        }
        let rows: Vec<_> = blocks
            .iter()
            .zip(counts)
            .flat_map(|(b, n)| b.iter().map(move |&(y, p, _)| (y, p, n as f64)))
            .collect();
        if let Some(value) = metrics::auroc(&rows)?.value {
            values.push(value);
        }
    }
    Ok(PairBootstrap {
        grouping: "stimulated_neuron".into(),
        clusters: blocks.len(),
        seed,
        replicates,
        auroc: interval(point, values),
    })
}
