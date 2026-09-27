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

/// Paired differences A minus B. Correlation uses only traces defined for both
/// models, so differing undefined-trace coverage cannot masquerade as improvement.
pub fn trace_difference(
    data: &Dataset,
    a: &Report,
    b: &Report,
    seed: u64,
    replicates: usize,
) -> Result<Vec<TraceBootstrap>> {
    if a.dataset_hash != data.content_hash()?
        || b.dataset_hash != a.dataset_hash
        || a.split_hash != b.split_hash
        || a.axis != super::Axis::StimulatedNeuron
        || b.axis != a.axis
        || std::mem::discriminant(&a.partition) != std::mem::discriminant(&b.partition)
    {
        return Err("incompatible paired trace reports".into());
    }
    let right: BTreeMap<_, _> = b
        .traces
        .iter()
        .map(|t| ((&t.trial, &t.neuron), t))
        .collect();
    let left: std::collections::BTreeSet<_> =
        a.traces.iter().map(|t| (&t.trial, &t.neuron)).collect();
    if right.len() != b.traces.len()
        || left.len() != a.traces.len()
        || left != right.keys().copied().collect()
    {
        return Err("paired trace coverage differs".into());
    }
    let trials: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let mut output = vec![];
    for by_target in [true, false] {
        let mut groups: BTreeMap<String, [f64; 4]> = BTreeMap::new();
        for x in &a.traces {
            let y = right[&(&x.trial, &x.neuron)];
            if x.confidence != y.confidence
                || x.scores.weight != y.scores.weight
                || x.scores.samples != y.scores.samples
            {
                return Err("paired trace weights differ".into());
            }
            let trial = trials.get(&x.trial).ok_or("unknown paired trial")?;
            let key = if by_target {
                trial.stimulated_neuron.as_ref().ok_or("missing stimulus")?
            } else {
                &trial.recording.animal_id
            };
            let g = groups.entry(key.clone()).or_default();
            match (x.scores.mse, y.scores.mse) {
                (Some(xm), Some(ym)) => {
                    g[0] += (xm - ym) * x.scores.weight;
                    g[1] += x.scores.weight;
                }
                (None, None) => {}
                _ => return Err("paired MSE availability differs".into()),
            }
            if let (Some(xc), Some(yc)) = (x.scores.correlation, y.scores.correlation) {
                g[2] += (xc - yc) * x.confidence;
                g[3] += x.confidence;
            }
        }
        let blocks: Vec<_> = groups.values().collect();
        let mut total = [0.; 4];
        for block in &blocks {
            for (a, b) in total.iter_mut().zip(*block) {
                *a += b;
            }
        }
        let mut mse = vec![];
        let mut corr = vec![];
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
                corr.push(sum[2] / sum[3]);
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
            pooled_mse: interval((total[1] > 0.).then(|| total[0] / total[1]), mse),
            macro_trace_correlation: interval((total[3] > 0.).then(|| total[2] / total[3]), corr),
        });
    }
    Ok(output)
}

/// Paired AUROC A minus B using the same resampled target identities.
/// Both inputs must first pass atlas::evaluate for the same partition.
pub fn pair_difference(
    evidence: &atlas::Evidence,
    a: &atlas::Predictions,
    b: &atlas::Predictions,
    seed: u64,
    replicates: usize,
) -> Result<PairBootstrap> {
    if a.evidence_hash != evidence.content_hash()?
        || b.evidence_hash != a.evidence_hash
        || a.split_hash != b.split_hash
    {
        return Err("incompatible paired atlas predictions".into());
    }
    let right: BTreeMap<_, _> = b
        .pairs
        .iter()
        .map(|p| ((&p.stimulated, &p.responding), p.score))
        .collect();
    let labels: BTreeMap<_, _> = evidence
        .pairs
        .iter()
        .map(|p| {
            (
                (&p.stimulated, &p.responding),
                p.q < evidence.detection_q_threshold,
            )
        })
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut groups: BTreeMap<String, Vec<(bool, f64, f64)>> = BTreeMap::new();
    for p in &a.pairs {
        let key = (&p.stimulated, &p.responding);
        if !seen.insert(key) {
            return Err("duplicate paired atlas prediction".into());
        }
        groups.entry(p.stimulated.clone()).or_default().push((
            *labels.get(&key).ok_or("unknown atlas pair")?,
            p.score,
            *right.get(&key).ok_or("paired atlas coverage differs")?,
        ));
    }
    if right.len() != b.pairs.len() || right.len() != seen.len() {
        return Err("paired atlas coverage differs".into());
    }
    let blocks: Vec<_> = groups.values().collect();
    let difference = |counts: &[usize]| -> Result<Option<f64>> {
        let mut x = vec![];
        let mut y = vec![];
        for (block, &count) in blocks.iter().zip(counts) {
            for &(label, sa, sb) in *block {
                x.push((label, sa, count as f64));
                y.push((label, sb, count as f64));
            }
        }
        Ok(metrics::auroc(&x)?
            .value
            .zip(metrics::auroc(&y)?.value)
            .map(|(a, b)| a - b))
    };
    let point = difference(&vec![1; blocks.len()])?;
    let mut values = vec![];
    for draw in draws(blocks.len(), seed, replicates)? {
        let mut counts = vec![0; blocks.len()];
        for i in draw {
            counts[i] += 1;
        }
        if let Some(d) = difference(&counts)? {
            values.push(d);
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
