//! Stabilized correlation of training pair-mean traces, with fluorescence adjoints.
use crate::{
    Result,
    data::{IndexedGraph, Recording},
};

#[derive(Debug)]
pub struct Loss {
    /// Sum of (1 - stabilized correlation) over eligible ordered pairs.
    pub value: f64,
    pub pairs: usize,
    pub fluorescence: Vec<Vec<f64>>,
}

/// Count pairs with at least two observed samples and nonconstant target values.
/// Eligibility depends only on observations, never on model predictions.
pub fn eligible_pairs(recording: &Recording) -> usize {
    recording
        .traces
        .iter()
        .filter(|trace| {
            if trace.provenance.id_confidence <= 0. {
                return false;
            }
            let mut values = trace.values.iter().flatten();
            let Some(first) = values.next() else {
                return false;
            };
            values.any(|v| v != first)
        })
        .count()
}

/// Each eligible pair contributes once. Confidence weights have already formed
/// the training pair mean; zero-confidence traces are excluded. Missing samples
/// are masked. Epsilon is a positive fluorescence standard-deviation floor.
/// This is not mean individual-trial Pearson correlation and not the test scorer.
pub fn loss(
    recording: &Recording,
    graph: &IndexedGraph,
    response: &[Vec<f64>],
    epsilon: f64,
) -> Result<Loss> {
    recording.validate(graph)?;
    let variance_floor = epsilon * epsilon;
    let n = graph.names.len();
    if !epsilon.is_finite()
        || epsilon <= 0.
        || !variance_floor.is_finite()
        || variance_floor <= 0.
        || response.len() != recording.times.len()
        || response
            .iter()
            .any(|row| row.len() != n || row.iter().any(|v| !v.is_finite()))
    {
        return Err("invalid correlation response or variance floor".into());
    }
    let mut out = Loss {
        value: 0.,
        pairs: 0,
        fluorescence: vec![vec![0.; n]; response.len()],
    };
    for trace in &recording.traces {
        if trace.provenance.id_confidence <= 0. {
            continue;
        }
        let samples: Vec<_> = trace
            .values
            .iter()
            .enumerate()
            .filter_map(|(t, v)| v.map(|y| (t, y)))
            .collect();
        if samples.len() < 2 || samples.iter().all(|(_, y)| *y == samples[0].1) {
            continue;
        }
        let i = graph.neuron(&trace.neuron)?;
        let count = samples.len() as f64;
        let mean_y = samples.iter().map(|(_, y)| y).sum::<f64>() / count;
        let mean_p = samples.iter().map(|(t, _)| response[*t][i]).sum::<f64>() / count;
        let mut vp = 0.;
        let mut vy = 0.;
        let mut cov = 0.;
        for &(t, y) in &samples {
            let p = response[t][i] - mean_p;
            let y = y - mean_y;
            vp += p * p / count;
            vy += y * y / count;
            cov += p * y / count;
        }
        let denominator = (vp + variance_floor).sqrt() * (vy + variance_floor).sqrt();
        let correlation = cov / denominator;
        out.value += 1. - correlation;
        out.pairs += 1;
        for &(t, y) in &samples {
            out.fluorescence[t][i] -= ((y - mean_y) / denominator
                - correlation * (response[t][i] - mean_p) / (vp + variance_floor))
                / count;
        }
    }
    if !out.value.is_finite() || out.fluorescence.iter().flatten().any(|v| !v.is_finite()) {
        return Err("nonfinite correlation objective".into());
    }
    Ok(out)
}
