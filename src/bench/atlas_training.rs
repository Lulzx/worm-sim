//! Training-only sufficient statistics for deterministic target-conditioned responses.
//! Complete response traces are required; no missing samples are silently dropped.
use super::{Axis, Dataset, Split};
use crate::{
    Result,
    data::{IndexedGraph, Provenance, Recording, Trace},
};
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct ResponseGroup {
    pub stimulated_neuron: String,
    pub recording: Recording,
    pub training_trials: Vec<String>,
    /// Original sum of confidence weights over all observed samples.
    pub sample_weight: f64,
    /// Add to the weighted mean-trace MSE to recover the original trial MSE.
    pub irreducible_mse: f64,
}
#[derive(Default)]
struct Moments {
    weight: f64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}
/// For any common prediction p, sum w(y-p)^2 equals sum w(y-mean)^2
/// plus sum w(mean-p)^2. This preserves the objective and its derivatives.
/// Only valid when the model uses the same initial state/drive/readout per target;
/// not suitable for trial-conditioned states, likelihood fitting or trial inputs.
pub fn aggregate(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
) -> Result<Vec<ResponseGroup>> {
    split.validate(data, graph)?;
    if split.axis != Axis::StimulatedNeuron {
        return Err("response aggregation requires stimulated-neuron split".into());
    }
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let mut targets: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for id in &split.train {
        let trial = indexed[id];
        targets
            .entry(trial.stimulated_neuron.clone().ok_or("missing stimulus")?)
            .or_default()
            .push(trial);
    }
    let mut out = vec![];
    for (target, mut trials) in targets {
        trials.sort_by(|a, b| a.id.cmp(&b.id));
        let times = &trials[0].recording.times;
        if times.len() < 2 || times[0].abs() > 1e-10 {
            return Err(
                "response aggregation requires at least two samples starting at stimulus zero"
                    .into(),
            );
        }
        let mut neurons: BTreeMap<String, Moments> = BTreeMap::new();
        for trial in &trials {
            if trial.forecast_origin.is_some() || trial.recording.times != *times {
                return Err(
                    "response aggregation requires identical stimulus-aligned grids per target"
                        .into(),
                );
            }
            for trace in &trial.recording.traces {
                let w = trace.provenance.id_confidence;
                if w == 0. {
                    continue;
                }
                if trace.values.iter().any(Option::is_none) {
                    return Err(
                        "response aggregation requires complete positive-confidence traces".into(),
                    );
                }
                let m = neurons.entry(trace.neuron.clone()).or_default();
                if m.mean.is_empty() {
                    m.mean = vec![0.; times.len()];
                    m.m2 = vec![0.; times.len()];
                }
                let total = m.weight + w;
                for (t, y) in trace.values.iter().enumerate() {
                    let y = y.unwrap();
                    let delta = y - m.mean[t];
                    m.mean[t] += w / total * delta;
                    m.m2[t] += w * delta * (y - m.mean[t]);
                }
                m.weight = total;
            }
        }
        let sample_weight = neurons
            .values()
            .map(|m| m.weight * times.len() as f64)
            .sum::<f64>();
        if sample_weight <= 0. {
            return Err("empty positive-confidence target group".into());
        }
        let irreducible_mse = neurons.values().flat_map(|m| &m.m2).sum::<f64>() / sample_weight;
        let max_weight = neurons.values().map(|m| m.weight).fold(0., f64::max);
        let traces=neurons.into_iter().map(|(neuron,m)|Trace {neuron,values:m.mean.into_iter().map(Some).collect(),provenance:Provenance {dataset:"training response sufficient statistics".into(),version:"1; confidence field encodes relative accumulated loss weight, not calibrated identity confidence".into(),id_confidence:m.weight/max_weight}}).collect();
        let recording = Recording {
            dataset: "training response sufficient statistics".into(),
            animal_id: "aggregate-not-an-animal".into(),
            condition: format!("stimulated {target}"),
            times: times.clone(),
            traces,
            behavior: BTreeMap::new(),
        };
        recording.validate(graph)?;
        if !irreducible_mse.is_finite() || !sample_weight.is_finite() {
            return Err("nonfinite response moments".into());
        }
        out.push(ResponseGroup {
            stimulated_neuron: target,
            recording,
            training_trials: trials.iter().map(|t| t.id.clone()).collect(),
            sample_weight,
            irreducible_mse,
        });
    }
    Ok(out)
}
