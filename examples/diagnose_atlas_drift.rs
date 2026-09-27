//! Validation-only decomposition of fixed-state responses into unforced drift and stimulus effect.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};
use wormsim::{
    Result,
    bench::{self, Dataset, Partition, Split, atlas, atlas_level0::AtlasModel},
    codec,
    initial_state::{self, Readout},
    model::{Inputs, Model},
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn write(p: impl AsRef<Path>, v: &impl serde::Serialize) -> Result<()> {
    fs::write(p, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: diagnose_atlas_drift GRAPH DATA SPLIT EVIDENCE MODEL NEW_OUTPUT_DIR (validation only)".into());
    }
    let graph = codec::decode(&fs::read(&a[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&a[2])?;
    let split: Split = read(&a[3])?;
    let evidence: atlas::Evidence = read(&a[4])?;
    let fitted: AtlasModel = read(&a[5])?;
    let driven = fitted.predict(&data, &graph, &split, Partition::Validation)?;
    let model = Model::new(graph.clone())?;
    let params = fitted.parameters.expand(&model)?;
    let prepared = model.prepare(&params)?;
    let mut derivative = vec![0.; model.state_len()];
    let mut scratch = vec![0.; model.n()];
    model.rhs(
        &prepared,
        &fitted.initial,
        &Inputs::new(model.n()),
        &mut scratch,
        &mut derivative,
    );
    let mut zero = driven.clone();
    let mut difference = driven.clone();
    zero.model.push_str("; diagnostic zero current");
    difference
        .model
        .push_str("; diagnostic driven minus zero current");
    for p in [&mut zero, &mut difference] {
        p.source_commit = option_env!("WORMSIM_COMMIT")
            .unwrap_or("unversioned")
            .into();
    }
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let mut cache = BTreeMap::new();
    let mut energies = [0.; 3];
    let mut sample_weight = 0.;
    let mut energy_cross = 0.;
    for ((p, z), d) in driven
        .trials
        .iter()
        .zip(&mut zero.trials)
        .zip(&mut difference.trials)
    {
        let frames = p.times.len();
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(frames) {
            e.insert(initial_state::response_with_currents(
                &model,
                &params,
                &fitted.initial,
                &p.times,
                &Readout::identity(model.n()),
                fitted.config.dt,
                &vec![vec![0.; model.n()]; frames],
            )?);
        }
        let baseline = &cache[&frames];
        for trace in &indexed[&p.id].recording.traces {
            let i = graph.neuron(&trace.neuron)?;
            let values = &p.fluorescence[&trace.neuron];
            let unforced: Vec<_> = baseline.iter().map(|r| r[i]).collect();
            let effect: Vec<_> = values.iter().zip(&unforced).map(|(a, b)| a - b).collect();
            for t in 0..frames {
                if trace.values[t].is_some() {
                    let w = trace.provenance.id_confidence;
                    energies[0] += w * values[t].powi(2);
                    energies[1] += w * unforced[t].powi(2);
                    energies[2] += w * effect[t].powi(2);
                    energy_cross += 2. * w * unforced[t] * effect[t];
                    sample_weight += w;
                }
            }
            z.fluorescence.insert(trace.neuron.clone(), unforced);
            d.fluorescence.insert(trace.neuron.clone(), effect);
        }
        for name in z.response_scores.clone().keys() {
            z.response_scores.insert(
                name.clone(),
                z.fluorescence[name].iter().map(|v| v.abs()).sum::<f64>() * fitted.sample_dt,
            );
            d.response_scores.insert(
                name.clone(),
                d.fluorescence[name].iter().map(|v| v.abs()).sum::<f64>() * fitted.sample_dt,
            );
        }
    }
    let out = Path::new(&a[6]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let mut scores = BTreeMap::new();
    for (name, p) in [
        ("driven", &driven),
        ("zero_current", &zero),
        ("stimulus_difference", &difference),
    ] {
        let traces = bench::evaluate(&data, &graph, &split, p, Partition::Validation)?;
        let pairs =
            atlas::rank_responses(&evidence, &data, &graph, &split, p, Partition::Validation)?;
        let classification = atlas::evaluate(
            &evidence,
            &data,
            &graph,
            &split,
            &pairs,
            Partition::Validation,
        )?;
        write(out.join(format!("{name}-predictions.json")), p)?;
        scores.insert(name,serde_json::json!({"mse":traces.pooled_trace_scores.mse,"correlation":traces.macro_trace_correlation,"defined_correlations":traces.defined_trace_correlations,"pair_auroc":classification.auroc.value}));
    }
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"model_source_commit":fitted.source_commit,"model_sha256":format!("{:x}",Sha256::digest(fs::read(&a[5]).map_err(|e|e.to_string())?)),"epoch":fitted.epoch,"dataset_hash":split.dataset_hash,"split_hash":split.content_hash()?,"partition":"validation","scores":scores,"observed_weighted_mean_square":{"driven":energies[0]/sample_weight,"zero_current":energies[1]/sample_weight,"stimulus_difference":energies[2]/sample_weight,"cross_term":energy_cross/sample_weight},"energy_identity_residual":(energies[0]-energies[1]-energies[2]-energy_cross)/sample_weight,"initial_unforced_derivative_l2":derivative.iter().map(|v|v*v).sum::<f64>().sqrt(),"scope":"Frozen model diagnostic on validation only. No refitting or test scoring. Driven = zero-current drift + stimulus difference exactly, up to floating-point error. Subtracted predictions are diagnostic and are not a newly trained model. Mean squares include only observed positive-weight validation samples; cross term means these are not orthogonal variance fractions."});
    write(out.join("report.json"), &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
