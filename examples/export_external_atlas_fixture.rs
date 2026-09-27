//! Small, explicitly synthetic inputs for the cross-language fitting smoke test.
use std::{collections::BTreeMap, fs, path::Path};
use wormsim::{
    Result,
    bench::{
        Axis, Dataset, Split, Trial,
        atlas_level0::{self, FitConfig},
    },
    codec,
    data::{Provenance, Recording, Trace},
    fixtures,
};
fn write(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec(value).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: export_external_atlas_fixture NEW_DIRECTORY".into());
    }
    let raw = fixtures::synthetic(3, 1, 1);
    let graph = raw.clone().compile()?;
    let data = Dataset {
        schema_version: 1,
        name: "synthetic external atlas smoke test".into(),
        graph_hash: graph.hash.clone(),
        source: "synthetic; not biological evidence".into(),
        trials: (0..6)
            .map(|k| Trial {
                id: format!("trial{k}"),
                stimulated_neuron: Some(graph.names[k % 3].clone()),
                forecast_origin: None,
                response_labels: BTreeMap::new(),
                recording: Recording {
                    dataset: "synthetic".into(),
                    animal_id: format!("animal{k}"),
                    condition: "stim".into(),
                    times: vec![0., 0.05, 0.1],
                    behavior: BTreeMap::new(),
                    traces: graph
                        .names
                        .iter()
                        .map(|n| Trace {
                            neuron: n.clone(),
                            values: vec![Some(0.), Some(0.01), Some(0.02)],
                            provenance: Provenance {
                                dataset: "synthetic".into(),
                                version: "1".into(),
                                id_confidence: 1.,
                            },
                        })
                        .collect(),
                },
            })
            .collect(),
    };
    let split = Split::generate(&data, &graph, Axis::StimulatedNeuron, 42, 1, 1)?;
    let config:FitConfig=serde_json::from_value(serde_json::json!({"epochs":1,"dt":0.005,"preparation_seconds":0.04,"learning_rate":0.001,"kernel_lags":2,"prior_strength":0.01,"sign_prior_strength":0.01,"kernel_prior_strength":0.01,"sharing":wormsim::parameters::Sharing::default()})).map_err(|e|e.to_string())?;
    let mut initial = None;
    atlas_level0::fit_select(&data, &graph, &split, config, |candidate, _| {
        if candidate.epoch == 0 {
            initial = Some(candidate.clone());
        }
        Ok(())
    })?;
    let mut model = initial.ok_or("missing epoch-zero fixture")?;
    model.config.epochs = 2;
    let out = Path::new(&args[1]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    write(out.join("graph.json"), &raw)?;
    fs::write(out.join("graph.wsc"), codec::encode(&graph)?).map_err(|e| e.to_string())?;
    write(out.join("data.json"), &data)?;
    write(out.join("split.json"), &split)?;
    write(out.join("model.json"), &model)?;
    Ok(())
}
