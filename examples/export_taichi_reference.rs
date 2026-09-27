//! Generate a short, exact Rust Euler/forward-AD reference for Taichi validation.
use std::{fs, time::Instant};
use wormsim::{
    Result, codec,
    data::{Provenance, Recording, Trace},
    fit, fixtures,
    model::Model,
    solve::{Config, Event, Method, simulate},
};
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or("runs/taichi-reference.json".into());
    let graph_path = args.next();
    if args.next().is_some() {
        return Err("usage: export_taichi_reference [OUTPUT [GRAPH.wsc]]".into());
    }
    let real_graph = graph_path.is_some();
    let graph = match graph_path {
        Some(path) => codec::decode(&fs::read(path).map_err(|e| e.to_string())?)?,
        None => fixtures::synthetic(4, 2, 1).compile()?,
    };
    let model = Model::new(graph)?;
    let n = model.n();
    let steps = 64;
    let dt = 0.001;
    let mut params = model.defaults();
    for i in 0..n {
        params.raw[n + i] = -0.6 + 0.1 * (i % 4) as f64;
    }
    let cfg = Config {
        duration: steps as f64 * dt,
        dt,
        save_dt: dt,
        method: Method::Euler,
        events: vec![Event::Stimulate {
            neuron: model.graph.names[0].clone(),
            start: 0.0,
            end: steps as f64 * dt,
            amplitude: 0.8,
        }],
    };
    let result = simulate(&model, &params, &cfg)?;
    let recording = Recording {
        dataset: "synthetic-gradient-audit".into(),
        animal_id: "synthetic".into(),
        condition: "current injection".into(),
        times: result.times.clone(),
        traces: (0..n)
            .map(|i| Trace {
                neuron: model.graph.names[i].clone(),
                values: result
                    .fluorescence
                    .iter()
                    .enumerate()
                    .map(|(t, row)| {
                        if (t + i) % 7 == 0 {
                            None
                        } else {
                            Some(row[i] + 0.03 * ((i % 4) as f64 + 1.0))
                        }
                    })
                    .collect(),
                provenance: Provenance {
                    dataset: "synthetic-gradient-audit".into(),
                    version: "1".into(),
                    id_confidence: 0.6 + 0.1 * (i % 4) as f64,
                },
            })
            .collect(),
    };
    let indices: Vec<_> = (0..params.raw.len()).collect();
    let mut current = vec![0.0; n];
    current[0] = 0.8;
    let start = Instant::now();
    let (loss, gradient) = fit::gradient(&model, &params, &cfg, &recording, &indices)?;
    let forward_ad_seconds = start.elapsed().as_secs_f64();
    let targets: Vec<Vec<_>> = (0..=steps)
        .map(|t| recording.traces.iter().map(|r| r.values[t]).collect())
        .collect();
    let output = serde_json::json!({"schema_version":1,"fixture":if real_graph { "imported anatomy; synthetic targets; full gradient audit only" } else { "synthetic; gradient audit only" },"graph_hash":model.graph.hash,"gradient_indices":indices,"method":"euler","neurons":n,"steps":steps,"dt":dt,"parameters_raw":params.raw,"pre":model.pre,"post":model.post,"counts":model.counts,"parameter_edge":model.parameter_edge,"incoming_offsets":model.incoming_offsets,"gap_a":model.gap_a,"gap_b":model.gap_b,"gap_sizes":model.gap_sizes,"current":current,"targets":targets,"confidence":recording.traces.iter().map(|r|r.provenance.id_confidence).collect::<Vec<_>>(),"reference":{"loss":loss,"gradient":gradient,"voltage":result.voltage,"fluorescence":result.fluorescence,"forward_ad_seconds":forward_ad_seconds}});
    fs::write(
        &path,
        serde_json::to_vec_pretty(&output).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "wrote {path}; {} parameters; loss={loss:.8e}",
        indices.len()
    );
    Ok(())
}
