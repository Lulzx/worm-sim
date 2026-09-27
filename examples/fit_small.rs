use wormsim::{
    Result,
    data::{Provenance, Recording, Trace},
    fit, fixtures,
    model::Model,
    solve::{Config, Event, simulate},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: fit_small [output.json]".into());
    }
    let graph = fixtures::synthetic(3, 1, 0);
    let model = Model::new(graph.clone().compile()?)?;
    let config = Config {
        duration: 0.6,
        events: vec![Event::Stimulate {
            neuron: "N000".into(),
            start: 0.1,
            end: 0.3,
            amplitude: 1.0,
        }],
        ..Config::default()
    };
    let truth = model.defaults();
    let target = simulate(&model, &truth, &config)?;
    let recording = Recording {
        behavior: Default::default(),
        dataset: "synthetic".into(),
        animal_id: "teacher".into(),
        condition: "pulse".into(),
        times: target.times.clone(),
        traces: (0..model.n())
            .map(|i| Trace {
                neuron: model.graph.names[i].clone(),
                values: target.fluorescence.iter().map(|row| Some(row[i])).collect(),
                provenance: Provenance {
                    dataset: "synthetic".into(),
                    version: "1".into(),
                    id_confidence: 1.0,
                },
            })
            .collect(),
    };
    let mut fitted = truth.clone();
    let index = 6 * model.n();
    fitted.raw[index] += 1.5;
    let initial = fitted.clone();
    let initial_trace = simulate(&model, &initial, &config)?;
    let history = fit::adam(&model, &mut fitted, &config, &recording, &[index], 80, 0.08)?;
    let fitted_trace = simulate(&model, &fitted, &config)?;
    println!(
        "synthetic training loss: {:.6e} -> {:.6e}",
        history[0],
        history.last().unwrap()
    );
    let held = Config {
        events: vec![Event::Stimulate {
            neuron: "N000".into(),
            start: 0.2,
            end: 0.4,
            amplitude: 0.6,
        }],
        ..config.clone()
    };
    let expected = simulate(&model, &truth, &held)?;
    let actual = simulate(&model, &fitted, &held)?;
    let mse: f64 = expected
        .fluorescence
        .iter()
        .flatten()
        .zip(actual.fluorescence.iter().flatten())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / (expected.times.len() * model.n()) as f64;
    println!("synthetic held-out pulse MSE: {mse:.6e}; not a biological benchmark");
    if let Some(path) = args.first() {
        let artifact = serde_json::json!({
            "schema_version": 1,
            "evidence": "synthetic parameter recovery; not a biological benchmark",
            "randomness": "none; deterministic fixture and initialization",
            "graph": graph,
            "training_config": config,
            "held_pulse_config": held,
            "optimizer": {"name": "adam", "steps": 80, "learning_rate": 0.08,
                "active_indices": [index], "active_group": "chemical_strength"},
            "parameters": {"truth": truth.raw, "initial": initial.raw, "fitted": fitted.raw},
            "loss_history": history,
            "loss_history_index": "number of completed Adam updates, including zero and final",
            "training": {"times": target.times, "truth": target.fluorescence,
                "initial": initial_trace.fluorescence, "fitted": fitted_trace.fluorescence},
            "held_pulse": {"times": expected.times, "truth": expected.fluorescence,
                "fitted": actual.fluorescence, "mse": mse},
        });
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&artifact).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
