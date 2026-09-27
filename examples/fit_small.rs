use wormsim::{
    Result,
    data::{Provenance, Recording, Trace},
    fit, fixtures,
    model::Model,
    solve::{Config, Event, simulate},
};
fn main() -> Result<()> {
    let model = Model::new(fixtures::synthetic(3, 1, 0).compile()?)?;
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
    let history = fit::adam(&model, &mut fitted, &config, &recording, &[index], 80, 0.08)?;
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
        ..config
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
    Ok(())
}
