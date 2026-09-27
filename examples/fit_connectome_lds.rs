//! Reproducible atlas population fit. Output directory must not already exist.
use std::{fs, path::Path};
use wormsim::{
    Result,
    bench::{
        self, Dataset, Partition, Split,
        connectome_fit::{self, FitConfig},
    },
    codec,
};
fn read<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn write(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: fit_connectome_lds GRAPH DATA SPLIT CONFIG NEW_OUTPUT_DIR".into());
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&args[2])?;
    let split: Split = read(&args[3])?;
    let config: FitConfig = read(&args[4])?;
    split.validate(&data, &graph)?;
    let output = Path::new(&args[5]);
    fs::create_dir(output).map_err(|e| e.to_string())?;
    write(output.join("config.json"), &config)?;
    let (model, candidates) =
        connectome_fit::fit_select(&data, &graph, &split, config, |model, candidate| {
            write(
                output.join(format!("iteration-{}.json", candidate.iteration)),
                model,
            )?;
            write(
                output.join(format!("iteration-{}.report.json", candidate.iteration)),
                candidate,
            )?;
            println!(
                "iteration {} validation MSE {:.9}, correlation {:?}, training seconds {:?}",
                candidate.iteration,
                candidate.validation_mse,
                candidate.validation_correlation,
                candidate.training_step.as_ref().map(|s| s.elapsed_seconds)
            );
            Ok(())
        })?;
    write(output.join("selected.json"), &model)?;
    write(output.join("selection.json"), &candidates)?;
    for (name, partition) in [
        ("validation", Partition::Validation),
        ("test", Partition::Test),
    ] {
        let prediction = model.predict(&data, &graph, &split, partition)?;
        let report = bench::evaluate(&data, &graph, &split, &prediction, partition)?;
        write(output.join(format!("{name}-predictions.json")), &prediction)?;
        write(output.join(format!("{name}-report.json")), &report)?;
        println!(
            "selected iteration {} {name} MSE {:?}, correlation {:?}",
            model.iteration, report.pooled_trace_scores.mse, report.macro_trace_correlation
        );
    }
    Ok(())
}
