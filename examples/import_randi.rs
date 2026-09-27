//! Reproducible native atlas import plus held-out stimulated-neuron split.
use std::{fs, path::Path};
use wormsim::{
    Result,
    bench::{Axis, Split},
    codec,
    recordings::randi,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: import_randi GRAPH.wsc SOURCE_DIR MANIFEST.json CONFIG.json OUTPUT_PREFIX"
                .into(),
        );
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let config = serde_json::from_slice(&fs::read(&args[4]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let (data, report) = randi::import(Path::new(&args[2]), Path::new(&args[3]), &graph, config)?;
    let split = Split::generate(&data, &graph, Axis::StimulatedNeuron, 42, 15, 15)?;
    for (suffix, value) in [
        ("data", serde_json::to_vec(&data)),
        ("import", serde_json::to_vec_pretty(&report)),
        ("split", serde_json::to_vec_pretty(&split)),
    ] {
        fs::write(
            format!("{}-{suffix}.json", args[5]),
            value.map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    println!(
        "{} trials, {} source recordings; train/validation/test {}/{}/{}; hash {}",
        data.trials.len(),
        report.source_recordings,
        split.train.len(),
        split.validation.len(),
        split.test.len(),
        report.dataset_hash
    );
    Ok(())
}
