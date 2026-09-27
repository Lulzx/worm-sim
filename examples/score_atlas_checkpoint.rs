//! Validation-only scoring boundary for external atlas fitters.
use std::{fs, io::Write, path::Path};
use wormsim::{
    Result,
    bench::{self, Dataset, Partition, Split, atlas_level0::AtlasModel},
    codec,
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn write(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?
        .write_all(&serde_json::to_vec(value).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: score_atlas_checkpoint GRAPH DATA SPLIT MODEL NEW_DIRECTORY".into());
    }
    let graph = codec::decode(&fs::read(&a[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&a[2])?;
    let split: Split = read(&a[3])?;
    let model: AtlasModel = read(&a[4])?;
    let prediction = model.predict(&data, &graph, &split, Partition::Validation)?;
    let report = bench::evaluate(&data, &graph, &split, &prediction, Partition::Validation)?;
    let mse = report
        .pooled_trace_scores
        .mse
        .ok_or("undefined validation MSE")?;
    let out = Path::new(&a[5]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    write(out.join("predictions.json"), &prediction)?;
    write(out.join("report.json"), &report)?;
    write(
        out.join("selection.json"),
        &serde_json::json!({"epoch":model.epoch,"validation_mse":mse,"scorer_source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"model_source_commit":model.source_commit,"partition":"validation"}),
    )?;
    println!("epoch {} validation MSE {mse}", model.epoch);
    Ok(())
}
