//! Export observation-free plans and independently score external predictions.
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
use wormsim::{
    Result,
    bench::{
        self, Dataset, Partition, Predictions, Split,
        atlas_level0::AtlasModel,
        external_atlas::{Checkpoint, PredictionPlan},
    },
    codec,
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn write(p: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(p)
        .map_err(|e| e.to_string())?
        .write_all(&serde_json::to_vec(value).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if !matches!(a.get(1).map(String::as_str), Some("plan" | "score"))
        || (a[1] == "plan" && a.len() != 8)
        || (a[1] == "score" && a.len() != 9)
    {
        return Err("usage: external_atlas plan GRAPH DATA SPLIT BASE_MODEL validation|test NEW_PLAN.json; external_atlas score GRAPH DATA SPLIT CHECKPOINT validation|test PREDICTIONS NEW_DIRECTORY".into());
    }
    let graph = codec::decode(&fs::read(&a[2]).map_err(|e| e.to_string())?)?;
    let data: Dataset = read(&a[3])?;
    let split: Split = read(&a[4])?;
    let partition = match a[6].as_str() {
        "validation" => Partition::Validation,
        "test" => Partition::Test,
        _ => return Err("invalid partition".into()),
    };
    if a[1] == "plan" {
        let model: AtlasModel = read(&a[5])?;
        write(
            &a[7],
            &PredictionPlan::new(&model, &data, &graph, &split, partition)?,
        )?;
    } else {
        let checkpoint: Checkpoint = read(&a[5])?;
        checkpoint.validate(&data, &graph, &split)?;
        let prediction: Predictions = read(&a[7])?;
        let checkpoint_hash = format!(
            "{:x}",
            Sha256::digest(fs::read(&a[5]).map_err(|e| e.to_string())?)
        );
        if prediction.model != format!("jax-atlas:{checkpoint_hash}")
            || prediction.training_trials != checkpoint.base_model.training_trials
            || prediction.selection_trials != checkpoint.base_model.selection_trials
            || prediction.source_commit != checkpoint.base_model.source_commit
        {
            return Err("external prediction/checkpoint lineage mismatch".into());
        }
        let report = bench::evaluate(&data, &graph, &split, &prediction, partition)?;
        let mse = report
            .pooled_trace_scores
            .mse
            .ok_or("undefined external atlas MSE")?;
        let out = Path::new(&a[8]);
        fs::create_dir(out).map_err(|e| e.to_string())?;
        write(out.join("report.json"), &report)?;
        write(
            out.join("selection.json"),
            &serde_json::json!({
                "epoch":checkpoint.base_model.epoch,"partition":partition,"mse":mse,
                "checkpoint_sha256":checkpoint_hash,
                "predictions_sha256":format!("{:x}",Sha256::digest(fs::read(&a[7]).map_err(|e|e.to_string())?)),
                "scorer_source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),
                "scope":"Independent trace scoring and declared lineage validation; Rust does not reproduce extension dynamics."
            }),
        )?;
        println!(
            "epoch {} {partition:?} MSE {mse}",
            checkpoint.base_model.epoch
        );
    }
    Ok(())
}
