//! Freeze a compact molecular-prior projection into an atlas fit configuration.
use std::{
    fs::{self, OpenOptions},
    io::Write,
};
use wormsim::{
    Result,
    bench::atlas_level0::FitConfig,
    codec,
    molecular::{Evidence, SignPriors},
};
fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: configure_molecular_fit GRAPH BASE_CONFIG MOLECULAR_EVIDENCE CONFIDENCE NEW_CONFIG".into());
    }
    let graph = codec::decode(&fs::read(&a[1]).map_err(|e| e.to_string())?)?;
    let mut config: FitConfig = read(&a[2])?;
    if config.molecular_sign_priors.is_some() {
        return Err("base configuration already has molecular priors".into());
    }
    let evidence: Evidence = read(&a[3])?;
    let confidence = a[4].parse().map_err(|_| "invalid confidence")?;
    config.molecular_sign_priors = Some(SignPriors::from_evidence(&evidence, &graph, confidence)?);
    let bytes = serde_json::to_vec_pretty(&config).map_err(|e| e.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a[5])
        .map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(())
}
