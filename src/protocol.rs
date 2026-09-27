//! Shared JSON/YAML declarative simulation protocols (specification section 9).
use crate::{Result, solve::Config};
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Json,
    Yaml,
}
/// Parse the same strict Config schema in either format. Neuron identities and
/// event intervals are checked by the simulator against the selected graph.
pub fn parse(bytes: &[u8], format: Format) -> Result<Config> {
    let config: Config = match format {
        Format::Json => serde_json::from_slice(bytes).map_err(|e| e.to_string())?,
        Format::Yaml => serde_yaml_ng::from_slice(bytes).map_err(|e| e.to_string())?,
    };
    config.validate()?;
    Ok(config)
}
/// `.yaml` and `.yml` (case-insensitive) select YAML; other paths retain the
/// existing JSON behavior. Parsing errors never trigger a fallback format.
pub fn read(path: impl AsRef<Path>) -> Result<Config> {
    let path = path.as_ref();
    let format = match path.extension().and_then(|e| e.to_str()) {
        Some(e) if e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml") => Format::Yaml,
        _ => Format::Json,
    };
    parse(&fs::read(path).map_err(|e| e.to_string())?, format)
}
