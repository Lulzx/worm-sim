//! Native import of pinned CeNGEN receptor expression and directed polarity evidence.
#[cfg(feature = "hdf5")]
fn main() -> wormsim::Result<()> {
    use std::{fs, path::Path};
    use wormsim::{
        Result, codec,
        molecular::{self, Catalog, CellMapping, Source},
    };
    fn read<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
        serde_json::from_slice(&fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    }
    fn write(p: impl AsRef<Path>, v: &impl serde::Serialize) -> Result<()> {
        fs::write(p, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())
    }
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: import_molecular_evidence GRAPH CATALOG CENGEN_H5 CELL_MAP THRESHOLD NEW_OUTPUT_DIRECTORY".into());
    }
    let graph = codec::decode(&fs::read(&a[1]).map_err(|e| e.to_string())?)?;
    let catalog: Catalog = read(&a[2])?;
    catalog.validate()?;
    let mapping: CellMapping = read(&a[4])?;
    let threshold = a[5].parse().map_err(|_| "invalid threshold")?;
    let source=Source {
        url:"https://raw.githubusercontent.com/francescorandi/wormneuroatlas/b2e13d88b670efcb3438aeacba2ad4bd6c383933/wormneuroatlas/data/cengen.h5".into(),
        sha256:"d8e6f6f2a25e05211676cfd9c3dd18b8a8bb5c3db4d673fd87b61ed91637ed23".into(),
        version:"CeNGEN 021821; Taylor et al. 2021; L4; Worm Neuro Atlas pinned mirror".into(),
        license:"Source data attribution to Taylor et al.; mirror repository GPL-3.0; no source software copied".into(),
    };
    let expression =
        molecular::import_cengen(Path::new(&a[3]), &catalog.genes(), threshold, source)?;
    let evidence = molecular::infer(&graph, &catalog, &expression, &mapping)?;
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for e in &evidence.edges {
        *counts
            .entry(
                serde_json::to_value(&e.state)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .into(),
            )
            .or_default() += 1;
    }
    let report = serde_json::json!({"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"graph_hash":graph.hash,"catalog_hash":evidence.catalog_hash,"expression_hash":evidence.expression_hash,"mapping_hash":evidence.mapping_hash,"evidence_hash":molecular::content_hash(&evidence)?,"threshold":threshold,"expression_classes":expression.classes.len(),"present_receptor_genes":expression.genes.len(),"missing_receptor_genes":expression.missing_genes,"mapped_cells":mapping.cell_to_class.len(),"unmapped_cells":mapping.unmapped,"edge_states":counts,"scope":"Qualitative ionotropic molecular evidence, not calibrated probabilities. Mapping is an explicit L4-to-anatomical-cell transfer assumption; no response labels used. Existing graph and benchmark datasets are unchanged."});
    let output = Path::new(&a[6]);
    fs::create_dir(output).map_err(|e| e.to_string())?;
    write(output.join("expression.json"), &expression)?;
    write(output.join("evidence.json"), &evidence)?;
    write(output.join("report.json"), &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
#[cfg(not(feature = "hdf5"))]
fn main() {
    eprintln!("requires --features hdf5");
    std::process::exit(1);
}
