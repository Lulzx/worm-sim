//! Native pair-level atlas labels bound to the existing trace dataset and neuron split.
#[cfg(feature = "hdf5")]
fn main() -> wormsim::Result<()> {
    use std::{fs, path::Path};
    use wormsim::{
        bench::{Dataset, Partition, Split, atlas},
        codec,
    };
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: import_atlas_pairs GRAPH.wsc DATA.json SPLIT.json FUNATLAS.h5 OUTPUT_PREFIX"
                .into(),
        );
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let data: Dataset = serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let split: Split = serde_json::from_slice(&fs::read(&args[3]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    split.validate(&data, &graph)?;
    let evidence = atlas::import_hdf5(
        Path::new(&args[4]),
        &data,
        &graph,
        "53a99055667b853e1d3d6be573ec2613d38c9f6989f302ec38ecd38dd50c7975",
    )?;
    let mut partitions = serde_json::Map::new();
    for (name, partition) in [
        ("train", Partition::Train),
        ("validation", Partition::Validation),
        ("test", Partition::Test),
    ] {
        let pairs = evidence.partition_pairs(&data, &split, partition)?;
        let detected = pairs
            .iter()
            .filter(|p| p.q < evidence.detection_q_threshold)
            .count();
        partitions.insert(name.into(),serde_json::json!({"pairs":pairs.len(),"detected":detected,"not_detected":pairs.len()-detected,"detected_and_equivalent":pairs.iter().filter(|p|p.q<0.05 && p.equivalence_q.is_some_and(|q|q<0.05)).count()}));
    }
    fs::write(
        format!("{}-pairs.json", args[5]),
        serde_json::to_vec(&evidence).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"dataset_hash":evidence.dataset_hash,"graph_hash":evidence.graph_hash,"split_hash":split.content_hash()?,"evidence_hash":evidence.content_hash()?,"source_sha256":evidence.source_sha256,"source_version":evidence.source_version,"detection_q_threshold":evidence.detection_q_threshold,"equivalence_threshold_in_source":evidence.equivalence_threshold,"partitions":partitions,"scope":"Classification evidence preparation only; no model fitted or scored. One ordered non-self pair per observation; finite published detection q defines detected versus not detected. Nonsignificance does not establish physiological absence. Published q is aggregated over source trials, not recomputed from the filtered trace windows. No class merging or left/right guesses."});
    fs::write(
        format!("{}-pair-import.json", args[5]),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{} source-bound pair labels", evidence.pairs.len());
    Ok(())
}
#[cfg(not(feature = "hdf5"))]
fn main() -> wormsim::Result<()> {
    Err("enable hdf5 to import atlas pairs".into())
}
