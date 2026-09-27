use std::collections::BTreeMap;
use wormsim::{
    fixtures,
    molecular::{
        self, Catalog, CellMapping, EvidenceState as State, Expression, Gene, Polarity, Receptor,
        Source, Transmitters,
    },
};
fn source() -> Source {
    Source {
        url: "https://example.org/synthetic".into(),
        sha256: "0".repeat(64),
        version: "synthetic".into(),
        license: "synthetic fixture".into(),
    }
}
fn fixture() -> (
    wormsim::data::IndexedGraph,
    Catalog,
    Expression,
    CellMapping,
) {
    let graph = fixtures::synthetic(4, 3, 0).compile().unwrap();
    let catalog = Catalog {
        schema_version: 1,
        source: source(),
        transmitters: graph
            .names
            .iter()
            .enumerate()
            .map(|(i, n)| Transmitters {
                neuron: n.clone(),
                source_neuron: n.clone(),
                dominant: Some("Glu".into()),
                alternative: None,
                source_row: i + 2,
            })
            .collect(),
        receptors: vec![
            Receptor {
                gene: "positive".into(),
                transmitter: "Glu".into(),
                polarity: Polarity::Excitatory,
                source_cell: "A2".into(),
            },
            Receptor {
                gene: "negative".into(),
                transmitter: "Glu".into(),
                polarity: Polarity::Inhibitory,
                source_cell: "B2".into(),
            },
        ],
    };
    let expression = Expression {
        schema_version: 1,
        source: source(),
        threshold: 4,
        classes: vec!["p".into(), "n".into(), "both".into(), "zero".into()],
        genes: vec![
            Gene {
                name: "positive".into(),
                wormbase_id: "gp".into(),
            },
            Gene {
                name: "negative".into(),
                wormbase_id: "gn".into(),
            },
        ],
        tpm: vec![1., 0., 0., 2., 1., 2., 0., 0.],
        missing_genes: vec![],
    };
    let mapping = CellMapping {
        schema_version: 1,
        graph_hash: graph.hash.clone(),
        provenance: "explicit synthetic mapping".into(),
        cell_to_class: graph
            .names
            .iter()
            .cloned()
            .zip(expression.classes.iter().cloned())
            .collect(),
        unmapped: BTreeMap::new(),
    };
    (graph, catalog, expression, mapping)
}
#[test]
fn polarity_is_directed_and_preserves_conflict_missing_and_unmapped_evidence() {
    let (graph, mut catalog, mut expression, mut mapping) = fixture();
    let evidence = molecular::infer(&graph, &catalog, &expression, &mapping).unwrap();
    assert_eq!(evidence.edges.len(), graph.chemical.len());
    for e in &evidence.edges {
        let expected = match e.post.as_str() {
            "N000" => State::Excitatory,
            "N001" => State::Inhibitory,
            "N002" => State::Conflicting,
            _ => State::NoDetectedReceptor,
        };
        assert_eq!(e.state, expected);
    }
    let probabilities = evidence.probabilities(0.9).unwrap();
    assert!(probabilities.iter().any(|p| (*p - 0.1).abs() < 1e-12));
    assert!(probabilities.contains(&0.9) && probabilities.contains(&0.5));
    assert!(evidence.probabilities(1.).is_err());
    assert!(evidence.probabilities(f64::NAN).is_err());
    mapping.cell_to_class.remove("N003");
    mapping
        .unmapped
        .insert("N003".into(), "unknown class".into());
    catalog.transmitters[0].dominant = None;
    let e = molecular::infer(&graph, &catalog, &expression, &mapping).unwrap();
    assert!(
        e.edges
            .iter()
            .filter(|e| e.pre == "N000")
            .all(|e| e.state == State::NoTransmitterEvidence)
    );
    assert!(
        e.edges
            .iter()
            .filter(|e| e.pre != "N000" && e.post == "N003")
            .all(|e| e.state == State::UnmappedPostsynapticClass)
    );
    // A missing negative receptor is unknown, not an observed zero.
    expression.genes.pop();
    expression.tpm = vec![1., 0., 1., 0.];
    expression.missing_genes.push("negative".into());
    let e = molecular::infer(&graph, &catalog, &expression, &mapping).unwrap();
    assert!(
        e.edges
            .iter()
            .filter(|e| e.pre != "N000" && e.post == "N000")
            .all(|e| e.state == State::IncompleteReceptors
                && e.missing_receptor_genes == ["negative"])
    );
    assert!(e.probabilities(0.9).unwrap().iter().all(|p| *p == 0.5));
}
#[test]
fn alternative_transmitters_can_create_conflicting_evidence() {
    let (graph, mut catalog, mut expression, mapping) = fixture();
    catalog.transmitters[1].alternative = Some("GABA".into());
    catalog.receptors.extend([
        Receptor {
            gene: "negative".into(),
            transmitter: "GABA".into(),
            polarity: Polarity::Excitatory,
            source_cell: "E2".into(),
        },
        Receptor {
            gene: "positive".into(),
            transmitter: "GABA".into(),
            polarity: Polarity::Inhibitory,
            source_cell: "F2".into(),
        },
    ]);
    let e = molecular::infer(&graph, &catalog, &expression, &mapping).unwrap();
    let edge = e
        .edges
        .iter()
        .find(|e| e.pre == "N001" && e.post == "N000")
        .unwrap();
    assert_eq!(edge.state, State::Conflicting);
    assert_eq!(edge.transmitters, vec!["GABA", "Glu"]);
    expression.tpm[0] = f32::NAN;
    assert!(molecular::infer(&graph, &catalog, &expression, &mapping).is_err());
}
#[test]
fn mapping_and_catalog_coverage_fail_closed() {
    let (graph, catalog, expression, mapping) = fixture();
    let mut bad = mapping.clone();
    bad.cell_to_class.insert("N000".into(), "unknown".into());
    assert!(molecular::infer(&graph, &catalog, &expression, &bad).is_err());
    let mut bad = mapping.clone();
    bad.cell_to_class.remove("N000");
    assert!(molecular::infer(&graph, &catalog, &expression, &bad).is_err());
    let mut bad = catalog.clone();
    bad.transmitters.pop();
    assert!(molecular::infer(&graph, &bad, &expression, &mapping).is_err());
    let mut bad = expression.clone();
    bad.missing_genes.push("positive".into());
    assert!(molecular::infer(&graph, &catalog, &bad, &mapping).is_err());
    let mut bad = catalog;
    bad.receptors.pop();
    assert!(bad.validate().is_err());
}
#[cfg(feature = "hdf5")]
#[test]
fn native_expression_import_checks_hash_orientation_missing_names_and_threshold() {
    use sha2::{Digest, Sha256};
    let path = std::env::temp_dir().join(format!("wormsim-expression-{}.h5", std::process::id()));
    {
        let file = hdf5::File::create(&path).unwrap();
        for (name, strings) in [
            ("neuron_ids", ["classB", "classA"]),
            ("gene_names_th4", ["b", "a"]),
            ("gene_wbids_th4", ["wb_b", "wb_a"]),
        ] {
            let values: Vec<hdf5::types::VarLenUnicode> =
                strings.iter().map(|s| s.parse().unwrap()).collect();
            file.new_dataset_builder()
                .with_data(&values)
                .create(name)
                .unwrap();
        }
        file.new_dataset::<f32>()
            .shape((2, 2))
            .create("tpm_th4")
            .unwrap()
            .write_raw(&[0., 2., 3., 0.])
            .unwrap();
    }
    let mut provenance = source();
    provenance.sha256 = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
    let requested = ["a", "b", "missing"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let e = molecular::import_cengen(&path, &requested, 4, provenance.clone()).unwrap();
    assert_eq!(e.classes, ["classB", "classA"]);
    assert_eq!(
        e.genes.iter().map(|g| g.name.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(e.tpm, [2., 0., 0., 3.]);
    assert_eq!(e.missing_genes, ["missing"]);
    assert!(molecular::import_cengen(&path, &requested, 0, provenance.clone()).is_err());
    provenance.sha256 = "0".repeat(64);
    assert!(molecular::import_cengen(&path, &requested, 4, provenance).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn compact_sign_projection_keeps_uncertain_edges_neutral_and_checks_lineage() {
    use wormsim::molecular::SignPriors;
    let (graph, catalog, expression, mapping) = fixture();
    let e = molecular::infer(&graph, &catalog, &expression, &mapping).unwrap();
    let p = SignPriors::from_evidence(&e, &graph, 0.75).unwrap();
    assert_eq!(
        p.probabilities(&graph).unwrap(),
        e.probabilities(0.75).unwrap()
    );
    assert_eq!(p.evidence_hash, molecular::content_hash(&e).unwrap());
    let mut broken = p.clone();
    broken.inhibitory_edges = broken.excitatory_edges.clone();
    assert!(broken.probabilities(&graph).is_err());
    let mut broken = p.clone();
    broken.graph_hash = "0".repeat(64);
    assert!(broken.probabilities(&graph).is_err());
    let mut broken = p;
    broken.excitatory_edges.push(graph.chemical.len());
    assert!(broken.probabilities(&graph).is_err());
    let mut broken = e.clone();
    broken.edges.swap(0, 1);
    assert!(SignPriors::from_evidence(&broken, &graph, 0.75).is_err());
    let mut broken = e.clone();
    let edge = broken
        .edges
        .iter_mut()
        .find(|e| e.state == State::Conflicting)
        .unwrap();
    edge.state = State::Excitatory;
    assert!(SignPriors::from_evidence(&broken, &graph, 0.75).is_err());
    assert!(SignPriors::from_evidence(&e, &graph, 0.5).is_err());
}
