use wormsim::{
    fixtures,
    model::Model,
    protocol::{self, Format},
    solve,
};
#[test]
fn yaml_and_json_share_schema_and_dynamics() {
    let yaml = include_bytes!("../examples/aval-waveform.yaml");
    let json = include_bytes!("../examples/aval-waveform.json");
    let a = protocol::parse(yaml, Format::Yaml).unwrap();
    let b = protocol::parse(json, Format::Json).unwrap();
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap()
    );
    let yaml = String::from_utf8(yaml.to_vec())
        .unwrap()
        .replace("AVAL", "N000");
    let json = String::from_utf8(json.to_vec())
        .unwrap()
        .replace("AVAL", "N000");
    let m = Model::new(fixtures::synthetic(1, 0, 0).compile().unwrap()).unwrap();
    let a = solve::simulate(
        &m,
        &m.defaults(),
        &protocol::parse(yaml.as_bytes(), Format::Yaml).unwrap(),
    )
    .unwrap();
    let b = solve::simulate(
        &m,
        &m.defaults(),
        &protocol::parse(json.as_bytes(), Format::Json).unwrap(),
    )
    .unwrap();
    assert_eq!(a.times, b.times);
    assert_eq!(a.voltage, b.voltage);
    assert_eq!(a.fluorescence, b.fluorescence);
}
#[test]
fn yaml_rejects_ambiguous_or_invalid_configurations() {
    let base = "duration: 1\ndt: 0.01\nsave_dt: 0.1\nmethod: rk4\n";
    for suffix in [
        "unknown: 1\n",
        "duration: 2\n",
        "---\nduration: 1\n",
        "events:\n - operation: unknown\n   neuron: N000\n",
        "events:\n - operation: ablate\n   neuron: N000\n   amplitude: 1\n",
    ] {
        assert!(
            protocol::parse(format!("{base}{suffix}").as_bytes(), Format::Yaml).is_err(),
            "{suffix}"
        );
    }
    for dt in [".nan", ".inf", "0", "-1"] {
        assert!(
            protocol::parse(
                base.replace("dt: 0.01", &format!("dt: {dt}")).as_bytes(),
                Format::Yaml
            )
            .is_err()
        );
    }
}
#[test]
fn invalid_neuron_is_rejected_by_simulation() {
    let cfg = protocol::parse(
        include_bytes!("../examples/aval-waveform.yaml"),
        Format::Yaml,
    )
    .unwrap();
    let m = Model::new(fixtures::synthetic(1, 0, 0).compile().unwrap()).unwrap();
    assert!(solve::simulate(&m, &m.defaults(), &cfg).is_err());
}
