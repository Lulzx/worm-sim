use std::{fs, time::Instant};
use wormsim::{
    Result, codec,
    data::Graph,
    fixtures,
    model::Model,
    solve::{Config, simulate},
    trace_codec::{self, Matrix},
};
fn load(path: &str) -> Result<wormsim::data::IndexedGraph> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.starts_with(b"WSC1") {
        codec::decode(&bytes)
    } else {
        serde_json::from_slice::<Graph>(&bytes)
            .map_err(|e| e.to_string())?
            .compile()
    }
}
fn write_json(path: &str, value: &impl serde::Serialize) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn benchmark(path: Option<&str>) -> Result<()> {
    let graph = if let Some(path) = path {
        load(path)?
    } else {
        fixtures::synthetic(302, 23, 3).compile()?
    };
    let json = serde_json::to_vec(&graph.graph).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let packed = codec::encode(&graph)?;
    let encode_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let restored = codec::decode(&packed)?;
    let decode_seconds = start.elapsed().as_secs_f64();
    if restored.hash != graph.hash {
        return Err("codec hash mismatch".into());
    }
    let zstd_json = zstd::stream::encode_all(json.as_slice(), 3).map_err(|e| e.to_string())?;
    let model = Model::new(graph)?;
    let params = model.defaults();
    let cfg = Config {
        duration: 100.0,
        save_dt: 1.0,
        ..Config::default()
    };
    simulate(
        &model,
        &params,
        &Config {
            duration: 0.1,
            ..cfg.clone()
        },
    )?;
    let mut seconds = Vec::new();
    let mut checksum = 0.0;
    for _ in 0..3 {
        let start = Instant::now();
        let output = std::hint::black_box(simulate(&model, &params, &cfg)?);
        seconds.push(start.elapsed().as_secs_f64());
        checksum += output.voltage.last().unwrap().iter().sum::<f64>();
    }
    let trace_cfg = Config {
        duration: 2.0,
        save_dt: 0.005,
        events: vec![wormsim::solve::Event::Stimulate {
            neuron: model.graph.names[0].clone(),
            start: 0.1,
            end: 0.7,
            amplitude: 1.0,
        }],
        ..Config::default()
    };
    let trace = simulate(&model, &params, &trace_cfg)?;
    let matrix = Matrix {
        rows: trace.times.len(),
        columns: model.n(),
        values: trace
            .fluorescence
            .iter()
            .flatten()
            .map(|&v| Some(v))
            .collect(),
    };
    let start = Instant::now();
    let archive = trace_codec::encode(&matrix)?;
    let trace_encode_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let decoded = trace_codec::decode_range(&archive, 0, matrix.rows)?;
    let trace_decode_seconds = start.elapsed().as_secs_f64();
    if decoded
        .values
        .iter()
        .zip(&matrix.values)
        .any(|(a, b)| a.map(f64::to_bits) != b.map(f64::to_bits))
    {
        return Err("trace roundtrip failed".into());
    }
    let raw: Vec<u8> = matrix
        .values
        .iter()
        .flat_map(|v| v.unwrap().to_le_bytes())
        .collect();
    let raw_zstd = zstd::stream::encode_all(raw.as_slice(), 3).map_err(|e| e.to_string())?;
    let output = serde_json::json!({
        "fixture":path.unwrap_or("synthetic; not biological data"),"neurons":model.n(),
        "chemical_edges":model.pre.len(),"gap_edges":model.gap_a.len(),
        "state_scalars":model.state_len(),"edge_gate_reference_state_scalars":2*model.n()+model.pre.len(),
        "graph_hash":model.graph.hash,"json_bytes":json.len(),"zstd_json_bytes":zstd_json.len(),
        "wsc1_bytes":packed.len(),"encode_seconds":encode_seconds,"decode_seconds":decode_seconds,
        "simulation":cfg,"wall_seconds":seconds,"checksum":checksum,"target_arch":std::env::consts::ARCH,
        "trace_benchmark":{"config":trace_cfg,"rows":matrix.rows,"columns":matrix.columns,"raw_f64_bytes":raw.len(),"zstd_raw_f64_bytes":raw_zstd.len(),"wst1_bytes":archive.len(),"encode_seconds":trace_encode_seconds,"decode_seconds":trace_decode_seconds}
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
    Ok(())
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("baseline-pack") if args.len()==4 => {
            let bundle:wormsim::baseline::Bundle=serde_json::from_slice(&fs::read(&args[2]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let bytes=wormsim::baseline::pack(&bundle)?;
            fs::write(&args[3],&bytes).map_err(|e|e.to_string())?;
            println!("packed {} models in {} bytes",bundle.models.len(),bytes.len());
        }
        Some("baseline-eval") if args.len()==4 => {
            let bundle=wormsim::baseline::unpack(&fs::read(&args[2]).map_err(|e|e.to_string())?)?;
            let report=wormsim::baseline::evaluate(&bundle)?;
            write_json(&args[3],&report)?;
            for model in &report.models {println!("{}: STAM r={:.6}, correlation r={:.6}, {:.3}s, parity={}",model.name,model.stams_test.correlation,model.correlation_test.correlation,model.prediction_seconds,model.parity_passed);}
            if !report.all_parity_passed {return Err("baseline differs from exported reference; see report".into());}
        }

        Some("import-c302") if args.len()==7 => {
            if args[6]!="--strict"&&args[6]!="--mean-mirrors" {return Err("choose --strict or --mean-mirrors".into());}
            let bytes=fs::read(&args[2]).map_err(|e|e.to_string())?;
            let names:Vec<String>=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let (graph,report)=wormsim::import::c302_csv(&bytes,&names,&args[4],args[6]=="--mean-mirrors")?;
            fs::write(&args[5],codec::encode(&graph)?).map_err(|e|e.to_string())?;
            write_json(&format!("{}.report.json",args[5]),&report)?;
            println!("imported {} neurons, {} chemical edges, {} gaps; {} mirror conflicts reported",graph.names.len(),graph.chemical.len(),graph.gaps.len(),report.gap_conflicts.len());
        }
        Some("pack") if args.len()==4 => {
            let graph=load(&args[2])?;let bytes=codec::encode(&graph)?;
            fs::write(&args[3],&bytes).map_err(|e|e.to_string())?;
            println!("{} bytes; graph {}",bytes.len(),graph.hash);
        }
        Some("unpack") if args.len()==4 => {write_json(&args[3],&load(&args[2])?.graph)?;}
        Some("simulate") if args.len()==5 => {
            let model=Model::new(load(&args[2])?)?;
            let cfg:Config=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let params=model.defaults();let start=Instant::now();let result=simulate(&model,&params,&cfg)?;
            let mut output=serde_json::json!({"schema_version":1,"graph_hash":model.graph.hash,"neuron_order":model.graph.names,"config":cfg,"parameters_raw":params.raw,"seed":0,"stochastic":false,"package_version":env!("CARGO_PKG_VERSION"),"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"elapsed_seconds":start.elapsed().as_secs_f64(),"times":result.times});
            if args[4].ends_with(".wst") {
                let matrix=Matrix {rows:result.times.len(),columns:2*model.n(),values:result.voltage.iter().zip(&result.fluorescence).flat_map(|(v,f)|v.iter().chain(f).map(|&v|Some(v))).collect()};
                fs::write(&args[4],trace_codec::encode(&matrix)?).map_err(|e|e.to_string())?;
                output["trace_columns"]=serde_json::json!("voltage in neuron_order, then fluorescence in neuron_order");
                write_json(&format!("{}.json",args[4]),&output)?;
            } else {
                output["voltage"]=serde_json::json!(result.voltage);output["fluorescence"]=serde_json::json!(result.fluorescence);
                write_json(&args[4],&output)?;
            }
            println!("saved {} samples to {}",result.times.len(),args[4]);
        }
        Some("bench") if args.len()<=3 => {benchmark(args.get(2).map(String::as_str))?;}
        _ => return Err("usage: wormsim baseline-pack BUNDLE.json BUNDLE.wsb | baseline-eval BUNDLE.wsb REPORT.json | import-c302 CSV IDS.json VERSION OUTPUT.wsc --strict|--mean-mirrors | pack GRAPH.json GRAPH.wsc | unpack GRAPH.wsc GRAPH.json | simulate GRAPH CONFIG.json OUTPUT.json|OUTPUT.wst | bench [GRAPH]".into()),
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
