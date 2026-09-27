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
        #[cfg(feature="hdf5")]
        Some("import-wormwideweb") if args.len()==8 => {
            let graph=load(&args[2])?;
            let config:wormsim::recordings::WindowConfig=serde_json::from_slice(&fs::read(&args[6]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let (data,report)=wormsim::recordings::wormwideweb::import(std::path::Path::new(&args[3]),std::path::Path::new(&args[4]),std::path::Path::new(&args[5]),&graph,&config)?;
            write_json(&args[7],&data)?;
            write_json(&format!("{}.import.json",args[7]),&report)?;
            println!("imported {} windows from {} animals; data {}",data.trials.len(),report.animals.len(),report.dataset_hash);
        }
        Some("level0-fit") if args.len()==7 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let config:wormsim::bench::population::FitConfig=serde_json::from_slice(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let (model,report)=wormsim::bench::population::fit(&data,&graph,&split,config,|model,epoch|{write_json(&format!("{}.epoch-{}.json",args[6],epoch.epoch),model)?;write_json(&format!("{}.epoch-{}.report.json",args[6],epoch.epoch),epoch)?;println!("epoch {} validation {:?}",epoch.epoch,epoch.validation_horizon_r2);Ok(())})?;
            write_json(&args[6],&model)?;write_json(&format!("{}.fit.json",args[6]),&report)?;
            println!("selected epoch {}; {} trainable population scalars",model.selected_epoch,model.free_parameters());
        }
        Some("level0-predict") if args.len()==8 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let model:wormsim::bench::population::PopulationModel=serde_json::from_slice(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let partition=match args[6].as_str(){"train"=>wormsim::bench::Partition::Train,"validation"=>wormsim::bench::Partition::Validation,"test"=>wormsim::bench::Partition::Test,_=>return Err("invalid partition".into())};
            write_json(&args[7],&model.predict(&data,&graph,&split,partition)?)?;
        }
        Some("level0-infer") if args.len()==7 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let audit=wormsim::bench::level0::infer_trial(&data,&graph,&split,&args[5],Default::default())?;
            println!("inferred {} state values from {} observed neurons; history objective {:?} -> {:?}; {} seconds",audit.inferred.forecast_state.len(),audit.inferred.observed_neurons,audit.inferred.history_objective.first(),audit.inferred.history_objective.last(),audit.elapsed_seconds);
            write_json(&args[6],&audit)?;
        }
        Some("gru-fit") if args.len()==7 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let config:wormsim::bench::gru::FitConfig=serde_json::from_slice(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let(model,report)=wormsim::bench::gru::fit_select(&data,&graph,&split,config,|m,c|{write_json(&format!("{}.epoch-{}.json",args[6],c.epoch),m)?;write_json(&format!("{}.epoch-{}.report.json",args[6],c.epoch),c)?;println!("epoch {} validation {:?}; training MSE {:?}",c.epoch,c.validation_horizon_r2,c.training_standardized_mse);Ok(())})?;
            write_json(&args[6],&model)?;write_json(&format!("{}.selection.json",args[6]),&report)?;
            println!("selected epoch {}; {} total scalars",model.epoch,model.free_parameters());
        }
        Some("gru-predict") if args.len()==8 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let model:wormsim::bench::gru::GruModel=serde_json::from_slice(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let partition=match args[6].as_str(){"train"=>wormsim::bench::Partition::Train,"validation"=>wormsim::bench::Partition::Validation,"test"=>wormsim::bench::Partition::Test,_=>return Err("invalid partition".into())};
            write_json(&args[7],&model.predict(&data,&graph,&split,partition)?)?;
        }
        Some("lds-fit") if args.len()==7 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let config:wormsim::bench::lds::FitConfig=serde_json::from_slice(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let(model,report)=wormsim::bench::lds::fit_select(&data,&graph,&split,config,|m,c|{write_json(&format!("{}.rank-{}-iteration-{}.json",args[6],c.rank,c.iteration),m)?;write_json(&format!("{}.rank-{}-iteration-{}.report.json",args[6],c.rank,c.iteration),c)?;println!("rank {} iteration {} validation {:?}; preceding training NLL {:?}",c.rank,c.iteration,c.validation_horizon_r2,c.preceding_training_nll_per_observation);Ok(())})?;
            write_json(&args[6],&model)?;write_json(&format!("{}.selection.json",args[6]),&report)?;
            println!("selected rank {} iteration {}; {} parameters",model.gaussian.dim,model.iteration,model.free_parameters());
        }
        Some("lds-predict") if args.len()==8 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let model:wormsim::bench::lds::LatentModel=serde_json::from_slice(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let partition=match args[6].as_str(){"train"=>wormsim::bench::Partition::Train,"validation"=>wormsim::bench::Partition::Validation,"test"=>wormsim::bench::Partition::Test,_=>return Err("invalid partition".into())};
            write_json(&args[7],&model.predict(&data,&graph,&split,partition)?)?;
        }
        Some("linear-fit") if args.len()==6 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let (model,report)=wormsim::bench::linear::fit_select(&data,&graph,&split,&[0.0001,0.001,0.01,0.1,1.0])?;
            write_json(&args[5],&model)?;write_json(&format!("{}.selection.json",args[5]),&report)?;
            println!("selected ridge={}; {} fitted scalars",model.ridge,model.free_parameters());
        }
        Some("linear-predict") if args.len()==8 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let model:wormsim::bench::linear::LinearModel=serde_json::from_slice(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let partition=match args[6].as_str() {"train"=>wormsim::bench::Partition::Train,"validation"=>wormsim::bench::Partition::Validation,"test"=>wormsim::bench::Partition::Test,_=>return Err("partition must be train, validation or test".into())};
            write_json(&args[7],&model.predict(&data,&graph,&split,partition)?)?;
        }
        Some("bench-control") if args.len()==8 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let partition=match args[5].as_str() {"train"=>wormsim::bench::Partition::Train,"validation"=>wormsim::bench::Partition::Validation,"test"=>wormsim::bench::Partition::Test,_=>return Err("invalid partition".into())};
            let control=wormsim::bench::controls::Control::parse(&args[6])?;
            write_json(&args[7],&wormsim::bench::controls::predict(&data,&graph,&split,partition,control)?)?;
        }
        Some("bench-persist") if args.len()==7 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let partition=match args[5].as_str() {"train"=>wormsim::bench::Partition::Train,"validation"=>wormsim::bench::Partition::Validation,"test"=>wormsim::bench::Partition::Test,_=>return Err("partition must be train, validation or test".into())};
            let predictions=wormsim::bench::persistence(&data,&graph,&split,partition)?;
            write_json(&args[6],&predictions)?;
        }
        Some("bench-split") if args.len()==9 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let axis=match args[4].as_str() {"neuron"=>wormsim::bench::Axis::StimulatedNeuron,"animal"=>wormsim::bench::Axis::Animal,_=>return Err("axis must be neuron or animal".into())};
            let split=wormsim::bench::Split::generate(&data,&graph,axis,args[5].parse().map_err(|_|"invalid seed")?,args[6].parse().map_err(|_|"invalid validation group count")?,args[7].parse().map_err(|_|"invalid test group count")?)?;
            write_json(&args[8],&split)?;
            println!("split {}: {} train, {} validation, {} test trials",split.content_hash()?,split.train.len(),split.validation.len(),split.test.len());
        }
        Some("bench-score") if args.len()==8 => {
            let graph=load(&args[2])?;
            let data:wormsim::bench::Dataset=serde_json::from_slice(&fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let split:wormsim::bench::Split=serde_json::from_slice(&fs::read(&args[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let predictions:wormsim::bench::Predictions=serde_json::from_slice(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let partition=match args[6].as_str() {"train"=>wormsim::bench::Partition::Train,"validation"=>wormsim::bench::Partition::Validation,"test"=>wormsim::bench::Partition::Test,_=>return Err("partition must be train, validation or test".into())};
            let report=wormsim::bench::evaluate(&data,&graph,&split,&predictions,partition)?;
            write_json(&args[7],&report)?;
            println!("scored {} trials; correlation={:?}; AUROC={:?}",report.trials,report.macro_trace_correlation,report.response_auroc.value);
        }
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
        _ => return Err("usage: wormsim gru-fit GRAPH DATA.json SPLIT.json CONFIG.json MODEL.json | gru-predict GRAPH DATA.json SPLIT.json MODEL.json PARTITION OUTPUT.json | lds-fit GRAPH DATA.json SPLIT.json CONFIG.json MODEL.json | lds-predict GRAPH DATA.json SPLIT.json MODEL.json PARTITION OUTPUT.json | level0-fit GRAPH DATA.json SPLIT.json CONFIG.json MODEL.json | level0-predict GRAPH DATA.json SPLIT.json MODEL.json PARTITION OUTPUT.json | level0-infer GRAPH DATA.json SPLIT.json TRIAL OUTPUT.json | bench-control GRAPH DATA.json SPLIT.json PARTITION history-mean|half-blend|training-mean|ar PREDICTIONS.json | linear-fit GRAPH DATA.json SPLIT.json MODEL.json | linear-predict GRAPH DATA.json SPLIT.json MODEL.json PARTITION PREDICTIONS.json | bench-persist GRAPH DATA.json SPLIT.json PARTITION PREDICTIONS.json | import-wormwideweb GRAPH H5_DIR LABELS.json RECEIPT.json CONFIG.json OUTPUT.json (hdf5 feature) | bench-split GRAPH DATA.json neuron|animal SEED VALIDATION_GROUPS TEST_GROUPS SPLIT.json | bench-score GRAPH DATA.json SPLIT.json PREDICTIONS.json train|validation|test REPORT.json | baseline-pack BUNDLE.json BUNDLE.wsb | baseline-eval BUNDLE.wsb REPORT.json | import-c302 CSV IDS.json VERSION OUTPUT.wsc --strict|--mean-mirrors | pack GRAPH.json GRAPH.wsc | unpack GRAPH.wsc GRAPH.json | simulate GRAPH CONFIG.json OUTPUT.json|OUTPUT.wst | bench [GRAPH]".into()),
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
