//! Local chemical gate-to-voltage sensitivity at the fitted preparation state.
use sha2::{Digest, Sha256};
use std::fs;
use wormsim::{Result, bench::atlas_level0::AtlasModel, codec, initial_state, model::Model};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: diagnose_atlas_coupling GRAPH MODEL OUTPUT.json".into());
    }
    let graph = codec::decode(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let bytes = fs::read(&args[2]).map_err(|e| e.to_string())?;
    let fitted: AtlasModel = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if graph.hash != fitted.graph_hash {
        return Err("model graph mismatch".into());
    }
    let model = Model::new(graph)?;
    let raw = fitted.parameters.expand(&model)?;
    let prepared = model.prepare(&raw)?;
    let (state, derivative) = initial_state::prepared_state(
        &model,
        &raw,
        &fitted.initial,
        fitted.config.dt,
        fitted.config.preparation_seconds,
    )?;
    let mut edges = vec![];
    let mut sum_squared = 0.;
    let mut zero = 0;
    let mut positive = 0;
    let mut negative = 0;
    for (post, voltage) in state.iter().enumerate().take(model.n()) {
        for e in model.incoming_offsets[post]..model.incoming_offsets[post + 1] {
            let driving_force = prepared.reversal[e] - voltage;
            let sensitivity = prepared.inv_tau[post] * prepared.weight[e] * driving_force;
            if !sensitivity.is_finite() {
                return Err("nonfinite gate sensitivity".into());
            }
            sum_squared += sensitivity * sensitivity;
            zero += usize::from(sensitivity == 0.);
            positive += usize::from(sensitivity > 0.);
            negative += usize::from(sensitivity < 0.);
            edges.push(serde_json::json!({"pre":model.graph.names[model.pre[e] as usize],"post":model.graph.names[post],"driving_force":driving_force,"d_voltage_rate_d_gate":sensitivity}));
        }
    }
    let report = serde_json::json!({"schema_version":1,"source_commit":option_env!("WORMSIM_COMMIT").unwrap_or("unversioned"),"model_source_commit":fitted.source_commit,"model_sha256":format!("{:x}",Sha256::digest(bytes)),"graph_hash":fitted.graph_hash,"epoch":fitted.epoch,"preparation_seconds":fitted.config.preparation_seconds,"dt":fitted.config.dt,"chemical_edges":edges.len(),"exact_zero_gate_sensitivity_edges":zero,"positive_gate_sensitivity_edges":positive,"negative_gate_sensitivity_edges":negative,"gate_sensitivity_l2":sum_squared.sqrt(),"prepared_derivative_l2":derivative.iter().map(|v|v*v).sum::<f64>().sqrt(),"edges":edges,"scope":"Local partial derivative of postsynaptic voltage rate with respect to presynaptic gate at the prepared state, with no perturbation masks. Includes chemical synapse counts and membrane time constants. Not the full recurrent transfer function, a parameter gradient, physiological polarity evidence or a held-out score. Nonzero gap-junction transmission and chemical shunting can remain when this derivative vanishes."});
    fs::write(
        &args[3],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "zero chemical gate derivatives {zero}/{}; L2 {}",
        edges.len(),
        sum_squared.sqrt()
    );
    Ok(())
}
