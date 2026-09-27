use std::collections::BTreeMap;
use wormsim::{
    data::{Provenance, Recording, Trace},
    fixtures,
    initial_state::{self, InferenceConfig, InferenceMethod, Readout},
    model::Model,
    parameters::forecast_defaults,
};
fn recording(values: Vec<Option<f64>>) -> Recording {
    Recording {
        dataset: "synthetic".into(),
        animal_id: "a".into(),
        condition: "filter".into(),
        times: vec![0.0, 0.1, 0.2],
        traces: vec![Trace {
            neuron: "N000".into(),
            values,
            provenance: Provenance {
                dataset: "synthetic".into(),
                version: "1".into(),
                id_confidence: 1.0,
            },
        }],
        behavior: BTreeMap::new(),
    }
}
#[test]
fn calcium_filter_matches_scalar_kalman_equations_and_ignores_future() {
    let model = Model::new(fixtures::synthetic(2, 0, 0).compile().unwrap()).unwrap();
    let mut params = forecast_defaults(&model);
    for i in 0..2 {
        params.raw[3 * model.n() + i] = -50.0;
    }
    let p = model.prepare(&params).unwrap();
    let readout = Readout::identity(2);
    let cfg = InferenceConfig {
        method: InferenceMethod::BlockEkf,
        dt: 0.01,
        ..Default::default()
    };
    let mut data = recording(vec![Some(0.65), Some(0.2), Some(999.0)]);
    let actual = initial_state::infer(&model, &params, &data, 0.1, &readout, &cfg).unwrap();
    let mut mean = 0.5;
    let mut variance = cfg.filter.initial_variance[1];
    let r = cfg.filter.observation_variance;
    let scale = p.calcium_scale[0];
    let k = variance / (variance + r);
    mean += k * (0.65 / scale - mean);
    variance *= 1.0 - k;
    assert!((actual.history_predictions[0][0] - scale * mean).abs() < 1e-12);
    // Independent closed-form ten-step propagation of scalar calcium AR(1).
    let a = 1.0 - cfg.dt * p.inv_calcium_tau[0];
    mean = 0.5 + a.powi(10) * (mean - 0.5);
    variance = a.powi(20) * variance
        + cfg.dt * cfg.filter.process_variance_rate[1] * (1.0 - a.powi(20)) / (1.0 - a * a);
    let k = variance / (variance + r);
    mean += k * (0.2 / scale - mean);
    variance *= 1.0 - k;
    assert!((actual.forecast_state[2] - mean).abs() < 1e-12);
    assert!(
        (actual
            .filter_diagnostics
            .as_ref()
            .unwrap()
            .forecast_covariance_blocks[0][4]
            - variance)
            .abs()
            < 1e-12
    );
    assert_eq!(actual.observed_neurons, 1);
    assert_eq!(actual.latent_neurons, 1);
    data.traces[0].values[2] = Some(-9999.0);
    let other = initial_state::infer(&model, &params, &data, 0.1, &readout, &cfg).unwrap();
    assert_eq!(actual.forecast_state, other.forecast_state);
    assert_eq!(actual.history_predictions, other.history_predictions);
    for p in &actual
        .filter_diagnostics
        .unwrap()
        .forecast_covariance_blocks
    {
        for i in 0..3 {
            assert!(p[i * 3 + i] >= 0.0);
            for j in 0..3 {
                assert!((p[i * 3 + j] - p[j * 3 + i]).abs() < 1e-14);
                assert!(p[i * 3 + i] * p[j * 3 + j] - p[i * 3 + j] * p[j * 3 + i] >= -1e-12);
            }
        }
    }
}
#[test]
fn filter_skips_missing_samples_and_rejects_invalid_noise() {
    let model = Model::new(fixtures::synthetic(2, 0, 0).compile().unwrap()).unwrap();
    let params = forecast_defaults(&model);
    let data = recording(vec![Some(0.5), None, Some(0.9)]);
    let mut cfg = InferenceConfig {
        method: InferenceMethod::BlockEkf,
        dt: 0.01,
        ..Default::default()
    };
    let out =
        initial_state::infer(&model, &params, &data, 0.1, &Readout::identity(2), &cfg).unwrap();
    assert_eq!(out.filter_diagnostics.unwrap().observation_updates, 1);
    cfg.filter.observation_variance = 0.0;
    assert!(
        initial_state::infer(&model, &params, &data, 0.1, &Readout::identity(2), &cfg).is_err()
    );
    let legacy: InferenceConfig = serde_json::from_str(
        r#"{"dt":0.01,"iterations":8,"learning_rate":0.02,"prior_weight":0.001}"#,
    )
    .unwrap();
    assert_eq!(legacy.method, InferenceMethod::Shooting);
}
