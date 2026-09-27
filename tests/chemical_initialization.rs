use wormsim::{
    fixtures,
    initial_state::{self, Readout},
    model::{Inputs, Model},
    parameters::forecast_defaults,
};

#[test]
fn neutral_reversal_at_zero_rest_blocks_chemical_only_cross_neuron_responses() {
    let model = Model::new(fixtures::synthetic(2, 1, 0).compile().unwrap()).unwrap();
    assert!(model.gap_a.is_empty());
    let mut params = forecast_defaults(&model);
    let sign_start = 6 * model.n() + model.pre.len();
    params.raw[sign_start..sign_start + model.pre.len()].fill(0.);
    let prepared = model.prepare(&params).unwrap();
    let initial = model.initial(&prepared);
    assert_eq!(&initial[..model.n()], &[0., 0.]);
    assert!(prepared.reversal.iter().all(|v| *v == 0.));
    let times = vec![0., 0.5, 1., 1.5, 2.];
    let mut currents = vec![vec![0.; model.n()]; times.len()];
    currents[0][0] = 0.5;
    let response = initial_state::prepared_response_with_currents(
        &model,
        &params,
        &initial,
        &times,
        &Readout::identity(model.n()),
        0.01,
        &currents,
        2.,
    )
    .unwrap();
    assert!(response.iter().any(|r| r[0] > 1e-6));
    assert!(response.iter().all(|r| r[1] == 0.));

    // A nonneutral reversal permits propagation after parameter-dependent prep.
    params.raw[sign_start..sign_start + model.pre.len()].fill((0.9_f64 / 0.1).ln());
    let response = initial_state::prepared_response_with_currents(
        &model,
        &params,
        &initial,
        &times,
        &Readout::identity(model.n()),
        0.01,
        &currents,
        60.,
    )
    .unwrap();
    assert!(response.iter().any(|r| r[1].abs() > 1e-7));
}

#[test]
fn chemical_gate_to_voltage_derivative_matches_rhs_finite_differences() {
    let model = Model::new(fixtures::synthetic(3, 1, 0).compile().unwrap()).unwrap();
    let raw = forecast_defaults(&model);
    let prepared = model.prepare(&raw).unwrap();
    let mut state = model.initial(&prepared);
    for (i, v) in state[..model.n()].iter_mut().enumerate() {
        *v += 0.1 * (i + 1) as f64;
    }
    let rhs = |state: &[f64]| {
        let mut dy = vec![0.; model.state_len()];
        model.rhs(
            &prepared,
            state,
            &Inputs::new(model.n()),
            &mut vec![0.; model.n()],
            &mut dy,
        );
        dy
    };
    for e in 0..model.pre.len() {
        let (pre, post) = (model.pre[e] as usize, model.post[e] as usize);
        let mut plus = state.clone();
        let mut minus = state.clone();
        plus[2 * model.n() + pre] += 1e-5;
        minus[2 * model.n() + pre] -= 1e-5;
        let derivative = (rhs(&plus)[post] - rhs(&minus)[post]) / 2e-5;
        let expected =
            prepared.inv_tau[post] * prepared.weight[e] * (prepared.reversal[e] - state[post]);
        assert!((derivative - expected).abs() < 1e-10);
    }
}
