use std::collections::BTreeMap;
use wormsim::{
    bench::atlas_correlation,
    data::{Provenance, Recording, Trace},
    fixtures,
};

#[test]
fn masked_pair_correlation_gradient_matches_finite_differences() {
    let graph = fixtures::synthetic(3, 1, 0).compile().unwrap();
    let mut recording = Recording {
        dataset: "synthetic".into(),
        animal_id: "aggregate".into(),
        condition: "stim".into(),
        times: vec![0., 1., 2., 3.],
        behavior: BTreeMap::new(),
        traces: graph
            .names
            .iter()
            .enumerate()
            .map(|(i, n)| Trace {
                neuron: n.clone(),
                values: if i == 0 {
                    vec![Some(1.), None, Some(-2.), Some(3.)]
                } else {
                    vec![Some(2.); 4]
                },
                provenance: Provenance {
                    dataset: "synthetic".into(),
                    version: "1".into(),
                    id_confidence: 0.5,
                },
            })
            .collect(),
    };
    recording.traces[2].values = vec![Some(-1.), Some(2.), Some(4.), Some(0.)];
    recording.traces[2].provenance.id_confidence = 0.;
    let response = vec![
        vec![0.1, 2., 4.],
        vec![9., 1., 2.],
        vec![0.3, 4., 8.],
        vec![-0.2, 3., 5.],
    ];
    let epsilon = 0.07;
    let g = atlas_correlation::loss(&recording, &graph, &response, epsilon).unwrap();
    assert_eq!(g.pairs, 1);
    assert_eq!(g.pairs, atlas_correlation::eligible_pairs(&recording));
    assert_eq!(g.fluorescence[1][0], 0.);
    for row in &g.fluorescence {
        assert_eq!(&row[1..], &[0., 0.]);
    }
    for t in 0..4 {
        for i in 0..3 {
            let mut plus = response.clone();
            let mut minus = response.clone();
            plus[t][i] += 1e-6;
            minus[t][i] -= 1e-6;
            let numeric = (atlas_correlation::loss(&recording, &graph, &plus, epsilon)
                .unwrap()
                .value
                - atlas_correlation::loss(&recording, &graph, &minus, epsilon)
                    .unwrap()
                    .value)
                / 2e-6;
            assert!((numeric - g.fluorescence[t][i]).abs() < 1e-8);
        }
    }
    let flat = vec![vec![0.; 3]; 4];
    let zero = atlas_correlation::loss(&recording, &graph, &flat, epsilon).unwrap();
    assert_eq!(zero.value, 1.);
    assert!(zero.fluorescence.iter().flatten().any(|v| v.abs() > 0.));
    let shifted: Vec<Vec<_>> = response
        .iter()
        .map(|r| r.iter().map(|v| v + 7.).collect())
        .collect();
    assert!(
        (atlas_correlation::loss(&recording, &graph, &shifted, epsilon)
            .unwrap()
            .value
            - g.value)
            .abs()
            < 1e-12
    );
    for invalid in [0., -1., f64::NAN, f64::INFINITY, 1e-300, 1e300] {
        assert!(atlas_correlation::loss(&recording, &graph, &response, invalid).is_err());
    }
    assert!(atlas_correlation::loss(&recording, &graph, &response[..3], epsilon).is_err());
    recording.traces[2].provenance.id_confidence = 1.;
    assert_eq!(atlas_correlation::eligible_pairs(&recording), 2);
    assert_eq!(
        atlas_correlation::loss(&recording, &graph, &response, epsilon)
            .unwrap()
            .pairs,
        2
    );

    // Verify composition with the actual prepared neural adjoint, rather than
    // only differentiating the loss in fluorescence space.
    use wormsim::{
        initial_state::{self, Readout},
        model::Model,
        parameters::forecast_defaults,
    };
    let network = Model::new(graph.clone()).unwrap();
    let params = forecast_defaults(&network);
    let seed = network.initial(&network.prepare(&params).unwrap());
    let mut readout = Readout::identity(network.n());
    readout.gain.fill(2.);
    let mut currents = vec![vec![0.; network.n()]; recording.times.len()];
    currents[0][0] = 0.4;
    let gradient = initial_state::prepared_response_objective_gradient(
        &network,
        &params,
        &readout,
        &seed,
        &recording.times,
        0.05,
        &currents,
        0.3,
        |response| {
            let loss = atlas_correlation::loss(&recording, &graph, response, epsilon)?;
            Ok((loss.value, loss.fluorescence))
        },
    )
    .unwrap();
    let objective = |p: &wormsim::model::Parameters<f64>, r: &Readout| {
        let response = initial_state::prepared_response_with_currents(
            &network,
            p,
            &seed,
            &recording.times,
            r,
            0.05,
            &currents,
            0.3,
        )
        .unwrap();
        atlas_correlation::loss(&recording, &graph, &response, epsilon)
            .unwrap()
            .value
    };
    for i in 0..params.raw.len() {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus.raw[i] += 1e-5;
        minus.raw[i] -= 1e-5;
        let numeric = (objective(&plus, &readout) - objective(&minus, &readout)) / 2e-5;
        assert!((numeric - gradient.parameters[i]).abs() < 1e-7);
    }
    let mut plus = readout.clone();
    let mut minus = readout.clone();
    plus.gain.iter_mut().for_each(|v| *v *= 1e-5_f64.exp());
    minus.gain.iter_mut().for_each(|v| *v *= (-1e-5_f64).exp());
    assert!(
        ((objective(&params, &plus) - objective(&params, &minus)) / 2e-5
            - gradient.readout_log_gain.iter().sum::<f64>())
        .abs()
            < 1e-7
    );
}
