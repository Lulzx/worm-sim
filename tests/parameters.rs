use wormsim::{
    fixtures,
    model::Model,
    parameters::{Sharing, TiedParameters, forecast_defaults},
};
#[test]
fn explicit_classes_tie_values_sum_gradients_and_match_prior_differences() {
    let model = Model::new(fixtures::synthetic(4, 2, 1).compile().unwrap()).unwrap();
    let initial = forecast_defaults(&model);
    let mut sharing = Sharing {
        allow_suffix_pairs: false,
        provenance: "synthetic test classes".into(),
        ..Default::default()
    };
    for (i, name) in model.graph.names.iter().enumerate() {
        sharing
            .classes
            .insert(name.clone(), format!("class-{}", i / 2));
    }
    let mut tied = TiedParameters::new(&model, &initial, sharing).unwrap();
    assert_eq!(tied.raw_to_group[0], tied.raw_to_group[1]);
    assert_ne!(tied.raw_to_group[0], tied.raw_to_group[2]);
    let k = tied.raw_to_group[0];
    tied.groups[k].value += 0.1;
    let expanded = tied.expand(&model).unwrap();
    assert_eq!(expanded.raw[0], expanded.raw[1]);
    let raw = vec![1.0; model.parameter_count()];
    let g = tied.reduce_gradient(&raw).unwrap();
    assert_eq!(g[k], 2.0);
    let (_, gradient) = tied.prior(&model, 0.2, 0.3).unwrap();
    for (i, derivative) in gradient.iter().enumerate() {
        if !tied.groups[i].trainable {
            continue;
        }
        let eps = 1e-5;
        let mut a = tied.clone();
        let mut b = tied.clone();
        a.groups[i].value += eps;
        b.groups[i].value -= eps;
        let fd = (a.prior(&model, 0.2, 0.3).unwrap().0 - b.prior(&model, 0.2, 0.3).unwrap().0)
            / (2.0 * eps);
        assert!((derivative - fd).abs() < 1e-9);
    }
}
#[test]
fn unknown_annotations_do_not_tie_the_whole_network_and_bad_maps_fail() {
    let mut graph = fixtures::synthetic(3, 1, 0);
    for n in &mut graph.neurons {
        n.class = "unannotated".into();
    }
    let model = Model::new(graph.compile().unwrap()).unwrap();
    let initial = forecast_defaults(&model);
    let tied = TiedParameters::new(&model, &initial, Sharing::default()).unwrap();
    assert_ne!(tied.raw_to_group[0], tied.raw_to_group[1]);
    let mut bad = Sharing::default();
    bad.classes.insert("not-a-neuron".into(), "test".into());
    assert!(TiedParameters::new(&model, &initial, bad).is_err());
    let mut bad = tied;
    bad.raw_to_group[0] = usize::MAX;
    assert!(bad.expand(&model).is_err());
}

#[test]
fn tied_current_projection_sums_shared_neuron_gradients() {
    use wormsim::parameters::TiedInputs;
    let model = Model::new(fixtures::synthetic(4, 1, 0).compile().unwrap()).unwrap();
    let mut sharing = Sharing::default();
    for name in &model.graph.names {
        sharing.classes.insert(name.clone(), "same".into());
    }
    let tied = TiedParameters::new(&model, &forecast_defaults(&model), sharing).unwrap();
    let mut weights = TiedInputs::new(&model, &tied, 2).unwrap();
    assert_eq!(weights.groups.len(), 1);
    weights.weights = vec![0.4, -0.2];
    let features = vec![vec![0.8, 1.0], vec![-0.3, 0.0]];
    let gradients = vec![vec![1.0, 2.0, 3.0, 4.0], vec![-1.0, 0.2, 0.3, 0.5]];
    let actual = weights.reduce_gradient(&features, &gradients).unwrap();
    let loss = |w: &TiedInputs| {
        w.currents(&features)
            .unwrap()
            .iter()
            .flatten()
            .zip(gradients.iter().flatten())
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };
    for (i, derivative) in actual.iter().enumerate() {
        let old = weights.weights[i];
        weights.weights[i] = old + 1e-6;
        let a = loss(&weights);
        weights.weights[i] = old - 1e-6;
        let b = loss(&weights);
        weights.weights[i] = old;
        assert!(((a - b) / 2e-6 - derivative).abs() < 1e-9);
    }
    weights.neuron_to_group[0] = 2;
    assert!(weights.currents(&features).is_err());
}
