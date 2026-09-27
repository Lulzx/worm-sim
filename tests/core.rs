use wormsim::{
    codec,
    data::*,
    fit, fixtures,
    math::{Scalar, inverse_softplus},
    model::{Inputs, Model},
    solve::*,
};
fn model(n: usize, c: usize, g: usize) -> Model {
    Model::new(fixtures::synthetic(n, c, g).compile().unwrap()).unwrap()
}
#[test]
fn canonical_order_hash_and_aliases() {
    let a = fixtures::synthetic(3, 2, 1);
    let mut b = a.clone();
    b.neurons.reverse();
    b.chemical.reverse();
    b.gaps.reverse();
    for e in &mut b.gaps {
        std::mem::swap(&mut e.a, &mut e.b);
    }
    let a = a.compile().unwrap();
    assert_eq!(a.hash, b.compile().unwrap().hash);
    let aliases = std::collections::BTreeMap::from([("old".into(), "N000".into())]);
    assert_eq!(
        reconcile(&["old".into()], &aliases, &a).unwrap()[0].canonical,
        "N000"
    );
    assert!(reconcile(&["unknown".into()], &aliases, &a).is_err());
}
#[test]
fn invalid_graph_is_rejected() {
    let mut g = fixtures::synthetic(3, 1, 1);
    g.chemical[0].post = "BAD".into();
    assert!(g.compile().is_err());
    let mut g = fixtures::synthetic(3, 1, 1);
    g.gaps.push(g.gaps[0].clone());
    assert!(g.compile().is_err());
}
#[test]
fn lossless_codec_and_corruption() {
    let mut g = fixtures::synthetic(302, 23, 3);
    g.chemical[0].synapse_count = f64::from_bits(0x3ff123456789abcd);
    let g = g.compile().unwrap();
    let packed = codec::encode(&g).unwrap();
    let roundtrip = codec::decode(&packed).unwrap();
    assert_eq!(roundtrip.hash, g.hash);
    assert_eq!(roundtrip.chemical[0].2.to_bits(), g.chemical[0].2.to_bits());
    for size in [0, 3, 20, 43, packed.len() - 1] {
        assert!(codec::decode(&packed[..size]).is_err());
    }
    let mut bad = packed;
    bad[12] ^= 1;
    assert!(codec::decode(&bad).is_err());
}
#[test]
fn gaps_conserve_current() {
    let m = model(2, 0, 1);
    let mut raw = m.defaults();
    raw.raw[2] = 0.0;
    raw.raw[3] = 0.0;
    let p = m.prepare(&raw).unwrap();
    let y = vec![0.3, -0.8, 0.0, 0.0, 0.0, 0.0];
    let mut dy = vec![0.0; 6];
    m.rhs(&p, &y, &Inputs::new(2), &mut [0.0; 2], &mut dy);
    let sum = (dy[0] / p.inv_tau[0] + y[0]) + (dy[1] / p.inv_tau[1] + y[1]);
    assert!(sum.abs() < 1e-14);
}
#[test]
fn passive_cell_matches_analytic_and_event_boundary() {
    let m = model(1, 0, 0);
    let mut p = m.defaults();
    p.raw[0] = inverse_softplus(0.1);
    p.raw[1] = 0.0;
    let cfg = Config {
        duration: 0.2,
        dt: 0.003,
        save_dt: 0.2,
        events: vec![Event::Stimulate {
            neuron: "N000".into(),
            start: 0.0,
            end: 0.1,
            amplitude: 1.0,
        }],
        ..Config::default()
    };
    let out = simulate(&m, &p, &cfg).unwrap();
    let tau = p.raw[0].softplus() + 1e-9;
    let expected = (1.0 - (-0.1 / tau).exp()) * (-0.1 / tau).exp();
    assert!((out.voltage[1][0] - expected).abs() < 1e-7);
    assert_eq!(out.times, vec![0.0, 0.2]);
}
#[test]
fn ablation_removes_outgoing_and_incoming_coupling() {
    let m = model(2, 1, 1);
    let cfg = Config {
        duration: 0.1,
        events: vec![
            Event::Ablate {
                neuron: "N000".into(),
            },
            Event::Stimulate {
                neuron: "N000".into(),
                start: 0.0,
                end: 0.1,
                amplitude: 100.0,
            },
        ],
        ..Config::default()
    };
    let out = simulate(&m, &m.defaults(), &cfg).unwrap();
    for row in out.voltage {
        assert_eq!(row, vec![-0.5, -0.5]);
    }
}
fn target(m: &Model, cfg: &Config) -> Recording {
    let out = simulate(m, &m.defaults(), cfg).unwrap();
    Recording {
        behavior: Default::default(),
        dataset: "synthetic".into(),
        animal_id: "a".into(),
        condition: "test".into(),
        times: out.times,
        traces: (0..m.n())
            .map(|i| Trace {
                neuron: m.graph.names[i].clone(),
                values: out.fluorescence.iter().map(|r| Some(r[i] + 0.03)).collect(),
                provenance: Provenance {
                    dataset: "synthetic".into(),
                    version: "1".into(),
                    id_confidence: 0.8,
                },
            })
            .collect(),
    }
}
#[test]
fn all_parameter_gradients_match_finite_differences() {
    let m = model(2, 1, 1);
    let cfg = Config {
        duration: 0.05,
        save_dt: 0.01,
        events: vec![Event::Stimulate {
            neuron: "N000".into(),
            start: 0.0,
            end: 0.03,
            amplitude: 0.8,
        }],
        ..Config::default()
    };
    let p = m.defaults();
    let rec = target(&m, &cfg);
    let indices: Vec<_> = (0..p.raw.len()).collect();
    let (_, grad) = fit::gradient(&m, &p, &cfg, &rec, &indices).unwrap();
    for (i, &g) in grad.iter().enumerate() {
        let mut plus = p.clone();
        let mut minus = p.clone();
        let h = 1e-5;
        plus.raw[i] += h;
        minus.raw[i] -= h;
        let fd = (fit::loss(&m, &simulate(&m, &plus, &cfg).unwrap(), &rec).unwrap()
            - fit::loss(&m, &simulate(&m, &minus, &cfg).unwrap(), &rec).unwrap())
            / (2.0 * h);
        assert!(
            (fd - g).abs() < 2e-8 + 1e-4 * fd.abs(),
            "index {i}: AD={g} FD={fd}"
        );
    }
}
#[test]
fn missing_and_zero_confidence_not_zero_observations() {
    let m = model(2, 0, 0);
    let cfg = Config::default();
    let out = simulate(&m, &m.defaults(), &cfg).unwrap();
    let mut r = target(&m, &cfg);
    r.traces[0].values.fill(None);
    r.traces[1].provenance.id_confidence = 0.0;
    assert!(fit::loss(&m, &out, &r).is_err());
}
#[test]
fn bad_config_rejected() {
    let m = model(1, 0, 0);
    let cfg = Config {
        dt: 0.0,
        ..Config::default()
    };
    assert!(simulate(&m, &m.defaults(), &cfg).is_err());
}

#[test]
fn importer_reports_mirrors_and_excluded_cells() {
    let bytes=b"Source,Target,Weight,Type\nN000,N001,4,electrical\nN001,N000,2,electrical\nN000,muscle,1,chemical\nN000,N001,3,chemical\n";
    let names = vec!["N000".into(), "N001".into()];
    assert!(wormsim::import::c302_csv(bytes, &names, "fixture", false).is_err());
    let (g, r) = wormsim::import::c302_csv(bytes, &names, "fixture", true).unwrap();
    assert_eq!(g.gaps[0].2, 3.0);
    assert_eq!(r.gap_conflicts.len(), 1);
    assert_eq!(r.excluded_unmapped_rows, 1);
    assert_eq!(g.chemical.len(), 1);
}
#[test]
fn shared_gates_match_independent_edge_gate_euler_reference() {
    let m = model(5, 3, 2);
    let p = m.prepare(&m.defaults()).unwrap();
    let n = m.n();
    let mut compact = m.initial(&p);
    let mut voltage = compact[..n].to_vec();
    let mut gates: Vec<_> = m.pre.iter().map(|&a| compact[2 * n + a as usize]).collect();
    let mut input = Inputs::new(n);
    input.current[0] = 0.7;
    let mut rhs = vec![0.0; m.state_len()];
    let mut scratch = vec![0.0; n];
    for _ in 0..400 {
        let releases: Vec<_> = (0..n)
            .map(|i| ((voltage[i] - p.threshold[i]) * p.slope[i]).sigmoid())
            .collect();
        let mut dv: Vec<_> = (0..n)
            .map(|i| -(voltage[i] - p.rest[i]) + input.current[i])
            .collect();
        for (e, gate) in gates.iter().enumerate() {
            let b = m.post[e] as usize;
            dv[b] += p.weight[e] * gate * (p.reversal[e] - voltage[b]);
        }
        for e in 0..m.gap_a.len() {
            let a = m.gap_a[e] as usize;
            let b = m.gap_b[e] as usize;
            let c = p.gap[e] * (voltage[b] - voltage[a]);
            dv[a] += c;
            dv[b] -= c;
        }
        m.rhs(&p, &compact, &input, &mut scratch, &mut rhs);
        for i in 0..n {
            voltage[i] += 0.0002 * dv[i] * p.inv_tau[i];
        }
        for (e, gate) in gates.iter_mut().enumerate() {
            let r = releases[m.pre[e] as usize];
            *gate += 0.0002 * (r * (1.0 - *gate) - *gate) * p.inv_synapse_tau;
        }
        for (state, derivative) in compact.iter_mut().zip(&rhs) {
            *state += 0.0002 * derivative;
        }
    }
    for (a, b) in voltage.iter().zip(&compact[..n]) {
        assert!((a - b).abs() < 1e-13);
    }
    for (e, &gate) in gates.iter().enumerate() {
        assert!((gate - compact[2 * n + m.pre[e] as usize]).abs() < 1e-13);
    }
}

#[test]
fn trace_codec_exact_bits_missing_and_partial_decode() {
    use wormsim::trace_codec::*;
    let values: Vec<_> = (0..600 * 3)
        .map(|i| {
            if i % 7 == 0 {
                None
            } else if i % 5 == 0 {
                Some(-0.0)
            } else {
                Some((i as f64 * 0.01).sin())
            }
        })
        .collect();
    let matrix = Matrix {
        rows: 600,
        columns: 3,
        values,
    };
    let packed = encode(&matrix).unwrap();
    for (start, end) in [(0, 600), (250, 270), (599, 600)] {
        let restored = decode_range(&packed, start, end).unwrap();
        for (a, b) in restored
            .values
            .iter()
            .zip(&matrix.values[start * 3..end * 3])
        {
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
        }
    }
    let mut bad = packed.clone();
    bad[16] ^= 1;
    assert!(decode_range(&bad, 0, 1).is_err());
    assert!(decode_range(&packed[..packed.len() - 1], 0, 600).is_err());
}

#[test]
fn importer_reconciles_zero_padded_motor_neurons() {
    let bytes=b"Source,Target,Weight,Type\nDA01,AVAL,2,chemical\nAVAL,DA01,3,electrical\nDA01,AVAL,3,electrical\n";
    let (g, r) =
        wormsim::import::c302_csv(bytes, &["AVAL".into(), "DA1".into()], "fixture", false).unwrap();
    assert_eq!(g.chemical.len(), 1);
    assert_eq!(g.gaps.len(), 1);
    assert_eq!(r.excluded_unmapped_rows, 0);
    assert_eq!(r.name_mappings[0].source, "DA01");
    assert_eq!(r.name_mappings[0].canonical, "DA1");
}

#[test]
fn pinned_c302_fixture_has_reconciled_motor_neurons() {
    let g = codec::decode(include_bytes!("../data/c302-herm.wsc")).unwrap();
    assert_eq!(g.names.len(), 302);
    assert_eq!(g.chemical.len(), 3638);
    assert_eq!(g.gaps.len(), 1080);
    let da1 = g.neuron("DA1").unwrap();
    assert!(g.chemical.iter().any(|e| e.0 == da1 || e.1 == da1));
    assert!(g.neuron("DA01").is_err());
}

#[test]
fn trace_codec_authenticates_dimensions() {
    use wormsim::trace_codec::*;
    let mut bytes = encode(&Matrix {
        rows: 2,
        columns: 3,
        values: vec![Some(1.0); 6],
    })
    .unwrap();
    bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
    bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
    assert!(decode_range(&bytes, 0, 3).is_err());
}
