//! Cluster bootstrap: resample animals, never individual windows.
use super::*;
type Neurons = BTreeMap<String, Moments>;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Interval {
    pub point: Option<f64>,
    pub lower_95: Option<f64>,
    pub upper_95: Option<f64>,
    pub defined_replicates: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimalScore {
    pub animal: String,
    pub macro_neuron_r2: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HorizonUncertainty {
    pub seconds: f64,
    pub confidence_weighted: Interval,
    pub unit_weighted: Interval,
    pub per_animal: Vec<AnimalScore>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimalBootstrap {
    pub seed: u64,
    pub replicates: usize,
    pub animals: usize,
    pub method: String,
    pub horizons: Vec<HorizonUncertainty>,
}
fn macro_r2(neurons: &Neurons) -> Option<f64> {
    let values: Vec<_> = neurons.values().filter_map(|m| m.scores().r2).collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}
fn combine<'a>(blocks: impl Iterator<Item = &'a Neurons>) -> Neurons {
    let mut out = Neurons::new();
    for block in blocks {
        for (name, m) in block {
            out.entry(name.clone()).or_default().merge(m);
        }
    }
    out
}
fn interval(point: Option<f64>, mut draws: Vec<f64>) -> Interval {
    draws.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        if draws.is_empty() {
            return None;
        }
        let x = p * (draws.len() - 1) as f64;
        let i = x.floor() as usize;
        Some(draws[i] + (draws[x.ceil() as usize] - draws[i]) * (x - i as f64))
    };
    Interval {
        point,
        lower_95: quantile(0.025),
        upper_95: quantile(0.975),
        defined_replicates: draws.len(),
    }
}
/// Called only after evaluate has checked coverage, lineage, finite values and grids.
pub(super) fn calculate(
    data: &Dataset,
    predictions: &Predictions,
    seed: u64,
    replicates: usize,
) -> Result<AnimalBootstrap> {
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    let mut blocks: BTreeMap<String, Vec<[Neurons; 2]>> = BTreeMap::new();
    for prediction in &predictions.trials {
        let trial = indexed[&prediction.id];
        let animal = blocks
            .entry(trial.recording.animal_id.clone())
            .or_insert_with(|| vec![Default::default(); 3]);
        let origin = trial.forecast_origin.ok_or("missing origin")?;
        for (h, seconds) in [1.0, 10.0, 30.0].iter().enumerate() {
            if let Some(i) = trial
                .recording
                .times
                .iter()
                .position(|t| (*t - origin - seconds).abs() <= 1e-9)
            {
                for trace in &trial.recording.traces {
                    if let Some(y) = trace.values[i] {
                        let p = prediction.fluorescence[&trace.neuron][i];
                        animal[h][0].entry(trace.neuron.clone()).or_default().push(
                            y,
                            p,
                            trace.provenance.id_confidence,
                        )?;
                        animal[h][1]
                            .entry(trace.neuron.clone())
                            .or_default()
                            .push(y, p, 1.0)?;
                    }
                }
            }
        }
    }
    let animals: Vec<_> = blocks.values().collect();
    let n = animals.len();
    if n == 0 {
        return Err("no animals for bootstrap".into());
    }
    // SplitMix64 plus rejection sampling gives platform-independent uniform indices.
    let mut state = seed;
    let mut draw_index = || {
        let bound = n as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            if z >= threshold {
                break (z % bound) as usize;
            }
        }
    };
    let draws: Vec<Vec<usize>> = (0..replicates)
        .map(|_| (0..n).map(|_| draw_index()).collect())
        .collect();
    let mut horizons = vec![];
    for (h, seconds) in [1.0, 10.0, 30.0].into_iter().enumerate() {
        let intervals: Vec<_> = (0..2)
            .map(|w| {
                let point = macro_r2(&combine(animals.iter().map(|a| &a[h][w])));
                let values = draws
                    .iter()
                    .filter_map(|draw| macro_r2(&combine(draw.iter().map(|&i| &animals[i][h][w]))))
                    .collect();
                interval(point, values)
            })
            .collect();
        horizons.push(HorizonUncertainty {
            seconds,
            confidence_weighted: intervals[0].clone(),
            unit_weighted: intervals[1].clone(),
            per_animal: blocks
                .iter()
                .map(|(name, a)| AnimalScore {
                    animal: name.clone(),
                    macro_neuron_r2: macro_r2(&a[h][0]),
                })
                .collect(),
        });
    }
    Ok(AnimalBootstrap{seed,replicates,animals:n,method:"Percentile 95% cluster bootstrap; draw N whole animals with replacement, retain all windows, re-center targets per neuron in every replicate, average defined neuron R². Same fixed draws for every model. Very few animals give weak population uncertainty estimates.".into(),horizons})
}
