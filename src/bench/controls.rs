//! Fixed forecasting controls. All estimated quantities use training animals only.
use super::*;

#[derive(Clone, Copy, Debug)]
pub enum Control {
    HistoryMean,
    HalfBlend,
    TrainingMean,
    Autoregressive,
}
impl Control {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "history-mean" => Ok(Self::HistoryMean),
            "half-blend" => Ok(Self::HalfBlend),
            "training-mean" => Ok(Self::TrainingMean),
            "ar" => Ok(Self::Autoregressive),
            _ => Err("control must be history-mean, half-blend, training-mean or ar".into()),
        }
    }
}
#[derive(Default)]
struct Fit {
    weight: f64,
    mean: f64,
    pair_weight: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
}
impl Fit {
    fn value(&mut self, y: f64, w: f64) {
        self.weight += w;
        self.mean += w / self.weight * (y - self.mean);
    }
    fn pair(&mut self, x: f64, y: f64, w: f64) {
        self.pair_weight += w;
        let dx = x - self.x;
        let dy = y - self.y;
        self.x += w / self.pair_weight * dx;
        self.y += w / self.pair_weight * dy;
        self.xx += w * dx * (x - self.x);
        self.xy += w * dx * (y - self.y);
    }
    fn ar(&self) -> (f64, f64) {
        // Fixed stationary AR(1): constrain slope to [-1,1], no validation tuning.
        let a = if self.xx > 0.0 {
            (self.xy / self.xx).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        (
            a,
            if self.pair_weight > 0.0 {
                self.y - a * self.x
            } else {
                self.mean
            },
        )
    }
}
pub fn predict(
    data: &Dataset,
    graph: &IndexedGraph,
    split: &Split,
    partition: Partition,
    control: Control,
) -> Result<Predictions> {
    // Reuse exact partition/history validation and a causal fallback for unseen neurons.
    let mut out = persistence(data, graph, split, partition)?;
    let fitted = matches!(control, Control::TrainingMean | Control::Autoregressive);
    let mut fits: BTreeMap<String, Fit> = BTreeMap::new();
    let mut dt: Option<f64> = None;
    if fitted {
        let mut trials: Vec<_> = data
            .trials
            .iter()
            .filter(|t| split.train.contains(&t.id))
            .collect();
        trials.sort_by(|a, b| a.id.cmp(&b.id));
        for trial in trials {
            for step in trial.recording.times.windows(2) {
                let delta = step[1] - step[0];
                let expected = *dt.get_or_insert(delta);
                if matches!(control, Control::Autoregressive) && (delta - expected).abs() > 1e-8 {
                    return Err("AR requires a uniform common sample interval".into());
                }
            }
            for trace in &trial.recording.traces {
                let w = trace.provenance.id_confidence;
                if w == 0.0 {
                    continue;
                }
                let fit = fits.entry(trace.neuron.clone()).or_default();
                for y in trace.values.iter().flatten() {
                    fit.value(*y, w);
                }
                for pair in trace.values.windows(2) {
                    if let (Some(x), Some(y)) = (pair[0], pair[1]) {
                        fit.pair(x, y, w);
                    }
                }
            }
        }
        fits.retain(|_, fit| fit.weight > 0.0);
        if fits.is_empty() {
            return Err("no training observations".into());
        }
    }
    let mut fallback_neurons = BTreeSet::new();
    let mut fallback_traces = 0;
    let indexed: BTreeMap<_, _> = data.trials.iter().map(|t| (&t.id, t)).collect();
    for prediction in &mut out.trials {
        let trial = indexed[&prediction.id];
        let origin = trial.forecast_origin.ok_or("missing origin")?;
        if matches!(control, Control::Autoregressive)
            && trial
                .recording
                .times
                .windows(2)
                .any(|p| (p[1] - p[0] - dt.unwrap_or(0.0)).abs() > 1e-8)
        {
            return Err("AR prediction interval differs from training".into());
        }
        for trace in &trial.recording.traces {
            if fitted && !fits.contains_key(&trace.neuron) {
                fallback_neurons.insert(trace.neuron.clone());
                fallback_traces += 1;
            }
            let history: Vec<_> = trial
                .recording
                .times
                .iter()
                .zip(&trace.values)
                .take_while(|(t, _)| **t <= origin)
                .filter_map(|(_, v)| *v)
                .collect();
            let last = *history.last().ok_or("no observed history")?;
            let mean = history.iter().sum::<f64>() / history.len() as f64;
            let values = prediction.fluorescence.get_mut(&trace.neuron).unwrap();
            match control {
                Control::HistoryMean => values.fill(mean),
                Control::HalfBlend => values.fill(0.5 * (last + mean)),
                Control::TrainingMean => {
                    values.fill(fits.get(&trace.neuron).map_or(last, |f| f.mean))
                }
                Control::Autoregressive => {
                    if let Some(f) = fits.get(&trace.neuron) {
                        let (a, b) = f.ar();
                        let mut state = f.mean;
                        for (i, &time) in trial.recording.times.iter().enumerate() {
                            if i > 0 {
                                state = a * state + b;
                            }
                            if time <= origin
                                && let Some(y) = trace.values[i]
                            {
                                state = y;
                            }
                            if !state.is_finite() {
                                return Err("nonfinite AR forecast".into());
                            }
                            values[i] = state;
                        }
                    }
                }
            }
        }
    }
    out.model = format!(
        "{control:?}; unseen training neurons use last observed value; fallback traces={fallback_traces}; neurons={fallback_neurons:?}"
    );
    out.free_parameters = if fitted {
        fits.len()
            * if matches!(control, Control::Autoregressive) {
                3
            } else {
                1
            }
    } else {
        0
    };
    if fitted {
        out.training_trials = split.train.clone();
    }
    Ok(out)
}
