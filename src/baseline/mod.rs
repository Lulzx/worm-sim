//! Reproduction of the pinned Creamer/Leifer/Pillow fitted-model evaluation.
//! This discrete linear model is separate from WormSim's biophysical Level 0.
pub mod operator;
use crate::Result;
use operator::Operator;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineModel {
    pub name: String,
    pub neurons: Vec<String>,
    pub sample_rate: f64,
    pub input_lags: usize,
    pub emission_input_lags: usize,
    pub learned_parameters: BTreeMap<String, usize>,
    pub dynamics_weights: Operator,
    pub dynamics_input_weights: Operator,
    pub dynamics_cov: Operator,
    pub emissions_weights: Operator,
    pub emissions_input_weights: Operator,
    pub emissions_cov: Operator,
    pub reference: Reference,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Probe {
    pub responding: usize,
    pub stimulated: usize,
    pub values: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub stams: Vec<f64>,
    pub correlation: Vec<f64>,
    pub probes: Vec<Probe>,
    pub stams_test_score: ReferenceScore,
    pub corr_test_score: ReferenceScore,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceScore {
    pub correlation: f64,
    pub pairs: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurements {
    pub stams_train: Vec<Option<f64>>,
    pub stams_test: Vec<Option<f64>>,
    pub corr_train: Vec<Option<f64>>,
    pub corr_test: Vec<Option<f64>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: String,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub repository: String,
    pub commit: String,
    pub artifacts: Vec<Artifact>,
    pub split: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub schema_version: u32,
    pub source: Source,
    pub pre_seconds: usize,
    pub post_seconds: usize,
    pub covariance_iterations: usize,
    pub models: Vec<BaselineModel>,
    pub measured: Measurements,
}
impl Bundle {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.models.is_empty()
            || self.models.len() > 32
            || self.pre_seconds != 15
            || self.post_seconds != 30
            || self.covariance_iterations != 100
        {
            return Err("unsupported baseline schema or evaluation protocol".into());
        }
        if self.source.repository.is_empty()
            || self.source.commit.len() != 40
            || self.source.artifacts.is_empty()
            || self.source.split.is_empty()
            || self.source.artifacts.iter().any(|a| {
                a.path.is_empty()
                    || a.sha256.len() != 64
                    || !a.sha256.bytes().all(|b| b.is_ascii_hexdigit())
            })
        {
            return Err("invalid baseline provenance".into());
        }
        let n = self.models[0].neurons.len();
        let mut names = BTreeSet::new();
        for model in &self.models {
            if !names.insert(&model.name)
                || model.name.is_empty()
                || n == 0
                || n > 1024
                || model.neurons != self.models[0].neurons
                || model.neurons.iter().collect::<BTreeSet<_>>().len() != n
                || model.neurons.iter().any(String::is_empty)
                || model.sample_rate != 2.0
                || model.input_lags == 0
                || model.emission_input_lags == 0
                || model.input_lags > 1000
                || model.emission_input_lags > 1000
            {
                return Err("invalid or inconsistent baseline model metadata".into());
            }
            for (op, shape) in [
                (&model.dynamics_weights, (n, n)),
                (&model.dynamics_input_weights, (n, n * model.input_lags)),
                (&model.dynamics_cov, (n, n)),
                (&model.emissions_weights, (n, n)),
                (
                    &model.emissions_input_weights,
                    (n, n * model.emission_input_lags),
                ),
                (&model.emissions_cov, (n, n)),
            ] {
                op.validate()?;
                if op.shape() != shape {
                    return Err("baseline operator shape mismatch".into());
                }
            }
            let r = &model.reference;
            if r.stams.len() != n * n
                || r.correlation.len() != n * n
                || r.stams.iter().chain(&r.correlation).any(|v| !v.is_finite())
                || r.probes.is_empty()
                || r.probes.iter().any(|p| {
                    p.responding >= n
                        || p.stimulated >= n
                        || p.values.len() != 60
                        || p.values.iter().any(|v| !v.is_finite())
                })
            {
                return Err("invalid reference matrix or impulse probes".into());
            }
            for score in [&r.stams_test_score, &r.corr_test_score] {
                if !score.correlation.is_finite()
                    || score.correlation.abs() > 1.0
                    || score.pairs > n * (n - 1)
                {
                    return Err("invalid reference metric".into());
                }
            }
        }
        for measurement in [
            &self.measured.stams_train,
            &self.measured.stams_test,
            &self.measured.corr_train,
            &self.measured.corr_test,
        ] {
            if measurement.len() != n * n || measurement.iter().flatten().any(|v| !v.is_finite()) {
                return Err("invalid measurement matrix".into());
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct Prediction {
    pub stams: Vec<f64>,
    pub correlation: Vec<f64>,
    pub probes: Vec<Probe>,
}
/// All single-neuron impulse trials run together as contiguous matrix columns.
pub fn predict(model: &BaselineModel) -> Result<Prediction> {
    let n = model.neurons.len();
    let mut x = vec![0.0; n * n];
    let mut next = x.clone();
    let mut y = x.clone();
    let mut stams = x.clone();
    let mut probes: Vec<_> = model
        .reference
        .probes
        .iter()
        .map(|p| Probe {
            responding: p.responding,
            stimulated: p.stimulated,
            values: Vec::with_capacity(60),
        })
        .collect();
    for tick in 0..60 {
        if tick > 0 {
            model.dynamics_weights.multiply(&x, n, &mut next)?;
            std::mem::swap(&mut x, &mut next);
        }
        if tick < model.input_lags {
            model
                .dynamics_input_weights
                .add_column_block(tick * n, n, &mut x)?;
        }
        model.emissions_weights.multiply(&x, n, &mut y)?;
        if tick < model.emission_input_lags {
            model
                .emissions_input_weights
                .add_column_block(tick * n, n, &mut y)?;
        }
        for (sum, &v) in stams.iter_mut().zip(&y) {
            *sum += v / model.sample_rate;
        }
        for probe in &mut probes {
            probe
                .values
                .push(y[probe.responding * n + probe.stimulated]);
        }
    }
    // Match upstream exactly: P0=W W^T+Q, then 100 updates, latent correlation.
    // Observation noise R is preserved but is not used in this particular metric.
    let w = model.dynamics_weights.dense();
    let q = model.dynamics_cov.dense();
    let mut wt = vec![0.0; n * n];
    transpose(&w, &mut wt, n);
    let mut covariance = vec![0.0; n * n];
    model.dynamics_weights.multiply(&wt, n, &mut covariance)?;
    for (p, &q) in covariance.iter_mut().zip(&q) {
        *p += q;
    }
    let mut temp = vec![0.0; n * n];
    let mut transposed = temp.clone();
    let mut output = temp.clone();
    for _ in 0..100 {
        model.dynamics_weights.multiply(&covariance, n, &mut temp)?;
        transpose(&temp, &mut transposed, n);
        model
            .dynamics_weights
            .multiply(&transposed, n, &mut output)?;
        transpose(&output, &mut covariance, n);
        for (p, &q) in covariance.iter_mut().zip(&q) {
            *p += q;
        }
    }
    let std: Vec<_> = (0..n).map(|i| covariance[i * n + i].sqrt()).collect();
    if std.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err("nonpositive or nonfinite predicted variance".into());
    }
    for r in 0..n {
        for c in 0..n {
            covariance[r * n + c] /= std[r] * std[c];
        }
    }
    if stams.iter().chain(&covariance).any(|v| !v.is_finite()) {
        return Err("nonfinite baseline prediction".into());
    }
    Ok(Prediction {
        stams,
        correlation: covariance,
        probes,
    })
}
fn transpose(input: &[f64], out: &mut [f64], n: usize) {
    for r in 0..n {
        for c in 0..n {
            out[c * n + r] = input[r * n + c];
        }
    }
}
#[derive(Debug, Serialize)]
pub struct Score {
    pub correlation: f64,
    pub pairs: usize,
    pub fisher_95: Option<[f64; 2]>,
}
/// Match upstream finite-pair Pearson after excluding self pairs. Confidence
/// limits use its Fisher approximation, not an animal-cluster bootstrap.
pub fn score(prediction: &[f64], measurement: &[Option<f64>], n: usize) -> Result<Score> {
    if n == 0 || prediction.len() != n * n || measurement.len() != n * n {
        return Err("score shape mismatch".into());
    }
    let mut count = 0;
    let (mut mx, mut my) = (0.0, 0.0);
    for (i, (&x, &y)) in prediction.iter().zip(measurement).enumerate() {
        if i / n != i % n
            && let Some(y) = y
        {
            if !x.is_finite() || !y.is_finite() {
                return Err("nonfinite score input".into());
            }
            count += 1;
            mx += x;
            my += y;
        }
    }
    if count < 2 {
        return Err("insufficient observed off-diagonal pairs".into());
    }
    mx /= count as f64;
    my /= count as f64;
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for (i, (&x, &y)) in prediction.iter().zip(measurement).enumerate() {
        if i / n != i % n
            && let Some(y) = y
        {
            let a = x - mx;
            let b = y - my;
            xy += a * b;
            xx += a * a;
            yy += b * b;
        }
    }
    if xx <= 0.0 || yy <= 0.0 {
        return Err("Pearson undefined for constant samples".into());
    }
    let correlation = (xy / xx.sqrt() / yy.sqrt()).clamp(-1.0, 1.0);
    if !correlation.is_finite() {
        return Err("nonfinite Pearson result".into());
    }
    let fisher_95 = if count > 3 {
        if correlation.abs() == 1.0 {
            Some([correlation, correlation])
        } else {
            let z = correlation.atanh();
            let d = 1.959963984540054 / ((count - 3) as f64).sqrt();
            Some([(z - d).tanh(), (z + d).tanh()])
        }
    } else {
        None
    };
    Ok(Score {
        correlation,
        pairs: count,
        fisher_95,
    })
}
#[derive(Debug, Serialize)]
pub struct ModelReport {
    pub name: String,
    pub neurons: usize,
    pub learned_parameters: usize,
    pub stored_parameter_scalars: usize,
    pub dense_parameter_scalars: usize,
    pub prediction_seconds: f64,
    pub stams_test: Score,
    pub correlation_test: Score,
    pub max_stams_error: f64,
    pub max_correlation_error: f64,
    pub max_probe_error: f64,
    pub max_score_error: f64,
    pub parity_passed: bool,
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub source: Source,
    pub bundle_sha256: String,
    pub protocol: String,
    pub stams_train_test: Score,
    pub correlation_train_test: Score,
    pub models: Vec<ModelReport>,
    pub all_parity_passed: bool,
}
fn error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}
fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-9 * b.abs())
}
pub fn evaluate(bundle: &Bundle) -> Result<Report> {
    bundle.validate()?;
    let n = bundle.models[0].neurons.len();
    let mut reports = Vec::new();
    for model in &bundle.models {
        let start = Instant::now();
        let p = predict(model)?;
        let prediction_seconds = start.elapsed().as_secs_f64();
        let stams_test = score(&p.stams, &bundle.measured.stams_test, n)?;
        let correlation_test = score(&p.correlation, &bundle.measured.corr_test, n)?;
        let reference = &model.reference;
        let mut max_probe_error = 0.0f64;
        let mut parity_passed =
            close(&p.stams, &reference.stams) && close(&p.correlation, &reference.correlation);
        for (actual, expected) in p.probes.iter().zip(&reference.probes) {
            max_probe_error = max_probe_error.max(error(&actual.values, &expected.values));
            parity_passed &= close(&actual.values, &expected.values);
        }
        let max_score_error = (stams_test.correlation - reference.stams_test_score.correlation)
            .abs()
            .max((correlation_test.correlation - reference.corr_test_score.correlation).abs());
        parity_passed &= max_score_error < 1e-10
            && stams_test.pairs == reference.stams_test_score.pairs
            && correlation_test.pairs == reference.corr_test_score.pairs;
        let ops = [
            &model.dynamics_weights,
            &model.dynamics_input_weights,
            &model.dynamics_cov,
            &model.emissions_weights,
            &model.emissions_input_weights,
            &model.emissions_cov,
        ];
        reports.push(ModelReport {
            name: model.name.clone(),
            neurons: n,
            learned_parameters: model.learned_parameters.values().sum(),
            stored_parameter_scalars: ops.iter().map(|o| o.scalar_count()).sum(),
            dense_parameter_scalars: ops
                .iter()
                .map(|o| {
                    let (r, c) = o.shape();
                    r * c
                })
                .sum(),
            prediction_seconds,
            stams_test,
            correlation_test,
            max_stams_error: error(&p.stams, &reference.stams),
            max_correlation_error: error(&p.correlation, &reference.correlation),
            max_probe_error,
            max_score_error,
            parity_passed,
        });
    }
    let paired_train = |train: &[Option<f64>], test: &[Option<f64>]| {
        let prediction: Vec<_> = train.iter().map(|v| v.unwrap_or(0.0)).collect();
        let measurement: Vec<_> = train
            .iter()
            .zip(test)
            .map(|(a, b)| if a.is_some() { *b } else { None })
            .collect();
        score(&prediction, &measurement, n)
    };
    Ok(Report {source:bundle.source.clone(),bundle_sha256:format!("{:x}",Sha256::digest(serde_json::to_vec(bundle).map_err(|e|e.to_string())?)),protocol:"Upstream 15s pre/30s post, 2Hz; off-diagonal STAM and latent correlation; P0=W W^T+Q then 100 updates; no refitting".into(),stams_train_test:paired_train(&bundle.measured.stams_train,&bundle.measured.stams_test)?,correlation_train_test:paired_train(&bundle.measured.corr_train,&bundle.measured.corr_test)?,all_parity_passed:reports.iter().all(|r|r.parity_passed),models:reports})
}
const MAX_BUNDLE: usize = 64 * 1024 * 1024;
pub fn pack(bundle: &Bundle) -> Result<Vec<u8>> {
    bundle.validate()?;
    let bytes = serde_json::to_vec(bundle).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_BUNDLE {
        return Err("baseline bundle too large".into());
    }
    let mut out = b"WSB1".to_vec();
    out.extend((bytes.len() as u64).to_le_bytes());
    out.extend(Sha256::digest(&bytes));
    out.extend(zstd::stream::encode_all(bytes.as_slice(), 3).map_err(|e| e.to_string())?);
    Ok(out)
}
pub fn unpack(bytes: &[u8]) -> Result<Bundle> {
    if bytes.len() < 44 || &bytes[..4] != b"WSB1" {
        return Err("invalid baseline archive header".into());
    }
    let length = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
    if length > MAX_BUNDLE as u64 {
        return Err("baseline archive exceeds limit".into());
    }
    let mut json = Vec::new();
    zstd::stream::read::Decoder::new(&bytes[44..])
        .map_err(|e| e.to_string())?
        .take(length + 1)
        .read_to_end(&mut json)
        .map_err(|e| e.to_string())?;
    if json.len() as u64 != length || Sha256::digest(&json)[..] != bytes[12..44] {
        return Err("baseline archive checksum or length mismatch".into());
    }
    let bundle: Bundle = serde_json::from_slice(&json).map_err(|e| e.to_string())?;
    bundle.validate()?;
    Ok(bundle)
}
