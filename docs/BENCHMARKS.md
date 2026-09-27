# Shared benchmark contracts and leakage checks

The Rust `bench` module implements common scoring and fixed group splits for
specification Tasks 1–2. It does not yet provide a real-data fit or demonstrate
that WormSim beats the prescribed baselines.

## CLI

```sh
wormsim bench-split GRAPH.wsc DATA.json neuron 42 10 20 SPLIT.json
wormsim bench-split GRAPH.wsc DATA.json animal 42 5 10 SPLIT.json
wormsim bench-score GRAPH.wsc DATA.json SPLIT.json PREDICTIONS.json test REPORT.json
```

The two numeric counts after the seed are validation groups and test groups;
they are not percentages or trial counts. All remaining groups form training.
There must be at least one training group and one test group. Validation may be
empty for protocols with no selection. The scorer also accepts `train` and
`validation`, and records the selected partition explicitly in its report.

The runnable CLI round-trip and synthetic edge-case fixtures are in
`tests/bench.rs`; run `cargo test --test bench`. These fixtures test the benchmark
contract and must never be used as evidence of biological prediction quality.

## Data contract

`Dataset`, `Trial`, `Predictions`, and `Split` are strict Serde schemas in
`src/bench/mod.rs`. Unknown JSON fields are rejected. A dataset declares schema
version 1, a descriptive name/source including version and preprocessing, the
compiled graph hash, and uniquely identified trials. Each trial contains:

- `id`: a stable trial/window identifier.
- `stimulated_neuron`: a canonical neuron ID for Task 1, otherwise null.
- `forecast_origin`: an exact observed timestamp for Task 2, otherwise null.
- `recording`: dataset, animal ID, condition, times, optional aligned behavior channels, and per-neuron traces using
  the existing `Recording` schema. Missing values are JSON null; each trace has
  dataset/version/ID-confidence provenance.
- `response_labels`: canonical responding neuron → measured boolean label.
  These labels must come from an explicitly recorded experimental criterion;
  the scorer never derives labels from predictions or invents them from traces.

The scorer requires predictions for every recorded neuron, including missing or
zero-confidence traces, so model-specific exclusions cannot change the sample
set silently. Every prediction contains the exact time grid, finite fluorescence
values, and one finite response ranking score for each measured label. Scores
may be arbitrary ranking values; they need not be calibrated probabilities.

Prediction bundles declare the dataset/split hashes, model name, number of free
parameters, source commit, seed, training trial IDs and model-selection trial
IDs. Training IDs must belong to the training partition; selection IDs must
belong to validation. Duplicate, missing or additional held-out trial predictions
are errors. The lineage is checked **as declared**: this is not proof of the
actual training process or protection against deliberately false metadata.

## Split identity and reproducibility

Groups are ranked by SHA-256 of the versioned split domain, axis, seed and group
name. All trials/windows for one stimulated neuron or animal stay together.
Animal identifiers are treated as globally scoped within the dataset, even if
multiple source datasets are combined; importers must reconcile them explicitly.
The manifest covers every trial exactly once. Validation rejects overlap at both
trial and group level, unknown IDs, omitted trials, stale graph hashes and
changed dataset content.

Dataset identity uses the `wormsim-benchmark-data-v2` canonical tuple (including
aligned behavior channels; earlier development manifests must be regenerated) of metadata, trials sorted by
ID, and traces sorted by canonical neuron name. Times and sample order retain
meaning. Hashing sorts references and streams serialized bytes into SHA-256;
it does not clone trace arrays or allocate another complete JSON serialization.
Split identity sorts its partition ID lists before hashing. Regenerate a split
only for a deliberately versioned dataset/protocol change, not to optimize scores.
The generated manifest should be published before fitting or model selection.

## Metric definitions

- Missing target samples are excluded, never filled with zero. Finite prediction
  values are still required at those positions. Zero-confidence observations
  contribute zero weight. Missing and zero-confidence counts remain in reports.
- MSE, Pearson correlation and R² use confidence-weighted streaming centered
  moments. This avoids the cancellation of raw second moments at large offsets.
- Task 1's macro trace correlation weights each defined per-trace correlation by
  its ID confidence. Each trace is one trial/responding-neuron curve; repeated
  experimental trials remain separate samples. The report also retains each
  trace's metrics and pooled diagnostics.
- Response AUROC compares labeled trial/responding-neuron pairs, weighted by ID
  confidence. Equal scores receive half credit. It is undefined if either class
  has no positive weight. No response threshold is selected on the test set.
- Task 2 excludes observations at/before `forecast_origin`. Horizon scores use
  samples at origin + 1, 10, and 30 seconds, with at most 1e-9 seconds grid
  tolerance. There is no implicit interpolation or nearest-sample substitution.
  Importers must document resampling. Absent timestamps/values are counted.
- Task 2's macro horizon R² averages defined per-neuron R² across forecast trials.
  Each neuron's R² uses confidence-weighted target centering. Pooled R² is also
  retained as a diagnostic but is not the macro score. Multiple forecast windows
  are necessary to define a neuron's variance; a one-sample or constant target
  yields null, not a fabricated zero or perfect score.
- Constant predictions make correlation undefined; constant targets make both
  correlation and R² undefined. Nulls and defined-score counts remain visible.
  Negative R² is allowed. Nonfinite inputs and accumulation overflow are errors.

These explicit conventions must be shared by nonlinear and baseline runs. They
are the initial benchmark protocol, not a claim that a source paper used exactly
the same aggregation. Parameter counts are reported for later matched-budget
comparisons; the scorer does not verify a model's claimed parameter count.

## Remaining data/training work

The [WormWideWeb importer](WORMWIDEWEB.md) now supplies a fixed animal split and
persistence control. Further source-specific importers still need to map stimuli, confidence, observed labels,
recording units and preprocessing into this contract. The current Creamer export
uses its upstream split and cannot substitute for a held-out-stimulated-neuron
training/evaluation run. The Task 2 LDS/GRU and nonlinear fits are recorded in the
[shared-input comparison](BEHAVIOR-INPUTS.md), including negative results.
[Randi ingestion](RANDI-ATLAS.md) now supplies individual traces and a fixed
neuron split. Published response labels, pulse calibration and the Task 1 fitted
comparison remain outstanding.

Primary source discovery for the next import stage:
[WormWideWeb datasets](https://wormwideweb.org/activity/dataset/),
[Flavell lab data manifests](https://github.com/flavell-lab/WormWideWeb-data), and
[their data tooling](https://github.com/flavell-lab/WormWideWebData.jl).
Read manifests and data formats directly; do not execute upstream code as part of
a downloader or equate a downloaded recording with a validated benchmark.

## Task 2 controls and fitting

See [TASK2-FITTING.md](TASK2-FITTING.md) for training-only controls, animal-cluster
bootstrap intervals, and the failed first dense linear experiment. Scorer reports
now include `animal_bootstrap` for animal splits; existing primary scores are
unchanged.
