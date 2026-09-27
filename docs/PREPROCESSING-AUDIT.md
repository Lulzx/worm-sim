# Task 2 preprocessing audit: retrospective signals

The current 504-window benchmark contains **retrospectively processed neural
signals**. Its animal split remains disjoint and its prediction code excludes
post-origin targets, but that does not establish end-to-end temporal causality.
All existing control, LDS, GRU and Level 0 receipts use this same dataset hash:
`966a77af4cd82d0085f45cdb013617546588136815a24aa818769fecc57873b5`.
Their numerical results are retained; their interpretation is limited accordingly.

## Direct numerical evidence

The [audit receipt](wormwideweb-preprocessing-audit.json) verifies every HDF5
against both the fetch and import receipts and checks all **2,917 exported ROI
traces from 21 animals**, including ROIs not used after NeuroPAL label filtering.
For every trace, the published `gcamp/trace_array` equals

```
(original - mean(original over the whole recording))
    / sample_std(original over the whole recording)
```

where `original` is `gcamp/trace_array_original`. The maximum absolute residual
across all files is **3.56e-14**. Units are whole-recording per-ROI z-scores, not
raw fluorescence, calibrated calcium, or ΔF/F.

A synthetic dependency check adds one original full-recording standard deviation
to each neuron's later-half values, leaving its first ten seconds unchanged, then
recomputes the normalization. The normalized initial prefix changes: per-animal
maximum changes range from **0.921 to 2.111 z-score units**. This demonstrates
future dependence of the transformation. It does not estimate an actual change
in benchmark R² or the direction/magnitude of score bias. In particular, an affine
transform applied to both targets and predictions can leave some R² definitions
unchanged; pooled cross-animal metrics and learned calibration need separate study.

## Source evidence and its limits

Sources are pinned and SHA-256 checked in
[data/wormwideweb-preprocessing-sources.json](../data/wormwideweb-preprocessing-sources.json).
The audit downloads text only and never executes upstream code.

- The [paper's HDF5 exporter](https://github.com/flavell-lab/AtanasKim-Cell2023/blob/f08dbd2eb85fbb97b759aa900f2bfdb05e9f14d3/src/CePNEM/ANTSUNDataJLD2.jl/src/data_h5.jl)
  maps the z-scored processed array to `trace_array` and processed fluorescence
  to `trace_array_original`.
- The [trace processing code](https://github.com/flavell-lab/AtanasKim-Cell2023/blob/f08dbd2eb85fbb97b759aa900f2bfdb05e9f14d3/src/ANTSUN/CaAnalysis.jl/src/noise_correction.jl)
  computes normalization over whole traces. The [reference notebook](https://github.com/flavell-lab/AtanasKim-Cell2023/blob/f08dbd2eb85fbb97b759aa900f2bfdb05e9f14d3/notebook/ANTSUN_NeuroPAL.ipynb),
  code cell 179, enables interpolation, marker division and bleaching correction
  before both exports; denoising is disabled in that call.
- The [behavior processing code](https://github.com/flavell-lab/AtanasKim-Cell2023/blob/f08dbd2eb85fbb97b759aa900f2bfdb05e9f14d3/src/ANTSUN/BehaviorDataNIR.jl/src/behaviors.jl)
  uses centered filtering for angular velocity and pumping, and imputation in
  several paths. Its [filter implementation](https://github.com/flavell-lab/AtanasKim-Cell2023/blob/f08dbd2eb85fbb97b759aa900f2bfdb05e9f14d3/src/ANTSUN/BehaviorDataNIR.jl/src/util.jl)
  accesses samples on both sides. Notebook cell 9 configures angular-velocity lag
  150 at 20 Hz: 7.5 seconds on each side for that configuration.
- The [export utility](https://github.com/flavell-lab/AtanasKim-Cell2023/blob/f08dbd2eb85fbb97b759aa900f2bfdb05e9f14d3/src/CePNEM/ANTSUNDataJLD2.jl/src/data_h5_utility.jl)
  interpolates velocity artifacts; these operations can use later samples.
- The [WormWideWeb integrity checker](https://github.com/flavell-lab/WormWideWebData.jl/blob/da73782cb00510a54f47f09199eedb49da1ca9f7/src/data_integrity.jl)
  expects whole-trace zero mean and unit sample standard deviation.

These are reference source snapshots, not a complete attestation of each animal's
historical preprocessing invocation. Whole-recording standardization is directly
confirmed in every file. The exact per-file bleach parameters, interpolated gaps,
filter settings and effective behavior lookahead are not recovered. It would be
incorrect to claim that every exported angular-velocity sample has exactly 7.5
seconds of lookahead merely from the notebook default.

## Consequences and implementation

The Rust scorer now adds a `preprocessing_assessment` to every report. A small
content-bound registry identifies this exact dataset and graph as `retrospective`
and supplies the evidence file's SHA-256. Unlisted or changed datasets are
`unaudited`; absence of an audit never means causal. Tests bind the registry to
its receipt and verify it does not transfer to a different dataset or graph.
Older reports without the field deserialize as unaudited. Old numerical receipts
are not rewritten; this evidence qualifies all receipts with the matching hash.

The following remain distinct:

1. Animal identity leakage: the split is disjoint and unchanged.
2. Model-code future access: existing future-target invariance checks still pass.
3. Upstream temporal processing: whole-recording z-scoring is noncausal, and other
   reference operations can also use future measurements.

Selecting `trace_array_original` or applying prefix-only normalization cannot
undo earlier interpolation or global bleaching fits. A prospective benchmark
needs earlier extraction-stage signals plus a documented causal calibration,
tracking/missing-data and behavior pipeline, followed by a new dataset hash and
refitting every model. Such a benchmark is not yet available here. This is a
benchmark-specific limit, not proof that the scientific forecasting task is
impossible, and not evidence that all scores are numerically invalid.

Further behavior-assisted experiments on the current data must be labeled
**retrospective processed-signal prediction**. All models should receive the same
behavior channels and history cutoff; future exported behavior must not be silently
provided as an input. Any experiment with actual future behavior is a separate
conditional/oracle diagnostic, not a free forecast. A finite history cutoff alone
does not repair the source's processing dependencies.

## Reproduce

The audit uses NumPy and h5py for independent inspection; simulation, fitting and
scoring remain Rust. It needs the previously fetched HDF5 files and import receipt:

```sh
python3 scripts/test_preprocessing_audit.py
python3 scripts/audit_wormwideweb_preprocessing.py
cargo test --locked --lib --test bench
```

Source cache entries are hash checked on every audit. Numerical tests reject
invalid arrays and distinguish whole-recording normalization from a prefix-only
transform. No raw recordings or third-party source code are redistributed.
