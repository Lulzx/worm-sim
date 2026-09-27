# Freely moving recording import and forecasting control

A native Rust HDF5 importer now converts the labeled baseline subset of
[Atanas & Kim et al. (Cell 2023)](https://doi.org/10.1016/j.cell.2023.07.035)
into the shared benchmark contract. This is a real-data pipeline and a
zero-parameter control, not evidence that the nonlinear model meets Task 2.

The source is [Zenodo record 19388374](https://zenodo.org/records/19388374), located
through the [Flavell lab manifest](https://github.com/flavell-lab/WormWideWeb-data/tree/0422546192ef58d0de4740a0aa91fd38de22da2e).
The live website API returned HTTP 403 in this environment; the published archive
was accessible. No upstream scripts were executed or copied into this repository.

## Reproduce

Install the HDF5 development library (macOS: `brew install hdf5`; Ubuntu:
`sudo apt-get install libhdf5-dev`). The crate's optional `hdf5` feature uses pinned
[hdf5-metno 0.14.1](https://docs.rs/crate/hdf5-metno/0.14.1) bindings. Core simulation
and benchmark scoring continue to build without this native dependency.

```sh
python3 scripts/fetch_wormwideweb.py
cargo build --locked --release --features hdf5
target/release/wormsim import-wormwideweb data/c302-herm.wsc \
  runs/wormwideweb-source/baseline \
  runs/wormwideweb-source/neuropal_label.json \
  docs/wormwideweb-fetch-receipt.json examples/wormwideweb-windows.json \
  runs/wormwideweb-benchmark.json
# The published manifest is already fixed; regenerate only to verify identity.
target/release/wormsim bench-split data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json animal 42 3 3 runs/verified-animal-split.json
target/release/wormsim bench-persist data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  test runs/wormwideweb-persistence.json
target/release/wormsim bench-score data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  runs/wormwideweb-persistence.json test runs/wormwideweb-persistence-report.json
```

The downloader checks pinned Git blob IDs, archive byte counts/MD5 checksums,
and the published SHA-256 for every extracted animal. It selects only manifest
rows tagged baseline with NeuroPAL labels, writes only expected regular-file
members, and records sources/hashes in `wormwideweb-fetch-receipt.json`.
It downloads the 568,776,589-byte processed archive and the 16,169-byte label
archive, not the much larger fitted-model archives. Downloads/extracted recordings
remain under ignored `runs/`. The inspected Zenodo record has no explicit license
field; the raw recordings are not redistributed in this repository.

## Import rules

- Read `timing/timestamp_confocal`, `gcamp/trace_array`, and available aligned
  velocity/head-angle/angular-velocity/pumping channels. Preserve the published
  trace scale and original behavior orientation. This importer does not apply
  dorsal/ventral sign corrections from the website's visualization pipeline.
- Check the HDF5 checksum again in Rust. Infer the time axis only when exactly one
  trace-array dimension matches the timestamp count. Reject ambiguous shapes.
- Interpret upstream ROI IDs as one-based. Accept only exact canonical identities
  that appear once; do not guess the side of labels such as `RIM?` or merge ROIs.
- Require an ordinal label rating at least 3/5. Weight accepted observations by
  rating/5. This is an explicit heuristic, **not calibrated identity probability**.
- Select 40-second windows on a predeclared stride, comprising 10 seconds of
  observed history and 30 seconds of forecast. Align each forecast origin to the
  first real source frame at/after its planned origin. Interpolate to a 0.5-second
  relative grid. No sample after the origin may contribute to the observed prefix.
- Do not extrapolate, interpolate across source gaps exceeding one second, or
  forecast across a recording discontinuity. Exclude such windows and report them.
  Nonfinite source samples become missing observations rather than zero.
- Preserve aligned behavior channels in `Recording.behavior`; these channels are
  included in dataset hashing. No behavior-state labels are inferred here.

The first draft used an arbitrary resampling origin. Reviewing it exposed possible
lookahead at the final interpolated history point. Origin alignment was corrected
and the dataset/control regenerated before any model fit. The group-assignment
seed and 15/3/3 animal allocation were unchanged. The published split binds the
corrected dataset; draft pre-alignment results are not benchmark evidence.

Full label decisions and per-animal diagnostics are written next to the imported
JSON as `.import.json`. The compact public receipt is
`wormwideweb-import-control.json`. The current JSON benchmark is 138,349,356 bytes;
compact, indexed training storage remains an optimization task.

## Initial fixed split and control

- 21 baseline animals; 504 windows (24 per animal).
- 1,768 accepted animal/neuron identities; 93 noncanonical/ambiguous labels excluded.
- Between 59 and 110 accepted neurons per animal.
- Training: 15 animals / 360 windows; validation: 3 / 72; test: 3 / 72.
- Fixed manifest: `data/wormwideweb-animal-split.json`.

The persistence control forecasts the last finite observed value at or before the
origin and has **zero fitted parameters**. Its test macro per-neuron R² is:

| Horizon | Macro neuron R² |
| --- | ---: |
| 1 second | 0.7064 |
| 10 seconds | -0.4660 |
| 30 seconds | -0.6506 |

Each horizon has 5,544 observed neuron/window samples and 126 neurons with defined
R², with no missing requested horizon samples in this subset. The scorer reports
null trace correlation because each persistence forecast is constant; response
AUROC is null because freely moving trials do not supply response labels.

These are pipeline-control results. They do not replace the required fitted LDS
and GRU baselines, nor demonstrate forecasting success by WormSim. The source's
own preprocessing still needs an explicit causality/unit audit before declaring
Task 2 acceptance; preserving a published trace does not prove that all upstream
normalization or filtering was causal. All later models must use the same fixed
observations, preprocessing declaration, and held-out animal manifest.

## Validation

Synthetic native-HDF5 tests exercise both array orientations, one-based ROI
mapping, confidence weighting, irregular timestamps with an aligned origin,
behavior hashing, missing samples, gap rejection, and checksum corruption.
A persistence regression alters all future targets and verifies unchanged
predictions. Linux CI installs HDF5 and runs these tests independently of the
local real-data import. The downloaded data are not required for CI.
