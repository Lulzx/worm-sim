# Randi stimulation-trial ingestion

The Task 1 target is held-out **stimulated neurons**, not the recording split used
by the published pretrained Creamer artifact. This importer prepares individual
trials for that target; it does not yet implement an atlas fit or establish a win
over the Creamer baseline.

## Pinned source and semantics

[Randi et al., Nature 2023](https://www.nature.com/articles/s41586-023-06683-4)
points to [OSF e2syt](https://osf.io/e2syt/). We use the wild-type processed text
archive `exported_data.tar.gz`, file version 1, uploaded May 10, 2023:
523,093,816 bytes, SHA-256
`d6e7b3d93175b40b7ae17bde2182835e9c2144388142c522ee9be3832f6ce836`.
The [source manifest](../data/randi-source-manifest.json) binds all 678 files
from 113 recordings. Acquisition writes only ignored `runs/` files; raw source
samples are not redistributed in this repository.

The six files per recording contain a time-by-ROI fluorescence matrix, timestamps,
ROI labels, stimulation-frame indices, stimulated-ROI indices and recording name.
The source [export method](https://github.com/leiferlab/pumpprobe/blob/1dbc5e0a2b609d54bc9b1c90c73d4e3bf183d3c7/pumpprobe/Funatlas.py#L279)
uses NumPy frame/ROI indices, i.e. zero-based indices. Its
[export command](https://github.com/leiferlab/pumpprobe/blob/1dbc5e0a2b609d54bc9b1c90c73d4e3bf183d3c7/scripts/fconnectivity/figures/paper/reminder_exporting_data.txt)
excludes mutant-tagged datasets; the
[driver](https://github.com/leiferlab/pumpprobe/blob/1dbc5e0a2b609d54bc9b1c90c73d4e3bf183d3c7/scripts/fconnectivity/funatlas_plot_intensity_map.py)
requests spike removal, smoothing and photobleaching correction. This code audit
explains the export pathway, but it is not an authenticated per-recording history
of every processing setting. These are processed fluorescence measurements,
not raw images or prospectively processed signals. No upstream Python code is
executed by the Rust importer.

## Declared trial construction

The [configuration](../configs/randi-import.json) uses ten seconds before the
stimulation frame for the fluorescence baseline and twenty seconds starting at
that frame for the response. No resampling or interpolation is added. Baseline
and response ranges are half-open. For each ROI, ΔF/F is `(F - mean_baseline) /
mean_baseline`; the baseline excludes the stimulation frame and all subsequent
samples. At least 80% of the baseline samples must be finite, its mean must be
positive, and at least two response samples must be finite. NaNs remain missing;
infinities or malformed matrix dimensions reject the source. Three recordings
have surplus label rows (39, 2 and 1 respectively), all blank. Only such trailing
blank rows beyond the matrix width are discarded, with counts in the receipt;
extra nonblank labels cause an error.

The complete baseline-plus-response ranges of retained events must not overlap.
Both events are rejected when their proposed ranges overlap, even if one event
has an unknown target identity. This prevents source-frame duplication across
stimulated-neuron partitions. It does not remove shared-recording dependencies,
upstream preprocessing dependencies, or residual activity from an earlier pulse.

Only unique, exact anatomical identities present in the graph are accepted.
Empty/unknown names, ambiguous classes and duplicate identities are excluded.
In particular AWCON/OFF are not arbitrarily assigned to AWCL/R. Every label and
event exclusion is counted. Unit identity weights are a declared convention,
not calibrated probabilities. The source supplies recording identifiers but no
verified recording-to-animal map: `animal_id` explicitly stores `recording:...`.
Do not report a bootstrap over those identifiers as a bootstrap over animals.

Stimulus onset is t=0 in every emitted trace. Per-event pulse duration, optical
power and membrane-current calibration are missing from this export. They are
not invented. Published response/nonresponse labels are also absent, so
`response_labels` remains empty and response AUROC is unavailable. A high or low
trace amplitude is not silently converted into a classification target.

The example generates a seed-42 stimulated-neuron split with 15 validation targets,
15 test targets and the remaining targets for training. Source-frame ranges and
stimulation metadata remain in the import receipt. The native benchmark hash binds
all transformed samples, identities and preprocessing configuration.

## Reproduction and checks

```sh
python3 scripts/fetch_randi.py
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example import_randi
target/release/examples/import_randi data/c302-herm.wsc runs/randi-source/exported_data data/randi-source-manifest.json configs/randi-import.json runs/randi
```

Targeted Rust tests exercise an independent hand-calculated ΔF/F fixture, frame
alignment, NaN preservation, duplicate/unknown identities, nonpositive baselines,
neighbor-event overlap, malformed timestamps/events, source hash corruption and
the stimulated-neuron split. Real-data receipt and independent sample checks are
recorded after running the committed importer.
