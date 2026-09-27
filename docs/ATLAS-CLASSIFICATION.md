# Task 1 response annotations and pair classification

Individual [Randi trial traces](RANDI-ATLAS.md) and pair-level response statistics
are different measurements. Repeating a pair's aggregate response label on every
trial would weight frequently stimulated pairs more heavily. The new
`bench::atlas` API therefore scores **one ordered non-self pair once**, with unit
weight and a tie-aware AUROC, separately from trace correlation.

## Frozen published evidence

The source is the unmerged `wt/q` matrix in
[`wormneuroatlas/data/funatlas.h5`](https://github.com/francescorandi/wormneuroatlas/blob/b2e13d88b670efcb3438aeacba2ad4bd6c383933/wormneuroatlas/data/funatlas.h5),
compiled June 28, 2023. Source SHA-256:
`53a99055667b853e1d3d6be573ec2613d38c9f6989f302ec38ecd38dd50c7975`.
It has 300 source identities. Matrices use **responding neurons in rows,
stimulated neurons in columns**. The native HDF5 importer uses exact identity
matches and never merges classes or guesses the anatomical side of AWCON/OFF.
The [upstream loader](https://github.com/francescorandi/wormneuroatlas/blob/b2e13d88b670efcb3438aeacba2ad4bd6c383933/wormneuroatlas/NeuroAtlas.py#L862)
warns against averaging q-values when merging identities; we do not perform that
operation.

The fixed detection criterion is **published q < 0.05**. A finite q at or above
0.05 is labeled **not detected**, not proven physiological absence. A missing q
is excluded, even if observation counts are nonzero. Self-pairs and pairs absent
from the imported trace cohort are excluded. The evidence artifact retains
published observation counts and equivalence-testing q-values separately.
Equivalence and detection tests can both be significant; equivalence evidence
does not overwrite the detection label. The source equivalence margin is retained
as uninterpreted source metadata, not confused with the q-value threshold.

These are externally published labels aggregated over source measurements, not
statistics recomputed from our filtered twenty-second trace windows. They include
source preprocessing and statistical decisions. Fitting code must partition them
by stimulated neuron using the existing split and must never treat held-out q
values as model inputs. Prediction metadata records training/selection trial IDs,
model parameter counts and source revision; the scorer checks their declared
partition membership, which is not proof of actual information access.

## Full recording metadata

[`export_randi_metadata.py`](../scripts/export_randi_metadata.py) verifies the
1,143,920,663-byte full OSF archive against SHA-256
`f59e8f1f74cc468559a230a3b44832ebe394680be0b73ee871633510a4df9165`.
It streams only selected metadata members and decodes NumPy arrays and inert
attribute containers through a restricted unpickler. It never imports or executes
upstream Fconn or recording methods, and does not extract the archive's roughly
21.6 GB of expanded image/segmentation-related contents.

The [metadata audit](randi-metadata-audit.json) matches **113 recordings and all
5,808 stimulation entries** exactly against the previously pinned text export.
It preserves **69,557 positive detector ROI entries**, detector settings,
detector observation windows, targeted-neuron-hit flags and raw optical pulse
counts/dividers/train counts. Per-ROI eligibility masks are not stored in the
Fconn objects; absence from a positive list is consequently **not** converted to
a negative label. Raw acquisition arrays are not yet a verified per-event optical
duration/power conversion or membrane-current calibration.

The [source detector](https://github.com/leiferlab/pumpprobe/blob/1dbc5e0a2b609d54bc9b1c90c73d4e3bf183d3c7/pumpprobe/Fconn.py)
uses signal-quality, amplitude and derivative criteria. This is why importing the
positive lists alone does not justify assigning every other observed cell a
negative response. The separate pair-level published statistical test gives a
precisely named classification target without making that substitution.

## Reproduction and validation

```sh
python3 scripts/export_randi_metadata.py
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --features hdf5 --example import_atlas_pairs
target/release/examples/import_atlas_pairs data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json runs/randi-source/funatlas.h5 runs/randi
```

The Rust tests cover HDF5 orientation, missing q-values, source hash validation,
exclusion of self-pairs, one-per-pair scoring despite repeated trials, duplicate
or missing predictions, and disallowed training-partition declarations. This
prepares classification evidence and scoring; a retrained held-out-neuron linear
baseline, a biological fit, uncertainty estimates and actual comparative Task 1
results remain outstanding.

## Full-corpus label audit

Committed importer `2329ba0` produced **23,316 ordered non-self pair labels**.
The [independent audit](randi-pair-label-audit.json) checks every detection q,
equivalence q, observation count, matrix direction and partition assignment
against the pinned HDF5 source. The original trace dataset hash and bytes are
unchanged. Pair labels are a separately hashed evidence artifact.

| Partition | Pairs | Detected (q < 0.05) | Not detected | Detected and equivalent |
| --- | ---: | ---: | ---: | ---: |
| Train | 19,833 | 984 | 18,849 | 292 |
| Validation | 1,725 | 48 | 1,677 | 13 |
| Test | 1,758 | 82 | 1,676 | 19 |

The last column is a diagnostic overlap between two different tests, not an
additional class. The original detection label is retained. No model or threshold
was selected using these class counts. Future comparisons must retain this class
imbalance and the shared source preprocessing; they must not silently change the
negative class to equivalence-only pairs or duplicate labels by trial count.

```sh
python3 scripts/audit_atlas_pairs.py
```
