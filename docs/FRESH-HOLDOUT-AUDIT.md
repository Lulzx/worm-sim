# Fresh holdout search: atlas provenance check

Checked 2026-09-28 using metadata only. No fresh confirmatory Task 1 cohort has
been secured. The existing validation and test targets remain exploratory.

The [Leifer lab publication list](https://leiferlab.princeton.edu/publications.php)
links the Randi 2023 paper to [DANDI 001075](https://dandiarchive.org/dandiset/001075).
That page lists a draft updated in May 2026 as well as published version
`0.240930.1859`. A recent metadata update does not establish new experiments.

The [reproducible overlap audit](dandi-atlas-overlap-audit.json) finds:

- Both complete inventories contain 223 assets; their downloaded JSON bytes have
  the same SHA-256. Asset paths, IDs, blob IDs and sizes are identical.
- All 113 DANDI subject/session pairs match the recording index and acquisition
  timestamp in the pinned OSF source's `*_ds_name.txt` files.
- No unknown subjects, conflicting timestamps or missing source recordings occur.

The script verifies every recording-name file against the existing source
manifest before comparing IDs. It reads no fluorescence, image arrays, NWB
payloads or held-out response values. NWB conversion may improve metadata and
processing access, but this inventory supplies no independent recording cohort.
DANDI's subject labels also do not independently verify the biological
recording-to-animal map; that limitation remains.

This rejects one apparent candidate. It does **not** establish that no suitable
external dataset exists. A genuine fresh cohort still requires recording-level
provenance, absence of overlap with previously inspected data, compatible
stimuli/readouts, and frozen eligibility and scoring rules before viewing outcomes.

## Reproduce

Only the metadata inventories are downloaded:

```sh
mkdir -p runs/dandi-holdout-metadata
curl -fL 'https://api.dandiarchive.org/api/dandisets/001075/versions/0.240930.1859/assets/?page_size=1000' \
  -o runs/dandi-holdout-metadata/0.240930.1859-assets.json
curl -fL 'https://api.dandiarchive.org/api/dandisets/001075/versions/draft/assets/?page_size=1000' \
  -o runs/dandi-holdout-metadata/draft-assets.json
python3 scripts/audit_dandi_overlap.py \
  --published-assets runs/dandi-holdout-metadata/0.240930.1859-assets.json \
  --draft-assets runs/dandi-holdout-metadata/draft-assets.json \
  --source-manifest data/randi-source-manifest.json \
  --source-directory runs/randi-source/exported_data \
  --output runs/dandi-overlap-new.json
```

The audit refuses incomplete pagination and duplicate or unrecognized paths.
The draft is mutable: future inventories may differ, which must trigger a new
provenance review rather than inheriting this conclusion.

## TWISP deposition: compatibility unverified

The [TWISP dataset, Figshare version 1](https://doi.org/10.6084/m9.figshare.23868972.v1)
was also screened through archive metadata. Its published ZIP is 758,409,991
bytes. The reproducible [directory receipt](twisp-inventory-audit.json) requests
only the 22-byte ZIP end record and 5,999-byte central directory. It lists eight
`*_behavior_data.mat` members and six supplementary `.docx` tables; the remaining
entries are directories or macOS metadata. No data member was decompressed or
interpreted. An initial exploratory tail request also received opaque compressed
bytes adjoining the directory; those were not decompressed or used. The committed
script reproduces the inventory using exact directory ranges only.

These filenames do not establish available labeled single-neuron stimulation
traces, nor do they provide recording timestamps for an overlap check. The
candidate is therefore **unverified**, not a secured holdout and not proof of
incompatibility. The published MD5 is recorded but not independently verified,
since the archive payload was not downloaded. Before opening data values, a
schema/recording manifest would need to establish compatibility and independence.

```sh
python3 scripts/inspect_twisp_inventory.py --output runs/twisp-inventory-new.json
```


## Dunn 2025: new candidate inventory, eligibility pending

A metadata-only screen of [DANDI 001623, published version 0.251015.0312](https://dandiarchive.org/dandiset/001623/0.251015.0312)
finds 95 NWB asset paths with distinct dated recording IDs. Every ID has a
matching `.pkl` filename in [Zenodo record 17353307](https://doi.org/10.5281/zenodo.17353307).
The [upstream repository](https://github.com/focolab/2025-dunn-et-al-curr-biol/tree/98016334b89ef087bdf39de938dad52a2cc8a47a)
links these raw and processed deposits; its README explains the recording-name
convention and analysis loading paths.

The [inventory receipt](dunn-holdout-inventory.json) pins the published DANDI
version, upstream commit, downloaded metadata hashes, all 95 asset IDs/paths,
and the deposited processed-file sizes/checksums. None of the 95 recording
names has an exact timestamp-string match among the 113 content-hash-verified
Randi source recording names. This is not a biological animal-identity audit or
a timezone-normalized comparison. Deposited checksums are metadata, not locally
verified data checksums, because no payloads were downloaded.

Task 1 compatibility is **not yet established**. The pinned
`lib/wbliveDataClass.py` source distinguishes localized and widefield stimuli,
loads onset/offset metadata, and supports identity/ROI-dependent stimulus filters.
Those capabilities do not establish which recordings contain eligible
single-neuron perturbations, stimulus amplitudes, target identity confidence,
actuator polarity, or compatible baseline/response windows. The first sampled
DANDI asset's session description is a placeholder, so that field cannot supply
these details. No response-based eligibility decision has been made.

Before admitting this candidate, extract a metadata-only per-event manifest,
verify recording/animal provenance and optogenetic calibration, then freeze
eligibility, preprocessing and scoring rules before accessing response values.
No NWB, pickle, CSV, ZIP member, image or calcium payload was opened in this
screen. Search snippets and public study descriptions were visible; this is not
a claim of blindness to all published findings. The confirmatory holdout remains
unsecured.

```sh
python3 scripts/audit_dunn_inventory.py \
  --source-manifest data/randi-source-manifest.json \
  --source-directory runs/randi-source/exported_data \
  --output-directory runs/dunn-holdout-metadata-new
```

The script downloads only JSON inventories and pinned repository source text,
rejects incomplete pagination or duplicate/conflicting recording identifiers,
and verifies the existing Randi recording-name files before comparison.


The processed-data ZIP was subsequently screened using exact directory ranges:
[archive inventory](dunn-archive-inventory.json). Only its 22-byte end record and
791-byte central directory were requested. Its eight entries comprise named
sine-fit CSVs, three compressed NumPy files, embedding accuracy, reversal labels,
and a reversal dataframe. None is separately named as an event manifest; filenames
alone do not prove that no useful metadata exists within these files. No member
payload was requested or decompressed, so per-event eligibility remains unresolved.

```sh
python3 scripts/inspect_dunn_archive.py --output runs/dunn-archive-inventory-new.json
```

## Bounded NWB schema inspection

`scripts/inspect_dunn_nwb_schema.py` selects the lexicographically first asset
path from the pinned inventory and inspects HDF5 object names, dataset shapes
and dtypes in selected root groups (by default `stimulus` and `intervals`). It does not index datasets or read attribute values. It uses exact
HTTP ranges, rejects servers that ignore ranges, and caps individual reads at
64 KiB, total received bytes at 2 MB, and requests at 500. Exact repeated byte ranges are cached. The output records
range offsets/hashes, asset metadata, software versions and the selected schema
inventory. Inspection errors retain an explicitly incomplete receipt and exit
with an error.

This is a separate phase from the JSON/directory-only inventory above: it reads
NWB structure bytes. Those ranges may include bytes colocated with metadata;
the claim is that no neural response values are decoded or displayed, not that
no raw data byte can enter a range response. A single asset's schema cannot
establish cohort-wide stimulus eligibility, calibration or animal independence.

```sh
uv run --no-project --with h5py==3.16.0 --python 3.12 \
  scripts/inspect_dunn_nwb_schema.py \
  --inventory docs/dunn-holdout-inventory.json \
  --output runs/dunn-first-nwb-schema-new.json
```

The optional h5py environment is separate from the pinned JAX fitter. Four tests cover seeking, exact reads, caching, limits, rejection of a full body
response before reading it, dataset-value access prevention during traversal,
and incomplete receipt retention after an HDF5 error. These checks do not
establish dataset compatibility.


### First targeted schema result

The original unrestricted object traversal stopped at the read-budget guard
(exit 1) and did not produce a complete receipt. Its exact received-byte/request
count was not retained; no schema-completeness conclusion is drawn from it.
Rather than increase its budget, the follow-up restricted traversal to the
standard `stimulus` and `intervals` root groups and retained partial receipts
on inspection failures.

The [targeted receipt](dunn-first-nwb-stimulus-schema.json) completes for
`sub-20220302-11-45-51_ses-20220302T114551.nwb`, the lexicographically first
inventory asset. It received **8,160 bytes** in exact ranges. The `stimulus`
group contains empty `presentation` and `templates` groups; `intervals` is
absent. No stimulus datasets exist in those selected locations for this file.
This is evidence that those standard locations cannot supply its per-event
stimulus manifest. It does not establish that metadata is absent elsewhere in
the NWB, in the processed pickle, or in the other 94 recordings. The published
repository describes the NWBs as raw data, so processed-recording metadata
remains a separate lead. No neural response values were indexed, decoded or
displayed; no confirmatory cohort has been secured.

### Converter source: recording identity is not animal identity

The pinned repository's [NWB converter](https://github.com/focolab/2025-dunn-et-al-curr-biol/blob/98016334b89ef087bdf39de938dad52a2cc8a47a/lib/clefNWB.py)
provides a more specific reason not to equate different NWB subjects with
independent animals. `create_nwbfile` (lines 25–60) derives the identifier and
session time from the recording name, using US/Pacific time. `create_subject`
(lines 63–92) assigns that identifier directly to `subject_id`. A separate
animal identifier is not supplied there. Several subject fields, including
birth date and growth stage, are defaults in this converter; they cannot be
used as measured evidence of biological independence.

The [source receipt](dunn-converter-source-audit.json) pins the source hash and
function locations. The sole `dc.md` access in this file supplies subject strain.
The conversion entry point creates subject, imaging and segmentation structures
and writes the file; there is no explicit stimulation-event export in this source.
This supports prioritizing the processed metadata's `stim_param_list` over
further broad NWB traversal when determining delivered events. It does not prove
that every published NWB was produced by this exact converter version.

The repository tree at that commit is complete (not truncated) and lists no
standalone JSON or CSV event manifest. The processed-recording metadata remains
the next lead. Establish per-event eligibility and animal/session relationships
before declaring any candidate subset confirmatory. This source inspection did
not decode recording arrays or inspect held-out outcomes.

### Acquisition traversal reached its request limit

A second targeted inspection selected only `acquisition` in the same first
asset. It terminated with exit 1 at the unchanged **500 range-request limit**,
after receiving **1,748,712 bytes**. The [incomplete receipt](dunn-first-nwb-acquisition-schema.json)
retains the exact ranges and hashes, partial object list, and exception. Its
byte sum, per-range bounds, and input/source hashes were checked after termination.

Only `acquisition` and `acquisition/CalciumImageSeries` were listed before the
limit. This does not establish the full contents of the acquisition group or
absence of events elsewhere. No dataset values were indexed or decoded; range
responses may contain colocated raw bytes. The process was not retried with a
larger limit. Together with the converter source audit, this favors using the
processed metadata to establish stimulus eligibility instead of spending more
requests traversing raw-image structures. Animal-level independence and a fresh
confirmatory cohort remain unverified.
