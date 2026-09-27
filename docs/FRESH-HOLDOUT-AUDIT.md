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
