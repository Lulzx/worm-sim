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
