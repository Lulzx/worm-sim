# Molecular evidence for synaptic priors

`molecular` imports source-thresholded receptor expression in native Rust and
combines it with separately attributed transmitter/receptor tables. It produces
qualitative evidence for existing directed chemical edges. It does not rewrite
the graph, change benchmark splits, or claim biological sign measurements.

## Sources and explicit mapping

The [Fenyves supplementary workbook](https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=10.1371/journal.pcbi.1007974.s003)
was downloaded from the publisher and compared byte-for-byte by SHA-256 with the
pinned Worm Neuro Atlas mirror: both are 983,199 bytes, SHA-256
`85959066fd7cbdbc2024d0ebb323b71c4365f4083bc85e555ee973f470697c47`.
The [paper](https://doi.org/10.1371/journal.pcbi.1007974) declares Creative Commons
Attribution. `scripts/fetch_molecular_sources.py` extracts literal cells from
`1. NT expr` columns A:C and `2. Receptor gene table` columns A:F, retaining source
row/cell coordinates. No formulas or upstream software execute. The
[derived catalog](../data/fenyves-transmitter-receptors.json) contains all 302
canonical cells and 62 receptor genes. Only numeric zero-padding aliases change;
AWC anatomical side labels remain intact. Empty transmitter fields mean no
catalog evidence for these three transmitter systems, not transmitter absence.

The pinned `cengen.h5` file is 32,852,608 bytes, SHA-256
`d8e6f6f2a25e05211676cfd9c3dd18b8a8bb5c3db4d673fd87b61ed91637ed23`.
It contains 128 expression clusters and 13,669 named genes at each of four
threshold categories. Requested genes retain their WormBase IDs and exact f32
TPM values. The importer verifies source bytes, identifier uniqueness, dimensions,
nonnegative finite TPMs and threshold choice. Requested gene names absent from
the source are listed separately, never zero-filled or silently aliased.

The [official CeNGEN L4 documentation](https://www.cengen.org/l4/) describes
threshold 4 as most stringent, with increased false negatives, and recommends
threshold 2 as a balance. These threshold categories are preprocessing settings,
not user-selected TPM cutoffs. L4 expression is not a direct observation of the
individual adult atlas worms. Threshold sensitivity remains relevant to any fit.

The [materialized cell map](../data/cengen-cell-map.json) assigns 300 anatomical
cells to source clusters and explicitly leaves AWCL and AWCR unmapped, since
ON/OFF identity is not fixed by side. It includes the DA9, DB1, VA12, VB1/2,
VC4/5, IL2, RMD, RME and pooled VD/DD exceptions. These are declared transfers of
class expression to anatomical cells; they do not establish cell-resolved
expression, identical dynamics, or obligatory parameter sharing. Inference uses
this exact reviewed table and performs no prefix matching.

## Evidence rule and uncertainty

For each existing chemical edge, use both dominant and alternative presynaptic
transmitters. Positive source-thresholded TPM identifies a detected receptor in
the mapped postsynaptic class. Preserve the genes supporting each polarity and
all missing candidate names. The mutually exclusive result is:

- `conflicting`: detected receptors support both polarities, potentially through
  different co-transmitters. This is not evidence of cancellation or zero current.
- `incomplete_receptors`: a candidate gene is absent from the source, and conflict
  has not already been demonstrated. No confident direction is assigned.
- `excitatory` or `inhibitory`: complete catalog coverage and detected receptors
  supporting only that direction, conditional on the source threshold.
- `no_detected_receptor`: complete coverage but no receptor passes the threshold;
  this is not proof of physiological absence.
- `no_transmitter_evidence` or `unmapped_postsynaptic_class`: the required source
  assignment is unavailable. The retained reason must not become an assumed sign.

At threshold 4, seven catalog names are missing: `acr-1`, `acr-10`, `acr-13`,
`acr-25`, `acr-4`, `eat-2`, and `lgc-48`. They may require separately sourced gene
alias resolution; this importer does not invent aliases. These gaps particularly
affect cholinergic coverage, so directional counts must not be interpreted as
the network's excitatory/inhibitory balance.

`Evidence::probabilities(confidence)` maps complete directional categories to
confidence or 1−confidence and all other categories to 0.5. The confidence must
lie strictly between 0.5 and 1 and is explicitly a modeling assumption, not a
probability calibrated by the source. The [atlas fitting integration](MOLECULAR-ATLAS-FIT.md) projects this evidence
into source-linked parameter initialization and sign penalties. Earlier atlas
runs retain their original neutral graph priors.

## Reproduction and checks

```sh
python3 scripts/fetch_molecular_sources.py
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --features hdf5 --example import_molecular_evidence
target/release/examples/import_molecular_evidence data/c302-herm.wsc data/fenyves-transmitter-receptors.json runs/molecular-source/cengen.h5 data/cengen-cell-map.json 4 runs/molecular-th4
python3 scripts/audit_molecular_evidence.py --run runs/molecular-th4 --output runs/molecular-th4-audit.json
```

Python dependencies `openpyxl`, `h5py` and NumPy support source extraction and an
independent audit; the HDF5 importer and inference rule are Rust. The audit checks
all catalog cells, all 7,040 selected class/gene TPM entries and all 3,638 directed
edge results against source data. Synthetic tests cover direction, co-transmitter
conflict, incomplete receptor coverage, missing class assignments, source hashes,
HDF5 orientation and threshold rejection. The explicit anatomical mapping remains
a modeling assumption rather than independently measured cell expression.

The [source audit](SIGN-PRIOR-SOURCE-AUDIT.md) describes software/data attribution
boundaries. The GPL upstream Python implementation was inspected for source
locations but has not been copied or executed in the MIT Rust core.


## Source-stamped corpus verification

Importer source `260ef29` was run separately at thresholds 2 and 4. The
[independent threshold-2 audit](molecular-th2-audit.json) and
[threshold-4 audit](molecular-th4-audit.json) each verify all 7,040 selected TPM
values exactly in f32, all 302 transmitter rows, all 62 receptor rows and all
3,638 directed edge results. Their receipts preserve full source and artifact
hashes. No upstream neural-response labels are accessed by import or inference.

| Evidence category | Threshold 2 | Threshold 4 |
| --- | ---: | ---: |
| Excitatory | 8 | 122 |
| Inhibitory | 257 | 440 |
| Conflicting | 2,743 | 1,829 |
| Incomplete receptor catalog coverage | 139 | 640 |
| No detected receptor | 4 | 120 |
| No transmitter evidence | 465 | 465 |
| Unmapped postsynaptic class | 22 | 22 |
| Total anatomical chemical edges | 3,638 | 3,638 |

The [paired category transition receipt](molecular-threshold-sensitivity.json)
shows **971 edges change category** between thresholds. A stricter threshold
can turn detected mixed receptor expression into apparently unopposed evidence;
that is not proof that the opposing physiological pathway is absent. Most edges
remain conflicting at either threshold. Missing genes, receptor localization,
subunit assembly, metabotropic effects and L4-to-adult transfer limit this simple
ionotropic expression rule. Do not collapse this table into a measured network
sign ratio or tune the threshold against held-out response labels.

All 93 Rust tests and strict all-target Clippy passed for the importer change.
These checks establish source fidelity and declared rule behavior. The fitting integration is described in [MOLECULAR-ATLAS-FIT.md](MOLECULAR-ATLAS-FIT.md);
no benchmark improvement is claimed from this import alone.
