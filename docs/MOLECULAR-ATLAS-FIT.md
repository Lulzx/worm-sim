# Molecular sign priors in atlas fitting

The Level 0 atlas fitter accepts an optional `molecular_sign_priors` overlay.
The graph, imported recordings, split and published response-label artifact keep
their original identities. The fit configuration records source/evidence hashes,
a declared confidence, and sorted disjoint lists of canonical chemical-edge
indices with complete excitatory or inhibitory expression evidence. Every other
edge has neutral prior probability 0.5. The full molecular artifact preserves
its uncertainty category and supporting/missing gene names separately.

The compact projection validates graph identity, edge ordering, evidence-state
consistency, hash shapes and index coverage. Its provenance is a declared link
to the separately [source-audited molecular evidence](MOLECULAR-PRIORS.md), not a
substitute for that source audit. It does not add anatomical edges or change
transmitter/receptor observations using neural-response labels.

## Initialization, tying and penalty

Each tied sign group starts at `logit(mean edge prior probability)`. This is the
minimum of that group's edge-uniform Bernoulli cross-entropy prior; averaging
logits would not have that property. This value is also the raw-coordinate
shrinkage center. Other parameter values and the shared forecast-default seed
remain unchanged; unforced preparation establishes the parameter-dependent
response baseline. A group containing both directional and neutral evidence
shares their mean initial probability. Individual neutral targets remain 0.5 in
the per-edge penalty even when tying makes their shared parameter non-neutral.

The existing sign penalty now uses the explicit molecular targets when present,
otherwise retaining the original graph-prior path exactly. The overlay replaces
all chemical sign-prior targets for that fit. Signs remain learned relaxed
variables, not hard fixed signs. The evidence adds no trainable parameters.
L/R suffix parameter sharing is retained for this comparison; transferring RNA
class expression does not automatically tie all neuronal dynamics by that class.

## Frozen initial comparison

The primary configuration is
[threshold 2](../configs/level0-atlas-molecular-th2-fit.json), with
[threshold 4](../configs/level0-atlas-molecular-th4-fit.json) as a predeclared
sensitivity run. Threshold 2 follows the balance recommended in the
[CeNGEN L4 documentation](https://www.cengen.org/l4/); threshold 4 previously
served as the atlas toolkit's most stringent default. Neither is selected against
test response labels. Directional confidence is **0.75**, an initial modeling
assumption rather than a source-calibrated probability. Inhibitory evidence gets
0.25 and other categories 0.5. The source categories contain 8 excitatory/257
inhibitory edges at threshold 2 and 122/440 at threshold 4.

Both runs preserve the preceding joint-classification protocol: five full-batch
Adam updates, dt 0.01 s, 60 s preparation, shared 39-lag positive input kernel,
classification weight 0.1, fixed unit calcium/readout scale and zero offset.
Minimum validation trace MSE selects a checkpoint within each run, including
initial epoch zero and retaining earlier exact ties. The threshold-4 sensitivity
run is not a mechanism for promoting whichever test result looks best. These
initial five-update runs do not establish optimization convergence.

Evaluate both with the same trace/pair scorer, paired LDS comparison, numerical
preparation/step sensitivity, and independent NumPy replay. The previously
inspected test cohort makes comparisons exploratory. The existing neutral-prior
joint run provides a matched architecture/protocol reference; a change in outcome
can be attributed to the combined sign initialization/regularization intervention,
not separately to either component.

## Reproduction and validation

Generate a fresh configuration from audited evidence with:

```sh
cargo run --locked --example configure_molecular_fit -- data/c302-herm.wsc configs/level0-atlas-classification-fit.json runs/molecular-th2/evidence.json 0.75 runs/molecular-th2-config.json
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example fit_level0_atlas
target/release/examples/fit_level0_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json configs/level0-atlas-molecular-th2-fit.json runs/level0-atlas-molecular-th2-fit runs/randi-pairs.json
```

Use a new output directory for threshold 4. The configuration generator rejects
an already populated molecular overlay and an existing output file. The fitter
rejects graph/overlay mismatches. Tests check mean-probability tying, exact legacy
prior parity, finite-difference prior gradients, uncertain-edge neutral targets,
evidence tampering, unchanged anatomy and unchanged optimization trajectories
when held-out test fluorescence changes.

The independent fit auditor requires `--molecular-evidence` and `--graph-json`
for a molecular-prior run. It verifies the evidence content hash and projection,
then checks every tied initial sign and raw shrinkage center against the mean
edge probability. Saved neural predictions and classifier outputs are audited by
the same independent replay/scoring path. It does not independently repeat the
optimizer or turn expression evidence into physiological ground truth.

## Completed comparison: no demonstrated Task 1 superiority

Both frozen runs completed from source `2e1210880e4ea6db77717f40f9de37cbebb292b6`,
with 6,783 trainable parameters, selecting epoch 5 by validation trace MSE.

| Test metric | Threshold 2 (primary) | Threshold 4 (sensitivity) | Neutral joint model | LDS |
| --- | ---: | ---: | ---: | ---: |
| Trace MSE | 0.04989662 | 0.04989649 | 0.04989685 | 0.04734484 |
| Macro trace correlation | -0.0031164 | -0.0030390 | 0.0021254 | 0.0452857 |
| Pair AUROC | 0.679623 | 0.690356 | 0.681006 | 0.686187 |

The target-cluster paired 95% AUROC difference intervals are:

| Comparison (first minus second) | Difference | 95% interval |
| --- | ---: | ---: |
| Threshold 2 minus LDS | -0.006563 | [-0.041703, +0.035707] |
| Threshold 4 minus LDS | +0.004169 | [-0.030572, +0.050191] |
| Threshold 2 minus neutral | -0.001383 | [-0.012412, +0.009475] |
| Threshold 4 minus neutral | +0.009350 | [-0.016264, +0.034611] |

All include zero. Neither superiority nor equivalence is established. Both
molecular models have worse MSE than LDS: differences about +0.002552, with
target-cluster intervals approximately [+0.001134, +0.003713]. Threshold 4 remains
a sensitivity run; its larger test AUROC does not make it the selected primary.
Thirteen targets have eligible pair labels, fifteen have trace scores, and all
12,588 test trace correlations are defined. Each bootstrap uses 2,000 paired
draws with seed 42, conditional on the frozen fit. Separate recording and target
intervals do not jointly correct crossed dependence; recordings are not verified
animals. The previously inspected test cohort remains exploratory.

Independent NumPy replay checks all 30 validation/test target grids for each
fit, with maximum fluorescence error below 2.23e-16. Source projection, tied
initialization, saved trace scores, pair ranking and classifier scores also pass.
This does not independently replay optimization or bootstrap draws. Extending
preparation from 60 to 120/240 seconds changes validation AUROC by at most
0.000112; threshold-4 AUROC is unchanged. Half-step prediction changes are about
5.32e-5 and MSE changes below 2.3e-8. Near-zero trace correlation counts remain
sensitive to duration and dt; stable MSE does not make those correlations robust.

Receipts:

- Threshold 2: [independent audit](level0-atlas-molecular-th2-fit-audit.json),
  [uncertainty](level0-atlas-molecular-th2-fit-uncertainty.json),
  [versus LDS](molecular-th2-versus-connectome-lds-first-fit.json),
  [versus neutral](molecular-th2-versus-level0-atlas-classification-fit.json),
  [numerical sensitivity](level0-atlas-molecular-th2-preparation-sensitivity.json).
- Threshold 4: [independent audit](level0-atlas-molecular-th4-fit-audit.json),
  [uncertainty](level0-atlas-molecular-th4-fit-uncertainty.json),
  [versus LDS](molecular-th4-versus-connectome-lds-first-fit.json),
  [versus neutral](molecular-th4-versus-level0-atlas-classification-fit.json),
  [numerical sensitivity](level0-atlas-molecular-th4-preparation-sensitivity.json).

The [training diagnostic](ATLAS-TRAINING-DIAGNOSTIC.md) finds substantial
underfitting and a validation benefit from training-only amplitude calibration.
Address fitting and observation amplitude before interpreting these five-update
runs as a limit on molecular priors or biological model capacity.
