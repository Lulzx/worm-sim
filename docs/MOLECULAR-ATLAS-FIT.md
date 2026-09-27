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
