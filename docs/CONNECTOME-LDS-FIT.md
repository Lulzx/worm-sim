# Connectome-constrained LDS training for held-out stimulation targets

Task 1 holds out entire stimulated-neuron identities. The pinned Creamer model
has a separate diagonal input kernel for each neuron (45 lags at 2 Hz). Under
our split, a test target's input column is never activated in training, so that
kernel is unidentifiable from training observations. Reusing pretrained kernels
would import information from the upstream recording split; leaving them at
arbitrary initialization would not be a meaningful retrained baseline.

The new Rust `ConnectomeLds` therefore uses **one stimulation kernel shared across
all target neurons**. This is a declared adaptation for the specified split, not
an unchanged reproduction of Creamer's published training run. The original
pretrained-model parity results remain separately documented in [BASELINE.md](BASELINE.md).

## Model and relation to the published baseline

For an impulse delivered to neuron s at time zero:

```
x[0] ~ Normal(0, P0)
x[t+1] = A x[t] + e_s b[t] + process_noise
y[t] = x[t] + observation_noise
```

`A` contains only self terms, directed chemical edges and both directions of gap
junctions from the specified graph. Presence, not synapse count, defines this
support. Coefficients may have either sign. The readout and input routing matrices
are fixed identities. Process and observation noise and the shared zero-mean
initial-state covariance are diagonal and fitted. **Posterior state covariance
is full**, with masked Kalman filtering and RTS smoothing across all latent cells.
Diagonal generative noise does not justify dropping posterior correlations.

The shared kernel `b[t]` is learned jointly with allowed `A` entries and is zero
beyond its declared lag count. The first observation is at t=0; `b[0]` affects the
first transition, not that initial observation. No held-out fluorescence enters
an impulse prediction. Kernel coefficients describe a latent fluorescence input
response, not a calibrated optical-to-membrane-current conversion.

The pinned [upstream implementation](https://github.com/Nondairy-Creamer/Creamer_LDS_2026/blob/bba43302d50a4947804d98b01779856e648237cc/ssm_classes.py)
and fitted artifact use synaptic support, diagonal dynamics input kernels,
diagonal process/observation noise and a fixed identity readout. Our differences
are explicit: shared input kernel; c302 support and all 302 latent cells rather
than the published 154-cell support; our ΔF/F event windows rather than upstream
continuous-recording preprocessing; shared zero-mean initial-state distribution;
positive ridge regularization, a transition operator-norm cap and noise floors.
The published update checks/warns about transition eigenvalues; it does not use
this same projection. These changes must be disclosed in any future comparison.

## Constrained parameter update

For each training sequence the full Gaussian posterior supplies first, second and
lagged state moments. The expected squared transition residual is weighted by
inverse current process variance. A ridge penalty equal to `ridge × transitions`
is applied to every allowed A entry and every shared kernel coefficient.

Each output row has its own anatomical regression coefficients; all rows share
kernel coefficients. The Rust implementation eliminates the row-local variables
and solves the resulting kernel normal system, then reconstructs each row. This
is algebraically the joint constrained normal solve, not a sequence of unrelated
per-neuron input fits. Unstimulated training identities still use the same learned
kernel when predicted.

If A exceeds the declared spectral/operator-norm cap, it is scaled uniformly.
The shared kernel is then refitted conditionally on the projected A. Diagonal Q
is updated from the expected residual under those final parameters, including
state uncertainty and lag covariance. R uses confidence-weighted expected
observation residuals; unobserved output noise retains its previous value.
P0 uses the average initial-state second moments. All fitted variances have a
1e−6 floor. This is a constrained EM-style update, not an unconstrained
maximum-likelihood EM monotonicity claim.

## Independent validation and workload check

The [NumPy reference generator](../scripts/generate_connectome_lds_fixture.py)
conditions one dense joint Gaussian over all times and solves a **full parameter
normal matrix**, with no Kalman recursion or row elimination. Rust agrees within
1e−10 for likelihood, allowed transition weights, the shared kernel and all noise
updates. Both inactive and active stability projection cases are tested. A target
never stimulated in the fixture's training sequences receives the learned shared
kernel; forbidden weights remain zero. Unsupported kernel lags are rejected.

```sh
python3 scripts/generate_connectome_lds_fixture.py
cargo test --locked --test connectome_lds
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example benchmark_connectome_lds
target/release/examples/benchmark_connectome_lds data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json runs/connectome-lds-workload.json
```

The workload command uses the lexically first **training** trial and measures one
full-state E/M step with 39 kernel lags for its 40-frame window. It does not fit a
population or evaluate validation/test data. The initial window timing must not
be presented as whole-fit throughput or biological acceptance. Population fitting,
validation selection, held-out trace/pair scoring and uncertainty remain pending.

### Exact covariance reuse

The constrained E-step groups sequences by their complete ordered, per-frame
output identities and confidence-weight bit patterns. For each group it prepares
Kalman scalar-update covariances/variances, RTS gains and smoothed/lag covariances
once. Observed values and stimulation inputs affect the conditional means and
likelihood but not these covariances. Each sequence still gets its own mean and
likelihood calculation. The model is borrowed immutably by the plan, preventing
stale reuse after parameter updates. Changed masks or weights are rejected.

Only one group's covariance plan is held at a time. Posterior covariances are
borrowed rather than copied for every trial. This preserves full covariance;
it introduces no diagonal/block approximation. Moment summation order changes
with grouping, so floating-point roundoff may differ from original trial order.
The original general Gaussian smoothing path remains available as a reference.
Tests compare both paths with dense readouts, correlated noise, missing frames,
repeated observations, varying confidence, changed data and changed inputs.

The workload report additionally times preparation and 16 repeated inferences on
one real training window, checking mean and covariance parity. This isolates
reuse cost; it is explicitly not a population throughput measurement.

Measured on the local Apple M4 Pro with release Rust, f64 arithmetic and source
`0e4bd682b3a5a161d67903d5a58b555df13a70e5`: preparation took 1.971 s;
16 subsequent mean/likelihood inferences including parity checks took 0.09663 s
(6.04 ms each). Means matched the reference exactly on this window, as did full
and lag covariance arrays. The one-window E/M step took 2.041 s. The
[receipt](connectome-lds-reuse-workload.json) pins the training trial, graph,
dataset, split and timing boundaries. Measurements use diagonal initialization;
trained-model and distinct-window population timing remain unmeasured.

## First population protocol

`configs/connectome-lds-first-fit.json` fixes three full training EM updates,
39 shared input lags, ridge 1e−4 and contraction cap 0.995. Initialization is also
a candidate. Checkpoint selection minimizes pooled confidence-weighted validation
MSE (earlier checkpoint on ties), not training likelihood or test performance.
MSE includes constant predictions whose trace correlations are undefined. The
common scorer separately reports correlation and its defined-trace count.

The fitter accepts only a held-out stimulated-neuron split, constructs EM
sequences exclusively from training trials, and predicts unconditioned impulses
for validation/test targets. No fluorescence values, behavior values or published
pair labels enter impulse prediction. Full-dataset hashes and structural validation
are checked; those checks do not supply outcome values to the fit. A regression
test changes every test fluorescence value and verifies identical fitted dynamics,
validation selection and test predictions after rebinding the dataset identity.

The model artifact records graph/data/split identities, config, source commit,
training and selection IDs, and iteration. The nominal fitted parameter count
includes supported A, the shared kernel, diagonal Q/P0 and training-observed R;
unobserved R is fixed. Initialization is reported with the same model-family
parameter count. Each candidate and score is persisted before the next update.
The run directory must be new, preventing accidental checkpoint replacement.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example fit_connectome_lds
target/release/examples/fit_connectome_lds data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json configs/connectome-lds-first-fit.json runs/connectome-lds-first-fit
```

This first protocol produces trace scores. Published pair classification and
cluster uncertainty remain separate required evaluation steps; a point score
alone is not evidence of improvement over the required baseline.

## Frozen-fit pair ranking and uncertainty

The separate `evaluate_connectome_atlas` example scores published pair detection
using `dt × sum(abs(predicted fluorescence))` over the common response grid.
This score is fixed without fitting a classifier or selecting against published
q-values. All trial impulse predictions for a given pair must agree exactly;
there is one score per ordered non-self pair. Mixed response grids are rejected.
The model remains the checkpoint chosen using validation trace MSE.

Uncertainty uses 2,000 percentile bootstrap replicates with seed 42. Trace MSE
and confidence-weighted mean defined trace correlation are computed separately
under stimulated-neuron cluster resampling and recording cluster resampling.
All traces/windows within a sampled cluster stay together. Pair AUROC resamples
stimulated identities, retaining all responding pairs and multiplying their
weights when a target is drawn repeatedly. A replicate containing only one
class has undefined AUROC and is excluded; the report gives defined counts.

These are **separate marginal intervals**, conditional on the frozen fitted
model. Targets and recordings form crossed dependencies; neither marginal
analysis accounts for both simultaneously. Source recording IDs are not verified
animal identities. Published aggregate pair labels lack an event-level recording
decomposition, so recording bootstrap is not claimed for pair classification.
Constant predictions have undefined trace correlation; report coverage alongside
correlation and MSE. These intervals alone do not prove a model comparison.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example evaluate_connectome_atlas
target/release/examples/evaluate_connectome_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json runs/randi-pairs.json runs/connectome-lds-first-fit/selected.json test runs/connectome-lds-first-fit-evaluation
```

## Independent completed-run audit

`scripts/audit_connectome_fit.py` reconstructs every candidate's validation MSE
using dense NumPy matrix-vector propagation and rechecks minimum-MSE selection.
It checks all selected validation/test impulse samples against native sparse
predictions, recomputes trace MSE/correlation, and reports the zero-response MSE
control. A zero response has undefined trace correlation; constant pair scores
have AUROC 0.5 when both classes are present. The audit verifies declared split
membership, disjoint target identities, training observation/transition counts,
checkpoint identity and source lineage, then hashes all result artifacts.
It does not independently rerun EM or prove absence of upstream leakage.

```sh
python3 scripts/test_connectome_audit.py
python3 scripts/audit_connectome_fit.py --output runs/connectome-lds-fit-audit.json
```

The audit requires a completed run and NumPy. Small analytical tests check dense
impulse timing/input routing, confidence weighting, missing samples and undefined
correlations. They run in CI using the existing pinned NumPy environment.
