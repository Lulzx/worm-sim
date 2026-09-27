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
