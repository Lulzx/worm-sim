# Level 0 initial-state inference

Task 2 now has a history-only inference path for the complete Level 0 state:
voltage, calcium and presynaptic gates for all 302 neurons (906 scalars). It uses
only finite, positive-confidence observations at or before the supplied origin;
unobserved neurons evolve through the same chemical/gap network. The resulting
state at the origin can initialize a free forecast with `solve::simulate_from_state`.

This is a regularized point estimate, **not a claim of hidden-state identifiability**.
It is not yet the fitted population model or a successful Task 2 forecast.

## Objective and algorithm

The objective is confidence-weighted history MSE plus
`prior_weight * mean((initial_state - default_state)^2)` over all state variables.
The output is an affine transform of the model's calcium fluorescence. The prior
is the existing default single-cell state; it is not asserted to be the coupled
network's equilibrium. The implementation uses Euler integration with exact sample
boundaries and a reverse pass through the actual discrete steps. The state adjoint
includes chemical and gap coupling and the voltage-to-calcium/gate derivatives.
No perturbation protocols are supported in this inference objective yet.

Adam proposes updates to the initial state, with calcium and gate components
projected to [1e-8, 1−1e-8]. Backtracking accepts only a lower objective and stops
when no proposal is accepted. All accepted objectives are recorded. The default
configuration is dt=0.005 seconds, at most 30 proposals, learning rate 0.02 and
prior weight 0.001. These are declared defaults, not validation-selected settings.
A forward tape stores every integration state for this short prefix. No backend
or checkpoint expansion is needed for this workload.

`Readout` makes offset/gain assumptions explicit. `bench::level0::training_readout`
uses only training-animal data: mean maps to calcium 0.5 and one standard deviation
to 0.2 calcium. This is an initialization convention, not measured GCaMP kinetics
or learned population readout parameters. Training-unseen identities use mean=0,
standard deviation=1 and are listed in the audit. Readout calibration parameters
must be counted in any future model comparison. Hierarchical animal readouts are
still outstanding.

## Real-data audit

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release
wormsim level0-infer data/c302-herm.wsc runs/wormwideweb-benchmark.json \
  data/wormwideweb-animal-split.json 2022-06-14-01-window-0000 \
  runs/level0-state-audit.json
```

Use `target/release/wormsim` if the executable is not on PATH. The command uses
frozen default Level 0 parameters and fits only the initial condition. It records
the full parameter vector, calibrated readout, inferred initial/origin states,
data/split/graph hashes, training provenance and source revision. The training
window is selected by sorted ID, not by fit outcome. A compact committed receipt
reports the training-animal audit; full states remain in ignored `runs/` files.

The frozen model has short default time constants. Optimizing its starting state
cannot generally explain new activity arriving late in a ten-second prefix. This
limitation is a reason to fit dynamics and observation parameters; it is not a
reason to infer states from future test targets.

## Checks and remaining work

Tests compare every state-gradient component against central finite differences
on a coupled network with gaps, partial observations and nontrivial readouts.
Other checks cover prefix-loss reduction, unchanged inference after future targets
are replaced, complete latent-state carry, and exact compatibility with the old
solver when initialized with its default state.

The current c302 graph has `class="unannotated"`, unknown side metadata, and
sign prior 0.5 for every chemical edge. That is missing biological annotation,
not a class-sharing instruction or evidence for excitatory/inhibitory identity.
Explicit sharing maps and source-backed sign priors remain to be added, followed
by a population parameter fit and common held-out scoring. A stable latent LDS
baseline is also still required; the failed dense linear experiment is retained.

The [committed-source training audit](initial-state-receipt.json) ran all 15
preselected windows using 67–110 observed neurons each. Every run retained all
906 state values and accepted 30 updates. Prefix objectives fell by 10.45–13.51%.
Median inference time was 1.60 s per window, 23.62 s summed across the 15 windows
on this M4 Pro CPU; these timings exclude data loading, training-readout calibration
and artifact writing. No parameter-gradient or forecast-throughput claim follows.
The modest prefix improvement under frozen defaults leaves substantial residuals;
shared dynamics/readout fitting is the next required modeling step.

Reproduce the complete training audit with:

```sh
python3 scripts/audit_initial_states.py --receipt runs/initial-state-receipt.json
```
