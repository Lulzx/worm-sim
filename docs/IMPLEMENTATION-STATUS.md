# Specification implementation and acceptance ledger

The active objective is the complete [specification](../SPEC.md), with a hybrid architecture following the user's revised direction: Rust owns
data, splits and common scorers; new fitting implementation uses JAX/Diffrax/
Equinox/Optax. The existing Rust core remains an independent numerical reference.
Optional Taichi acceleration remains separately audited. A software feature, a synthetic numerical
check, and a successful biological benchmark are distinct acceptance levels.
This ledger does not declare scientific targets achieved merely because code exists.

Current priority: **feature development is paused** while Level 0 training capacity
is tested with non-neutral initialization and learned per-neuron gains. See the
[training-only capacity protocol](LEVEL0-CAPACITY-PROTOCOL.md). Existing held-out
comparisons are exploratory; a fresh confirmatory cohort is not yet secured.

| Spec area | Current evidence | Work required for full acceptance |
| --- | --- | --- |
| Data (§3) | Canonical identities, aliases/provenance, content-hashed c302 anatomy, WSC1/WST1 codecs, pinned Creamer operators; native HDF5 import of 21 freely moving animals with behavior; source audit confirms retrospective whole-recording normalization; native Randi text import of 3,166 event trials; native CeNGEN receptor import and explicit molecular polarity evidence (MOLECULAR-PRIORS.md) | Direct Cook/Witvliet variation, broader CeNGEN channel/peptide annotations and gene-alias coverage, peptide network, per-event atlas observation eligibility/pulse calibration, further moving cohorts and prospectively processed signals; graph exports |
| Neurons (§4) | Differentiable Level 0 | Mixed Level 1 conductance channels and published Level 2 adapters with source-specific tests |
| Coupling (§5) | Anatomical chemistry, relaxed signs, symmetric gaps with optional JAX conservative rectification (GAP-RECTIFICATION.md), shared equivalent gates; optional JAX sparse species/compartment concentrations and tied receptor gain/leak/weight feedback, checked on synthetic dynamics and gradients (NEUROMODULATION.md); explicit budgeted JAX off-connectome chemical candidates with tied strengths/signs and physical-strength L1 penalty (DARK-EDGES.md); optional type-tied rate-adapted depression/facilitation with exact source/type state sharing (PLASTICITY.md); all four modules have JAX fitting/checkpoint reload and independent Rust prediction scoring, verified in a synthetic cross-language pipeline (EXTENDED-FITTING.md) | Source-backed modulation maps/priors, dark-edge candidate selection, plasticity biological assignments, rectification assignments, and biological validation of the integrated extensions |
| Numerics (§6) | Euler/RK4, exact events, Rust forward AD, Taichi reverse AD, segment checkpoints; optional JAX/Diffrax Tsit5 and Kvaerno5 with PID control and interval-local forcing, tested on stiff passive dynamics/gradients and one 302-neuron tolerance-convergence comparison (ADAPTIVE-SOLVERS.md) | Adaptive fitting/checkpoint settings and Rust prediction scoring are integrated (EXTENDED-FITTING.md); broader full-network adaptive validation and performance remain; full-network multirate error/cost validation (held-concentration splitting, synthetic convergence and gradients, and fitting/replay implemented in MULTIRATE.md), long-duration and full-network memory/accuracy validation of selectable Diffrax checkpoint/continuous adjoints (ADJOINTS.md; small-system gradients and fit/reload checked), truncated training, heterogeneous batches, multi-GPU orchestration |
| Parameters (§7) | Positive/bounded transforms, named parameter groups, explicit class maps and optional L/R suffix sharing, raw shrinkage and graph-sign priors | Source-backed molecular evidence and fitted-prior initialization/penalties implemented with explicit cell mapping (MOLECULAR-ATLAS-FIT.md); gene alias coverage and hierarchical animal offsets remain, annealed discrete choices |
| Training (§8) | Masked confidence-weighted MSE, small synthetic Adam fit; full-network history-only shooting and block-projected Kalman state inference; joint atlas pair BCE and trace MSE with full preparation adjoint (LEVEL0-ATLAS-CLASSIFICATION-FIT.md) | Unconditioned shooting/filtering fits selected epoch zero; behavior-driven fit selected epoch two but fails both long-horizon test hurdles (BEHAVIOR-INPUTS.md); optional global atlas observation gain implemented with frozen longer-fit comparison (ATLAS-OBSERVATION-GAIN.md; completed gain/control comparison improves over unit gain but does not establish superiority over LDS); optional training pair-mean correlation loss and adjoint implemented (ATLAS-CORRELATION-LOSS.md; no real-data fit yet); JAX per-neuron positive observation gains and a 1,000-update training-only capacity test completed (LEVEL0-LONG-RUN.md; final 74.8%, best observed 76.7% of available zero-start energy captured on two targets, 90% gate unmet; late instability and large gains remain); animal calibration, behavior losses; atlas cosine schedule and optional AdamW implemented (ATLAS-OPTIMIZATION.md), seeded tied-sign restart initialization and frozen three-seed pilot implemented (ATLAS-SIGN-RESTARTS.md; 25-update runs complete, training losses audited, final comparisons unaudited); group-rate multipliers with zero-rate freezing are implemented and checked against Optax (GROUP-LEARNING-RATES.md); broader restart/ensemble validation remains, ensembles/Laplace/SBI, held-out evidence |
| Perturbations (§9) | Current pulses and stage-correct piecewise-linear current/conductance waveforms, voltage-trace clamps, silence, ablation; JSON/YAML protocols (PERTURBATIONS.md) | Gene mappings, drug/modulator input, source-backed phenotype registry |
| Body (§10) | Not implemented | Motor/muscle map, differentiable reduced body, sensor/environment interface, replay and Sibernetic adapter |
| Tasks 1–2 (§11) | Pinned pretrained Creamer inference reproduced; Rust group-disjoint splits and common Task 1/2 scorers implemented; upstream Creamer split differs from requested held-out neurons | Controls, AR(1), animal bootstrap, stable latent LDS and GRU are scored (LATENT-LDS.md, GRU.md); long-horizon intervals include zero for both fitted baselines; shared history-only behavior AR, GRU inputs, controlled LDS and tied Level 0 current inputs implemented (BEHAVIOR-INPUTS.md); three-model behavior comparison completed, with Level 0 long-horizon failure; Randi held-out-neuron split and separately audited published pair labels now fixed (RANDI-ATLAS.md, ATLAS-CLASSIFICATION.md); shared-kernel full-state constrained LDS E/M and exact covariance reuse implemented with independent numerical validation (CONNECTOME-LDS-FIT.md); population atlas fit runner with validation-only MSE selection and leakage test implemented; first population atlas LDS scored: test trace correlation 0.0453 and pair AUROC 0.686 with cluster intervals (CONNECTOME-LDS-FIT.md); relative Level 0 response gradients, exact training aggregation, shared-current population fitting and validation-only selection implemented (LEVEL0-ATLAS-FIT.md); first nonlinear atlas fit scored and failed LDS comparison (test AUROC 0.418 versus 0.686; paired intervals in atlas-first-comparison.json); differentiable preparation removes gross drift but 30 s prepared fit still fails LDS comparison and ranking needs at least 60 s preparation in validation diagnostics; independent NumPy dynamics replay passes; joint pair-BCE/trace fit completed with test AUROC 0.681 versus LDS 0.686 (paired difference interval includes zero), worse MSE and independently verified dynamics/calibration (LEVEL0-ATLAS-CLASSIFICATION-FIT.md); source-backed molecular signs imported and independently audited, with fitting integration and frozen threshold comparisons (MOLECULAR-ATLAS-FIT.md; both fits audited, neither establishes superiority over LDS; training diagnostics reveal substantial underfitting (ATLAS-TRAINING-DIAGNOSTIC.md)); useful biological forecasting and Task 1 baseline superiority remain |
| Tasks 3–5 (§11) | Not demonstrated | Behavior decoder comparison, mutant/ablation direction scoring, untrained closed-loop gait statistics |
| Performance (§12) | Native CPU/Metal and storage receipts; checkpoint tradeoff measured | Spec workload targets: GPU 100 s <1 s, batch 64 gradient <2 s, atlas fit <12 h, ensemble runs |
| Engineering (§12) | Public MIT repository, local/CI regression tests, manifests and receipts | Complete tutorial notebook, experiment tracking adapter, package/API documentation and reproducible full benchmark runs |

## Build sequence and completion gates

The latest priority supersedes the earlier implementation-first sequence:
**train and diagnose Level 0 before adding dynamical features**. The hybrid
migration gate is complete: the frozen forward result and full five-update
trajectory reproduce through the Rust scorer ([migration receipt](JAX-MIGRATION.md)).
This verifies the fitting implementation, not biological adequacy.

1. **Training capacity, in progress.** Non-neutral signs/rest and positive
   per-neuron gains are implemented. The two-target 1,000-update fit missed the
   unchanged 90% capacity gate. Both subsequent 200-update warm starts failed
   before completion. Their complete finite histories and terminal failures are
   retained in [the warm-start report](CAPACITY-WARM-START.md).
2. **Numerical robustness, in progress.** The higher-rate failure is reproduced
   at its failing parameters. Independent replay localizes it to coarse Euler
   preparation leaving the continuous model's voltage bounds. Refined steps
   give finite trajectories and gradients, with directional finite-difference
   checks. The separately declared refined-step fit completed 200 updates with
   independent replay and numerical controls; final capture is 76.69%, still
   below the 90% capacity gate. The lower-rate failure's parameters remain unaudited. New runs
   retain finite failing parameters automatically on nonfinite evaluations.
3. **Broader fitting, pending.** The refined fit is audited and remains underfit. An equal-evaluation
   [L-BFGS comparison](CAPACITY-LBFGS.md) raises capture to 77.13% but exhausts
   its budget without convergence; a longer fixed-model diagnostic is declared. Continue
   optimization/capacity diagnosis until the small-target gate passes; then
   replicate across unrelated training targets and fit the training population.
   A failed small-target gate alone does not identify model capacity as the cause.
4. **Scientific comparison, pending.** Preserve the LDS comparison and report
   uncertainty at the independent target/animal level. Existing validation/test
   observations have been inspected; treat subsequent comparisons as exploratory.
   The [fresh-cohort audit](FRESH-HOLDOUT-AUDIT.md) has not secured an independent
   confirmatory cohort. Do not relabel old recordings as a fresh test set.
5. **Deferred specification work.** After the fitting priority is resolved,
   finish the missing data/biological assignments, mixed-fidelity neurons,
   gene/drug protocols, body/sensory interfaces, Tasks 3–5, inference/uncertainty,
   API/tutorial and performance requirements listed above. Neuromodulation is
   the first scientific extension to evaluate; existing synthetic extension
   checks are not biological acceptance.
6. **Full acceptance.** Close each original specification requirement with
   source-backed implementation and scope-matched validation, including the
   complete uncertainty/performance batteries. Neither the capacity gate nor a
   successful Level 0 benchmark would complete the whole specification.

Performance claims must specify workload, hardware, precision, timing boundaries,
and excluded storage. Benchmark success requires actual held-out data and named
baselines. Unknown metadata, missing measurements, unavailable adapters, or failed
thresholds remain visible; none may be replaced by synthetic success claims.
