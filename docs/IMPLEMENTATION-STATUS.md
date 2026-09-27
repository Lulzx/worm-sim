# Specification implementation and acceptance ledger

The active objective is the complete [specification](../SPEC.md), with Rust
replacing its proposed Python/JAX core at the user's request. Optional Taichi
acceleration remains separately audited. A software feature, a synthetic numerical
check, and a successful biological benchmark are distinct acceptance levels.
This ledger does not declare scientific targets achieved merely because code exists.

| Spec area | Current evidence | Work required for full acceptance |
| --- | --- | --- |
| Data (§3) | Canonical identities, aliases/provenance, content-hashed c302 anatomy, WSC1/WST1 codecs, pinned Creamer operators; native HDF5 import of 21 freely moving animals with behavior; source audit confirms retrospective whole-recording normalization; native Randi text import of 3,166 event trials; native CeNGEN receptor import and explicit molecular polarity evidence (MOLECULAR-PRIORS.md) | Direct Cook/Witvliet variation, broader CeNGEN channel/peptide annotations and gene-alias coverage, peptide network, per-event atlas observation eligibility/pulse calibration, further moving cohorts and prospectively processed signals; graph exports |
| Neurons (§4) | Differentiable Level 0 | Mixed Level 1 conductance channels and published Level 2 adapters with source-specific tests |
| Coupling (§5) | Anatomical chemistry, relaxed signs, symmetric gaps, shared equivalent gates | Dark edges/penalty, plasticity, rectification, peptide/monoamine compartments and receptor modulation |
| Numerics (§6) | Euler/RK4, exact events, Rust forward AD, Taichi reverse AD, segment checkpoints | Stiff/adaptive and multirate solvers, recursive checkpoints/continuous adjoint, truncated training, heterogeneous batches, multi-GPU orchestration |
| Parameters (§7) | Positive/bounded transforms, named parameter groups, explicit class maps and optional L/R suffix sharing, raw shrinkage and graph-sign priors | Source-backed molecular evidence and fitted-prior initialization/penalties implemented with explicit cell mapping (MOLECULAR-ATLAS-FIT.md); gene alias coverage and hierarchical animal offsets remain, annealed discrete choices |
| Training (§8) | Masked confidence-weighted MSE, small synthetic Adam fit; full-network history-only shooting and block-projected Kalman state inference; joint atlas pair BCE and trace MSE with full preparation adjoint (LEVEL0-ATLAS-CLASSIFICATION-FIT.md) | Unconditioned shooting/filtering fits selected epoch zero; behavior-driven fit selected epoch two but fails both long-horizon test hurdles (BEHAVIOR-INPUTS.md); optional global atlas observation gain implemented with frozen longer-fit comparison (ATLAS-OBSERVATION-GAIN.md; results pending); optional training pair-mean correlation loss and adjoint implemented (ATLAS-CORRELATION-LOSS.md; no real-data fit yet); per-neuron/animal observation calibration, behavior losses; atlas cosine schedule implemented (ATLAS-OPTIMIZATION.md), AdamW/group rates and random restarts remain, ensembles/Laplace/SBI, held-out evidence |
| Perturbations (§9) | Current pulses, silence, ablation; JSON protocols | Waveforms/conductance input, gene mappings, drug/modulator input, voltage clamps, YAML, source-backed phenotype registry |
| Body (§10) | Not implemented | Motor/muscle map, differentiable reduced body, sensor/environment interface, replay and Sibernetic adapter |
| Tasks 1–2 (§11) | Pinned pretrained Creamer inference reproduced; Rust group-disjoint splits and common Task 1/2 scorers implemented; upstream Creamer split differs from requested held-out neurons | Controls, AR(1), animal bootstrap, stable latent LDS and GRU are scored (LATENT-LDS.md, GRU.md); long-horizon intervals include zero for both fitted baselines; shared history-only behavior AR, GRU inputs, controlled LDS and tied Level 0 current inputs implemented (BEHAVIOR-INPUTS.md); three-model behavior comparison completed, with Level 0 long-horizon failure; Randi held-out-neuron split and separately audited published pair labels now fixed (RANDI-ATLAS.md, ATLAS-CLASSIFICATION.md); shared-kernel full-state constrained LDS E/M and exact covariance reuse implemented with independent numerical validation (CONNECTOME-LDS-FIT.md); population atlas fit runner with validation-only MSE selection and leakage test implemented; first population atlas LDS scored: test trace correlation 0.0453 and pair AUROC 0.686 with cluster intervals (CONNECTOME-LDS-FIT.md); relative Level 0 response gradients, exact training aggregation, shared-current population fitting and validation-only selection implemented (LEVEL0-ATLAS-FIT.md); first nonlinear atlas fit scored and failed LDS comparison (test AUROC 0.418 versus 0.686; paired intervals in atlas-first-comparison.json); differentiable preparation removes gross drift but 30 s prepared fit still fails LDS comparison and ranking needs at least 60 s preparation in validation diagnostics; independent NumPy dynamics replay passes; joint pair-BCE/trace fit completed with test AUROC 0.681 versus LDS 0.686 (paired difference interval includes zero), worse MSE and independently verified dynamics/calibration (LEVEL0-ATLAS-CLASSIFICATION-FIT.md); source-backed molecular signs imported and independently audited, with fitting integration and frozen threshold comparisons (MOLECULAR-ATLAS-FIT.md; both fits audited, neither establishes superiority over LDS; training diagnostics reveal substantial underfitting (ATLAS-TRAINING-DIAGNOSTIC.md)); useful biological forecasting and Task 1 baseline superiority remain |
| Tasks 3–5 (§11) | Not demonstrated | Behavior decoder comparison, mutant/ablation direction scoring, untrained closed-loop gait statistics |
| Performance (§12) | Native CPU/Metal and storage receipts; checkpoint tradeoff measured | Spec workload targets: GPU 100 s <1 s, batch 64 gradient <2 s, atlas fit <12 h, ensemble runs |
| Engineering (§12) | Public MIT repository, local/CI regression tests, manifests and receipts | Complete tutorial notebook, experiment tracking adapter, package/API documentation and reproducible full benchmark runs |

## Build sequence and completion gates

1. Prioritize Task 2 fitting: controls and animal-level intervals, stable LDS,
   302-neuron initial-state inference, tied parameters/sign priors, then a real
   Level 0 fit through the same scorer. Defer backend/performance expansion.
2. Add equal behavior inputs to the fitted models under the explicitly retrospective
   benchmark; retain prospective preprocessing as a separate acceptance requirement,
   then ingest the Task 1 atlas and publish
   its held-out stimulated-neuron split; complete staged comparisons.
3. Extend coupling with slow modulation; compare against the same held-out tasks.
4. Add mixed-fidelity neuron models and stiff/multirate numerical support.
5. Complete perturbation protocols and the measured-phenotype registry.
6. Add reduced-body/replay/high-fidelity interfaces and Tasks 3–5.
7. Optimize measured bottlenecks, execute uncertainty/performance batteries,
   document reproducibility, and close every specification requirement individually.

Performance claims must specify workload, hardware, precision, timing boundaries,
and excluded storage. Benchmark success requires actual held-out data and named
baselines. Unknown metadata, missing measurements, unavailable adapters, or failed
thresholds remain visible; none may be replaced by synthetic success claims.
