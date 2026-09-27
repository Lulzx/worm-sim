# WormSim: Differentiable C. elegans Nervous System Simulator — Technical Specification

Sep 27, 2026 · @Lulzx

## 1. Purpose and scope

WormSim is a GPU-accelerated, end-to-end differentiable simulator of the 302-neuron *C. elegans* nervous system, built so that every biological parameter can be fit to activity data by gradient descent and every prediction can be scored against held-out experiments.

**Goals**

- Simulate all 302 neurons with chemical synapses, gap junctions, and slow neuropeptide/monoamine signaling in one model.
- Support multiple fidelity levels per neuron (rate unit to conductance-based) inside one simulation.
- Fit parameters jointly to stimulation atlases and freely moving whole-brain recordings, with uncertainty estimates.
- Expose a perturbation API (stimulate, silence, ablate, mutate) that mirrors real experiments.
- Couple to a body/environment model through a clean sensor and motor interface.

**Non-goals (v1)**

- Spatially detailed multi-compartment morphology (point or few-compartment neurons only).
- Muscle, pharynx, and gut physiology beyond a motor-output interface.
- Development, learning over hours, or gene regulatory dynamics.

**Primary users:** computational neuroscientists fitting models, and experimental labs testing model predictions.

## 2. Architecture

WormSim is a pipeline of pure functions, so gradients flow from the loss back to every parameter in one pass.

&#91;embedded content: WormSim architecture · data to loss, with gradient loop\]

The perturbation API and body model plug into the solver without changing the model core.

**Package layout (Python, JAX)**

- `wormsim.data` — loaders, neuron identity mapping, unified graph.
- `wormsim.params` — parameter pytrees, priors, transforms, constraints.
- `wormsim.neurons` — neuron model library.
- `wormsim.coupling` — chemical synapses, gap junctions, neuromodulation.
- `wormsim.solve` — integration, checkpointing, adjoints.
- `wormsim.observe` — calcium and behavior observation models.
- `wormsim.fit` — losses, optimizers, uncertainty.
- `wormsim.experiments` — perturbation protocols, virtual experiments.
- `wormsim.body` — body/environment interface and adapters.
- `wormsim.bench` — evaluation tasks and metrics.

## 3. Data layer

All datasets are merged into one versioned graph keyed by canonical neuron names (e.g. AVAL, RIMR), so every downstream module reads a single source of truth.

**Inputs**

| Source | Provides | Used for |
| --- | --- | --- |
| Cook 2019, Witvliet 2021 connectomes | Chemical synapse counts, gap junction sizes, per-animal variation | Graph edges, weight priors |
| CeNGEN expression atlas | Neurotransmitter, receptor, ion channel genes per neuron class | Synapse sign priors, channel priors |
| Ripoll-Sánchez 2023 neuropeptide network | Peptide–receptor pairs between neurons | Neuromodulation edges |
| Randi 2023 signal-propagation atlas | Responses to single-neuron optogenetic stimulation | Training and held-out tests |
| Freely moving whole-brain datasets (e.g. Atanas 2023, via WormWideWeb) | Calcium traces with neuron IDs and behavior | Training and held-out tests |

**Unified graph schema**

- `Neuron`: `id`, `class`, `side`, `type` (sensory/inter/motor), `neurotransmitters[]`, `receptors[]`, `channels[]`, `peptides_released[]`, `peptide_receptors[]`.
- `ChemicalEdge`: `pre`, `post`, `synapse_count`, `datasets[]`, `sign_prior` (P(excitatory)), `receptor_candidates[]`.
- `GapEdge`: `a`, `b`, `size`, `datasets[]` (undirected).
- `PeptideEdge`: `sender`, `receiver`, `peptide`, `receptor`, `evidence_score`.
- `Recording`: `dataset`, `animal_id`, `condition`, `t[]`, `traces{neuron: array}`, `id_confidence{neuron: float}`, `behavior{}`.

**Requirements**

- Neuron name reconciliation across datasets, with a logged mapping table and unit tests.
- Every edge and trace carries provenance and an ID-confidence score, so low-confidence data can be down-weighted.
- Export to dense JAX arrays (adjacency, masks, prior tensors) with a fixed neuron ordering.
- Datasets versioned by content hash; a model run records the exact graph version.

## 4. Neuron models

Each neuron picks a model from a shared library, so one simulation can mix cheap rate units with detailed conductance models where the data justifies them.

**Level 0 — graded rate unit (default)**

```latex
\tau_i \frac{dV_i}{dt} = -(V_i - E_{L,i}) + I^{\text{chem}}_i + I^{\text{gap}}_i + I^{\text{ext}}_i
```

Output is a smooth sigmoid of voltage, matching graded (non-spiking) release. Parameters per neuron: τ, resting potential, sigmoid threshold and slope.

**Level 1 — reduced conductance model**

```latex
C_i \frac{dV_i}{dt} = -\sum_k g_{k,i}\, m_k^{p}\, h_k^{q}\, (V_i - E_k) + I^{\text{syn}}_i + I^{\text{ext}}_i
```

A small set of channel types (leak, K, Ca, optional Na-like plateau currents) whose presence is gated by CeNGEN expression. This captures bistability and plateau potentials seen in neurons like RIM and AVA.

**Level 2 — published detailed models**

Adapters for existing single-neuron models (e.g. RMD, AWA, AIY) so validated biophysics can be dropped in.

**Interface contract**

- `init_state(params) -> state` and `dstate_dt(t, state, inputs, params) -> dstate`.
- `release(state, params) -> r` in \[0, 1\], the neuron's transmitter output.
- Pure functions, no side effects, vectorized over neurons of the same model type with `vmap`.
- A per-neuron `fidelity` config chooses the level; the default is Level 0 everywhere.

## 5. Coupling

Three coupling channels run in parallel: fast chemical synapses, instantaneous gap junctions, and slow neuromodulation that changes how the fast network behaves.

**Chemical synapses**

```latex
I^{\text{chem}}_i = \sum_j w_{ij}\, s_{ij}\, (E_{ij} - V_i), \qquad \tau_s \frac{ds_{ij}}{dt} = r_j (1 - s_{ij}) - s_{ij}
```

- w\_ij = softplus(θ\_ij) × synapse\_count\_ij, so strength scales with anatomy but is learnable.
- Reversal E\_ij is excitatory or inhibitory; sign is learned as a relaxed binary variable initialized from the CeNGEN sign prior.
- Only connectome edges exist (sparse mask); an optional "dark edge" mode allows a small L1-penalized set of extra edges to test off-connectome signaling.
- Optional short-term plasticity (depression/facilitation) per edge type.

**Gap junctions**

```latex
I^{\text{gap}}_i = \sum_j g_{ij} (V_j - V_i), \qquad g_{ij} = g_{ji} \ge 0
```

Symmetric by construction, conductance scaled from junction size, optional rectification flag.

**Neuromodulation (slow layer)**

```latex
\tau_p \frac{dc_p}{dt} = \sum_{j \in \text{senders}(p)} \alpha_{jp}\, r_j - c_p
```

- c\_p is the concentration of peptide or monoamine p, in a well-mixed or coarse spatial compartment model (head, nerve ring, body).
- Receptor activation on neuron i modulates its parameters: gain, leak, or synaptic weights, via learned sensitivities β\_ip.
- Time constants of seconds to minutes, separated from the millisecond fast layer.
- Monoamines (serotonin, dopamine, tyramine, octopamine) use the same mechanism with their own release maps.

**Sensory and external inputs**

I\_ext carries sensory drive (from the body model or recorded stimuli) and optogenetic stimulation from the perturbation API.

## 6. Numerical integration and differentiability

The system is stiff (millisecond membranes, minute-scale modulation), so the solver must handle multiple timescales while keeping gradients stable over long recordings.

**Solvers (via Diffrax)**

- Default: adaptive implicit-explicit or Kværnø 5(4) for Level 0; implicit (Kværnø/ESDIRK) when Level 1 channels make the system stiff.
- Fixed-step Euler/RK4 mode for fast training sweeps and debugging.
- Multi-rate option: integrate the slow neuromodulation layer on a coarser step, held constant within each fast window.

**Gradients**

- Backprop through the solver with recursive checkpointing for runs up to \~1,000 s.
- Continuous adjoint as an alternative when memory is the bottleneck.
- Truncated backprop through time for long freely moving recordings, with state carried between windows.

**Stability measures**

- All positive parameters passed through softplus or exp transforms; bounded ones through scaled sigmoids.
- Gradient clipping and per-parameter-group learning rates.
- Relaxed discrete choices (synapse sign, channel presence) via Gumbel-softmax or straight-through estimators, annealed to hard values.
- Numerical checks: gradient-vs-finite-difference tests on small subnetworks in CI.

**Batching**

`vmap` over animals, stimulation trials, and parameter samples; `pmap` or sharding across GPUs for large ensembles.

## 7. Parameters, priors, constraints

Biology enters as priors, not hard-coded values, so data can overrule anatomy where they disagree and the disagreement itself becomes a finding.

| Parameter group | Approx. count (Level 0) | Prior source | Sharing |
| --- | --- | --- | --- |
| Chemical synapse weights | \~7,000 edges | Synapse counts (log-normal around scaled count) | Optionally tied by neuron class pair |
| Synapse signs | \~7,000 edges | CeNGEN transmitter + receptor expression | Tied by transmitter–receptor type |
| Gap junction conductances | \~900 pairs | Junction size | Symmetric |
| Neuron time constants, thresholds, gains | 302 × 3–4 | Literature ranges | Left/right pairs tied by default |
| Neuromodulation release and sensitivity | Sparse, per peptide edge | Peptide network evidence score | Tied by peptide–receptor pair |
| Observation model (calcium kernel, scale) | Per neuron × per animal | GCaMP kinetics | Hierarchical across animals |

**Mechanisms**

- Parameters stored as a JAX pytree with named groups; each group declares its transform, prior, and sharing rule.
- Class tying (e.g. AVAL = AVAR) on by default, removable per group to test asymmetry.
- Hierarchical model: shared population parameters plus small per-animal offsets with a shrinkage prior.
- Prior strength is a config knob, so the same code runs "anatomy-trusting" and "data-driven" fits for comparison.

## 8. Training and inference

Fitting runs in stages of increasing difficulty, and every stage reports held-out performance, never just training fit.

**Observation model**

Simulated voltage is mapped to fluorescence through a per-neuron nonlinearity and a learned calcium kernel; losses compare in fluorescence space (ΔF/F), with missing or low-confidence neurons masked.

**Losses**

- Stimulation atlas: MSE or correlation loss on response traces per stimulated–responding pair, plus a classification loss on whether a pair responds at all.
- Freely moving: multi-step prediction loss (teacher-forced for short windows, free-running for longer), plus behavior-state prediction when a body or readout model is attached.
- Regularizers: prior log-densities, L1 on dark edges, smoothness on per-animal offsets.

**Fitting procedure**

1. Fit to the stimulation atlas only (clean causal data).
2. Add freely moving data, jointly with the atlas.
3. Enable neuromodulation layer and compare against the model without it.
4. Raise fidelity for neurons where Level 0 residuals remain large.

Optimizer: Adam/AdamW via Optax with cosine schedule; multiple random restarts to detect degenerate solutions.

**Uncertainty**

- Deep ensembles (10–50 fits from different seeds and data splits) as the default.
- Laplace approximation around the fitted solution for cheap local uncertainty.
- Simulation-based inference (neural posterior estimation) for small parameter subsets where full posteriors matter.
- Output: which parameters are well constrained, and which predictions disagree across the ensemble (inputs to experiment design).

## 9. Perturbation and virtual experiment API

Every real experiment type has a matching simulated operation, so any published result can be rerun on the model and scored automatically.

```python
exp = Experiment(model, params)
exp.stimulate("AVAL", t=(10.0, 12.0), amplitude=1.0)   # optogenetic pulse
exp.silence(["AIB"], t=(0, None))                        # chemogenetic block
exp.ablate("RIM")                                         # remove cell + edges
exp.mutate("unc-7")                                       # remove gene-dependent gap junctions
exp.mutate("flp-18", mode="knockout")                    # remove peptide release
exp.drug("serotonin", concentration=1.0, t=(0, None))
result = exp.run(duration=120.0, dt_save=0.1)
```

**Operations**

- `stimulate` / `inhibit`: current injection or conductance change with configurable waveform.
- `silence` / `ablate`: clamp output or remove the neuron and its edges.
- `mutate`: gene-to-model mapping table (e.g. innexin → gap junction subsets, receptor gene → synapse subsets, peptide gene → release terms).
- `drug`: bath application of a modulator into the slow layer.
- `clamp`: fix voltage traces from recordings (for partial-observation fitting).

**Requirements**

- All operations differentiable where meaningful, so perturbation outcomes can be used as training targets.
- Declarative protocol files (YAML) so labs can share experiment definitions.
- A registry of published experiments with their measured outcomes, feeding the benchmark suite.

## 10. Body and environment interface

The nervous system talks to any body model through two fixed arrays, so a fast differentiable body can be used for training and a high-fidelity one for validation.

**Interface**

- `motor_out`: activations of the \~75 ventral cord and head motor neurons, mapped to 95 body-wall muscle activations via a fixed or learned neuromuscular map.
- `sensory_in`: per-sensory-neuron drive computed from body state and environment (touch, chemical gradients, temperature, proprioception from body curvature).
- `BodyModel` protocol: `step(body_state, muscle_act, dt) -> body_state`, `sense(body_state, env) -> sensory_in`.

**Adapters**

1. Differentiable reduced body: a 2D chain of \~50 segments with viscoelastic mechanics and resistive-force fluid drag, in JAX, fast enough for gradient training.
2. OpenWorm Sibernetic: high-fidelity 3D fluid/body simulation, non-differentiable, used for validation only.
3. Replay mode: feed recorded body posture and stimuli in open loop when fitting neural data alone.

**Coupling**

Lockstep co-simulation at a shared step (default 1 ms) or at the slower body step with interpolated neural outputs. Proprioceptive feedback is required for undulation, so closed-loop mode is the target for behavior benchmarks.

## 11. Evaluation and success criteria

No public benchmark yet scores worm *models* on held-out perturbations and behavior, so WormSim ships its own evaluation suite, built on existing data and baselines.

**What exists today**

- [WormID-Bench](https://www.biorxiv.org/content/10.1101/2025.01.06.631621v4) (2025) benchmarks neuron detection, identification, and tracking — extracting activity from images, not modeling it.
- [ZAPBench](https://arxiv.org/pdf/2503.02618) (2025) benchmarks whole-brain activity forecasting, but in larval zebrafish.
- [Creamer et al.](https://pubmed.ncbi.nlm.nih.gov/41040343/) fit a connectome-constrained linear model to the [signal-propagation atlas](https://pubmed.ncbi.nlm.nih.gov/37914938/); it is the baseline to beat on Task 1.

**WormSim benchmark tasks**

| Task | Data | Metric | Must beat |
| --- | --- | --- | --- |
| 1. Held-out stimulations | Signal-propagation atlas, held-out stimulated neurons | Trace correlation; response yes/no AUROC | Connectome-constrained linear model |
| 2. Held-out animals, freely moving | Whole-brain recordings, held-out worms | Multi-step forecast R² at 1, 10, 30 s | Linear dynamical system, GRU with no connectome |
| 3. Behavior states | Recordings with behavior labels | Forward/reverse/turn prediction accuracy | Decoder trained on recorded activity |
| 4. Virtual ablations and mutants | Published ablation/mutant phenotypes | Fraction of directional effects matched | Random and anatomy-only predictions |
| 5. Closed-loop locomotion | Body model + tracking data | Undulation frequency, wavelength, reversal rate | Recorded wild-type statistics |

**Success criteria**

- v1: beats both baselines on Tasks 1–2 with the same number of free parameters or fewer.
- v2: neuromodulation layer measurably improves off-connectome responses in Task 1.
- v3: correct direction on at least 70% of Task 4 perturbations (target, to revise once the registry is built).
- "Solved" milestone: Task 5 behavior emerges from the fitted network without being trained on it.

Splits are fixed by neuron and by animal, published with the code, and never used for tuning.

## 12. Performance, engineering, and risks

A full Level 0 fit should run overnight on one consumer GPU, so a solo developer can iterate daily.

**Performance targets**

| Workload | Target |
| --- | --- |
| Level 0, 302 neurons, 100 s simulated, forward pass | < 1 s on one GPU |
| One gradient step, batch of 64 atlas trials | < 2 s |
| Full atlas fit, Level 0 | < 12 h on one 24 GB GPU |
| 20-member ensemble | Parallel on a small multi-GPU node, or sequential over a few days |

**Engineering**

- Stack: Python 3.11+, JAX, Diffrax, Equinox (modules), Optax (optimizers), NumPyro (priors, SBI helpers), Hydra (configs).
- Tests: unit tests per neuron/coupling model, gradient checks, conservation checks (gap junction symmetry), regression tests on fixed seeds.
- Reproducibility: every run logs config, data hash, code commit, and seed; results tracked with Weights & Biases or MLflow.
- Open source (MIT or Apache 2.0), with docs and a tutorial notebook that fits a small circuit end to end.

**Build order**

Data layer and Level 0 model first, then benchmark Tasks 1–2; neuromodulation next; Level 1 neurons, perturbation registry, and body coupling after v1 results are in.

**Risks and mitigations**

| Risk | Mitigation |
| --- | --- |
| Too many free parameters; many models fit equally well | Strong priors, class tying, ensembles, and held-out perturbation tests |
| Stiff dynamics break gradients | Implicit solvers, multi-rate integration, truncated BPTT |
| Neuron ID errors in recordings corrupt fits | Use ID confidence as loss weights; exclude low-confidence neurons |
| Neuromodulation data too sparse to constrain | Start with coarse, tied parameters; test against mutant and drug data |
| Model fits data but explains nothing | Interpretability pass on every release: ablate pathways in-silico, report mechanisms |

**Open questions**

- Which datasets share compatible preprocessing, and how much re-extraction is needed?
- How much does an animal's history (feeding state, age) need to be a model input?

**Sources**

- [Neural signal propagation atlas of C. elegans (Randi et al., Nature 2023)](https://pubmed.ncbi.nlm.nih.gov/37914938/)
- [Bridging the gap between the connectome and whole-brain activity in C. elegans (preprint, 2025)](https://pubmed.ncbi.nlm.nih.gov/41040343/)
- [WormID-Bench (bioRxiv, 2025)](https://www.biorxiv.org/content/10.1101/2025.01.06.631621v4)
- [ZAPBench (arXiv, 2025)](https://arxiv.org/pdf/2503.02618)
- [WormID unified datasets (Sprague et al.)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11092512/)
