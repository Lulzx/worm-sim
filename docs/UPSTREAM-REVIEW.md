# Upstream review and reuse plan

Checked 2026-09-27. Scope: the named projects' papers, repository documentation,
selected numerical/data-loading source files, licenses, and current default-branch
commits. This is a source audit, not an independent reproduction of their results
or an exhaustive review of every repository file. Pinned commits and dates are in
[upstream-lock.json](upstream-lock.json).

## Corrections to the starting assumptions

- OpenWorm is active: the inspected c302 and Sibernetic heads are dated August 3
  and May 28, 2026. Their published stack connects neuronal output to a moving
  body. A blanket statement that it cannot move is too strong; movement demos
  also do not establish autonomous, quantitatively validated behavior or feeding.
  [OpenWorm stack](https://github.com/openworm/OpenWorm),
  [Sibernetic coupling](https://github.com/openworm/sibernetic/blob/development/sibernetic_c302.py).
- BAAIWorm's paper explicitly limits its neural model to 136 neurons and zigzag
  locomotion. Its 80 motor-neuron outputs map to 96 muscle forces through a
  trained readout, fitted using a 10-second simulated movement. This matters:
  those results do not meet our spec's criterion of locomotion emerging without
  behavior training. “Most advanced” is not a ranking established by this audit.
  [Paper, methods and discussion](https://www.nature.com/articles/s43588-024-00738-w).
- The Creamer, Leifer, and Pillow preprint was updated May 18, 2026.
  [Version 4](https://www.biorxiv.org/content/10.1101/2024.09.22.614271v4).
- The September whole-body title corresponds to **Lee T., worm-whisperer**,
  rather than a Leifer/Pillow-authored paper. The repository's license names
  Taehee Lee. The primary repository links DOI 10.64898/2026.09.06.749731;
  bioRxiv's page was inaccessible during this audit, so the exact September 9
  posting date was not independently confirmed from that primary page.
  [Author repository](https://github.com/kairess/worm-whisperer),
  [license](https://github.com/kairess/worm-whisperer/blob/main/LICENSE).

## What to reuse, and what each result establishes

| Project | Inspected implementation/assets | Use in WormSim |
| --- | --- | --- |
| OpenWorm c302 | Canonical identity list, connectome CSV, multiscale NeuroML generation | Canonical identities and pinned anatomy imported now; later use generated cell models as cross-simulator references |
| Sibernetic | C++/OpenCL PCISPH physics and Python/NEURON coupling | External high-fidelity body validation adapter after neural validation; not the differentiable training backend |
| BAAIWorm | NMODL channel mechanisms, NEURON/CuPy fitting, brain/body interface | Candidate Level 1 reference equations and transfer-impedance/block-memory ideas; compare fitting memory versus exact AD before porting |
| Creamer/Leifer/Pillow | Trained models, STAM evaluation example, anatomy loader | Highest-priority baseline reproduction, data preprocessing and masks |
| worm-whisperer | Model description, neural solver, controls, manuscript, license | Experimental designs and failure modes; no source copied into this MIT core |
| Worm Neuro Atlas | Dataset aggregation and exact/merged identity APIs | Offline source-data export candidate, with explicit identity mapping and dataset-specific provenance |

Sources: [c302](https://github.com/openworm/c302),
[Sibernetic](https://github.com/openworm/sibernetic),
[BAAIWorm README](https://github.com/Jessie940611/BAAIWorm/blob/main/README.md),
[BAAIWorm fitting source](https://github.com/Jessie940611/BAAIWorm/blob/main/eworm_learn/run_eworm_v4.py),
[Creamer code](https://github.com/Nondairy-Creamer/Creamer_LDS_2026),
[worm-whisperer](https://github.com/kairess/worm-whisperer),
[Worm Neuro Atlas](https://github.com/francescorandi/wormneuroatlas).

### Causal fitting comes first

The latest Creamer preprint reports that multi-hop anatomical paths explain
interactions between neurons lacking direct anatomical connections, shuffled
connectomes perform worse, and additional connections do not improve the fitted
model. **Our inference:** test indirect propagation and the observation model
before interpreting residuals as evidence for dark edges or peptide signaling.
[Preprint](https://www.biorxiv.org/content/10.1101/2024.09.22.614271v4).

The repository supplies connectome-constrained, fully connected, and shuffled
models. Its quick-start evaluates held-out recordings, excludes diagonal pairs,
and compares stimulus-triggered averages and correlations. Its anatomy loader
uses White JSH/N2U and Witvliet animals 7/8, with an OR/AND combination option;
that differs from simply loading Cook anatomy. Preserve this distinction in
benchmark manifests. The published CSV of weights alone is not a full dynamical
model: retain input filters, noise, observation mapping, timebase, and neuron
ordering when making an interchange export.
[STAM example](https://github.com/Nondairy-Creamer/Creamer_LDS_2026/blob/main/quick_start_examples/predict_stams.py),
[anatomy loader](https://github.com/Nondairy-Creamer/Creamer_LDS_2026/blob/main/analysis_utilities.py),
[model documentation](https://github.com/Nondairy-Creamer/Creamer_LDS_2026/blob/main/README.md).

### Distinguish neural behavior from controller behavior

worm-whisperer combines a learned external stimulation policy, fixed fitted
network, and an assumed motor/body layer. It reports failure with sensory-only
stimulation and a large dissociation between chemical and gap-junction wiring
controls. Its README also describes timestep sensitivity in transit times.
**Our inference:** carry over these controls, not the headline reach rate as a
whole-brain validation result. Require sensory-only trials, separate pathway
ablations, multiple seeds, body/readout controls, and timestep convergence.
[README and limitations](https://github.com/kairess/worm-whisperer/blob/main/README.md).

For Apple Silicon, BAAIWorm's documented Ubuntu/CUDA/OptiX stack is not a drop-in
runtime. Reuse scientific reference mechanisms and data contracts first; avoid
making the Rust core depend on its renderer.
[System requirements](https://github.com/Jessie940611/BAAIWorm/blob/main/README.md).

Worm Neuro Atlas already integrates expression, peptide/receptor, anatomical,
monoamine and propagation data. Its bilateral/dorsoventral/numbered merging
options must be recorded, never silently treated as 302 individually identified
neurons. Preserve exact identities in the storage format and attach class
aggregation only as a separate mapping.
[Identity and data APIs](https://github.com/francescorandi/wormneuroatlas/blob/main/README.md).

## License and actual reuse record

These are repository declarations, not a blanket statement about every embedded
third-party dataset. Creamer inference equations are reimplemented with its MIT notice retained; no upstream fitting source is executed.

| Repository | Observed declaration | Current handling |
| --- | --- | --- |
| c302 | MIT | Canonical IDs and anatomy imported; notice retained in `licenses/c302-MIT.txt` |
| Sibernetic | MIT in LICENSE (API reported NOASSERTION) | Reference only |
| BAAIWorm | Apache-2.0 | Reference only; preserve notices for any future port |
| Creamer_LDS_2026 | MIT | Pinned numeric export and Rust inference reproduction; MIT notice retained |
| worm-whisperer | PolyForm Noncommercial 1.0.0; documentation CC BY-NC 4.0 | Reference only; do not silently include in MIT distribution |
| wormneuroatlas | GPL-3.0 | Reference only; evaluate separate offline exporter and individual dataset terms |

[Sibernetic license](https://github.com/openworm/sibernetic/blob/development/LICENSE),
[BAAIWorm license](https://github.com/Jessie940611/BAAIWorm/blob/main/LICENSE),
[Creamer license](https://github.com/Nondairy-Creamer/Creamer_LDS_2026/blob/main/LICENSE),
[worm-whisperer license](https://github.com/kairess/worm-whisperer/blob/main/LICENSE),
[Worm Neuro Atlas license](https://github.com/francescorandi/wormneuroatlas/blob/main/LICENSE).

## Concrete consequences for implementation

1. **Done:** 302 c302 neuron IDs; Rust CSV importer; packed anatomy; source hashes;
   log zero-padded motor-neuron aliases; report unmapped rows and zero-current self gaps; explicitly
   average 12 unequal mirrored gap pairs. No invented neuron-type annotations.
2. **Done:** exact shared-presynaptic gates, incoming sparse rows, f64 reference
   solver, finite-difference gradient tests, conservation tests, two lossless
   storage formats. These are original implementations of our specified equations.
3. **Done:** pinned Creamer inference export and Rust/NumPy parity; see
   [baseline results](BASELINE.md). **Next:** add our neuron-held-out split as a
   separately labeled task. Never substitute one split for the other.
4. **Next:** atlas preprocessing and calcium calibration, immutable split
   manifests, linear baseline; require held-out benefit before adding parameters.
5. **Next:** checkpointed reverse-mode gradients and batched Metal kernels,
   measured against the Rust CPU reference. Then stiff integration, peptide
   ablations, and body coupling, in that order.

The pretrained Creamer models have now been exported and evaluated; upstream
training has not been rerun. A small [Taichi Metal gradient audit](TAICHI.md)
also passes; full-network training and checkpointing remain open.
