# Source audit for biological sign priors

The graph used by the current fits (`data/c302-herm.wsc`, graph hash
`0db7c2fd8b83bc15cc17dd4b952b1bac4d1fed52ec524ba5cab403ecf3ea720e`)
contains 3,638 chemical edges representing 20,589 synapses. Every chemical sign
prior is 0.5. All 302 neuron classes are `unannotated`, with no transmitter
annotations. These counts were checked against the canonical decoded graph;
current L/R parameter sharing is a suffix assumption rather than CeNGEN evidence.
The joint-classification experiment retains this graph unchanged.

## Sources inspected

[Fenyves et al. (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007974)
combine presynaptic transmitter and postsynaptic ionotropic receptor evidence to
predict chemical synapse polarity. Their network has the same reported edge and
synapse counts as our current graph; equality of counts alone does not prove
identity of every edge. Their supplementary tables are linked from the paper,
which declares a Creative Commons Attribution licence. A transmitter alone is
insufficient to assign a sign: receptor identity matters. Predictions from
expression are not direct physiological measurements or calibrated probabilities.

The already pinned Worm Neuro Atlas revision
`b2e13d88b670efcb3438aeacba2ad4bd6c383933` exposes the following source interfaces:

- [SynapseSign.py](https://github.com/francescorandi/wormneuroatlas/blob/b2e13d88b670efcb3438aeacba2ad4bd6c383933/wormneuroatlas/SynapseSign.py)
  reads `journal.pcbi.1007974.s003.xlsx`, sheets `1. NT expr` and
  `2. Receptor gene table`. It retains dominant/alternative transmitter labels and
  excitatory/inhibitory receptor lists for Glu, ACh and GABA. File SHA-256:
  `bef2502ab1ac787a7897c78b29be2195bd85b727cdb0fb3ed8fd55cfa38850e4`.
- [Cengen.py](https://github.com/francescorandi/wormneuroatlas/blob/b2e13d88b670efcb3438aeacba2ad4bd6c383933/wormneuroatlas/Cengen.py)
  describes Taylor et al. 2021 expression data, intended version `021821`, and
  four threshold-specific gene-name, WormBase-ID and TPM arrays in `cengen.h5`.
  Default expression queries use threshold 4. File SHA-256:
  `1ca9d965ee3cfca63d2b893fce9f5ed3ce4d5b9e66b61fe56b0db2a45e89f73d`.
- [NeuroAtlas.py](https://github.com/francescorandi/wormneuroatlas/blob/b2e13d88b670efcb3438aeacba2ad4bd6c383933/wormneuroatlas/NeuroAtlas.py)
  combines transmitter and receptor evidence into potential directed signs,
  independently of anatomical connectivity. Its detailed result distinguishes
  unknown from conflicting evidence; its convenience signed-connectome function
  leaves many such edges positive. That convenience representation is unsuitable
  for preserving our prior uncertainty.

These source files were read, not imported or executed. The repository
[licence](https://github.com/francescorandi/wormneuroatlas/blob/b2e13d88b670efcb3438aeacba2ad4bd6c383933/LICENSE)
is GPL-3.0 (file SHA-256
`3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`).
No upstream implementation has been copied into the MIT Rust core. Data provenance
and attribution must be recorded separately from software provenance.

## Required import boundaries

The initial source audit preceded import. The subsequent native importer and
independent source checks are documented in [MOLECULAR-PRIORS.md](MOLECULAR-PRIORS.md). A native implementation must retain threshold/version,
source hashes, transmitter alternatives, receptor candidates, and unknown versus
conflicting evidence. Missing expression cannot be silently turned into a known
inhibitory or excitatory connection. Only existing anatomical edges receive
priors; expression does not establish an anatomical edge.

Class-to-cell expansion requires an explicit, audited mapping. In particular,
AWC ON/OFF identity must not be equated with fixed anatomical left/right. Numeric
zero-padding aliases can be declared separately from biological class expansion.
Any conversion of a qualitative polarity to P(excitatory), such as 0.9/0.1,
would be a declared modeling assumption, not a probability supplied by the source.
Metabotropic and peptide effects remain separate mechanisms. The import preserves these boundaries. Integration into parameter initialization
and fitted penalties remains required before claiming implemented molecular
priors in a benchmark run.
