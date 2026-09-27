# Fenyves transmitter and receptor data attribution

`data/fenyves-transmitter-receptors.json` is a structured extraction of factual
transmitter and receptor tables from supplementary file S1 Data
(`journal.pcbi.1007974.s003.xlsx`) accompanying:

Bánk G. Fenyves, Gábor S. Szilágyi, Zsolt Vassy, Csaba Sőti and Peter Csermely
(2020), “Synaptic polarity and sign-balance prediction using gene expression data
in the Caenorhabditis elegans chemical synapse neuronal connectome network”,
PLOS Computational Biology 16(12): e1007974.
https://doi.org/10.1371/journal.pcbi.1007974

Publisher supplement:
https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=10.1371/journal.pcbi.1007974.s003

The article declares [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). This attribution
applies separately from the MIT licence of WormSim software. Changes made:
extracted the first three columns of `1. NT expr` and first six columns of
`2. Receptor gene table`, converted numeric zero-padding neuron aliases to c302
identities, retained source coordinates, and serialized to JSON. Empty transmitter
cells remain unknown, and no expression measurements or polarities were changed.
The original workbook and source metadata are described in
[MOLECULAR-PRIORS.md](../docs/MOLECULAR-PRIORS.md).
