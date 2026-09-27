# Pinned Creamer linear baseline

The Rust baseline module reproduces inference from three pretrained models in
[Creamer_LDS_2026](https://github.com/Nondairy-Creamer/Creamer_LDS_2026), commit
`bba43302d50a4947804d98b01779856e648237cc`. Source hashes and the MIT notice are
retained. The downloader verifies Git blob identities. A restricted numeric
unpickler exports arrays without importing upstream model classes or executing
upstream Python source. Only fetch the pinned assets; the exporter is not a
general untrusted-pickle sandbox.

```sh
python3 scripts/fetch_creamer.py
# NumPy is available in the optional Taichi environment; see TAICHI.md.
.venv-taichi/bin/python scripts/export_creamer.py runs/creamer-baseline.json
cargo run --release -- baseline-pack runs/creamer-baseline.json runs/creamer-baseline.wsb
cargo run --release -- baseline-eval runs/creamer-baseline.wsb runs/creamer-report.json
```

All six inference operators are retained: dynamics W, input H, process covariance
Q, emissions C, direct input D, and observation covariance R. The 154-neuron
models have 45 input lags at 2 Hz. Identity, zero, diagonal/lagged-diagonal, CSR,
and dense representations encode exact structure; no coefficients are quantized.
Training histories and initialization copies are excluded. Learned parameter
counts include all 6,930 input coefficients, including fitted zeros.

The WSB1 bundle uses checksummed zstd-compressed neutral JSON with bounded decode.
Three models, measurements, and independent references pack into 2,089,477 bytes
from 5,007,824 bytes of compact JSON. This is a structured inference export, not
a byte-for-byte archival replacement for the original pickles.

## Evaluation and limits

Rust matches independent NumPy einsum calculations for all response matrices,
correlation matrices, selected temporal probes, and Pearson scores. STAMs use
all 154 impulse trials, 30 seconds post-stimulus, and sum/sample-rate integration.
Correlations follow the upstream 100-step covariance recurrence and latent
normalization; observation noise R is retained but not added to this metric.
Missing pairs and diagonals are excluded, giving 11,010 STAM and 17,936 correlation
pairs. Approximate Fisher intervals are not animal-level bootstrap uncertainty.

| Model | Learned parameters | Test STAM r | Test correlation r |
| --- | ---: | ---: | ---: |
| Connectome constrained | 9,403 | 0.140284 | 0.281957 |
| Fully connected | 30,954 | 0.119643 | 0.324723 |
| Shuffled constrained | 9,403 | 0.040332 | 0.115433 |

Receipts: `creamer-fetch-receipt.json`, `creamer-numpy-reference.json`,
`creamer-rust-report.json`. The upstream recording train/test split is preserved.
This does not implement the specification's held-out stimulated-neuron split,
retrain the models, reproduce every published figure, or fit our nonlinear
Level 0 model to these data. The scores describe the pinned assets and protocol;
we have not independently rerun the upstream training pipeline.
