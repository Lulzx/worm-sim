# Neutral chemical reversal and atlas initialization

The initial neutral-prior atlas model has **zero local chemical gate-to-voltage
sensitivity on all 3,638 chemical edges**. This follows from the actual Level 0
equation, not from absence of anatomical edges:

```text
chemical_current_ab = conductance_ab * gate_a * (reversal_ab - voltage_b)
d(voltage_rate_b)/d(gate_a)
    = inverse_tau_b * conductance_ab * (reversal_ab - voltage_b)
```

The initialization puts voltage and leak-rest at zero. Neutral sign probabilities
of 0.5 map to reversal `2*probability-1 = 0`. Unforced preparation therefore leaves
the voltages at zero, and every chemical driving force is zero. Presynaptic gate
changes have no first-order effect on postsynaptic voltage at that state.

This does **not** turn off every chemical effect: postsynaptic shunting remains,
and gap junctions can still transmit voltage changes. Once parameters and resting
voltages move, chemical transmission can emerge. These are local derivatives,
not a complete recurrent transfer function or a claim that all optimization
gradients vanish.

A Rust test isolates a two-neuron chemical-only circuit with no gaps. Stimulating
one neuron gives a nonzero response there and exactly zero fluorescence response
in its neighbor with neutral reversals and zero resting voltage. A separate
nonneutral-reversal run, after parameter-dependent preparation, permits a
neighbor response. Another test verifies the derivative above against central
differences of the actual RHS, including synapse-count and membrane-time scaling.

The native diagnostic `diagnose_atlas_coupling`, source `fbaa0cf`, was applied to
the actual 302-neuron graph. Independent NumPy preparation and edge calculations
check all coefficients, edge identities, zero/positive/negative counts and norms:

| Checkpoint | Exactly zero derivatives | L2 norm of all edge derivatives |
| --- | ---: | ---: |
| Neutral joint initialization | 3,638 / 3,638 | 0 |
| Learned-gain model, epoch 8 | 0 / 3,638 | 0.2032823 |
| Molecular threshold-2 initialization | 35 / 3,638 | 0.3784930 |

These norms are diagnostics, not a ranking of biological quality. In particular,
the molecular-prior comparison already failed to establish superiority over LDS;
nonzero initial coupling alone does not establish a useful fit. The coefficient
sign reflects driving force at this state, not independently established
physiological neurotransmitter polarity.

Independent receipts: [neutral initialization](coupling-neutral-initial-audit.json),
[gain epoch 8](coupling-gain-epoch8-audit.json),
[molecular initialization](coupling-molecular-th2-initial-audit.json).
They retain source/model/graph/native-receipt hashes. Reproduce each pair with:

```sh
target/release/examples/diagnose_atlas_coupling data/c302-herm.wsc runs/level0-atlas-classification-fit/epoch-0.json runs/coupling-neutral-new.json
python3 scripts/audit_atlas_coupling.py --model runs/level0-atlas-classification-fit/epoch-0.json --native runs/coupling-neutral-new.json --output runs/coupling-neutral-new-audit.json
```

## Intermediate training fit, not a final comparison

Both 25-update runs from `dcdd0ff` were still running when their fixed epoch-8
checkpoints were diagnosed. No test responses or interim test scores were used.
Training-only decomposition and independent replay give:

| Epoch-8 training metric | Unit-gain control | Learned gain |
| --- | ---: | ---: |
| Original-trial weighted MSE | 0.07535556 | 0.07485749 |
| Fraction of training mean-trace energy captured | 0.9179% | 4.9122% |
| Pair-mean shape loss, epsilon 0.01 | 1.00219678 | 1.00305549 |

Gain improves amplitude fit, but substantial training underfitting and poor
pair-uniform shape agreement remain at this checkpoint. The shape diagnostic
was not part of these runs' training objective. Different trace weighting allows
its deterioration to coexist with lower MSE. These interim measurements neither
replace validation checkpoint selection nor imply a final test outcome.

Receipts: [control native](longer-epoch8-training.json),
[control independent](longer-epoch8-training-audit.json),
[gain native](gain-epoch8-training.json),
[gain independent](gain-epoch8-training-audit.json).
Both independent audits replay all 161 training target grids, verify the MSE
decomposition within 1e-10 and check 24,635 eligible pair means.

The next initialization experiment should test nonneutral starting signs under
multiple predeclared seeds while retaining the declared prior centers. This is
an optimization hypothesis, not a reason to invent physiological sign labels or
alter the two already frozen runs. It also does not establish initialization as
the sole cause of underfitting; model capacity, observation/input calibration,
loss weighting and optimization duration remain open factors.
