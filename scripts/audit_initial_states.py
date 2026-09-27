#!/usr/bin/env python3
"""Audit history-only Level 0 state inference on the first window of each training animal."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='target/release/wormsim')
parser.add_argument('--graph', default='data/c302-herm.wsc')
parser.add_argument('--data', default='runs/wormwideweb-benchmark.json')
parser.add_argument('--split', default='data/wormwideweb-animal-split.json')
parser.add_argument('--output', default='runs/initial-state-audit')
parser.add_argument('--receipt', default='runs/initial-state-receipt.json')
args = parser.parse_args()
data = json.loads(Path(args.data).read_text())
split = json.loads(Path(args.split).read_text())
train = set(split['train'])
selected = {}
for trial in sorted(data['trials'], key=lambda t: t['id']):
    if trial['id'] in train:
        selected.setdefault(trial['recording']['animal_id'], trial['id'])
del data
Path(args.output).mkdir(parents=True, exist_ok=True)
rows = []
for animal, trial in sorted(selected.items()):
    path = Path(args.output) / (trial + '.json')
    subprocess.run([args.binary, 'level0-infer', args.graph, args.data, args.split, trial, str(path)], check=True)
    audit = json.loads(path.read_text())
    assert audit['partition'] == 'train' and audit['trial'] == trial and audit['animal'] == animal
    state = audit['inferred']
    rows.append({
        'animal': animal, 'trial': trial, 'partition': 'train',
        'state_values': len(state['forecast_state']),
        'observed_neurons': state['observed_neurons'], 'latent_neurons': state['latent_neurons'],
        'history_samples': state['history_samples'],
        'initial_objective': state['history_objective'][0],
        'final_objective': state['history_objective'][-1],
        'accepted_steps': len(state['history_objective']) - 1,
        'elapsed_seconds': audit['elapsed_seconds'], 'full_audit': str(path),
    })
receipt = {
    'schema_version': 1,
    **{key: audit[key] for key in ['dataset_hash', 'split_hash', 'graph_hash', 'source_commit', 'description', 'config']},
    'selection': 'First trial in sorted ID order per training animal; no outcome selection; validation/test animals not evaluated.',
    'readout_training_trials': len(audit['readout_training_trials']),
    'readout_unseen_neurons': len(audit['readout_unseen_neurons']),
    'animals': rows,
    'limitations': ['Frozen default network parameters; no population parameter fit.',
                    'Prefix reconstruction only, not forecast scores or hidden-state recovery ground truth.',
                    'One window per training animal is an engineering audit, not the complete training cohort.'],
}
Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
print(args.receipt)
