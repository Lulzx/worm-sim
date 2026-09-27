#!/usr/bin/env python3
"""Score the validation-selected latent LDS and record a compact reproducible receipt."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='target/release/wormsim')
parser.add_argument('--graph', default='data/c302-herm.wsc')
parser.add_argument('--data', default='runs/wormwideweb-benchmark.json')
parser.add_argument('--split', default='data/wormwideweb-animal-split.json')
parser.add_argument('--model', default='runs/latent-lds.json')
parser.add_argument('--prefix', default='runs/latent-lds')
parser.add_argument('--receipt', default='runs/latent-lds-receipt.json')
args = parser.parse_args()
inputs = [args.graph, args.data, args.split]
model_bytes = Path(args.model).read_bytes()
model = json.loads(model_bytes)
selection = json.loads(Path(args.model + '.selection.json').read_text())
reports = {}
for partition in ['validation', 'test']:
    prediction = args.prefix + '-' + partition + '-predictions.json'
    report = args.prefix + '-' + partition + '-report.json'
    subprocess.run([args.binary, 'lds-predict', *inputs, args.model, partition, prediction], check=True)
    subprocess.run([args.binary, 'bench-score', *inputs, prediction, partition, report], check=True)
    full = json.loads(Path(report).read_text())
    assert full['dataset_hash'] == model['dataset_hash'] == selection['dataset_hash']
    assert full['split_hash'] == model['split_hash'] == selection['split_hash']
    reports[partition] = {key: full[key] for key in ['dataset_hash', 'split_hash', 'graph_hash', 'partition', 'model', 'free_parameters', 'prediction_source_commit', 'scorer_source_commit', 'animal_bootstrap']}

data = json.loads(Path(args.data).read_text())
indexed = {t['id']: t for t in data['trials']}
predictions = json.loads(Path(args.prefix + '-validation-predictions.json').read_text())
history = {'description': 'Independent weighted per-neuron reconstruction R² at the first and final prefix frames; validation only. Unknown training identities retain the disclosed persistence fallback.'}
for location in ['history_start', 'forecast_origin']:
    groups = defaultdict(list)
    for pred in predictions['trials']:
        trial = indexed[pred['id']]
        at = 0 if location == 'history_start' else next(i for i, t in enumerate(trial['recording']['times']) if abs(t - trial['forecast_origin']) < 1e-9)
        for trace in trial['recording']['traces']:
            y, weight = trace['values'][at], trace['provenance']['id_confidence']
            if y is not None and weight > 0:
                groups[trace['neuron']].append((y, pred['fluorescence'][trace['neuron']][at], weight))
    scores = []
    for pairs in groups.values():
        mean = math.fsum(y*w for y, p, w in pairs) / math.fsum(w for y, p, w in pairs)
        variance = math.fsum(w*(y-mean)**2 for y, p, w in pairs)
        if variance > 0:
            scores.append(1 - math.fsum(w*(y-p)**2 for y, p, w in pairs) / variance)
    history[location] = {'macro_neuron_r2': math.fsum(scores)/len(scores) if scores else None, 'defined_neurons': len(scores)}

receipt = {
    'schema_version': 1,
    'model_sha256': hashlib.sha256(model_bytes).hexdigest(),
    'model_training_source_commit': model['source_commit'],
    'selection': selection,
    **reports,
    'validation_history_diagnostic': history,
    'timed_preparation_and_candidate_seconds': selection['training_preparation_seconds'] + sum(c['elapsed_seconds'] for c in selection['candidates']),
    'timing_exclusions': 'File loading, per-rank PCA initialization, candidate artifact writing, and final held-out scoring are outside the summed timers.',
    'limitations': [
        'Three test animals only; marginal bootstrap intervals at both long horizons include zero.',
        'No paired-bootstrap significance claim against AR(1).',
        'Training-unseen neuron outputs use persistence; this differs from Level 0 readout assumptions.',
        'Constrained EM-style updates with fixed process jitter, observation floor, ridge and operator-norm cap.',
        'Source preprocessing causality/units remain to be audited.',
        'GRU and a successful fitted biological model remain outstanding.',
    ],
}
Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
print(args.receipt)
