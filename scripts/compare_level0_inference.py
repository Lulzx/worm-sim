#!/usr/bin/env python3
"""Compare fixed-parameter Level 0 inference methods on validation only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='target/release/wormsim')
parser.add_argument('--graph', default='data/c302-herm.wsc')
parser.add_argument('--data', default='runs/wormwideweb-benchmark.json')
parser.add_argument('--split', default='data/wormwideweb-animal-split.json')
parser.add_argument('--shooting', default='runs/level0-first-fit.json.epoch-0.json')
parser.add_argument('--filter', default='runs/level0-filter-fit.json.epoch-0.json')
parser.add_argument('--prefix', default='runs/level0-inference-comparison')
parser.add_argument('--receipt', default='runs/level0-inference-comparison.json')
args = parser.parse_args()

def load(path):
    return json.loads(Path(path).read_text())

models = {name: load(path) for name, path in [('shooting', args.shooting), ('filter', args.filter)]}
matched = ['dataset_hash', 'split_hash', 'graph_hash', 'parameters', 'readout', 'calibration',
           'readout_fitted', 'training_trials', 'selection_trials', 'selected_epoch']
for key in matched:
    if models['shooting'][key] != models['filter'][key]:
        raise ValueError('Incomparable initial models: ' + key)
if models['shooting']['selected_epoch'] != 0:
    raise ValueError('Expected untrained epoch zero')
inputs = [args.graph, args.data, args.split]
trials = {t['id']: t for t in load(args.data)['trials']}
receipt = {'schema_version': 1, 'partition': 'validation', 'identical_fields': matched, 'methods': {}}
for name, path in [('shooting', args.shooting), ('filter', args.filter)]:
    pred = args.prefix + '-' + name + '-predictions.json'
    report = args.prefix + '-' + name + '-report.json'
    subprocess.run([args.binary, 'level0-predict', *inputs, path, 'validation', pred], check=True)
    subprocess.run([args.binary, 'bench-score', *inputs, pred, 'validation', report], check=True)
    scored = load(report)
    history = {}
    predictions = load(pred)
    for location in ['history_start', 'forecast_origin']:
        groups = defaultdict(list)
        for p in predictions['trials']:
            t = trials[p['id']]
            at = 0 if location == 'history_start' else next(i for i, x in enumerate(t['recording']['times']) if abs(x - t['forecast_origin']) < 1e-9)
            for trace in t['recording']['traces']:
                y, w = trace['values'][at], trace['provenance']['id_confidence']
                if y is not None and w > 0:
                    groups[trace['neuron']].append((y, p['fluorescence'][trace['neuron']][at], w))
        r2 = []
        for rows in groups.values():
            mean = math.fsum(y*w for y, p, w in rows)/math.fsum(w for y, p, w in rows)
            variance = math.fsum(w*(y-mean)**2 for y, p, w in rows)
            if variance > 0:
                r2.append(1-math.fsum(w*(y-p)**2 for y, p, w in rows)/variance)
        history[location] = {'macro_neuron_r2': math.fsum(r2)/len(r2), 'defined_neurons': len(r2)}
    receipt['methods'][name] = {
        'model_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'inference_config': models[name]['config']['inference'],
        'history_reconstruction': history,
        'score': {k: scored[k] for k in ['dataset_hash', 'split_hash', 'graph_hash', 'model', 'free_parameters', 'prediction_source_commit', 'scorer_source_commit', 'animal_bootstrap']},
    }
receipt['limitations'] = [
    'History reconstruction uses assimilated observations; it is not forecast accuracy.',
    'Comparison changes only inference configuration and uses no test targets.',
    'Block filter omits cross-neuron covariance and covariance effects of mean projection.',
    'Fixed variance assumptions were not estimated from biological noise.',
]
Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
print(args.receipt)
