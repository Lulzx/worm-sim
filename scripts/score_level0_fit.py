#!/usr/bin/env python3
"""Score a validation-selected Level 0 artifact and audit its validation timestep sensitivity."""
import argparse
import hashlib
import math
from collections import defaultdict
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='target/release/wormsim')
parser.add_argument('--graph', default='data/c302-herm.wsc')
parser.add_argument('--data', default='runs/wormwideweb-benchmark.json')
parser.add_argument('--split', default='data/wormwideweb-animal-split.json')
parser.add_argument('--model', default='runs/level0-first-fit.json')
parser.add_argument('--prefix', default='runs/level0-first')
parser.add_argument('--receipt', default='runs/level0-first-receipt.json')
args = parser.parse_args()
inputs = [args.graph, args.data, args.split]
model_bytes = Path(args.model).read_bytes()
model = json.loads(model_bytes)

def load(path):
    return json.loads(Path(path).read_text())

def score(path, partition, label):
    pred = args.prefix + '-' + label + '-predictions.json'
    report = args.prefix + '-' + label + '-report.json'
    subprocess.run([args.binary, 'level0-predict', *inputs, path, partition, pred], check=True)
    subprocess.run([args.binary, 'bench-score', *inputs, pred, partition, report], check=True)
    r = load(report)
    return {key: r[key] for key in ['dataset_hash', 'split_hash', 'graph_hash', 'partition', 'model', 'free_parameters', 'prediction_source_commit', 'scorer_source_commit', 'preprocessing_assessment', 'animal_bootstrap']}

validation = score(args.model, 'validation', 'validation')
# Numerical audit only, not another trained candidate or a test-based choice.
model['config']['inference']['dt'] *= 0.5
half = args.prefix + '-half-dt-model.json'
Path(half).write_text(json.dumps(model) + '\n')
refined = score(half, 'validation', 'half-dt-validation')
test = score(args.model, 'test', 'test')
delta = [b['confidence_weighted']['point'] - a['confidence_weighted']['point']
         for a, b in zip(validation['animal_bootstrap']['horizons'], refined['animal_bootstrap']['horizons'])]
# Diagnose whether history reconstruction survives to the forecast origin.
data = load(args.data)
indexed = {trial['id']: trial for trial in data['trials']}
predictions = load(args.prefix + '-validation-predictions.json')
history = {'description': 'Independent Python weighted per-neuron R², matching the scorer aggregation, at the first and final observed prefix frames. The ten-second history is used by state inference; these are reconstruction scores, not held-out forecasts.'}
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
    history[location] = {'macro_neuron_r2': math.fsum(scores) / len(scores) if scores else None, 'defined_neurons': len(scores)}
receipt = {
    'schema_version': 1,
    'model_sha256': hashlib.sha256(model_bytes).hexdigest(),
    'fit': load(args.model + '.fit.json'),
    'behavior_forecast': model.get('behavior'),
    'validation': validation,
    'test': test,
    'validation_history_diagnostic': history,
    'numerical_audit': {
        'partition': 'validation',
        'original_dt': 2 * model['config']['inference']['dt'],
        'refined_dt': model['config']['inference']['dt'],
        'macro_r2_delta_1_10_30': delta,
        'scope': 'Complete inference-and-forecast pipeline with selected weights fixed. No refitting, timestep selection or test-set numerical tuning.',
        'refined_validation': refined,
    },
    'limitations': [
        ('Selected epoch zero means population training failed to improve the declared validation criterion.' if model['selected_epoch'] == 0 else 'The selected trained epoch improved validation; this alone does not establish biological recovery or positive held-out forecasting.'),
        'Conditional parameter gradients hold history-inferred states fixed.',
        'Suffix-based bilateral sharing is an assumption; graph class and transmitter annotations remain missing.',
        'Graph sign priors are all neutral 0.5; these are not CeNGEN-informed sign priors.',
        'Only three test animals, and baseline results on them were previously inspected.',
        'LDS and GRU are scored separately; parameter-matched comparisons remain outstanding.',
    ],
}
Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
print(args.receipt)
