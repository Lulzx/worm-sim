#!/usr/bin/env python3
"""Score the validation-selected GRU and record a compact reproducible receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='target/release/wormsim')
parser.add_argument('--graph', default='data/c302-herm.wsc')
parser.add_argument('--data', default='runs/wormwideweb-benchmark.json')
parser.add_argument('--split', default='data/wormwideweb-animal-split.json')
parser.add_argument('--model', default='runs/gru-first-fit.json')
parser.add_argument('--prefix', default='runs/gru-first')
parser.add_argument('--receipt', default='runs/gru-first-receipt.json')
args = parser.parse_args()
inputs = [args.graph, args.data, args.split]
model_bytes = Path(args.model).read_bytes()
model = json.loads(model_bytes)
selection = json.loads(Path(args.model + '.selection.json').read_text())
reports = {}
for partition in ['validation', 'test']:
    prediction = args.prefix + '-' + partition + '-predictions.json'
    report = args.prefix + '-' + partition + '-report.json'
    subprocess.run([args.binary, 'gru-predict', *inputs, args.model, partition, prediction], check=True)
    subprocess.run([args.binary, 'bench-score', *inputs, prediction, partition, report], check=True)
    full = json.loads(Path(report).read_text())
    assert full['dataset_hash'] == model['dataset_hash'] == selection['dataset_hash']
    assert full['split_hash'] == model['split_hash'] == selection['split_hash']
    reports[partition] = {key: full[key] for key in ['dataset_hash', 'split_hash', 'graph_hash', 'partition', 'model', 'free_parameters', 'prediction_source_commit', 'scorer_source_commit', 'preprocessing_assessment', 'animal_bootstrap']}

receipt = {
    'schema_version': 1,
    'model_sha256': hashlib.sha256(model_bytes).hexdigest(),
    'model_training_source_commit': model['source_commit'],
    'selection': selection,
    'behavior_forecast': model.get('behavior'),
    **reports,
    'timed_training_and_validation_seconds': sum(c['elapsed_seconds'] for c in selection['candidates']),
    'timing_exclusions': 'File loading, training preparation/initialization, candidate artifact writing, and final held-out scoring are outside the summed timers.',
    'limitations': [
        'Only three test animals, already inspected during previous experiments; results are exploratory.',
        'One fixed hidden size and random seed, not an exhaustive hyperparameter search.',
        'Marginal animal-bootstrap intervals are not paired-difference significance tests.',
        'Training-unseen identities use persistence, differing from Level 0 default readouts.',
        'Standardized forecast MSE differs from the Level 0 raw fluorescence training objective; common selection/scoring uses unstandardized fluorescence.',
        ('Shared training-fitted behavior AR inputs; actual future behavior excluded.' if model.get('behavior') else 'No behavior inputs.') + ' Consult preprocessing_assessment for source causality evidence.',
        'Benchmark code and engineering tests do not establish biological validity.',
    ],
}
Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
print(args.receipt)
