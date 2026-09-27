#!/usr/bin/env python3
"""Run Rust Task 2 controls and a fixed linear candidate grid; emit compact receipt."""
import argparse
import json
import pathlib
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='target/release/wormsim')
parser.add_argument('--graph', default='data/c302-herm.wsc')
parser.add_argument('--data', default='runs/wormwideweb-benchmark.json')
parser.add_argument('--split', default='data/wormwideweb-animal-split.json')
parser.add_argument('--prefix', default='runs/wormwideweb-controls')
parser.add_argument('--receipt', default='runs/wormwideweb-controls-receipt.json')
args = parser.parse_args()
inputs = [args.graph, args.data, args.split]
pathlib.Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)

def run(*command):
    subprocess.run([args.binary, *command], check=True)

def read(path):
    return json.loads(pathlib.Path(path).read_text())

model = args.prefix + '-linear-model.json'
run('linear-fit', *inputs, model)
rows = []
for name in ['persistence', 'history-mean', 'half-blend', 'training-mean', 'ar', 'linear']:
    prediction = args.prefix + '-' + name + '.json'
    if name == 'persistence':
        run('bench-persist', *inputs, 'test', prediction)
    elif name == 'linear':
        run('linear-predict', *inputs, model, 'test', prediction)
    else:
        run('bench-control', *inputs, 'test', name, prediction)
    report = prediction + '.report.json'
    run('bench-score', *inputs, prediction, 'test', report)
    result = read(report)
    rows.append({
        'name': name,
        **{key: result[key] for key in ['model', 'free_parameters', 'prediction_source_commit', 'scorer_source_commit', 'preprocessing_assessment', 'trials', 'animal_bootstrap']},
        'report': report,
    })
receipt = {
    'schema_version': 1,
    'dataset_hash': result['dataset_hash'],
    'split_hash': result['split_hash'],
    'graph_hash': result['graph_hash'],
    'partition': 'test',
    'linear_selection': read(model + '.selection.json'),
    'models': rows,
    'limitations': [
        'Three held-out animals: bootstrap intervals have very limited population coverage.',
        'Linear model is fluorescence-state VAR(1), not a latent Kalman-EM LDS.',
        'Model fitting uses training animals only; fixed ridge grid selected using validation animals only.',
        'Two test neuron identities (AVBL, URAVR) are absent from training; fitted controls and linear model use persistence for these.',
        'Initial unconstrained linear experiment failed at long horizons. No positive biological conclusion.',
    ],
}
pathlib.Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
print(args.receipt)
