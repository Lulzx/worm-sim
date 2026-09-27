"""Predict from a saved JAX envelope and Rust's observation-free atlas plan."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
from extensions import restore
from level0 import response


def predict(checkpoint, graph, plan, checkpoint_hash):
    model = checkpoint['base_model']
    if plan['schema_version'] != 1 or plan['partition'] not in ('validation', 'test'):
        raise ValueError('invalid prediction plan')
    for key in ('graph_hash', 'dataset_hash', 'split_hash', 'training_trials', 'selection_trials', 'sample_dt'):
        if plan[key] != model[key]:
            raise ValueError(f'prediction plan lineage mismatch: {key}')
    names = sorted(n['id'] for n in graph['neurons'])
    index = {name: i for i, name in enumerate(names)}
    chemistry = sorted(graph['chemical'], key=lambda e: (e['pre'], e['post']))
    gaps = sorted(graph['gaps'], key=lambda e: (e['a'], e['b']))
    if names != plan['names'] or [[index[e['pre']], index[e['post']], e['synapse_count']] for e in chemistry] != plan['chemical_topology'] or [[index[e['a']], index[e['b']], e['size']] for e in gaps] != plan['gap_topology']:
        raise ValueError('prediction graph differs from authoritative Rust topology')
    trials = []; cache = {}; engines = {}; count = None; seen = set()
    for trial in plan['trials']:
        if trial['id'] in seen or not 0 <= trial['target'] < len(names):
            raise ValueError('duplicate trial or invalid target')
        seen.add(trial['id'])
        grid = tuple(trial['times'])
        if grid not in engines:
            engine, theta, active = restore(checkpoint, graph, grid)
            engines[grid] = (engine, theta)
            count = sum(int(np.asarray(v).sum()) for v in jax.tree.leaves(active))
        key = (grid, trial['target'])
        if key not in cache:
            engine, theta = engines[grid]
            cache[key] = np.asarray(response(engine, theta, jax.numpy.asarray(trial['target'])))
            if not np.isfinite(cache[key]).all():
                raise ValueError('nonfinite JAX prediction')
        values = {name: cache[key][:, index[name]].tolist() for name in trial['neurons']}
        trials.append({'id': trial['id'], 'times': trial['times'], 'fluorescence': values,
                       'response_scores': {name: model['sample_dt'] * float(np.abs(values[name]).sum()) for name in trial['response_neurons']}})
    if not trials:
        raise ValueError('empty prediction plan')
    return {'schema_version': 1, 'dataset_hash': model['dataset_hash'], 'split_hash': model['split_hash'],
            'model': 'jax-atlas:' + checkpoint_hash, 'source_commit': model['source_commit'],
            'free_parameters': count, 'training_trials': model['training_trials'],
            'selection_trials': model['selection_trials'], 'seed': 0, 'trials': trials}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'graph', 'plan', 'output']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    raw = Path(a.checkpoint).read_bytes()
    result = predict(json.loads(raw), json.loads(Path(a.graph).read_text()), json.loads(Path(a.plan).read_text()), hashlib.sha256(raw).hexdigest())
    with open(a.output, 'x') as f:
        json.dump(result, f, allow_nan=False)

if __name__ == '__main__':
    main()
