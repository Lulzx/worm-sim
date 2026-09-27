#!/usr/bin/env python3
"""Independently audit frozen atlas LDS selection, impulses and trace scores.

Requires a completed run. Does not fit or select on test data. NumPy is used only
for direct matrix-vector propagation and elementary score calculations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def impulse(model, target, frames):
    gaussian = model['dynamics']['gaussian']
    n = gaussian['dim']
    matrix = np.asarray(gaussian['transition']).reshape(n, n)
    response = np.zeros((frames, n))
    for t in range(frames - 1):
        # Independent dense propagation; native prediction uses sparse support.
        response[t + 1] = np.einsum('ij,j->i', matrix, response[t])
        kernel = model['dynamics']['kernel']
        if t < len(kernel):
            response[t + 1, target] += kernel[t]
    return response


def trace_scores(trials, prediction):
    squared, zero_squared, weight_sum = 0.0, 0.0, 0.0
    correlations, correlation_weights = [], []
    for pred in prediction['trials']:
        trial = trials[pred['id']]
        assert pred['times'] == trial['recording']['times']
        assert set(pred['fluorescence']) == {t['neuron'] for t in trial['recording']['traces']}
        for trace in trial['recording']['traces']:
            weight = trace['provenance']['id_confidence']
            observed = np.array([x is not None for x in trace['values']])
            y = np.array([0.0 if x is None else x for x in trace['values']])[observed]
            p = np.array(pred['fluorescence'][trace['neuron']])[observed]
            assert np.isfinite(y).all() and np.isfinite(p).all()
            if weight <= 0 or not len(y):
                continue
            squared += weight * np.sum((y-p)**2)
            zero_squared += weight * np.sum(y*y)
            weight_sum += weight * len(y)
            dy, dp = y-y.mean(), p-p.mean()
            denominator = np.sqrt(np.sum(dy*dy)*np.sum(dp*dp))
            if denominator > 0:
                correlations.append(float(np.sum(dy*dp)/denominator))
                correlation_weights.append(weight)
    return {
        'pooled_mse': float(squared/weight_sum),
        'zero_response_mse': float(zero_squared/weight_sum),
        'macro_trace_correlation': float(np.average(correlations, weights=correlation_weights)) if correlations else None,
        'defined_trace_correlations': len(correlations),
    }


def direct_predictions(model, indexed, ids, name_index):
    cache, trials = {}, []
    for identity in ids:
        trial = indexed[identity]
        times = trial['recording']['times']
        key = (trial['stimulated_neuron'], len(times))
        if key not in cache:
            cache[key] = impulse(model, name_index[key[0]], key[1])
        trials.append({'id': identity, 'times': times, 'fluorescence': {
            trace['neuron']: cache[key][:, name_index[trace['neuron']]].tolist()
            for trace in trial['recording']['traces']}})
    return {'trials': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/connectome-lds-first-fit')
    parser.add_argument('--data', default='runs/randi-data.json')
    parser.add_argument('--split', default='data/randi-neuron-split.json')
    parser.add_argument('--ids', default='data/c302-neuron-ids.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    model = load(run/'selected.json')
    selection = load(run/'selection.json')
    config = load(run/'config.json')
    data, split, names = load(args.data), load(args.split), load(args.ids)
    assert model['config'] == config
    assert [c['iteration'] for c in selection] == list(range(config['iterations']+1))
    best = min(selection, key=lambda c: (c['validation_mse'], c['iteration']))
    assert model['iteration'] == best['iteration']
    assert model == load(run/f"iteration-{model['iteration']}.json")
    assert model['dataset_hash'] == split['dataset_hash']
    assert model['graph_hash'] == split['graph_hash'] == data['graph_hash']
    assert model['training_trials'] == split['train']
    assert model['selection_trials'] == split['validation']
    indexed = {t['id']: t for t in data['trials']}
    target_sets = [{indexed[i]['stimulated_neuron'] for i in split[p]} for p in ['train','validation','test']]
    assert all(not target_sets[i] & target_sets[j] for i in range(3) for j in range(i))
    n = model['dynamics']['gaussian']['dim']
    assert len(names) == len(set(names)) == n and names == sorted(names)
    name_index = {name: i for i, name in enumerate(names)}
    expected_observations = sum(sum(x is not None for x in t['values']) for i in split['train'] for t in indexed[i]['recording']['traces'] if t['provenance']['id_confidence'] > 0)
    expected_transitions = sum(len(indexed[i]['recording']['times'])-1 for i in split['train'])
    file_hashes = {}
    for candidate in selection:
        iteration = candidate['iteration']
        assert candidate == load(run/f'iteration-{iteration}.report.json')
        checkpoint = load(run/f'iteration-{iteration}.json')
        assert checkpoint['training_trials'] == split['train']
        assert checkpoint['selection_trials'] == split['validation']
        assert checkpoint['config'] == config and checkpoint['iteration'] == iteration
        assert checkpoint['source_commit'] == model['source_commit']
        if iteration:
            assert candidate['training_step']['observations'] == expected_observations
            assert candidate['training_step']['transitions'] == expected_transitions
        independent_validation = trace_scores(indexed, direct_predictions(checkpoint, indexed, split['validation'], name_index))
        assert abs(independent_validation['pooled_mse']-candidate['validation_mse']) < 1e-10
        file_hashes[f'iteration-{iteration}.json'] = digest(run/f'iteration-{iteration}.json')
    results = {}
    max_error = 0.0
    for partition in ['validation', 'test']:
        prediction = load(run/f'{partition}-predictions.json')
        report = load(run/f'{partition}-report.json')
        assert len(prediction['trials']) == len(split[partition])
        assert {t['id'] for t in prediction['trials']} == set(split[partition])
        assert prediction['training_trials'] == split['train']
        assert prediction['selection_trials'] == split['validation']
        assert prediction['source_commit'] == model['source_commit']
        cache = {}
        for pred in prediction['trials']:
            key = (indexed[pred['id']]['stimulated_neuron'], len(pred['times']))
            if key not in cache:
                cache[key] = impulse(model, name_index[key[0]], key[1])
            for name, values in pred['fluorescence'].items():
                error = float(np.max(np.abs(cache[key][:, name_index[name]]-values)))
                max_error = max(max_error, error)
                assert error < 1e-10
        scores = trace_scores(indexed, prediction)
        assert abs(scores['pooled_mse']-report['pooled_trace_scores']['mse']) < 1e-10
        assert scores['defined_trace_correlations'] == report['defined_trace_correlations']
        if scores['macro_trace_correlation'] is not None:
            assert abs(scores['macro_trace_correlation']-report['macro_trace_correlation']) < 1e-10
        else:
            assert report['macro_trace_correlation'] is None
        results[partition] = scores
        for suffix in ['predictions', 'report']:
            file_hashes[f'{partition}-{suffix}.json'] = digest(run/f'{partition}-{suffix}.json')
    assert abs(results['validation']['pooled_mse']-best['validation_mse']) < 1e-10
    receipt = {'schema_version':1, 'source_commit':model['source_commit'], 'selected_iteration':model['iteration'], 'config':config, 'selection':selection, 'independent_trace_scores':results, 'max_dense_impulse_error':max_error, 'nominal_free_parameters':report['free_parameters'], 'dataset_file_sha256':digest(args.data), 'split_file_sha256':digest(args.split), 'neuron_ids_sha256':digest(args.ids), 'selected_model_sha256':digest(run/'selected.json'), 'audit_script_sha256':digest(__file__), 'artifacts':file_hashes, 'limitations':'Checks numerical propagation, trace scores, declared lineage and validation checkpoint selection. Does not independently refit EM, establish biological acceptance, or prove absence of upstream preprocessing leakage. Zero-response control has undefined trace correlation and constant-score AUROC 0.5 when both classes exist.'}
    Path(args.output).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
