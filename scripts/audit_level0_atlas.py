#!/usr/bin/env python3
"""Audit selected Level 0 atlas artifacts and independently recompute saved-output scores.

Does not independently replay the nonlinear ODE or refit parameters.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_connectome_fit import load, digest, trace_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/level0-atlas-first-fit')
    parser.add_argument('--evaluation', default='runs/level0-atlas-first-fit-evaluation')
    parser.add_argument('--data', default='runs/randi-data.json')
    parser.add_argument('--split', default='data/randi-neuron-split.json')
    parser.add_argument('--evidence', default='runs/randi-pairs.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, evaluation = Path(args.run), Path(args.evaluation)
    model, selection = load(run/'selected.json'), load(run/'selection.json')
    config = load(run/'config.json')
    data, split, evidence = load(args.data), load(args.split), load(args.evidence)
    assert model['config'] == config
    assert [c['epoch'] for c in selection] == list(range(config['epochs']+1))
    best = min(selection, key=lambda c: (c['validation_mse'], c['epoch']))
    assert model['epoch'] == best['epoch']
    assert model == load(run/f"epoch-{model['epoch']}.json")
    assert model['training_trials'] == split['train'] and model['selection_trials'] == split['validation']
    assert model['dataset_hash'] == split['dataset_hash'] == evidence['dataset_hash']
    indexed = {t['id']: t for t in data['trials']}
    targets = [{indexed[i]['stimulated_neuron'] for i in split[p]} for p in ['train','validation','test']]
    assert all(not targets[i] & targets[j] for i in range(3) for j in range(i))
    hashes = {}
    for c in selection:
        name = f"epoch-{c['epoch']}"
        assert c == load(run/f'{name}.report.json')
        checkpoint = load(run/f'{name}.json')
        assert checkpoint['config'] == config and checkpoint['epoch'] == c['epoch']
        assert checkpoint['training_trials'] == split['train'] and checkpoint['selection_trials'] == split['validation']
        assert checkpoint['initial'] == model['initial'] and checkpoint['source_commit'] == model['source_commit']
        hashes[f'{name}.json'] = digest(run/f'{name}.json')
    scores = {}
    for partition in ['validation','test']:
        pred = load(run/f'{partition}-predictions.json')
        report = load(run/f'{partition}-report.json')
        assert len(pred['trials']) == len(split[partition])
        assert {t['id'] for t in pred['trials']} == set(split[partition])
        assert pred['training_trials'] == split['train'] and pred['selection_trials'] == split['validation']
        assert pred['source_commit'] == model['source_commit']
        score = trace_scores(indexed, pred)
        assert abs(score['pooled_mse']-report['pooled_trace_scores']['mse']) < 1e-10
        assert abs(score['macro_trace_correlation']-report['macro_trace_correlation']) < 1e-10
        assert score['defined_trace_correlations'] == report['defined_trace_correlations']
        scores[partition] = score
        for suffix in ['report','predictions']:
            hashes[f'{partition}-{suffix}.json'] = digest(run/f'{partition}-{suffix}.json')
    assert abs(scores['validation']['pooled_mse']-best['validation_mse']) < 1e-10
    predicted_pairs = load(evaluation/'pair-predictions.json')
    expected = {(p['stimulated'],p['responding']):p for p in evidence['pairs'] if p['stimulated'] in targets[2]}
    actual = {(p['stimulated'],p['responding']):p['score'] for p in predicted_pairs['pairs']}
    assert len(actual) == len(predicted_pairs['pairs']) and set(actual) == set(expected)
    areas = {}
    for trial in pred['trials']:
        stim = indexed[trial['id']]['stimulated_neuron']
        for neuron, values in trial['fluorescence'].items():
            area = float(np.abs(values).sum()*model['sample_dt'])
            key = (stim,neuron)
            if key in areas:
                assert area == areas[key]
            areas[key] = area
    area_error = max(abs(value-areas[key]) for key,value in actual.items())
    assert area_error < 1e-10
    positive = np.array([actual[k] for k,p in expected.items() if p['q']<evidence['detection_q_threshold']])
    negative = np.array([actual[k] for k,p in expected.items() if p['q']>=evidence['detection_q_threshold']])
    auc = float(np.mean((positive[:,None]>negative).astype(float)+0.5*(positive[:,None]==negative)))
    assert abs(auc-load(evaluation/'pair-report.json')['auroc']['value']) < 1e-12
    receipt = {'schema_version':1,'source_commit':model['source_commit'],'dataset_hash':model['dataset_hash'],'split_hash':model['split_hash'],'selected_epoch':model['epoch'],'config':config,'selection':selection,'free_parameters':report['free_parameters'],'independent_saved_trace_scores':scores,'independent_pair_auroc':auc,'max_pair_area_error':area_error,'validation_half_step':load(run/'validation-half-step.json'),'selected_model_sha256':digest(run/'selected.json'),'dataset_file_sha256':digest(args.data),'split_file_sha256':digest(args.split),'evidence_file_sha256':digest(args.evidence),'audit_script_sha256':digest(__file__),'shared_score_script_sha256':digest(Path(__file__).with_name('audit_connectome_fit.py')),'artifacts':hashes,'limitations':'Independent recomputation of saved-output MSE, correlation, response area and direct pairwise AUROC. Selection and declared lineage checked. Nonlinear ODE, optimization and bootstrap draws are not independently replayed here. See gradient tests and selected-checkpoint half-step check for separate numerical evidence. Fixed common initial state, assumed positive shared input, neutral unannotated signs and previously inspected test cohort remain limitations.'}
    Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(scores,indent=2))
    print('pair AUROC',auc)


if __name__ == '__main__':
    main()
