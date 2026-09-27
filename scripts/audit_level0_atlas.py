#!/usr/bin/env python3
"""Audit selected Level 0 atlas artifacts and independently recompute saved-output scores.

Optionally replays the nonlinear ODE using independent NumPy dynamics; never refits parameters.
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
    parser.add_argument('--graph-json', help='Canonical graph JSON from wormsim unpack; enables independent ODE replay')
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
    assert model['graph_hash'] == data['graph_hash'] == split['graph_hash']
    replay = None
    if args.graph_json:
        from replay_level0_atlas import Replay
        replay = Replay(model, load(args.graph_json))
    replay_cache, max_replay_error = {}, 0.0
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
        if replay is not None:
            for trial in pred['trials']:
                target = indexed[trial['id']]['stimulated_neuron']
                key = (target,tuple(trial['times']))
                if key not in replay_cache:
                    replay_cache[key] = replay.response(target,trial['times'])
                for neuron,values in trial['fluorescence'].items():
                    error = float(np.max(np.abs(replay_cache[key][:,replay.index[neuron]]-values)))
                    max_replay_error = max(max_replay_error,error)
                    assert error < 1e-10
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
    test_prediction = load(run/'test-predictions.json')
    for trial in test_prediction['trials']:
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
    receipt = {'schema_version':1,'source_commit':model['source_commit'],'dataset_hash':model['dataset_hash'],'split_hash':model['split_hash'],'selected_epoch':model['epoch'],'config':config,'selection':selection,'free_parameters':report['free_parameters'],'independent_saved_trace_scores':scores,'independent_pair_auroc':auc,'max_pair_area_error':area_error,'validation_half_step':load(run/'validation-half-step.json'),'selected_model_sha256':digest(run/'selected.json'),'dataset_file_sha256':digest(args.data),'split_file_sha256':digest(args.split),'evidence_file_sha256':digest(args.evidence),'audit_script_sha256':digest(__file__),'shared_score_script_sha256':digest(Path(__file__).with_name('audit_connectome_fit.py')),'artifacts':hashes,'limitations':'Independent recomputation of saved-output MSE, correlation, response area and direct pairwise AUROC. Selection and declared lineage checked. Nonlinear ODE, optimization and bootstrap draws are not independently replayed here. See gradient tests and selected-checkpoint half-step check for separate numerical evidence. Shared fixed preparation seed, assumed positive shared input, neutral unannotated signs and previously inspected test cohort remain limitations.'}
    if config.get('preparation_seconds',0.) > 0:
        preparation_check = load(run/'validation-preparation-check.json')
        assert preparation_check['epoch'] == model['epoch']
        assert preparation_check['source_commit'] == model['source_commit']
        assert preparation_check['preparation_seconds'] == config['preparation_seconds']
        assert preparation_check['longer_seconds'] == 2*config['preparation_seconds']
        assert abs(preparation_check['selected_duration_mse']-scores['validation']['pooled_mse']) < 1e-10
        receipt['validation_preparation_check'] = preparation_check
        receipt['artifacts']['validation-preparation-check.json'] = digest(run/'validation-preparation-check.json')
    if replay is not None:
        norm = float(np.linalg.norm(replay.rhs(replay.state,None,0.)))
        if config.get('preparation_seconds',0.) > 0:
            assert abs(norm-preparation_check['derivative_l2']) < 1e-10
        receipt['independent_ode_replay'] = {'script_sha256':digest(Path(__file__).with_name('replay_level0_atlas.py')), 'graph_json_sha256':digest(args.graph_json),'target_grids_replayed':len(replay_cache),'max_fluorescence_error':max_replay_error,'prepared_unforced_derivative_l2':norm,'scope':'Independent NumPy Euler dynamics including unforced preparation, positive parameter transforms, chemical and gap currents, calcium/synapse dynamics and baseline-relative readout. All saved selected validation/test predictions checked. Does not refit parameters, replay optimization or independently bootstrap.'}
        receipt['limitations'] = receipt['limitations'].replace('Nonlinear ODE, optimization and bootstrap draws are not independently replayed here.', 'Nonlinear ODE replay is audited separately below; optimization and bootstrap draws are not independently replayed.')
    Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(scores,indent=2))
    print('pair AUROC',auc)


if __name__ == '__main__':
    main()
