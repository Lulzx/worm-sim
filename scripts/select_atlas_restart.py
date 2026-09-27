#!/usr/bin/env python3
"""Select a completed, predeclared restart cohort using validation data only."""
import argparse
import copy
import json
import math
from pathlib import Path
from audit_connectome_fit import digest, load, trace_scores


def normalized_config(config):
    config=copy.deepcopy(config)
    for key,value in {'molecular_sign_priors':None,'correlation':None,
                      'learning_rate_schedule':{'kind':'constant'},
                      'optimizer':{'kind':'adam'},'observation_gain':None,
                      'classification':None,'preparation_seconds':0.,
                      'sign_initialization':None}.items():
        config.setdefault(key,value)
    return config


def select(manifest, root):
    root=Path(root)
    assert manifest['schema_version']==1
    entries=manifest['runs']
    seeds=[e['seed'] for e in entries]
    assert len(seeds)>=2 and len(seeds)==len(set(seeds)), 'Incomplete or duplicate declared cohort'
    assert digest(root/manifest['data'])==manifest['data_file_sha256']
    assert digest(root/manifest['split'])==manifest['split_file_sha256']
    data,split=load(root/manifest['data']),load(root/manifest['split'])
    indexed={t['id']:t for t in data['trials']}
    assert len(indexed)==len(data['trials'])
    assert split['dataset_hash']==manifest['dataset_hash']
    assert data['graph_hash']==manifest['graph_hash']
    candidates=[]
    shared_config=None
    for entry in entries:
        path=root/entry['directory']
        assert digest(root/entry['config'])==entry['config_sha256'], 'Declared config changed'
        declared=normalized_config(load(root/entry['config']))
        config=load(path/'config.json')
        assert normalized_config(config)==declared
        assert config['sign_initialization']['seed']==entry['seed']
        common=copy.deepcopy(declared)
        del common['sign_initialization']['seed']
        if shared_config is None:shared_config=common
        assert common==shared_config, 'Restart cohort differs beyond its seed'
        selection=load(path/'selection.json')
        assert [s['epoch'] for s in selection]==list(range(config['epochs']+1)), 'Unfinished epoch sequence'
        assert all(math.isfinite(s['validation_mse']) and s['validation_mse']>=0 for s in selection)
        best=min(selection,key=lambda s:(s['validation_mse'],s['epoch']))
        model=load(path/'selected.json')
        assert model==load(path/f"epoch-{best['epoch']}.json")
        assert model['epoch']==best['epoch'] and model['config']==config
        for key in ('dataset_hash','split_hash','graph_hash'):
            assert model[key]==manifest[key]
        assert model['source_commit']==manifest['fit_source_commit']
        assert model['training_trials']==split['train'] and model['selection_trials']==split['validation']
        epoch_hashes={}
        for record in selection:
            epoch=record['epoch']
            assert record==load(path/f'epoch-{epoch}.report.json')
            checkpoint=load(path/f'epoch-{epoch}.json')
            for key in ('config','dataset_hash','split_hash','graph_hash','source_commit','training_trials','selection_trials'):
                assert checkpoint[key]==model[key]
            assert checkpoint['epoch']==epoch
            epoch_hashes[str(epoch)]=digest(path/f'epoch-{epoch}.json')
        prediction=load(path/'validation-predictions.json')
        assert len(prediction['trials'])==len(split['validation'])
        assert {t['id'] for t in prediction['trials']}==set(split['validation'])
        assert prediction['training_trials']==split['train'] and prediction['selection_trials']==split['validation']
        assert prediction['source_commit']==manifest['fit_source_commit']
        assert prediction['dataset_hash']==manifest['dataset_hash'] and prediction['split_hash']==manifest['split_hash']
        assert prediction['seed']==entry['seed']
        score=trace_scores(indexed,prediction)['pooled_mse']
        report=load(path/'validation-report.json')
        assert score is not None and abs(score-best['validation_mse'])<1e-10
        assert abs(score-report['pooled_trace_scores']['mse'])<1e-10
        candidates.append({'seed':entry['seed'],'directory':entry['directory'],
                           'selected_epoch':best['epoch'],'validation_mse':best['validation_mse'],
                           'independently_rescored_validation_mse':score,
                           'selected_model_sha256':digest(path/'selected.json'),
                           'selection_sha256':digest(path/'selection.json'),
                           'validation_predictions_sha256':digest(path/'validation-predictions.json'),
                           'epoch_model_sha256':epoch_hashes})
    candidates.sort(key=lambda c:c['seed'])
    best=min(candidates,key=lambda c:(c['validation_mse'],c['seed']))
    return {'schema_version':1,'fit_source_commit':manifest['fit_source_commit'],
            'candidates':candidates,'selected_seed':best['seed'],'selected_directory':best['directory'],
            'criterion':'Minimum saved validation trace MSE; earlier epoch within run, then smaller seed across exact ties. All declared runs must be complete.',
            'scope':'Validation-only selection and independent rescoring of saved validation predictions. No test files opened. No independent dynamics or optimization replay; those audits remain required. A small restart cohort is not an uncertainty ensemble.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--root',default='.')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=select(load(args.manifest),args.root)
    report.update(manifest_sha256=digest(args.manifest),selector_script_sha256=digest(__file__))
    with Path(args.output).open('x') as file:json.dump(report,file,indent=2);file.write('\n')
    print('Selected seed',report['selected_seed'],'from',len(report['candidates']),'completed validation records')


if __name__=='__main__':main()
