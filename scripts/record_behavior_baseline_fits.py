#!/usr/bin/env python3
"""Record training/validation-only behavior-assisted fits; require identical common inputs."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lds',default='runs/lds-behavior-fit.json')
parser.add_argument('--gru',default='runs/gru-behavior-fit.json')
parser.add_argument('--output',default='runs/behavior-baseline-selection.json')
args=parser.parse_args()
models={};reports={};hashes={}
for name,path in [('lds',args.lds),('gru',args.gru)]:
    content=Path(path).read_bytes();models[name]=json.loads(content)
    reports[name]=json.loads(Path(path+'.selection.json').read_text())
    hashes[name]=hashlib.sha256(content).hexdigest()
    for key in ['dataset_hash','split_hash','source_commit']:
        if models[name][key]!=reports[name][key]:raise ValueError(name+' model/selection mismatch: '+key)
    report=reports[name]
    if name=='lds':
        chosen=[c for c in report['candidates'] if c['rank']==models[name]['gaussian']['dim'] and c['iteration']==models[name]['iteration']]
    else:
        chosen=[c for c in report['candidates'] if c['epoch']==models[name]['epoch']]
    if len(chosen)!=1 or chosen[0]['validation_criterion']!=max(c['validation_criterion'] for c in report['candidates']):
        raise ValueError(name+' selection not validation-optimal')
for key in ['dataset_hash','split_hash','graph_hash','training_trials','selection_trials','behavior']:
    if models['lds'][key]!=models['gru'][key]:raise ValueError('Models have unequal common field: '+key)
behavior=models['lds']['behavior']
if behavior is None:raise ValueError('Expected behavior-assisted models')
registry=json.loads(Path('data/benchmark-preprocessing-audits.json').read_text())
evidence=[e for e in registry if e['dataset_hash']==models['lds']['dataset_hash'] and e['graph_hash']==models['lds']['graph_hash']]
for e in evidence:
    if hashlib.sha256(Path(e['evidence_path']).read_bytes()).hexdigest()!=e['evidence_sha256']:raise ValueError('Preprocessing evidence changed')
receipt={'schema_version':1,'scope':'Training/validation selection only. This script does not generate or score behavior-assisted test predictions. Level 0 input integration and equal-input test comparison remain pending.',
         'dataset_hash':models['lds']['dataset_hash'],'split_hash':models['lds']['split_hash'],'graph_hash':models['lds']['graph_hash'],
         'model_sha256':hashes,'common_behavior_artifacts_exactly_equal':True,'common_behavior_forecast':behavior,'selection':reports,
         'preprocessing_evidence':evidence,
         'limitations':['Retrospective processed-signal benchmark, not a prospectively processed forecast.','Validation scores used to select candidates are optimistic estimates, not held-out test results.','Actual future behavior is excluded; future behavior uncertainty is not marginalized.','Added input-matrix weights and common behavior scalars must be counted in model comparisons.']}
Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n')
print(args.output)
