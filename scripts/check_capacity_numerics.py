#!/usr/bin/env python3
"""Frozen-parameter step/preparation checks for a training capacity checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from audit_capacity_fit import digest
from replay_level0_atlas import Replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','manifest','training','graph','output']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args()
    saved,manifest,training,graph=[json.loads(Path(getattr(a,k)).read_text()) for k in ['checkpoint','manifest','training','graph']]
    assert saved['format']==manifest['format']=='wormsim-training-capacity-diagnostic'
    assert saved['targets']==manifest['targets']
    assert digest(a.training)==manifest['input_sha256']['training']
    assert digest(a.graph)==manifest['input_sha256']['graph']
    packed=saved['model'];base=packed['base_model']
    assert set(packed['configuration']['extensions'])=={'observation'}
    assert packed['configuration']['solver'] is None
    assert not base['selection_trials']
    groups=[g for g in training['groups'] if training['names'][g['target']] in saved['targets']]
    assert sorted(t for g in groups for t in g['training_trials'])==base['training_trials']==manifest['training_trials']
    gains=np.exp(packed['extension_parameters']['observation']['log_gain']['values'])
    total=sum(g['sample_weight'] for g in groups)
    variants={'reference':{},'half_step':{'dt':base['config']['dt']/2},
              'double_preparation':{'preparation_seconds':base['config']['preparation_seconds']*2}}
    reference={};rows=[]
    for name,changes in variants.items():
        model=copy.deepcopy(base);model['config'].update(changes)
        replay=Replay(model,graph);mse=0.;maxdiff=0.
        for group in groups:
            target=training['names'][group['target']]
            pred=replay.response(target,group['recording']['times'])*gains/replay.gain
            if name=='reference':reference[target]=pred
            maxdiff=max(maxdiff,float(np.max(np.abs(pred-reference[target]))))
            traces=group['recording']['traces'];weights=np.array([t['provenance']['id_confidence'] for t in traces])
            mean=np.array([t['values'] for t in traces]).T
            observed=pred[:,[replay.index[t['neuron']] for t in traces]]
            mse+=group['sample_weight']/total*(float(np.sum((observed-mean)**2*weights)/(weights.sum()*len(pred)))+group['irreducible_mse'])
        rows.append({'variant':name,'dt':replay.dt,'preparation_seconds':replay.preparation,
                     'training_mse':mse,'max_absolute_prediction_difference':maxdiff,
                     'unforced_prepared_derivative_max':float(np.max(np.abs(replay.rhs(replay.state,None,0.))))})
    assert abs(rows[0]['training_mse']-saved['metrics']['mse'])<1e-10
    receipt={'checkpoint_sha256':digest(a.checkpoint),'manifest_sha256':digest(a.manifest),
        'training_sha256':digest(a.training),'graph_sha256':digest(a.graph),
        'script_sha256':digest(__file__),'replay_script_sha256':digest(Path(__file__).with_name('replay_level0_atlas.py')),
        'variants':rows,'scope':'Frozen parameters, independent NumPy replay on training targets only. Half-step and doubled preparation are diagnostics, not refits or held-out scores.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
