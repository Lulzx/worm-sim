#!/usr/bin/env python3
"""Independent NumPy replay and training-subset bound check for capacity fits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from replay_level0_atlas import Replay


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    packed=saved['model'];model=packed['base_model']
    assert set(packed['configuration']['extensions'])=={'observation'}
    assert packed['configuration']['solver'] is None
    assert not model['selection_trials']
    for key in ['graph_hash','dataset_hash','split_hash']:
        assert model[key]==training[key]
    names=training['names'];wanted=set(saved['targets'])
    groups=[g for g in training['groups'] if names[g['target']] in wanted]
    assert len(groups)==len(wanted)
    assert sorted(t for g in groups for t in g['training_trials'])==model['training_trials']==manifest['training_trials']
    assert set(model['training_trials'])<=set(training['training_trials'])
    gain=packed['extension_parameters']['observation']['log_gain']
    assert gain['shape']==[len(names)]
    gains=np.exp(gain['values']);assert np.isfinite(gains).all()
    replay=Replay(model,graph)
    assert replay.names==names
    total=sum(g['sample_weight'] for g in groups)
    mse=zero=floor=start_zero=0.
    for group in groups:
        prediction=replay.response(names[group['target']],group['recording']['times'])*gains/replay.gain
        traces=group['recording']['traces'];nt=len(prediction)
        weights=np.array([t['provenance']['id_confidence'] for t in traces])
        mean=np.array([t['values'] for t in traces]).T
        pred=prediction[:,[replay.index[t['neuron']] for t in traces]]
        scale=group['sample_weight']/total
        weights=weights/(weights.sum()*nt)*scale
        residual_floor=scale*group['irreducible_mse']
        mse+=float(np.sum((pred-mean)**2*weights))+residual_floor
        zero+=float(np.sum(mean**2*weights))+residual_floor
        floor+=residual_floor
        start_zero+=float(np.sum(mean[0]**2*weights))+residual_floor
    for key,value in [('zero_response_mse',zero),('mean_response_bound',floor),('start_zero_mean_response_bound',start_zero)]:
        assert abs(value-manifest['bounds'][key])<1e-12
    assert abs(mse-saved['metrics']['mse'])<1e-10,(mse,saved['metrics']['mse'])
    assert mse>=start_zero-1e-12
    coefficients=replay.weight*(replay.reversal-replay.state[replay.post])*replay.inv_tau[replay.post]
    receipt={'checkpoint_sha256':digest(a.checkpoint),'manifest_sha256':digest(a.manifest),
        'training_sha256':digest(a.training),'graph_sha256':digest(a.graph),
        'audit_script_sha256':digest(__file__),'replay_script_sha256':digest(Path(__file__).with_name('replay_level0_atlas.py')),
        'epoch':model['epoch'],'targets':saved['targets'],'training_trials':len(model['training_trials']),
        'numpy_mse':mse,'jax_mse':saved['metrics']['mse'],'mse_absolute_difference':abs(mse-saved['metrics']['mse']),
        'bounds':manifest['bounds'],'captured_start_zero_energy':(zero-mse)/(zero-start_zero),
        'nonzero_prepared_chemical_coefficients':int(np.count_nonzero(coefficients)),
        'chemical_edges':len(coefficients),
        'unforced_prepared_derivative_max':float(np.max(np.abs(replay.rhs(replay.state,None,0.)))),
        'scope':'Independent NumPy dynamics and weighted scoring of Rust-exported training sufficient statistics; no held-out observations or generalization estimate.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
