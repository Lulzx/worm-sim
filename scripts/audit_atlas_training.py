#!/usr/bin/env python3
"""Training-only shared-response error bound and independent model replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from audit_connectome_fit import digest, load
from replay_level0_atlas import Replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='runs/randi-data.json')
    p.add_argument('--split',default='data/randi-neuron-split.json')
    p.add_argument('--graph',default='runs/c302-audit.json')
    p.add_argument('--model',required=True)
    p.add_argument('--native',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--check-global-gain',action='store_true',help='Fit one nonnegative gain on training predictions, then check validation MSE only')
    a=p.parse_args()
    data,split,model,native=map(load,[a.data,a.split,a.model,a.native])
    assert native['partition']=='train' and native['model_sha256']==digest(a.model)
    assert model['dataset_hash']==split['dataset_hash']==native['dataset_hash']
    train=set(split['train'])
    trials=[t for t in data['trials'] if t['id'] in train]
    assert len(trials)==len(train)==native['training_trials']
    replay=Replay(model,load(a.graph))
    predicted={}
    values=defaultdict(list)
    zero_error=weight=model_error=cross=power=0.
    for t in trials:
        target=t['stimulated_neuron']
        key=(target,tuple(t['recording']['times']))
        if key not in predicted:
            predicted[key]=replay.response(target,t['recording']['times'])
        for trace in t['recording']['traces']:
            w=trace['provenance']['id_confidence']
            if w==0: continue
            assert all(v is not None for v in trace['values'])
            y=np.asarray(trace['values'])
            values[(target,trace['neuron'])].append((w,y))
            zero_error+=w*float(y@y)
            weight+=w*len(y)
            pred=predicted[key][:,replay.index[trace['neuron']]]
            cross+=w*float(y@pred)
            power+=w*float(pred@pred)
            residual=y-pred
            model_error+=w*float(residual@residual)
    within=energy=origin_energy=0.
    target_sums=defaultdict(lambda:np.zeros(3))
    for (target,neuron),rows in values.items():
        weights=np.asarray([w for w,_ in rows])
        y=np.stack([v for _,v in rows])
        mean=np.average(y,axis=0,weights=weights)
        residual=y-mean
        error=float(np.sum(weights[:,None]*residual**2))
        signal=float(weights.sum()*(mean@mean))
        within+=error; energy+=signal
        origin_energy+=float(weights.sum()*mean[0]**2)
        target_sums[target]+=np.array([weights.sum()*y.shape[1],error,signal])
    for g in native['groups']:
        total,error,signal=target_sums[g['stimulated_neuron']]
        assert abs(total-g['sample_weight'])<1e-8
        assert abs(error/total-g['within_target_trial_mse'])<1e-10
        assert abs(signal/total-g['weighted_mean_trace_energy'])<1e-10
    scores={'direct_zero_response_mse':zero_error/weight,'shared_target_response_lower_bound_mse':within/weight,
            'weighted_mean_trace_energy':energy/weight,'model_mse':model_error/weight}
    for name,value in scores.items(): assert abs(value-native[name])<1e-10,(name,value,native[name])
    assert abs((within+energy-zero_error)/weight)<1e-10
    zero_origin_bound=(within+origin_energy)/weight
    assert model_error/weight>=zero_origin_bound-1e-10
    receipt={'schema_version':1,'native_source_commit':native['source_commit'],'model_source_commit':model['source_commit'],
             'native_receipt_sha256':digest(a.native),'model_sha256':digest(a.model),'graph_sha256':digest(a.graph),
             'dataset_file_sha256':digest(a.data),'split_file_sha256':digest(a.split),'audit_script_sha256':digest(__file__),
             'replay_script_sha256':digest(Path(__file__).with_name('replay_level0_atlas.py')),'training_trials':len(trials),
             'target_grids_replayed':len(predicted),'sample_weight':weight,'independent_scores':scores,
             'zero_origin_extra_mse':origin_energy/weight,'zero_origin_shared_target_lower_bound_mse':zero_origin_bound,
             'fraction_mean_trace_energy_captured':(zero_error-model_error)/energy,
             'scope':'Training-only two-pass weighted means/residuals and independent NumPy neural replay. Per-target empirical means are a lower bound for the shared-response model, not a held-out baseline or irreducible biological noise estimate. The tighter bound additionally requires the initial predicted fluorescence to be zero, as in the current relative readout. Trial-specific context could explain within-target variation; no held-out labels or model selection here.'}
    if a.check_global_gain:
        assert np.isfinite(power) and power > 0., 'Global gain is unidentified for zero predicted energy'
        gain=max(0.,cross/power)
        assert np.isfinite(gain)
        validation=set(split['validation'])
        val_before=val_after=val_weight=0.
        for t in data['trials']:
            if t['id'] not in validation: continue
            key=(t['stimulated_neuron'],tuple(t['recording']['times']))
            if key not in predicted:
                predicted[key]=replay.response(t['stimulated_neuron'],t['recording']['times'])
            for trace in t['recording']['traces']:
                w=trace['provenance']['id_confidence']
                if w==0:continue
                valid=np.array([v is not None for v in trace['values']])
                y=np.array([v for v in trace['values'] if v is not None])
                pred=predicted[key][valid,replay.index[trace['neuron']]]
                val_before+=w*float((y-pred)@(y-pred))
                val_after+=w*float((y-gain*pred)@(y-gain*pred))
                val_weight+=w*len(y)
        calibrated=(zero_error-2*gain*cross+gain*gain*power)/weight
        receipt['training_fitted_global_gain_probe']={'gain':gain,'added_parameters':1,
            'training_scaled_mse':calibrated,'training_original_mse':model_error/weight,
            'validation_original_mse':val_before/val_weight,'validation_scaled_mse':val_after/val_weight,
            'scope':'One nonnegative least-squares observation gain, estimated from training data only; frozen neural dynamics and kernel. Validation-only diagnostic with no test scoring or q-label use. Does not retrain the joint classifier or improve waveform shape; not a replacement for the original fit artifact.'}
        print('training-fitted global gain probe',receipt['training_fitted_global_gain_probe'])
    Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(scores,indent=2))
    print('captured mean-trace energy',receipt['fraction_mean_trace_energy_captured'],'zero-origin bound',zero_origin_bound)


if __name__=='__main__':main()
