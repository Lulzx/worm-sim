#!/usr/bin/env python3
"""Training-only, frozen-dynamics calibration and residual diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_capacity_fit import digest
from replay_level0_atlas import Replay


def calibration_scores(weighted_signal, cross, power):
    """Shared per-neuron rescaling across all selected training targets."""
    multiplier=np.divide(cross,power,out=np.zeros_like(cross),where=power>0)
    positive=np.maximum(0.,multiplier)
    error=weighted_signal-2*cross+power
    positive_error=weighted_signal-2*positive*cross+positive**2*power
    signed_error=weighted_signal-2*multiplier*cross+multiplier**2*power
    return multiplier, positive, error, positive_error, signed_error


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
    assert packed['configuration']['solver'] is None and not model['selection_trials']
    names=training['names'];n=len(names)
    groups=[g for g in training['groups'] if names[g['target']] in saved['targets']]
    assert sorted(t for g in groups for t in g['training_trials'])==model['training_trials']==manifest['training_trials']
    gains=np.exp(packed['extension_parameters']['observation']['log_gain']['values'])
    replay=Replay(model,graph);assert replay.names==names
    total=sum(g['sample_weight'] for g in groups)
    signal=np.zeros(n);cross=np.zeros(n);power=np.zeros(n);observed=np.zeros(n,dtype=int)
    floor=0.;pairs=[]
    for group in groups:
        target=names[group['target']]
        prediction=replay.response(target,group['recording']['times'])*gains/replay.gain
        traces=group['recording']['traces'];nt=len(prediction)
        normalizer=sum(t['provenance']['id_confidence'] for t in traces)*nt
        scale=group['sample_weight']/total
        floor+=scale*group['irreducible_mse']
        for trace in traces:
            i=replay.index[trace['neuron']];weight=scale*trace['provenance']['id_confidence']/normalizer
            if weight<=0:continue
            y=np.asarray(trace['values']);pred=prediction[:,i]
            signal[i]+=weight*float(y@y);cross[i]+=weight*float(y@pred);power[i]+=weight*float(pred@pred)
            observed[i]+=1
            yc=y-y.mean();pc=pred-pred.mean();denom=np.linalg.norm(yc)*np.linalg.norm(pc)
            pairs.append({'target':target,'neuron':names[i],
                'weighted_mean_trace_squared_error':weight*float((pred-y)@(pred-y)),
                'mean_trace_correlation':float(yc@pc/denom) if denom>0 else None})
    signed,positive,error,pos_error,signed_error=calibration_scores(signal,cross,power)
    mse=floor+float(error.sum())
    assert abs(mse-saved['metrics']['mse'])<1e-10
    pos_mse=floor+float(pos_error.sum());signed_mse=floor+float(signed_error.sum())
    assert signed_mse<=pos_mse+1e-12<=mse+2e-12
    rows=[{'neuron':names[i],'observed_target_count':int(observed[i]),
           'gain':float(gains[i]),'positive_gain_multiplier':float(positive[i]),
           'signed_gain_multiplier':float(signed[i]),'weighted_mean_trace_squared_error':float(error[i]),
           'positive_recalibrated_error':float(pos_error[i])} for i in range(n) if observed[i]]
    rows.sort(key=lambda row:-row['weighted_mean_trace_squared_error'])
    pairs.sort(key=lambda row:-row['weighted_mean_trace_squared_error'])
    receipt={'checkpoint_sha256':digest(a.checkpoint),'manifest_sha256':digest(a.manifest),
        'training_sha256':digest(a.training),'graph_sha256':digest(a.graph),'script_sha256':digest(__file__),
        'replay_script_sha256':digest(Path(__file__).with_name('replay_level0_atlas.py')),
        'targets':saved['targets'],'epoch':model['epoch'],'training_mse':mse,
        'frozen_dynamics_positive_recalibration_mse':pos_mse,
        'frozen_dynamics_signed_recalibration_mse':signed_mse,
        'positive_calibration_fraction_of_remaining_bound_gap':(mse-pos_mse)/(mse-manifest['bounds']['start_zero_mean_response_bound']),
        'observed_neurons':len(rows),'neurons':rows,'target_neuron_pairs':pairs,
        'scope':'Training-only least squares shared across selected targets with frozen dynamics. Positive scale is an optimistic calibration diagnostic; a zero multiplier is a limiting value for log gains. Signed scale is an inadmissible readout diagnostic, not a proposed model. No held-out scoring or checkpoint selection.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['neurons','target_neuron_pairs']},indent=2))

if __name__=='__main__':main()
