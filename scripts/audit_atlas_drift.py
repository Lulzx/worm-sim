#!/usr/bin/env python3
"""Audit validation-only drift decomposition and saved-output scores, not the ODE."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_connectome_fit import load, digest, trace_scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic',required=True)
    parser.add_argument('--data',default='runs/randi-data.json')
    parser.add_argument('--split',default='data/randi-neuron-split.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.diagnostic)
    report=load(root/'report.json')
    data,split=load(args.data),load(args.split)
    indexed={t['id']:t for t in data['trials']}
    names=['driven','zero_current','stimulus_difference']
    predictions={name:load(root/f'{name}-predictions.json') for name in names}
    maps={name:{t['id']:t for t in pred['trials']} for name,pred in predictions.items()}
    for name,pred in predictions.items():
        assert pred['dataset_hash']==split['dataset_hash']==report['dataset_hash']
        assert set(maps[name])==set(split['validation']) and len(maps[name])==len(pred['trials'])
        assert not set(maps[name]) & set(split['test'])
        scores=trace_scores(indexed,pred)
        assert abs(scores['pooled_mse']-report['scores'][name]['mse'])<1e-10
        assert scores['defined_trace_correlations']==report['scores'][name]['defined_correlations']
        if scores['macro_trace_correlation'] is None:
            assert report['scores'][name]['correlation'] is None
        else:
            assert abs(scores['macro_trace_correlation']-report['scores'][name]['correlation'])<1e-10
    totals=np.zeros(4);weight=0.;max_error=0.;zero_by_neuron={}
    for identity in split['validation']:
        for trace in indexed[identity]['recording']['traces']:
            neuron=trace['neuron']
            p,z,d=[np.asarray(maps[name][identity]['fluorescence'][neuron]) for name in names]
            max_error=max(max_error,float(np.max(np.abs(p-z-d))))
            if neuron in zero_by_neuron:
                np.testing.assert_array_equal(z,zero_by_neuron[neuron])
            zero_by_neuron[neuron]=z
            mask=np.array([v is not None for v in trace['values']])
            w=trace['provenance']['id_confidence']
            totals+=w*np.array([np.sum(p[mask]**2),np.sum(z[mask]**2),np.sum(d[mask]**2),np.sum(2*z[mask]*d[mask])])
            weight+=w*mask.sum()
    assert max_error<1e-12
    expected=report['observed_weighted_mean_square']
    for key,value in zip(names+['cross_term'],totals/weight):
        assert abs(value-expected[key])<1e-12
    receipt={'schema_version':1,'native_report':report,'independent_audit':{'script_sha256':digest(__file__),'shared_score_script_sha256':digest(Path(__file__).with_name('audit_connectome_fit.py')),'max_decomposition_error':max_error,'zero_current_identical_across_stimulated_targets':True,'only_validation_trial_ids':True,'dataset_file_sha256':digest(args.data),'split_file_sha256':digest(args.split),'prediction_sha256':{name:digest(root/f'{name}-predictions.json') for name in names},'scope':'Independent saved-output decomposition, weighted mean squares and trace scores. No independent ODE replay or pair-label/AUROC audit in this script.'}}
    Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print('audited validation trials',len(split['validation']),'maximum decomposition error',max_error)


if __name__=='__main__':
    main()
