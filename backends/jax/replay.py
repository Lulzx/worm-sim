"""Recompute a frozen atlas prediction artifact with JAX; never fit held-out data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import jax
import diffrax
import equinox
from level0 import Level0, parameters, response


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--graph',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)
    out.mkdir(exist_ok=False)
    model,graph,data,reference=map(load,[a.model,a.graph,a.data,a.reference])
    for key in ['graph_hash','dataset_hash','split_hash','training_trials','selection_trials']:
        if key in reference and reference[key]!=model[key]:
            raise ValueError(f'model/prediction mismatch: {key}')
    if data['graph_hash']!=model['graph_hash']:
        raise ValueError('data graph mismatch')
    indexed={t['id']:t for t in data['trials']}
    names=sorted(n['id'] for n in graph['neurons'])
    times=reference['trials'][0]['times']
    engine=Level0(model,graph,times)
    theta=parameters(model)
    predictions={k:v for k,v in reference.items() if k!='trials'}
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    predictions.update(model=reference['model']+'-jax-diffrax-replay',source_commit=source,trials=[])
    cache={}
    maximum=0.
    seconds=[]
    for trial in reference['trials']:
        if trial['times']!=times or indexed[trial['id']]['recording']['times']!=times:
            raise ValueError('this replay requires one common time grid')
        target=indexed[trial['id']]['stimulated_neuron']
        if target not in cache:
            start=time.perf_counter()
            cache[target]=np.asarray(response(engine,theta,jax.numpy.asarray(names.index(target))))
            seconds.append(time.perf_counter()-start)
            if not np.isfinite(cache[target]).all():
                raise ValueError('nonfinite JAX response')
            print(f'{target}: {seconds[-1]:.3f}s',flush=True)
        values={name:cache[target][:,names.index(name)].tolist() for name in trial['fluorescence']}
        maximum=max(maximum,max(float(np.max(np.abs(np.asarray(values[n])-v))) for n,v in trial['fluorescence'].items()))
        predictions['trials'].append({'id':trial['id'],'times':times,'fluorescence':values,'response_scores':{}})
    (out/'predictions.json').write_text(json.dumps(predictions,allow_nan=False))
    receipt={'schema_version':1,'source_commit':source,
             'source_worktree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
             'inputs':{k:digest(getattr(a,k)) for k in ['model','graph','data','reference']},
             'prediction_sha256':digest(out/'predictions.json'),
             'jax':jax.__version__,'diffrax':diffrax.__version__,'equinox':equinox.__version__,
             'devices':[str(d) for d in jax.devices()], 'precision':'float64',
             'targets':len(cache),'trials':len(predictions['trials']),
             'maximum_absolute_prediction_error':maximum,'target_seconds_including_first_compile':seconds,
             'scope':'Frozen parameter forward replay, same Euler grid; no refit, no new biological result, no optimizer parity claim. Reference predictions used only for output schema and post-hoc numerical comparison.'}
    (out/'replay.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    if maximum>1e-9:
        raise ValueError(f'forward parity failed: {maximum}')
    print(f'maximum error: {maximum:.3g}')

if __name__=='__main__':
    main()
