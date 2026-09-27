"""Reproduce a terminated capacity fit to retain its nonfinite-step parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from overfit import prepare, warm_parameters
from objective import build, evaluate
from fit import make_optimizer, checkpoint
from extensions import pack


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','graph','training','parent','reference-run','output']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args();reference=Path(a.reference_run)
    manifest=json.loads((reference/'manifest.json').read_text())
    paths={k:Path(getattr(a,k)) for k in ['model','graph','training','parent']}
    for key,path in paths.items():
        expected=manifest['input_sha256'][{'graph':'graph','parent':'warm_start'}.get(key,key)]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('reference input hash differs: '+key)
    for name,digest in manifest['backend_source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('reference backend source differs: '+name)
    recorded=[json.loads(line) for line in (reference/'progress.jsonl').read_text().splitlines()]
    if not recorded or [r['epoch'] for r in recorded]!=list(range(len(recorded))):
        raise ValueError('reference history is empty or incomplete')
    if (reference/'result.json').exists():raise ValueError('reference run completed normally')
    model,graph,training,parent=[json.loads(paths[k].read_text()) for k in ['model','graph','training','parent']]
    c=manifest['fit_config']
    model,training,configuration=prepare(model,training,manifest['targets'],c['epochs'],c['learning_rate'],manifest['rest_initialization'],c['preparation_seconds'])
    if model['config']!=c or configuration!=manifest['configuration']:raise ValueError('reference configuration differs')
    _,active,groups,data,prior=build(model,graph,training,configuration)
    theta=warm_parameters(parent,model,graph,training,configuration,manifest['targets'])
    optimizer=make_optimizer(c,active);state=optimizer.init(theta)
    out=Path(a.output);out.mkdir(exist_ok=False)
    history=[]
    for epoch in range(len(recorded)+1):
        value,gradient,metrics=evaluate(theta,groups,data,prior)
        finite_value=bool(np.isfinite(float(value)))
        invalid=sum(int(np.count_nonzero(~np.isfinite(np.asarray(g)))) for g in jax.tree.leaves(gradient))
        row={'epoch':epoch,'objective_finite':finite_value,'nonfinite_gradient_coordinates':invalid,
             'metrics':{k:v if np.isfinite(v) else None for k,v in metrics.items()}}
        history.append(row)
        if epoch<len(recorded):
            if not finite_value or invalid or abs(metrics['mse']-recorded[epoch]['mse'])>1e-10:
                raise ValueError('reference trajectory did not reproduce at epoch '+str(epoch))
        else:
            if finite_value and invalid==0:raise ValueError('reference nonfinite step did not reproduce')
            snapshot={'format':'wormsim-capacity-failure-replay','epoch':epoch,
                'model':pack(checkpoint(model,theta,epoch,manifest['source_commit']),theta,configuration),
                'warm_start':manifest['warm_start'],'reference_manifest_sha256':hashlib.sha256((reference/'manifest.json').read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'matched_finite_epochs':len(recorded),'history':history,
                'scope':'Diagnostic replay of the observed failed run with the original backend source and inputs. Finite parameters retained before evaluation of the reproduced nonfinite step; not a successful fit.'}
            with (out/'failure.json').open('x') as f:json.dump(snapshot,f,indent=2,allow_nan=False)
            print(json.dumps(row),flush=True)
            return
        gradient=jax.tree.map(lambda g,mask:jnp.where(mask,g,0.),gradient,active)
        updates,state=optimizer.update(gradient,state,theta)
        candidate=optax.apply_updates(theta,updates)
        theta=jax.tree.map(lambda new,old,mask:jnp.where(mask,new,old),candidate,theta,active)
        if epoch%5==0:print('Matched finite epoch',epoch,flush=True)

if __name__=='__main__':main()
