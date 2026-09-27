"""Compute a frozen training objective and its automatic gradient, without fitting."""
import argparse
import hashlib
import subprocess
import optax
import jax.numpy as jnp
import json
import time
from pathlib import Path
import jax
import numpy as np
from objective import build, evaluate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['model','graph','training','output']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--reference-next',help='Optional frozen Rust epoch-one checkpoint for first Adam update parity')
    a=parser.parse_args()
    start=time.perf_counter()
    model,graph,training=[json.loads(Path(p).read_text()) for p in [a.model,a.graph,a.training]]
    hashes={k:hashlib.sha256(Path(getattr(a,k)).read_bytes()).hexdigest() for k in ['model','graph','training']}
    if training['model_sha256']!=hashes['model']:
        raise ValueError('training export belongs to another checkpoint')
    theta,active,groups,data,prior=build(model,graph,training)
    value,gradient,metrics=evaluate(theta,groups,data,prior,lambda i,n: print(f"target {i}/{n}",flush=True) if i%20==0 or i==n else None)
    leaves=jax.tree.leaves(gradient)
    if not all(np.isfinite(np.asarray(v)).all() for v in leaves):
        raise ValueError('nonfinite gradient')
    report={'objective':float(value),'components':metrics,'gradient_l2':float(np.sqrt(sum(np.sum(np.asarray(g)**2) for g in leaves))), 'seconds_including_compilation':time.perf_counter()-start,'groups':len(groups),'scope':'Training-only frozen objective and reverse gradient; no optimizer updates or validation/test selection.'}
    report.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_worktree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),inputs_sha256=hashes)
    if a.reference_next:
        config=model['config']
        if model['epoch']!=0 or config.get('optimizer',{'kind':'adam'})['kind']!='adam':
            raise ValueError('first-update parity requires epoch zero and Adam')
        optimizer=optax.chain(optax.clip_by_global_norm(10.),optax.adam(config['learning_rate'],b1=.9,b2=.999,eps=1e-8))
        updates,_=optimizer.update(gradient,optimizer.init(theta),theta)
        updated=optax.apply_updates(theta,updates)
        updated=jax.tree.map(lambda v,old,a:jnp.where(a,v,old),updated,theta,active)
        next_model=json.loads(Path(a.reference_next).read_text())
        if next_model['epoch']!=1 or next_model['config']!=model['config']:
            raise ValueError('reference is not epoch one of the same configuration')
        from level0 import parameters
        expected=parameters(next_model)
        if next_model.get('classifier'):
            expected['classifier']=jnp.asarray([next_model['classifier']['bias'],next_model['classifier']['raw_slope']])
        errors={k:float(np.max(np.abs(np.asarray(updated[k])-np.asarray(expected[k])))) for k in updated}
        report['first_adam_update_maximum_absolute_errors']=errors
        report['reference_next_sha256']=hashlib.sha256(Path(a.reference_next).read_bytes()).hexdigest()
        if max(errors.values())>1e-8:
            raise ValueError(f'first update differs from Rust: {errors}')
    with open(a.output,'x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report))
if __name__=='__main__':
    main()
