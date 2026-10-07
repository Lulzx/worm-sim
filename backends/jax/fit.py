"""Population atlas fitting with Optax; Rust alone scores validation checkpoints."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from objective import build, evaluate
from extensions import pack
from optimization import rate_multipliers


def make_optimizer(config, active, multipliers=None):
    epochs=config['epochs']
    if not isinstance(epochs,int) or epochs<0:
        raise ValueError('epochs must be nonnegative')
    rate=config['learning_rate']
    if not np.isfinite(rate) or rate<=0:
        raise ValueError('invalid learning rate')
    schedule=config.get('learning_rate_schedule',{'kind':'constant'})
    if schedule['kind']=='cosine':
        minimum=schedule['minimum_fraction']
        if not np.isfinite(minimum) or not 0<=minimum<=1:
            raise ValueError('invalid cosine minimum')
        if epochs>1:
            rate=optax.cosine_decay_schedule(rate,epochs-1,alpha=minimum)
    elif schedule['kind']!='constant':
        raise ValueError('unsupported schedule')
    maximum_multiplier=1.
    if multipliers is not None:
        if jax.tree.structure(multipliers)!=jax.tree.structure(active):
            raise ValueError('learning-rate multiplier structure mismatch')
        for scale,mask in zip(jax.tree.leaves(multipliers),jax.tree.leaves(active),strict=True):
            values=np.asarray(scale);mask=np.asarray(mask)
            if values.shape!=mask.shape or not np.isfinite(values).all() or np.any(values<0):
                raise ValueError('invalid learning-rate multiplier array')
        maximum_multiplier=max((float(np.max(np.asarray(s)[np.asarray(a)])) for s,a in zip(jax.tree.leaves(multipliers),jax.tree.leaves(active),strict=True) if np.asarray(a).any()),default=0.)
    if not np.isfinite(config['learning_rate']*maximum_multiplier):
        raise ValueError('effective learning rate is nonfinite')
    optimizer=config.get('optimizer',{'kind':'adam'})
    if optimizer['kind']=='adam':
        algorithm=optax.adam(rate,b1=.9,b2=.999,eps=1e-8)
    elif optimizer['kind']=='adamw':
        decay=optimizer['weight_decay']
        if not np.isfinite(decay) or decay<0 or config['learning_rate']*decay*max(1.,maximum_multiplier)>1:
            raise ValueError('invalid decoupled decay')
        algorithm=optax.adamw(rate,b1=.9,b2=.999,eps=1e-8,weight_decay=decay)
    else:
        raise ValueError('unsupported optimizer')
    return optax.chain(optax.clip_by_global_norm(10.),algorithm)


def checkpoint(template, theta, epoch, source):
    out=copy.deepcopy(template)
    if theta['groups'].shape!=(len(out['parameters']['groups']),) or theta['kernel'].shape!=(len(out['kernel_raw']),):
        raise ValueError('checkpoint parameter dimensions changed')
    for group,value in zip(out['parameters']['groups'],np.asarray(theta['groups']),strict=True):
        if not group['trainable'] and float(value)!=group['value']:
            raise ValueError('frozen coordinate changed')
        group['value']=float(value)
    out['kernel_raw']=np.asarray(theta['kernel']).tolist()
    if out.get('classifier'):
        out['classifier'].update(bias=float(theta['classifier'][0]),raw_slope=float(theta['classifier'][1]))
    if out.get('observation_log_gain') is not None:
        out['observation_log_gain']=float(theta['log_gain'])
    out.update(epoch=epoch,source_commit=source)
    json.dumps(out,allow_nan=False)
    return out


def fit(model,graph,training,source,score_and_save,progress=None,configuration=None):
    if training.get('synthetic') is not None:
        raise ValueError('planted-truth exports are diagnostics, never benchmark fitting inputs')
    if model['epoch']!=0:
        raise ValueError('fit requires epoch zero; optimizer resume is not implemented')
    theta,active,groups,data,prior=build(model,graph,training,configuration)
    multipliers=rate_multipliers(model,theta,configuration)
    optimizer=make_optimizer(model['config'],active,multipliers)
    state=optimizer.init(theta)
    reports=[];selected=None;best=float('inf')
    for epoch in range(model['config']['epochs']+1):
        start=time.perf_counter();metrics=None
        if epoch:
            value,gradient,metrics=evaluate(theta,groups,data,prior,progress)
            if not np.isfinite(float(value)) or not all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradient)):
                raise ValueError('nonfinite objective or gradient')
            gradient=jax.tree.map(lambda g,a:jnp.where(a,g,0.),gradient,active)
            updates,state=optimizer.update(gradient,state,theta)
            updates=jax.tree.map(lambda update,scale:update*scale,updates,multipliers)
            updated=optax.apply_updates(theta,updates)
            theta=jax.tree.map(lambda new,old,a:jnp.where(a,new,old),updated,theta,active)
        candidate=checkpoint(model,theta,epoch,source)
        if configuration is not None:
            candidate=pack(candidate,theta,configuration)
        score=score_and_save(candidate)
        if not np.isfinite(score):
            raise ValueError('nonfinite validation MSE')
        if score<best:
            best=score;selected=candidate
        reports.append({'epoch':epoch,'validation_mse':score,'preceding_training_components':metrics,'elapsed_seconds':time.perf_counter()-start})
    return selected,reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','graph-json','graph','data','split','training','scorer','output']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args()
    paths={k:Path(getattr(a,k)) for k in ['model','graph_json','graph','data','split','training','scorer']}
    hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    model,graph,training=[json.loads(paths[k].read_text()) for k in ['model','graph_json','training']]
    if hashes['model']!=training['model_sha256']:
        raise ValueError('training export belongs to another checkpoint')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    out=Path(a.output);out.mkdir(exist_ok=False)
    def write(path,value):
        with open(path,'x') as f:
            json.dump(value,f,allow_nan=False,indent=2)
    write(out/'manifest.json',{'source_commit':source,'source_worktree_dirty':dirty,'input_sha256':hashes,'selection':'minimum Rust validation MSE; earliest epoch breaks ties','jax':jax.__version__,'optax':optax.__version__,'devices':[str(d) for d in jax.devices()],'scope':'New fit from saved epoch-zero initialization; no test scoring or optimizer resume.'})
    def score(candidate):
        epoch=candidate['epoch'];path=out/f'epoch-{epoch}.json'
        write(path,candidate)
        evaluation=out/f'validation-{epoch}'
        subprocess.run([str(paths['scorer'].resolve()),str(paths['graph']),str(paths['data']),str(paths['split']),str(path),str(evaluation)],check=True)
        result=json.loads((evaluation/'selection.json').read_text())
        if result['epoch']!=epoch or result['partition']!='validation':
            raise ValueError('unexpected scorer response')
        return result['validation_mse']
    selected,reports=fit(model,graph,training,source,score,lambda i,n:print(f'target {i}/{n}',flush=True) if i%20==0 or i==n else None)
    write(out/'selected.json',selected)
    write(out/'selection.json',reports)
    print(f"selected epoch {selected['epoch']}",flush=True)

if __name__=='__main__':
    main()
