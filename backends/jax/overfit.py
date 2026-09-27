"""Training-only Level 0 capacity diagnostic. Never loads validation/test responses."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from objective import build, evaluate
from extensions import pack
from fit import checkpoint, make_optimizer


def prepare(model, training, targets, steps, rate, rest, preparation_seconds=None):
    """Keep an explicit subset lineage; these are diagnostic, not benchmark models."""
    if model['epoch'] != 0 or not model['config'].get('sign_initialization'):
        raise ValueError('requires epoch-zero, non-neutral sign initialization')
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('choose distinct training targets')
    if type(steps) is not int or steps < 1 or not np.isfinite(rate) or rate <= 0:
        raise ValueError('invalid optimization budget')
    if not np.isfinite(rest) or rest == 0:
        raise ValueError('rest must be finite and nonzero')
    for key in ['graph_hash', 'dataset_hash', 'split_hash', 'training_trials']:
        if model[key] != training[key]:
            raise ValueError('input lineage mismatch: ' + key)
    available = {training['names'][g['target']]: g for g in training['groups']}
    if any(t not in available for t in targets):
        raise ValueError('requested target is not in the training export')
    if preparation_seconds is not None and (not np.isfinite(preparation_seconds) or preparation_seconds <= 0):
        raise ValueError('preparation duration must be finite and positive')
    m, t = copy.deepcopy(model), copy.deepcopy(training)
    if preparation_seconds is not None:
        m['config']['preparation_seconds'] = float(preparation_seconds)
    t['groups'] = [copy.deepcopy(available[name]) for name in targets]
    trials = [trial for g in t['groups'] for trial in g['training_trials']]
    if len(set(trials)) != len(trials) or not set(trials) <= set(model['training_trials']):
        raise ValueError('invalid subset trial membership')
    m['training_trials'] = t['training_trials'] = sorted(trials)
    m['selection_trials'] = []
    # Capacity test isolates trace fitting from classification and regularization.
    m['classifier'] = None
    m['classification_evidence_hash'] = None
    t['classification_pairs'] = 0
    for g in t['groups']:
        g['labels'] = []
    m['config'].update(epochs=steps, learning_rate=rate,
        learning_rate_schedule={'kind':'constant'}, optimizer={'kind':'adam'},
        classification=None, correlation=None, prior_strength=0.,
        sign_prior_strength=0., kernel_prior_strength=0.,
        observation_gain={'initial_gain':10., 'prior_strength':0.})
    m['observation_log_gain'] = float(np.log(10.))
    for group in m['parameters']['groups']:
        if group['name'].startswith('rest/'):
            group['value'] = rest
    n = len(t['names'])
    m['initial'][:n] = [rest]*n
    configuration = {'schema_version':1, 'solver':None,
        'extensions':{'observation':{'initial_gain':10., 'prior_strength':0.}}}
    return m, t, configuration


def bounds(groups):
    floor = sum(float(g[3]) for g in groups)
    zero = floor + sum(float(jnp.sum(g[1]**2*g[2])) for g in groups)
    start_zero = floor + sum(float(jnp.sum(g[1][0]**2*g[2])) for g in groups)
    if zero <= start_zero:
        raise ValueError('no explainable response energy in chosen targets')
    return {'zero_response_mse':zero, 'mean_response_bound':floor,
            'start_zero_mean_response_bound':start_zero}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'graph', 'training', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--targets', nargs='+', required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--learning-rate', type=float, default=.01)
    p.add_argument('--rest', type=float, default=-.2)
    p.add_argument('--preparation-seconds', type=float, default=None)
    a = p.parse_args()
    paths = {k:Path(getattr(a,k)) for k in ['model','graph','training']}
    hashes = {k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    model, graph, training = [json.loads(paths[k].read_text()) for k in ['model','graph','training']]
    if hashes['model'] != training['model_sha256']:
        raise ValueError('training export belongs to another checkpoint')
    model, training, config = prepare(model, training, a.targets, a.steps, a.learning_rate, a.rest, a.preparation_seconds)
    theta, active, groups, data, prior = build(model, graph, training, config)
    reference = bounds(groups)
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    out = Path(a.output);out.mkdir(exist_ok=False)
    def write(name, value):
        with (out/name).open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
    write('manifest.json', {'format':'wormsim-training-capacity-diagnostic',
        'source_commit':source, 'source_worktree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'input_sha256':hashes, 'targets':a.targets, 'training_trials':training['training_trials'],
        'process_id':os.getpid(),
        'backend_source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob('*.py'))},
        'configuration':config, 'fit_config':model['config'], 'rest_initialization':a.rest,
        'bounds':reference, 'jax':jax.__version__, 'optax':optax.__version__,
        'devices':[str(d) for d in jax.devices()],
        'selection':'minimum training MSE, earliest tie; no held-out scoring',
        'scope':'Unregularized subset capacity diagnostic; checkpoints cannot be submitted as full-training benchmark fits.'})
    optimizer = make_optimizer(model['config'],active)
    state = optimizer.init(theta)
    best = float('inf');best_epoch = None;start = time.perf_counter()
    with (out/'progress.jsonl').open('x',buffering=1) as progress:
        for epoch in range(a.steps+1):
            value, gradient, metrics = evaluate(theta,groups,data,prior)
            if not np.isfinite(float(value)) or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)):
                raise ValueError(f'nonfinite objective/gradient at epoch {epoch}; prior progress retained')
            gains = np.exp(np.asarray(theta['observation']['log_gain']))
            captured = (reference['zero_response_mse']-metrics['mse'])/(reference['zero_response_mse']-reference['start_zero_mean_response_bound'])
            report = dict(epoch=epoch, **metrics, captured_start_zero_energy=captured,
                gradient_norm=float(optax.global_norm(gradient)), gain_min=float(gains.min()),
                gain_max=float(gains.max()), elapsed_seconds=time.perf_counter()-start)
            progress.write(json.dumps(report,allow_nan=False)+'\n')
            if metrics['mse'] < best:
                best=metrics['mse'];best_epoch=epoch
            if epoch%50 == 0 or epoch==a.steps:
                saved=pack(checkpoint(model,theta,epoch,source),theta,config)
                write(f'epoch-{epoch}.json', {'format':'wormsim-training-capacity-diagnostic',
                    'targets':a.targets, 'model':saved, 'metrics':report})
            if epoch%10 == 0 or epoch==a.steps:
                print(json.dumps(report,allow_nan=False),flush=True)
            if epoch<a.steps:
                gradient=jax.tree.map(lambda g,mask:jnp.where(mask,g,0.),gradient,active)
                updates,state=optimizer.update(gradient,state,theta)
                candidate=optax.apply_updates(theta,updates)
                theta=jax.tree.map(lambda new,old,mask:jnp.where(mask,new,old),candidate,theta,active)
    write('result.json',{'completed_steps':a.steps,'best_training_mse':best,'best_epoch':best_epoch,
        'final':report,'bounds':reference,'capacity_gate':captured>=.9,
        'capacity_gate_definition':'final iterate captures at least 90% of the zero-start mean-response energy; no generalization claim'})

if __name__=='__main__':
    main()
