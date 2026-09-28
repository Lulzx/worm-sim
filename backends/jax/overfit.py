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
from extensions import pack, restore
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



def warm_parameters(saved, model, graph, training, configuration, targets, allow_step_change=False, allow_preparation_change=False, initial_override=None):
    """Reuse only fitted parameters; keep the new run's initialization and objective."""
    if saved.get('format') != 'wormsim-training-capacity-diagnostic' or saved.get('targets') != targets:
        raise ValueError('warm start requires the same diagnostic target subset')
    packed = saved['model']
    if packed['configuration'] != configuration:
        raise ValueError('warm-start configuration differs')
    base = packed['base_model']
    if type(base['epoch']) is not int or base['epoch'] < 0 or saved['metrics']['epoch'] != base['epoch']:
        raise ValueError('invalid warm-start epoch')
    _, theta, _ = restore(packed, graph, training['groups'][0]['recording']['times'])
    # This also rejects modifications to coordinates frozen in the original model.
    reconstructed = checkpoint(model, theta, base['epoch'], base['source_commit'])
    expected = copy.deepcopy(base)
    for key in ['epochs', 'learning_rate']:
        expected['config'][key] = model['config'][key]
    if allow_step_change:
        dt = model['config']['dt']
        if not np.isfinite(dt) or dt <= 0 or dt > base['config']['dt']:
            raise ValueError('warm-start step override must be positive and no larger than parent')
        expected['config']['dt'] = dt
    if allow_preparation_change:
        duration = model['config']['preparation_seconds']
        if not np.isfinite(duration) or duration <= 0 or duration < base['config']['preparation_seconds']:
            raise ValueError('warm-start preparation override must be positive and no shorter than parent')
        expected['config']['preparation_seconds'] = duration
    if initial_override is not None:
        seed = np.asarray(initial_override, dtype=float)
        if seed.shape != np.asarray(base['initial']).shape or not np.isfinite(seed).all():
            raise ValueError('initial override must be a finite state with unchanged dimensions')
        expected['initial'] = seed.tolist()
    if reconstructed != expected:
        raise ValueError('warm-start lineage, objective, initialization, or parameter layout differs')
    return theta


def atomic_best(path, value):
    """Replace only the best artifact in this run's newly created output directory."""
    encoded = json.dumps(value, indent=2, allow_nan=False)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)



def failure_artifact(model, theta, config, epoch, source, value, gradient, metrics, targets, warm_info):
    """Keep finite failing parameters without serializing NaNs as valid checkpoints."""
    invalid_parameters = sum(int(np.count_nonzero(~np.isfinite(np.asarray(x)))) for x in jax.tree.leaves(theta))
    invalid_gradients = sum(int(np.count_nonzero(~np.isfinite(np.asarray(x)))) for x in jax.tree.leaves(gradient))
    return {'format':'wormsim-capacity-failure', 'epoch':epoch, 'targets':targets,
        'model':None if invalid_parameters else pack(checkpoint(model,theta,epoch,source),theta,config),
        'warm_start':warm_info, 'objective_finite':bool(np.isfinite(float(value))),
        'nonfinite_parameter_coordinates':invalid_parameters,
        'nonfinite_gradient_coordinates':invalid_gradients,
        'metrics':{key:float(v) if np.isfinite(float(v)) else None for key,v in metrics.items()},
        'scope':'Failed evaluation, not a completed fit or eligible best checkpoint. Parameters omitted if nonfinite.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'graph', 'training', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--targets', nargs='+', required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--learning-rate', type=float, default=.01)
    p.add_argument('--rest', type=float, default=-.2)
    p.add_argument('--preparation-seconds', type=float, default=None)
    p.add_argument('--dt', type=float, help='Explicit Euler step refinement; distinct numerical configuration')
    p.add_argument('--warm-start', help='Diagnostic checkpoint; reuses parameters with fresh Adam state')
    a = p.parse_args()
    paths = {k:Path(getattr(a,k)) for k in ['model','graph','training']}
    hashes = {k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    model, graph, training = [json.loads(paths[k].read_text()) for k in ['model','graph','training']]
    if hashes['model'] != training['model_sha256']:
        raise ValueError('training export belongs to another checkpoint')
    model, training, config = prepare(model, training, a.targets, a.steps, a.learning_rate, a.rest, a.preparation_seconds)
    if a.dt is not None:
        if not np.isfinite(a.dt) or a.dt <= 0 or a.dt > model['config']['dt']:
            raise ValueError('dt must be positive and no larger than the source step')
        model['config']['dt'] = a.dt
    theta, active, groups, data, prior = build(model, graph, training, config)
    warm = None; warm_info = None
    if a.warm_start:
        raw = Path(a.warm_start).read_bytes()
        hashes['warm_start'] = hashlib.sha256(raw).hexdigest()
        warm = json.loads(raw)
        theta = warm_parameters(warm, model, graph, training, config, a.targets, allow_step_change=a.dt is not None)
        warm_info = {'checkpoint_sha256':hashes['warm_start'],
            'source_epoch':warm['model']['base_model']['epoch'],
            'source_commit':warm['model']['base_model']['source_commit'],
            'parent_dt':warm['model']['base_model']['config']['dt'],
            'run_dt':model['config']['dt'],
            'parent_training_mse':warm['metrics']['mse'],
            'optimizer_state':'reset', 'epoch_convention':'additional updates in this run'}
    reference = bounds(groups)
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    out = Path(a.output);out.mkdir(exist_ok=False)
    def write(name, value):
        with (out/name).open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
    write('manifest.json', {'format':'wormsim-training-capacity-diagnostic',
        'source_commit':source, 'source_worktree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'input_sha256':hashes, 'targets':a.targets, 'training_trials':training['training_trials'],
        'warm_start':warm_info,
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
                write('failure.json', failure_artifact(model,theta,config,epoch,source,value,gradient,metrics,a.targets,warm_info))
                raise ValueError(f'nonfinite objective/gradient at epoch {epoch}; failure.json and prior progress retained')
            if epoch == 0 and warm is not None and warm_info['parent_dt'] == warm_info['run_dt']:
                parent_mse = warm['metrics']['mse']
                if not np.isfinite(parent_mse) or abs(metrics['mse'] - parent_mse) > 1e-10:
                    raise ValueError('warm-start initial score differs from parent checkpoint')
            gains = np.exp(np.asarray(theta['observation']['log_gain']))
            captured = (reference['zero_response_mse']-metrics['mse'])/(reference['zero_response_mse']-reference['start_zero_mean_response_bound'])
            report = dict(epoch=epoch, **metrics, captured_start_zero_energy=captured,
                gradient_norm=float(optax.global_norm(gradient)), gain_min=float(gains.min()),
                gain_max=float(gains.max()), elapsed_seconds=time.perf_counter()-start)
            progress.write(json.dumps(report,allow_nan=False)+'\n')
            improved = metrics['mse'] < best
            if improved or epoch%50 == 0 or epoch==a.steps:
                saved=pack(checkpoint(model,theta,epoch,source),theta,config)
                artifact={'format':'wormsim-training-capacity-diagnostic',
                    'targets':a.targets, 'model':saved, 'metrics':report, 'warm_start':warm_info}
                if improved:
                    atomic_best(out/'best.json', artifact)
                    best=metrics['mse'];best_epoch=epoch
                if epoch%50 == 0 or epoch==a.steps:
                    write(f'epoch-{epoch}.json', artifact)
            if epoch%10 == 0 or epoch==a.steps:
                print(json.dumps(report,allow_nan=False),flush=True)
            if epoch<a.steps:
                gradient=jax.tree.map(lambda g,mask:jnp.where(mask,g,0.),gradient,active)
                updates,state=optimizer.update(gradient,state,theta)
                candidate=optax.apply_updates(theta,updates)
                theta=jax.tree.map(lambda new,old,mask:jnp.where(mask,new,old),candidate,theta,active)
    write('result.json',{'completed_steps':a.steps,'best_training_mse':best,'best_epoch':best_epoch,
        'best_checkpoint_sha256':hashlib.sha256((out/'best.json').read_bytes()).hexdigest(),
        'warm_start':warm_info,
        'final':report,'bounds':reference,'capacity_gate':captured>=.9,
        'capacity_gate_definition':'final iterate captures at least 90% of the zero-start mean-response energy; no generalization claim'})

if __name__=='__main__':
    main()
