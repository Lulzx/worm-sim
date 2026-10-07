"""Planted-truth controls for the training capacity gate. Training trials only.

`noise` measures how much of a target subset's mean-response energy is trial
sampling noise, which no deterministic shared-response model can capture.
`make` replaces those targets' mean responses with a known model's response,
optionally plus real trial residuals, and writes a marked synthetic export that
`overfit.py` fits unchanged. `score` compares a fitted checkpoint with the
planted signal. Synthetic exports never enter benchmark fitting (fit.py rejects them).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
from extensions import initialize, restore
from level0 import response
from overfit import prepare

FORMAT = 'wormsim-planted-truth'


def trial_residuals(group, trials, names):
    """Per neuron: confidence weights and residuals of training trials about the export mean."""
    mean = {t['neuron']: np.asarray(t['values'], float) for t in group['recording']['traces']}
    per = {}
    for trial_id in group['training_trials']:
        trial = trials[trial_id]
        if trial['stimulated_neuron'] != names[group['target']]:
            raise ValueError('trial does not belong to this target')
        for trace in trial['recording']['traces']:
            w = trace['provenance']['id_confidence']
            if w > 0:
                per.setdefault(trace['neuron'], []).append((w, np.asarray(trace['values'], float)))
    if set(per) != set(mean):
        raise ValueError('trial neurons differ from export traces')
    out = {}
    for neuron, obs in per.items():
        w = np.asarray([o[0] for o in obs]); y = np.stack([o[1] for o in obs])
        m = (w[:, None]*y).sum(0)/w.sum()
        if not np.allclose(m, mean[neuron], rtol=0, atol=1e-9):
            raise ValueError('trial data do not reproduce the export mean')
        out[neuron] = (w, y-m)
    return out


def noise_variance(w, r):
    """Unbiased per-sample variance of the weighted mean; None for single trials."""
    if len(w) < 2:
        return None
    W = w.sum(); k = (w**2).sum()/W**2
    return (w[:, None]*r**2).sum(0)/(W-(w**2).sum()/W)*k


def group_weights(groups):
    """Per-trace objective weights, exactly as objective.build normalizes them."""
    total = sum(g['sample_weight'] for g in groups)
    out = []
    for g in groups:
        conf = np.asarray([t['provenance']['id_confidence'] for t in g['recording']['traces']])
        scale = g['sample_weight']/total
        out.append((conf/(conf.sum()*len(g['recording']['times']))*scale, scale*g['irreducible_mse']))
    return out


def energies(groups):
    """Zero-response MSE, start-zero bound and explainable energy (overfit.bounds)."""
    zero = start = 0.
    for g, (w, floor) in zip(groups, group_weights(groups)):
        y = np.asarray([t['values'] for t in g['recording']['traces']])
        zero += floor+(w[:, None]*y**2).sum()
        start += floor+(w*y[:, 0]**2).sum()
    return zero, start


def noise_ceiling(training, data, targets):
    names = training['names']
    by_name = {names[g['target']]: g for g in training['groups']}
    if any(t not in by_name for t in targets):
        raise ValueError('requested target is not in the training export')
    groups = [by_name[t] for t in targets]
    trials = {t['id']: t for t in data['trials']}
    zero, start = energies(groups)
    noise = 0.; rows = []
    for g, (w, _) in zip(groups, group_weights(groups)):
        res = trial_residuals(g, trials, names)
        variances = {n: noise_variance(*res[n]) for n in res}
        pooled = np.mean([v for v in variances.values() if v is not None], axis=0)
        e = 0.; single = 0
        for trace, wi in zip(g['recording']['traces'], w):
            v = variances[trace['neuron']]
            if v is None:
                single += 1; v = pooled
            e += wi*v.sum()
        noise += e
        rows.append({'target': names[g['target']], 'trials': len(g['training_trials']),
            'observed_neurons': len(w), 'single_trial_neurons': single, 'noise_energy': e})
    explainable = zero-start
    return {'targets': targets, 'zero_response_mse': zero, 'start_zero_mean_response_bound': start,
        'explainable_energy': explainable, 'noise_energy': noise, 'noise_share': noise/explainable,
        'expected_truth_capture': 1-noise/explainable, 'groups': rows,
        'assumptions': 'Noise energy is the expected squared sampling error of each confidence-weighted '
            'trial mean (unbiased weighted variance x sum(w^2)/W^2), summed with objective weights. '
            'Single-trial neurons use the target\'s pooled per-sample variance. Trial-to-trial and '
            'animal-to-animal variation both count as noise for a shared deterministic response.'}


def predict(engine, theta, groups, names):
    index = {n: i for i, n in enumerate(names)}
    out = []
    for g in groups:
        p = np.asarray(response(engine, theta, jnp.asarray(g['target'])))
        if not np.isfinite(p).all():
            raise ValueError('nonfinite planted response')
        out.append(np.stack([p[:, index[t['neuron']]] for t in g['recording']['traces']]))
    return out


def metrics(groups, prediction, signal=None):
    """Training MSE and capture as overfit.py reports them; signal recovery if planted."""
    zero, start = energies(groups)
    mse = 0.; err = energy = absorbed = noise = 0.
    for g, p, (w, floor), k in zip(groups, prediction, group_weights(groups), range(len(groups))):
        y = np.asarray([t['values'] for t in g['recording']['traces']])
        mse += floor+(w[:, None]*(p-y)**2).sum()
        if signal is not None:
            s = signal[k]; e = y-s
            err += (w[:, None]*(p-s)**2).sum(); energy += (w[:, None]*s**2).sum()
            absorbed += (w[:, None]*(p-s)*e).sum(); noise += (w[:, None]*e**2).sum()
    out = {'mse': mse, 'captured_start_zero_energy': (zero-mse)/(zero-start)}
    if signal is not None:
        out.update(signal_recovery=1-err/energy,
            noise_absorbed=absorbed/noise if noise > 0 else None, signal_energy=energy, noise_energy=noise)
    return out


def truth_engine(truth, graph, times, model, configuration, extension):
    """Fitted-class engine with the truth's parameters, or the same plus a fixed extension."""
    packed = truth['model'] if truth.get('format') == 'wormsim-training-capacity-diagnostic' else truth
    base = packed['base_model']
    for key in ['dt', 'preparation_seconds']:
        if base['config'][key] != model['config'][key]:
            raise ValueError(f'truth numerical configuration differs: {key}')
    if [g['name'] for g in base['parameters']['groups']] != [g['name'] for g in model['parameters']['groups']]:
        raise ValueError('truth parameter layout differs from the fitted model')
    if packed['configuration'] != configuration:
        raise ValueError('truth configuration differs from the fit configuration')
    engine, theta, _ = restore(packed, graph, times)
    if extension is None:
        return engine, theta
    name, spec = extension
    extended = copy.deepcopy(configuration); extended['extensions'][name] = spec
    engine, fresh, _ = initialize(base, graph, times, extended)
    for key in theta:
        fresh[key] = theta[key]
    return engine, fresh


def make(model, graph, training, data, targets, preparation, truth, extension, noise, seed, signal_energy=None):
    m, t, configuration = prepare(model, training, targets, 1, .01, -.2, preparation)
    times = t['groups'][0]['recording']['times']
    engine, theta = truth_engine(truth, graph, times, m, configuration, extension)
    signal = predict(engine, theta, t['groups'], t['names'])
    gain = 1.
    if signal_energy is not None:
        # Response is linear in the observation gain, so one shared factor matches
        # energy exactly and stays inside the truth's parameter class.
        current = sum((w[:, None]*x**2).sum() for x, (w, _) in zip(signal, group_weights(t['groups'])))
        if not np.isfinite(signal_energy) or signal_energy <= 0 or current <= 0:
            raise ValueError('invalid signal energy target')
        gain = float(np.sqrt(signal_energy/current)); signal = [x*gain for x in signal]
    real_zero, real_start = energies(t['groups'])
    trials = {x['id']: x for x in data['trials']} if noise == 'residual' else None
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(training)
    out['groups'] = []
    for g, s in zip(t['groups'], signal):
        g = copy.deepcopy(g)
        if noise == 'residual':
            res = trial_residuals(g, trials, t['names'])
            variances = [noise_variance(*res[n]) for n in res]
            pooled = np.mean([v for v in variances if v is not None], axis=0)
        for i, trace in enumerate(g['recording']['traces']):
            values = s[i].copy()
            if noise == 'residual':
                w, r = res[trace['neuron']]
                if len(w) > 1:
                    # Random-sign weighted residual mean, corrected so its variance
                    # equals the sampling variance of the real mean (exact for equal weights).
                    k = (w**2).sum()/w.sum()**2
                    eps = rng.choice([-1., 1.], size=len(w))
                    values += (w*eps) @ r/w.sum()/np.sqrt(1-k)
                else:
                    values += rng.normal(size=len(values))*np.sqrt(pooled)
            trace['values'] = values.tolist()
        g['planted_signal'] = s.tolist()
        out['groups'].append(g)
    zero, start = energies(out['groups'])
    oracle = metrics(out['groups'], signal, signal)
    return out, {'targets': targets, 'noise': noise, 'seed': seed,
        'extension': None if extension is None else {'name': extension[0], 'spec': extension[1]},
        'preparation_seconds': m['config']['preparation_seconds'], 'dt': m['config']['dt'],
        'signal_gain_factor': gain,
        'real_explainable_energy': real_zero-real_start, 'synthetic_explainable_energy': zero-start,
        'oracle': oracle,
        'scope': 'Synthetic planted-truth targets for training-capacity diagnosis. Not data; never a benchmark input.'}


def sign_agreement(model, truth_theta, fit_theta):
    names = [g['name'] for g in model['parameters']['groups']]
    signs = np.asarray([n.startswith('chemical_sign/') for n in names])
    if not signs.any():
        return None
    a = np.asarray(truth_theta['groups'])[signs]; b = np.asarray(fit_theta['groups'])[signs]
    return float(np.mean(np.sign(a) == np.sign(b)))


def score(graph, export, checkpoint, truth=None):
    planted = export.get('synthetic')
    if not planted or planted.get('format') != FORMAT:
        raise ValueError('score requires a planted-truth export')
    packed = checkpoint['model'] if checkpoint.get('format') == 'wormsim-training-capacity-diagnostic' else checkpoint
    if checkpoint.get('targets', planted['targets']) != planted['targets']:
        raise ValueError('checkpoint targets differ from the planted export')
    names = export['names']
    by_name = {names[g['target']]: g for g in export['groups']}
    groups = [by_name[t] for t in planted['targets']]
    times = groups[0]['recording']['times']
    engine, theta, _ = restore(packed, graph, times)
    signal = [np.asarray(g['planted_signal']) for g in groups]
    out = metrics(groups, predict(engine, theta, groups, names), signal)
    out['oracle'] = planted['oracle']
    if truth is not None:
        tp = truth['model'] if truth.get('format') == 'wormsim-training-capacity-diagnostic' else truth
        _, truth_theta, _ = restore(tp, graph, times)
        out['chemical_sign_agreement'] = sign_agreement(packed['base_model'], truth_theta, theta)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    n = sub.add_parser('noise'); m = sub.add_parser('make'); s = sub.add_parser('score')
    for q in (n, m):
        q.add_argument('--training', required=True); q.add_argument('--data', required=True)
        q.add_argument('--targets', nargs='+', required=True); q.add_argument('--output', required=True)
    for key in ['model', 'graph', 'truth']:
        m.add_argument('--'+key, required=True)
    m.add_argument('--preparation-seconds', type=float, default=None)
    m.add_argument('--extension', help='JSON {"name": ..., "spec": ...} added to the truth only')
    m.add_argument('--noise', choices=['none', 'residual'], required=True)
    m.add_argument('--seed', type=int, default=0)
    m.add_argument('--signal-energy', type=float, help='Rescale the planted signal to this weighted energy')
    for key in ['graph', 'export', 'checkpoint', 'output']:
        s.add_argument('--'+key, required=True)
    s.add_argument('--truth')
    a = p.parse_args()
    keys = {'noise': ['training', 'data'], 'make': ['model', 'graph', 'training', 'data', 'truth'],
        'score': ['graph', 'export', 'checkpoint']}[a.command]
    if a.command == 'make' and a.extension:
        keys = keys+['extension']
    if a.command == 'score' and a.truth:
        keys = keys+['truth']
    raw = {k: Path(getattr(a, k)).read_bytes() for k in keys}
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
    loaded = {k: json.loads(v) for k, v in raw.items()}
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    provenance = {'source_commit': source, 'source_worktree_dirty': dirty, 'input_sha256': hashes,
        'jax': jax.__version__, 'devices': [str(d) for d in jax.devices()]}
    if a.command == 'noise':
        result = dict(format=FORMAT+'-noise-ceiling', **provenance,
            **noise_ceiling(loaded['training'], loaded['data'], a.targets))
    elif a.command == 'make':
        if hashes['model'] != loaded['training']['model_sha256']:
            raise ValueError('training export belongs to another checkpoint')
        extension = None
        if a.extension:
            e = loaded['extension']; extension = (e['name'], e['spec'])
        result, block = make(loaded['model'], loaded['graph'], loaded['training'], loaded['data'],
            a.targets, a.preparation_seconds, loaded['truth'], extension, a.noise, a.seed, a.signal_energy)
        result['synthetic'] = dict(format=FORMAT, **provenance, **block)
    else:
        result = dict(format=FORMAT+'-score', **provenance,
            **score(loaded['graph'], loaded['export'], loaded['checkpoint'], loaded.get('truth')))
    with open(a.output, 'x') as f:
        json.dump(result, f, indent=None if a.command == 'make' else 2, allow_nan=False)
    if a.command != 'make':
        print(json.dumps({k: v for k, v in result.items() if k not in ('groups', 'input_sha256')}, indent=2))
    else:
        print(json.dumps(result['synthetic']['oracle'], indent=2))


if __name__ == '__main__':
    main()
