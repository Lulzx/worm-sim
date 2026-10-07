"""Trial-split fit-quality check for trial-mean targets. Training trials only.

Each target's training trials are split by sorted trial ID into even and odd
positions. `make` writes a fit-half export, which `overfit.py` fits unchanged,
and a held-half export. `score` reports each checkpoint's capture on both halves
against the held half's noise-ceiling expectation, with a trial bootstrap
interval. Checkpoints are selected on held-half capture only. See
docs/PLANTED-TRUTH.md, "Replacement gate".
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from extensions import restore
from planted_truth import energies, metrics, predict, noise_variance, group_weights

FORMAT = 'wormsim-trial-split'


def aggregate(trials, target, names, times):
    """Python mirror of src/bench/atlas_training.rs::aggregate for one target."""
    trials = sorted(trials, key=lambda t: t['id'])
    moments = {}
    for trial in trials:
        if trial['stimulated_neuron'] != names[target] or trial['recording']['times'] != times:
            raise ValueError('trial does not match the target or its grid')
        for trace in trial['recording']['traces']:
            w = trace['provenance']['id_confidence']
            if w == 0:
                continue
            if any(v is None for v in trace['values']):
                raise ValueError('incomplete positive-confidence trace')
            m = moments.setdefault(trace['neuron'], [0., np.zeros(len(times)), np.zeros(len(times))])
            y = np.asarray(trace['values'], float)
            total = m[0]+w; delta = y-m[1]
            m[1] = m[1]+w/total*delta
            m[2] = m[2]+w*delta*(y-m[1])
            m[0] = total
    if not moments:
        raise ValueError('empty positive-confidence target group')
    sample_weight = sum(m[0]*len(times) for m in moments.values())
    irreducible = sum(float(m[2].sum()) for m in moments.values())/sample_weight
    top = max(m[0] for m in moments.values())
    traces = [{'neuron': n, 'values': moments[n][1].tolist(),
        'provenance': {'dataset': 'training response sufficient statistics',
            'version': '1; confidence field encodes relative accumulated loss weight, not calibrated identity confidence',
            'id_confidence': moments[n][0]/top}} for n in sorted(moments)]
    return {'irreducible_mse': irreducible, 'labels': [],
        'recording': {'animal_id': 'aggregate-not-an-animal', 'behavior': {},
            'condition': f'stimulated {names[target]}', 'dataset': 'training response sufficient statistics',
            'times': list(times), 'traces': traces},
        'sample_weight': sample_weight, 'target': target, 'training_trials': [t['id'] for t in trials]}


def check_mirror(training, data, targets):
    """The Python aggregate must reproduce the Rust export on the full trial set."""
    names = training['names']; trials = {t['id']: t for t in data['trials']}
    for g in training['groups']:
        if names[g['target']] not in targets:
            continue
        a = aggregate([trials[i] for i in g['training_trials']], g['target'], names, g['recording']['times'])
        if a['training_trials'] != g['training_trials'] or [t['neuron'] for t in a['recording']['traces']] != [t['neuron'] for t in g['recording']['traces']]:
            raise ValueError('aggregate membership differs from the Rust export')
        for x, y in zip(a['recording']['traces'], g['recording']['traces']):
            if not np.allclose(x['values'], y['values'], rtol=0, atol=1e-12) or abs(x['provenance']['id_confidence']-y['provenance']['id_confidence']) > 1e-12:
                raise ValueError('aggregate means differ from the Rust export')
        for key in ['sample_weight', 'irreducible_mse']:
            if abs(a[key]-g[key]) > 1e-9*max(1., abs(g[key])):
                raise ValueError(f'aggregate {key} differs from the Rust export')


def split(training, data, targets, parity):
    """Fit-half export (positions == parity) and held-half export (the others)."""
    check_mirror(training, data, targets)
    names = training['names']; trials = {t['id']: t for t in data['trials']}
    halves = {'fit': copy.deepcopy(training), 'held': copy.deepcopy(training)}
    for h in halves.values():
        h['groups'] = []; h['classification_pairs'] = 0
    for g in training['groups']:
        if names[g['target']] not in targets:
            continue
        ids = sorted(g['training_trials'])
        parts = {'fit': [i for k, i in enumerate(ids) if k % 2 == parity],
                 'held': [i for k, i in enumerate(ids) if k % 2 != parity]}
        for half, members in parts.items():
            halves[half]['groups'].append(aggregate([trials[i] for i in members], g['target'], names, g['recording']['times']))
    for half, h in halves.items():
        h['trial_split'] = {'format': FORMAT, 'half': half, 'parity': parity, 'targets': targets,
            'rule': 'per target, training trials sorted by ID; fit half takes positions with index % 2 == parity',
            'scope': 'Training-trial split for fit-quality checks. Never a benchmark input.'}
    return halves['fit'], halves['held']


def expectation(groups, trials):
    """Capture a perfect shared-response predictor would reach on these means."""
    zero, start = energies(groups)
    noise = 0.
    for g, (w, _) in zip(groups, group_weights(groups)):
        per = {}
        for tid in g['training_trials']:
            for tr in trials[tid]['recording']['traces']:
                if tr['provenance']['id_confidence'] > 0:
                    per.setdefault(tr['neuron'], []).append((tr['provenance']['id_confidence'], np.asarray(tr['values'], float)))
        variances = {}
        for n, obs in per.items():
            ww = np.asarray([o[0] for o in obs]); y = np.stack([o[1] for o in obs])
            m = (ww[:, None]*y).sum(0)/ww.sum()
            variances[n] = noise_variance(ww, y-m)
        pooled = np.mean([v for v in variances.values() if v is not None], axis=0)
        for trace, wi in zip(g['recording']['traces'], w):
            v = variances[trace['neuron']]
            noise += wi*(pooled if v is None else v).sum()
    return 1-noise/(zero-start)


def bootstrap(held, data, reps, seed):
    """Trial bootstrap of the held-half expectation (resample trials within each target)."""
    rng = np.random.default_rng(seed); names = held['names']
    trials = {t['id']: t for t in data['trials']}
    values = []
    for _ in range(reps):
        groups = []; pool = {}
        for g in held['groups']:
            ids = g['training_trials']
            draw = [ids[i] for i in rng.integers(0, len(ids), len(ids))]
            members = []
            for k, i in enumerate(draw):
                clone = copy.deepcopy(trials[i]); clone['id'] = f'{i}#{k}'
                pool[clone['id']] = clone; members.append(clone)
            groups.append(aggregate(members, g['target'], names, g['recording']['times']))
        values.append(expectation(groups, pool))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def score(graph, fit_half, held_half, data, checkpoints, reps, seed):
    trials = {t['id']: t for t in data['trials']}
    targets = fit_half['trial_split']['targets']
    if held_half['trial_split']['targets'] != targets or {fit_half['trial_split']['half'], held_half['trial_split']['half']} != {'fit', 'held'}:
        raise ValueError('exports are not the two halves of one split')
    names = held_half['names']
    order = lambda h: [{names[g['target']]: g for g in h['groups']}[t] for t in targets]
    fit_groups, held_groups = order(fit_half), order(held_half)
    times = held_groups[0]['recording']['times']
    out = {'targets': targets, 'parity': fit_half['trial_split']['parity'],
        'held_expectation': expectation(held_groups, trials),
        'held_expectation_interval': bootstrap(held_half, data, reps, seed),
        'fit_expectation': expectation(fit_groups, trials), 'checkpoints': []}
    for name, c in checkpoints:
        packed = c['model'] if c.get('format') == 'wormsim-training-capacity-diagnostic' else c
        engine, theta, _ = restore(packed, graph, times)
        row = {'checkpoint': name, 'epoch': packed['base_model']['epoch']}
        for half, groups in [('fit', fit_groups), ('held', held_groups)]:
            row[half+'_capture'] = metrics(groups, predict(engine, theta, groups, names))['captured_start_zero_energy']
        out['checkpoints'].append(row)
    out['checkpoints'].sort(key=lambda r: r['epoch'])
    best = max(out['checkpoints'], key=lambda r: (r['held_capture'], -r['epoch']))
    lo, hi = out['held_expectation_interval']
    out['selected'] = best
    out['adequate'] = lo <= best['held_capture'] <= hi or best['held_capture'] > hi
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('make'); s = sub.add_parser('score')
    for key in ['training', 'data', 'fit-output', 'held-output']:
        m.add_argument('--'+key, required=True)
    m.add_argument('--targets', nargs='+', required=True)
    m.add_argument('--parity', type=int, choices=[0, 1], required=True)
    for key in ['graph', 'fit-half', 'held-half', 'data', 'output']:
        s.add_argument('--'+key, required=True)
    s.add_argument('--checkpoints', nargs='+', required=True)
    s.add_argument('--bootstrap', type=int, default=200)
    s.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    if a.command == 'make':
        raw = {k: Path(getattr(a, k)).read_bytes() for k in ['training', 'data']}
        training, data = [json.loads(raw[k]) for k in ['training', 'data']]
        fit_half, held_half = split(training, data, a.targets, a.parity)
        for h in (fit_half, held_half):
            h['trial_split'].update(source_commit=source, source_worktree_dirty=dirty,
                input_sha256={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()})
        for path, h in [(a.fit_output, fit_half), (a.held_output, held_half)]:
            with open(path, 'x') as f:
                json.dump(h, f, allow_nan=False)
        print(json.dumps({h['trial_split']['half']: {training['names'][g['target']]: len(g['training_trials']) for g in h['groups']} for h in (fit_half, held_half)}))
        return
    paths = {'graph': a.graph, 'fit_half': a.fit_half, 'held_half': a.held_half, 'data': a.data}
    raw = {k: Path(v).read_bytes() for k, v in paths.items()}
    loaded = {k: json.loads(v) for k, v in raw.items()}
    checkpoints = [(c, json.loads(Path(c).read_text())) for c in a.checkpoints]
    result = dict(format=FORMAT+'-score', source_commit=source, source_worktree_dirty=dirty,
        input_sha256={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
        **score(loaded['graph'], loaded['fit_half'], loaded['held_half'], loaded['data'], checkpoints, a.bootstrap, a.seed))
    with open(a.output, 'x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_sha256', 'checkpoints')}, indent=2))


if __name__ == '__main__':
    main()
