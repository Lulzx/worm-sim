#!/usr/bin/env python3
"""Test whether fitted chemical sign movement agrees with molecular evidence.

Reads saved checkpoints only; it fits nothing and opens no recordings. The unit is
the tied `chemical_sign` group, because tied edges share one fitted value. A group
is labeled when its directional member edges all point the same way; groups whose
directional members disagree are excluded and counted.

The null permutes labels among groups within strata of presynaptic transmitter
signature and initial sign. Stratifying on initial sign removes the sign-prior
shrinkage confound: shrinkage moves every positive start down and every negative
start up, whatever the label. Agreement here is evidence about training data and
the Level 0 model class, not a measurement of biological signs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base(checkpoint):
    return checkpoint['model']['base_model'] if 'model' in checkpoint else checkpoint


def sign_values(path, n, m):
    model = base(load(path))
    groups = model['parameters']['groups']
    mapping = np.asarray(model['parameters']['raw_to_group'])
    rows = mapping[6*n+m:6*n+2*m]
    names = [groups[i]['name'] for i in rows]
    assert all(name.startswith('chemical_sign/') for name in names)
    raw = np.asarray([g['value'] for g in groups])
    return rows, 2/(1+np.exp(-raw))-1, model['graph_hash']


def group_table(rows, evidence):
    """Per tied group: label (+1, -1, 0 none, None mixed) and transmitter signature."""
    members = {}
    for edge, group in enumerate(rows):
        members.setdefault(int(group), []).append(edge)
    keys = sorted(members)
    labels, strata, mixed = [], [], 0
    for key in keys:
        states = {evidence[e]['state'] for e in members[key]} & {'excitatory', 'inhibitory'}
        if len(states) == 2:
            labels.append(None)
            mixed += 1
        else:
            labels.append({'excitatory': 1, 'inhibitory': -1}.get(next(iter(states), None), 0))
        strata.append('+'.join(sorted({t for e in members[key] for t in evidence[e]['transmitters']})) or 'none')
    return keys, labels, strata, mixed


def alignment(labels, delta):
    return float(np.mean(labels*delta))


def permutation_test(labels, delta, strata, permutations, rng):
    """Two-sided stratified label-permutation test of mean label*delta."""
    observed = alignment(labels, delta)
    buckets = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(permutations)
    shuffled = labels.copy()
    for k in range(permutations):
        for b in buckets:
            shuffled[b] = rng.permutation(labels[b])
        null[k] = alignment(shuffled, delta)
    centered = np.abs(null-null.mean())
    p = (1+np.sum(centered >= abs(observed-null.mean())))/(permutations+1)
    return observed, float(null.mean()), float(null.std()), float(p)


def planted(lab, delta, stratum, fraction, permutations, rng):
    """Power reference: force a fraction of labeled groups to move toward their label."""
    forced = delta.copy()
    labeled = np.flatnonzero(lab != 0)
    chosen = rng.choice(labeled, size=int(round(fraction*len(labeled))), replace=False)
    forced[chosen] = lab[chosen]*np.abs(delta[chosen])
    obs, mu, sd, p = permutation_test(lab, forced, stratum, permutations, rng)
    return {'fraction_forced': fraction, 'z': (obs-mu)/sd if sd > 0 else None, 'p_two_sided': p}


def compare(name, start, end, keys, labels, strata, permutations, seed, fractions):
    rows0, r0, h0 = start
    rows1, r1, h1 = end
    assert h0 == h1 and np.array_equal(rows0, rows1)
    r0, r1 = r0[keys], r1[keys]
    delta = r1-r0
    keep = np.asarray([lab is not None for lab in labels])
    lab = np.asarray([0 if lab is None else lab for lab in labels], dtype=float)
    init_sign = np.where(r0 >= 0, 'pos', 'neg')
    stratum = np.asarray([f'{s}|{i}' for s, i in zip(strata, init_sign)])
    # Shrinkage-adjusted movement: residual after a pooled linear fit on the start.
    slope = float(np.dot(r0, delta)/np.dot(r0, r0))
    residual = delta-slope*r0
    rng = np.random.default_rng(seed)
    labeled = keep & (lab != 0)
    big = np.abs(delta) >= np.quantile(np.abs(delta[keep]), 0.5)
    toward = labeled & big
    out = {
        'comparison': name,
        'groups': int(keep.sum()),
        'labeled_groups': int(labeled.sum()),
        'labeled_excitatory': int((lab[keep] > 0).sum()),
        'labeled_inhibitory': int((lab[keep] < 0).sum()),
        'abs_delta_quantiles': {q: float(np.quantile(np.abs(delta[keep]), float(q))) for q in ['0.5', '0.9', '0.99', '1.0']},
        'sign_flips_all': int(np.sum(np.sign(r1[keep]) != np.sign(r0[keep]))),
        'sign_flips_labeled': int(np.sum(np.sign(r1[labeled]) != np.sign(r0[labeled]))),
        'flips_toward_label': int(np.sum((np.sign(r1) != np.sign(r0)) & labeled & (np.sign(r1) == lab))),
        'shrinkage_slope': slope,
        'groups_moving_against_shrinkage': int(np.sum(keep & (np.sign(delta) == np.sign(r0)))),
        'labeled_above_median_movement': int(toward.sum()),
        'labeled_above_median_moving_toward_label': int(np.sum(toward & (np.sign(delta) == lab))),
    }
    for key, values in [('raw', delta), ('shrinkage_adjusted', residual)]:
        obs, mu, sd, p = permutation_test(lab[keep], values[keep], stratum[keep], permutations, rng)
        out[f'alignment_{key}'] = {'observed': obs, 'null_mean': mu, 'null_sd': sd,
                                   'z': (obs-mu)/sd if sd > 0 else None, 'p_two_sided': p}
    # Forcing changes only direction, so a fraction near 0.5 of all labeled groups
    # reproduces chance; the reference shows what agreement this design can see.
    out['planted_power'] = [planted(lab[keep], delta[keep], stratum[keep], f, permutations, rng) for f in fractions]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', default='runs/c302-audit.json')
    p.add_argument('--evidence', action='append', required=True, help='label=path')
    p.add_argument('--pair', action='append', required=True, help='name=start.json,end.json')
    p.add_argument('--planted', type=float, nargs='*', default=[0.25, 0.5, 1.0],
                   help='fractions of labeled groups forced toward their label for the power reference')
    p.add_argument('--permutations', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert not Path(a.output).exists(), 'output must be new'
    graph = load(a.graph)
    n, m = len(graph['neurons']), len(graph['chemical'])
    chem = sorted(graph['chemical'], key=lambda e: (e['pre'], e['post']))
    pairs = [(name, *paths.split(',')) for name, paths in (x.split('=', 1) for x in a.pair)]
    cache = {}
    values = lambda path: cache.setdefault(path, sign_values(path, n, m))
    report = {'format': 'wormsim-molecular-sign-probe', 'graph_sha256': digest(a.graph),
              'permutations': a.permutations, 'seed': a.seed, 'inputs': {}, 'evidence': {}}
    for path in sorted({x for _, s, e in pairs for x in (s, e)}):
        report['inputs'][path] = digest(path)
    for item in a.evidence:
        label, path = item.split('=', 1)
        evidence = load(path)
        assert evidence['graph_hash'] == values(pairs[0][1])[2]
        edges = evidence['edges']
        assert [(e['pre'], e['post']) for e in edges] == [(e['pre'], e['post']) for e in chem]
        rows = values(pairs[0][1])[0]
        keys, labels, strata, mixed = group_table(rows, edges)
        entry = {'sha256': digest(path), 'sign_groups': len(keys), 'mixed_direction_groups_excluded': mixed,
                 'comparisons': [compare(name, values(start), values(end), keys, labels, strata,
                                         a.permutations, a.seed, a.planted) for name, start, end in pairs]}
        report['evidence'][label] = entry
    Path(a.output).write_text(json.dumps(report, indent=1)+'\n')


if __name__ == '__main__':
    main()
