#!/usr/bin/env python3
"""Compiler v0: molecular evidence and wiring reliability to Level 0 sign initialization.

A fixed, audited rule; nothing is learned. Starting from a random-sign epoch-zero
model, each tied chemical sign group with a unanimous directional molecular label
resting on reproducible wiring is set to that direction, at the seed's own
magnitude. Every other value is copied unchanged.

Wiring is reproducible for a group when at least one directional member edge has
a present left/right mirror or no defined mirror (midline cells cannot be tested).
Labels whose directional edges all lack an existing mirror are not compiled.

The shuffled arm assigns the same labels to other groups, permuted within strata of
presynaptic transmitter signature and wiring status, so the two arms differ only in
which groups receive which labels. Magnitudes, label counts and stratum
composition match. Labels are expression evidence, not measured signs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

RULE = 'compiler-v0: unanimous directional label on reproducible wiring sets sign group to label at seed magnitude'


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wiring_status(edges, reliability):
    flags = [reliability[e] for e in edges]
    if any(r['mirror_present'] for r in flags):
        return 'confirmed'
    if any(not r['mirror_defined'] for r in flags):
        return 'untestable'
    return 'unconfirmed'


def groups(model, graph, evidence, reliability):
    """Per tied sign group: index, name, label, stratum. Order follows group index."""
    n, m = len(graph['neurons']), len(graph['chemical'])
    mapping = np.asarray(model['parameters']['raw_to_group'])
    rows = mapping[6*n+m:6*n+2*m]
    chem = sorted(graph['chemical'], key=lambda e: (e['pre'], e['post']))
    edges = [(e['pre'], e['post']) for e in chem]
    assert [(e['pre'], e['post']) for e in evidence['edges']] == edges
    members = {}
    for k, g in enumerate(rows):
        members.setdefault(int(g), []).append(k)
    table = []
    for g in sorted(members):
        name = model['parameters']['groups'][g]['name']
        assert name.startswith('chemical_sign/')
        states = {k: evidence['edges'][k]['state'] for k in members[g]}
        directional = [k for k, s in states.items() if s in ('excitatory', 'inhibitory')]
        kinds = {states[k] for k in directional}
        label = 0 if not kinds else (None if len(kinds) == 2 else (1 if 'excitatory' in kinds else -1))
        basis = directional or members[g]
        status = wiring_status([edges[k] for k in basis], reliability)
        transmitters = '+'.join(sorted({t for k in members[g] for t in evidence['edges'][k]['transmitters']})) or 'none'
        table.append({'index': g, 'name': name, 'label': label, 'status': status,
                      'stratum': f'{transmitters}|{status}', 'edges': len(members[g])})
    return table


def compile_labels(table, arm, permutation_seed):
    """Return the label assigned to each group under the requested arm."""
    eligible = np.array([0 if t['label'] is None or t['status'] == 'unconfirmed' else t['label'] for t in table])
    if arm == 'molecular':
        return eligible
    if arm != 'shuffled':
        raise ValueError('arm must be molecular or shuffled')
    rng = np.random.default_rng(permutation_seed)
    strata = np.array([t['stratum'] for t in table])
    out = eligible.copy()
    for s in sorted(set(strata)):
        idx = np.flatnonzero(strata == s)
        out[idx] = rng.permutation(eligible[idx])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='random-sign epoch-zero model')
    p.add_argument('--graph', default='runs/c302-audit.json')
    p.add_argument('--evidence', default='runs/molecular-th2/evidence.json')
    p.add_argument('--reliability', default='data/c302-edge-reliability.json')
    p.add_argument('--arm', choices=['molecular', 'shuffled'], required=True)
    p.add_argument('--permutation-seed', type=int, help='required for the shuffled arm')
    p.add_argument('--output', required=True)
    p.add_argument('--receipt', required=True)
    a = p.parse_args()
    for path in [a.output, a.receipt]:
        assert not Path(path).exists(), 'outputs must be new'
    if (a.arm == 'shuffled') != (a.permutation_seed is not None):
        raise ValueError('a permutation seed is required for, and only for, the shuffled arm')
    model, graph, evidence, rel = load(a.model), load(a.graph), load(a.evidence), load(a.reliability)
    if model['epoch'] != 0 or not model['config'].get('sign_initialization') or model['config'].get('molecular_sign_priors'):
        raise ValueError('requires an epoch-zero random-sign model without molecular priors')
    if not (model['graph_hash'] == evidence['graph_hash']) or rel['graph_sha256'] != digest(a.graph):
        raise ValueError('graph lineage differs')
    reliability = {tuple(e['pair']): e for e in rel['chemical']}
    table = groups(model, graph, evidence, reliability)
    values = np.array([model['parameters']['groups'][t['index']]['value'] for t in table])
    magnitude = abs(values[0])
    if not np.allclose(np.abs(values), magnitude, rtol=0, atol=1e-12) or magnitude == 0:
        raise ValueError('seed sign values must share one nonzero magnitude')
    labels = compile_labels(table, a.arm, a.permutation_seed)
    changed = []
    for t, value, label in zip(table, values, labels):
        if label == 0:
            continue
        new = float(label*magnitude)
        if new != value:
            model['parameters']['groups'][t['index']]['value'] = new
            changed.append({'name': t['name'], 'seed_value': float(value), 'compiled_value': new})
    with open(a.output, 'x') as f:
        json.dump(model, f)
    assigned = [t for t, label in zip(table, labels) if label != 0]
    receipt = {
        'format': 'wormsim-molecular-init-receipt', 'rule': RULE, 'arm': a.arm,
        'permutation_seed': a.permutation_seed,
        'script_sha256': digest(__file__),
        'input_sha256': {k: digest(getattr(a, k)) for k in ['model', 'graph', 'evidence', 'reliability']},
        'output_sha256': digest(a.output),
        'seed': model['config']['sign_initialization']['seed'],
        'seed_magnitude': float(magnitude),
        'sign_groups': len(table),
        'labeled_groups': sum(t['label'] not in (0, None) for t in table),
        'mixed_label_groups': sum(t['label'] is None for t in table),
        'labels_dropped_unconfirmed_wiring': sum(t['label'] not in (0, None) and t['status'] == 'unconfirmed' for t in table),
        'assigned_groups': len(assigned),
        'assigned_excitatory': int(np.sum(labels > 0)),
        'assigned_inhibitory': int(np.sum(labels < 0)),
        'assigned_by_status': {s: sum(t['status'] == s for t in assigned) for s in ['confirmed', 'untestable']},
        'changed_groups': len(changed),
        'changes': changed,
        'scope': 'Initialization only; capacity fits disable priors. The model keeps its seed in config because unassigned groups retain the seed draw; this receipt identifies the compiled values.',
    }
    Path(a.receipt).write_text(json.dumps(receipt, indent=1)+'\n')


if __name__ == '__main__':
    main()
