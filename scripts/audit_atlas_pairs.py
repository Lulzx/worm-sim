#!/usr/bin/env python3
"""Check every native pair evidence entry directly against the pinned HDF5 matrices."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / name).read_text())


def digest(name):
    with (ROOT / name).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


data = load('runs/randi-data.json')
split = load('data/randi-neuron-split.json')
evidence = load('runs/randi-pairs.json')
report = load('runs/randi-pair-import.json')
assert evidence['dataset_hash'] == split['dataset_hash'] == report['dataset_hash']
assert evidence['source_sha256'] == digest('runs/randi-source/funatlas.h5')
observed = {(t['stimulated_neuron'], r['neuron']) for t in data['trials'] for r in t['recording']['traces'] if t['stimulated_neuron'] != r['neuron']}
actual = {(p['stimulated'], p['responding']): p for p in evidence['pairs']}
assert len(actual) == len(evidence['pairs'])
with h5py.File(ROOT / 'runs/randi-source/funatlas.h5', 'r') as f:
    ids = [s.decode() for s in f['neuron_ids'][:]]
    q, eq, counts = (f[f'wt/{k}'][:] for k in ['q', 'q_eq', 'occ1'])
    expected = {(s, r) for j, s in enumerate(ids) for i, r in enumerate(ids) if (s, r) in observed and np.isfinite(q[i, j])}
    assert set(actual) == expected
    assert evidence['detection_q_threshold'] == 0.05
    assert evidence['equivalence_threshold'] == f['wt/q_eq_th'][()]
    assert evidence['source_version'] == f.attrs['time_compiled'].decode()
    for (s, r), pair in actual.items():
        i, j = ids.index(r), ids.index(s)
        assert pair['q'] == q[i, j]
        assert pair['observations'] == counts[i, j] > 0
        assert pair['equivalence_q'] == (float(eq[i, j]) if np.isfinite(eq[i, j]) else None)
    missing_q_observed = len({(s, r) for j, s in enumerate(ids) for i, r in enumerate(ids) if (s, r) in observed and not np.isfinite(q[i, j])})
    source_missing_id_pairs = len(observed - {(s, r) for s in ids for r in ids})

covered = set()
for partition in ['train', 'validation', 'test']:
    ids = set(split[partition])
    targets = {t['stimulated_neuron'] for t in data['trials'] if t['id'] in ids}
    pairs = {key for key in actual if key[0] in targets}
    assert not pairs & covered
    covered |= pairs
    summary = report['partitions'][partition]
    assert summary['pairs'] == len(pairs)
    assert summary['detected'] == sum(actual[key]['q'] < 0.05 for key in pairs)
    assert summary['not_detected'] == sum(actual[key]['q'] >= 0.05 for key in pairs)
assert covered == set(actual)
report['independent_audit'] = {
    'script_sha256': digest('scripts/audit_atlas_pairs.py'),
    'evidence_file_sha256': digest('runs/randi-pairs.json'),
    'split_file_sha256': digest('data/randi-neuron-split.json'),
    'pairs_checked_against_source_exactly': len(actual),
    'observed_pairs_excluded_for_missing_q': missing_q_observed,
    'observed_pairs_excluded_for_source_identity_absence': source_missing_id_pairs,
    'pair_partitions_disjoint_and_complete': True,
    'source_trace_dataset_unchanged': digest('runs/randi-data.json') == load('docs/randi-import-audit.json')['dataset_file_sha256'],
}
assert report['independent_audit']['source_trace_dataset_unchanged']
path = ROOT / 'docs/randi-pair-label-audit.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(path)
print(report['partitions'])
