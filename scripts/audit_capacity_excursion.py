"""Reconstruct and audit a threshold loss-excursion probe; never fit parameters."""
import argparse
import copy
import json
from pathlib import Path
import sys
BACKEND = Path(__file__).resolve().parents[1] / 'backends' / 'jax'
sys.path.insert(0, str(BACKEND))
import jax
import numpy as np
from audit_capacity_stationarity import digest
from extensions import restore, pack
from fit import checkpoint
import subprocess
from objective import build, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint', 'manifest', 'training', 'graph', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--directions', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError('output already exists')
    saved, manifest, training, graph = [json.loads(Path(getattr(a, k)).read_text())
                                      for k in ['checkpoint', 'manifest', 'training', 'graph']]
    if saved['format'] != manifest['format'] or saved['format'] != 'wormsim-training-capacity-diagnostic':
        raise ValueError('unexpected checkpoint format')
    if saved['targets'] != manifest['targets'] or manifest['jax'] != jax.__version__:
        raise ValueError('target or JAX version differs')
    for key in ['training', 'graph']:
        if digest(getattr(a, key)) != manifest['input_sha256'][key]:
            raise ValueError('input hash differs: ' + key)
    for name, sha in manifest['backend_source_sha256'].items():
        if digest(BACKEND / name) != sha:
            raise ValueError('fitting source changed: ' + name)
    packed = saved['model']; base = packed['base_model']
    if base['selection_trials'] or packed['configuration'] != manifest['configuration']:
        raise ValueError('selection or configuration differs')
    training['groups'] = [g for g in training['groups'] if training['names'][g['target']] in saved['targets']]
    trials = sorted(t for g in training['groups'] for t in g['training_trials'])
    if trials != base['training_trials'] or trials != manifest['training_trials']:
        raise ValueError('training subset differs')
    training['training_trials'] = trials
    training['classification_pairs'] = 0
    for group in training['groups']:
        group['labels'] = []
    _, theta, active = restore(packed, graph, training['groups'][0]['recording']['times'])
    directions = json.loads(Path(a.directions).read_text())
    for key in ['checkpoint', 'manifest', 'training', 'graph']:
        if directions['input_sha256'][key] != digest(getattr(a, key)):
            raise ValueError('direction receipt input mismatch: ' + key)
    if directions['script_sha256'] != digest(BACKEND / 'diagnose_capacity_curvature.py'):
        raise ValueError('direction diagnostic source changed')
    _, _, groups, data, prior = build(base, graph, training, packed['configuration'])
    value, gradient, metrics = evaluate(theta, groups, data, prior)
    if abs(metrics['mse'] - saved['metrics']['mse']) > 1e-10:
        raise ValueError('endpoint does not reproduce')
    selected = np.array([g['name'].split('/')[0] == 'threshold'
                         for g in base['parameters']['groups']]) & np.asarray(active['groups'], dtype=bool)
    vector = np.where(selected, np.asarray(gradient['groups']), 0.)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('invalid threshold gradient')
    # Previous diagnostic used negative gradient direction; its negative probe
    # is therefore a positive-gradient displacement of length 1e-5.
    probe = dict(theta)
    probe['groups'] = theta['groups'] + 1e-5 * vector / norm
    value, _, metrics = evaluate(probe, groups, data, prior)
    expected = next(f for f in directions['probes'] if f['family'] == 'native/threshold')
    expected = next(p for p in expected['probes'] if p['step'] == 1e-5)['negative_probe_loss']
    if not np.isfinite(float(value)) or abs(metrics['mse'] - expected) > 1e-10:
        raise ValueError('excursion does not reproduce')
    output = Path(a.output)
    probe_path = output.with_suffix('.probe.json')
    numerics_path = output.with_suffix('.numerics.json')
    if probe_path.exists() or numerics_path.exists():
        raise ValueError('probe artifacts already exist')
    artifact = copy.deepcopy(saved)
    artifact['model'] = pack(checkpoint(base, probe, base['epoch'], 'diagnostic-probe'),
                             probe, packed['configuration'])
    artifact['metrics'] = metrics
    artifact['probe_provenance'] = {'parent_sha256': digest(a.checkpoint),
        'family': 'native/threshold', 'signed_negative_gradient_step': -1e-5,
        'scope': 'Unaccepted diagnostic perturbation; not a fitted checkpoint.'}
    with probe_path.open('x') as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
    print(json.dumps({'probe_mse': metrics['mse'], 'expected_mse': expected}), flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('check_capacity_numerics.py')),
        '--checkpoint', str(probe_path), '--manifest', a.manifest, '--training', a.training,
        '--graph', a.graph, '--output', str(numerics_path)], check=True)
    result = {'schema_version': 1,
        'input_sha256': {k: digest(getattr(a, k)) for k in ['checkpoint', 'manifest', 'training', 'graph', 'directions']},
        'script_sha256': digest(__file__), 'jax': jax.__version__,
        'probe_sha256': digest(probe_path), 'numerics_sha256': digest(numerics_path),
        'probe_provenance': artifact['probe_provenance'], 'probe_metrics': metrics,
        'numerics': json.loads(numerics_path.read_text()),
        'scope': 'Reconstructed training-only unaccepted threshold perturbation. Independent NumPy step/preparation controls; no optimization or held-out selection.'}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
