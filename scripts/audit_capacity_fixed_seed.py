"""Audit a fixed prepared-state seed without fitting parameters."""
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
from extensions import restore
from jax.flatten_util import ravel_pytree
from diagnose_capacity_curvature import directional_summary
from audit_preparation_gradients import compare_vectors
from replay_level0_atlas import Replay
from objective import build, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint', 'manifest', 'training', 'graph', 'output']:
        p.add_argument('--' + key, required=True)
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
    _, _, groups, data, prior = build(base, graph, training, packed['configuration'])
    _, old_gradient, old_metrics = evaluate(theta, groups, data, prior)
    if abs(old_metrics['mse'] - saved['metrics']['mse']) > 1e-10:
        raise ValueError('original score differs')
    replay = Replay(base, graph)
    seed = replay.state.copy()
    if seed.shape != (3*replay.n,) or not np.isfinite(seed).all():
        raise ValueError('invalid prepared state')
    fixed_base = copy.deepcopy(base)
    fixed_base['initial'] = seed.tolist()
    _, _, groups, data, prior = build(fixed_base, graph, training, packed['configuration'])
    flat, unravel = ravel_pytree(theta)
    mask, _ = ravel_pytree(active)
    def objective(x):
        value, gradient, metrics = evaluate(unravel(x), groups, data, prior)
        vector, _ = ravel_pytree(gradient)
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(vector)).all():
            raise ValueError('nonfinite probe')
        return float(value), np.asarray(vector), metrics
    value, gradient, metrics = objective(flat)
    old_vector, _ = ravel_pytree(old_gradient)
    comparison = compare_vectors(np.asarray(old_vector)[np.asarray(mask, dtype=bool)],
                                 gradient[np.asarray(mask, dtype=bool)])
    labels = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(theta)[0]:
        name = '/'.join(str(p.key) for p in path)
        if name == 'groups':
            labels.extend('native/' + g['name'].split('/')[0] for g in base['parameters']['groups'])
        else:
            labels.extend([name] * np.asarray(leaf).size)
    labels = np.asarray(labels)
    families = []
    for family in sorted(set(labels)):
        direction = np.where((labels == family) & np.asarray(mask, dtype=bool), -gradient, 0.)
        norm = np.linalg.norm(direction)
        if norm > 0:
            families.append((float(norm), family, direction/norm))
    families.sort(key=lambda row: (-row[0], row[1]))
    rows = []
    for norm, family, direction in families[:3]:
        vp, gp, _ = objective(flat + 1e-6*direction)
        vm, gm, _ = objective(flat - 1e-6*direction)
        row = {'family': family, **directional_summary(value, gradient, direction, (vp,gp), (vm,gm), 1e-6)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Reconstruct the exact old high-loss direction, using the original gradient.
    old_threshold = np.where((labels == 'native/threshold') & np.asarray(mask, dtype=bool), np.asarray(old_vector), 0.)
    excursion_value, _, excursion_metrics = objective(flat + 1e-5*old_threshold/np.linalg.norm(old_threshold))
    repeated_value, repeated_gradient, _ = objective(flat)
    result = {'schema_version': 1,
        'input_sha256': {k: digest(getattr(a,k)) for k in ['checkpoint','manifest','training','graph']},
        'script_sha256': digest(__file__),
        'replay_script_sha256': digest(Path(__file__).with_name('replay_level0_atlas.py')),
        'jax': jax.__version__, 'objective_evaluations': 10,
        'fixed_initial_state': seed.tolist(), 'preparation_seconds': base['config']['preparation_seconds'],
        'original_metrics': old_metrics, 'fixed_seed_metrics': metrics,
        'active_gradient_comparison': comparison, 'directional_probes': rows,
        'old_excursion_with_fixed_seed': excursion_metrics,
        'repeat_loss_difference': repeated_value-value,
        'repeat_gradient_linf_difference': float(np.max(np.abs(repeated_gradient-gradient))),
        'scope': 'Frozen training-only parameters. Initial vector fixed once from parent NumPy preparation; differentiated preparation remains enabled. No fitting, mutable seed, branch selection per evaluation or held-out scoring.'}
    with Path(a.output).open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
