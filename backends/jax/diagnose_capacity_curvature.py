"""Frozen training-only directional curvature probes; never update a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from jax.flatten_util import ravel_pytree
from extensions import restore
from objective import build, evaluate


def directional_summary(value, gradient, direction, plus, minus, step):
    """Central value derivative and gradient secant along a unit direction."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError('step must be finite and positive')
    gradient, direction = np.asarray(gradient), np.asarray(direction)
    vp, gp = plus
    vm, gm = minus
    if gradient.shape != direction.shape or not np.isclose(np.linalg.norm(direction), 1):
        raise ValueError('expected matching gradient and unit direction')
    if not all(np.isfinite(x).all() for x in [value, gradient, direction, vp, gp, vm, gm]):
        raise ValueError('nonfinite curvature probe')
    slope = float(gradient @ direction)
    curvature = float((np.asarray(gp) - np.asarray(gm)) @ direction / (2 * step))
    return {'step': step, 'gradient_slope': slope,
            'central_value_slope': float((vp - vm) / (2 * step)),
            'gradient_secant_curvature': curvature,
            'central_value_curvature': float((vp - 2 * value + vm) / step**2),
            'positive_probe_loss': float(vp), 'negative_probe_loss': float(vm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint', 'manifest', 'training', 'graph', 'output']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('output exists')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    saved, manifest, training, graph = [json.loads(Path(getattr(args, k)).read_text())
                                      for k in ['checkpoint', 'manifest', 'training', 'graph']]
    if saved['format'] != manifest['format'] or saved['format'] != 'wormsim-training-capacity-diagnostic':
        raise ValueError('unexpected checkpoint format')
    if saved['targets'] != manifest['targets'] or manifest['jax'] != jax.__version__:
        raise ValueError('target or JAX version differs')
    for key in ['training', 'graph']:
        if digest(getattr(args, key)) != manifest['input_sha256'][key]:
            raise ValueError('input hash differs: ' + key)
    for name, sha in manifest['backend_source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
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
    flat, unravel = ravel_pytree(theta)
    mask, _ = ravel_pytree(active)
    def objective(x):
        value, gradient, metrics = evaluate(unravel(x), groups, data, prior)
        grad, _ = ravel_pytree(gradient)
        return float(value), np.asarray(grad), metrics
    value, gradient, metrics = objective(flat)
    if abs(metrics['mse'] - saved['metrics']['mse']) > 1e-10:
        raise ValueError('checkpoint score does not reproduce')
    labels = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(theta)[0]:
        name = '/'.join(str(p.key) for p in path)
        if name == 'groups':
            labels.extend('native/' + g['name'].split('/')[0] for g in base['parameters']['groups'])
        else:
            labels.extend([name] * np.asarray(leaf).size)
    labels = np.asarray(labels)
    if labels.shape != gradient.shape:
        raise ValueError('family alignment differs')
    # Fixed rule: three families with largest active gradient L2; ties alphabetical.
    families = []
    for family in sorted(set(labels)):
        selected = (labels == family) & np.asarray(mask, dtype=bool)
        direction = np.where(selected, -gradient, 0.)
        norm = np.linalg.norm(direction)
        if norm > 0:
            families.append((float(norm), family, direction / norm))
    families.sort(key=lambda row: (-row[0], row[1]))
    probes = []
    for norm, family, direction in families[:3]:
        rows = []
        for step in [0.001, 0.0005]:
            vp, gp, _ = objective(flat + step * direction)
            vm, gm, _ = objective(flat - step * direction)
            rows.append(directional_summary(value, gradient, direction, (vp, gp), (vm, gm), step))
        probes.append({'family': family, 'active_gradient_l2': norm, 'probes': rows})
        print(json.dumps(probes[-1]), flush=True)
    result = {'schema_version': 1, 'input_sha256': {k: digest(getattr(args, k)) for k in ['checkpoint', 'manifest', 'training', 'graph']},
              'script_sha256': digest(__file__), 'jax': jax.__version__, 'metrics': metrics,
              'objective_evaluations': 1 + 4 * len(probes), 'probes': probes,
              'scope': 'Frozen training checkpoint. Top three active-gradient L2 families, normalized negative-gradient direction, fixed central steps 0.001 and 0.0005. Local directional curvature only; not Hessian eigenvalues, a condition number, a fitted model, or evidence of generalization.'}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
