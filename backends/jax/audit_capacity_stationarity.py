"""Report active-coordinate gradients at a frozen capacity checkpoint; no refit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
from extensions import restore
from objective import build, evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(gradient, active, group_names):
    if jax.tree.structure(gradient) != jax.tree.structure(active):
        raise ValueError('gradient/mask structure mismatch')
    categories = defaultdict(list)
    leaves, _ = jax.tree_util.tree_flatten_with_path(gradient)
    masks = jax.tree.leaves(active)
    for (path, values), mask in zip(leaves, masks):
        name = '/'.join(str(p.key) for p in path)
        values, mask = np.asarray(values), np.asarray(mask)
        if values.shape != mask.shape or mask.dtype != bool or not np.isfinite(values).all():
            raise ValueError('invalid gradient or mask')
        if name == 'groups':
            if values.shape != (len(group_names),):
                raise ValueError('native group-name alignment differs')
            for label, value, enabled in zip(group_names, values, mask):
                if enabled:
                    categories['native/' + label.split('/')[0]].append(float(value))
        else:
            categories[name].extend(values[mask].reshape(-1).tolist())
    rows = []
    for name, values in sorted(categories.items()):
        if values:
            rows.append({'family': name, 'active_coordinates': len(values),
                         'gradient_l2': float(np.linalg.norm(values)),
                         'gradient_linf': float(np.max(np.abs(values)))})
    if not rows:
        raise ValueError('no active coordinates')
    return {'active_coordinates': sum(r['active_coordinates'] for r in rows),
            'active_gradient_l2': float(np.sqrt(sum(r['gradient_l2']**2 for r in rows))),
            'active_gradient_linf': max(r['gradient_linf'] for r in rows), 'families': rows}


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
    value, gradient, metrics = evaluate(theta, groups, data, prior)
    if not np.isfinite(float(value)) or abs(metrics['mse'] - saved['metrics']['mse']) > 1e-10:
        raise ValueError('checkpoint score does not reproduce')
    result = summarize(gradient, active, [g['name'] for g in base['parameters']['groups']])
    gtol = manifest['fitting_optimizer']['gtol']
    result.update({'schema_version': 1, 'metrics': metrics, 'epoch': base['epoch'],
                   'declared_gtol': gtol,
                   'active_gradient_linf_meets_declared_gtol': result['active_gradient_linf'] <= gtol,
                   'input_sha256': {k: digest(getattr(a, k)) for k in ['checkpoint', 'manifest', 'training', 'graph']},
                   'script_sha256': digest(__file__), 'jax': jax.__version__,
                   'scope': 'One frozen training checkpoint, existing objective and raw active parameter coordinates. No parameter updates, finite-difference claim or held-out selection. A small gradient does not prove global optimality or sufficient model capacity; this is not an optimizer termination report.'})
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
