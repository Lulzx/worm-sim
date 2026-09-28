"""Compare frozen capacity gradients under longer unforced preparation."""
import argparse
import copy
import json
from pathlib import Path
import sys
BACKEND = Path(__file__).resolve().parents[1] / 'backends' / 'jax'
sys.path.insert(0, str(BACKEND))
import jax
import numpy as np
from jax.flatten_util import ravel_pytree
from audit_capacity_stationarity import digest, summarize
from extensions import restore
from objective import build, evaluate


def compare_vectors(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.ndim != 1 or candidate.shape != reference.shape or not reference.size:
        raise ValueError('matching nonempty vectors required')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('nonfinite gradient')
    a, b = np.linalg.norm(reference), np.linalg.norm(candidate)
    delta = candidate - reference
    return {'difference_l2': float(np.linalg.norm(delta)),
            'difference_linf': float(np.max(np.abs(delta))),
            'relative_difference_l2': float(np.linalg.norm(delta)/a) if a else None,
            'cosine': float(reference @ candidate/(a*b)) if a and b else None}


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
    mask, _ = ravel_pytree(active)
    mask = np.asarray(mask, dtype=bool)
    names = [g['name'] for g in base['parameters']['groups']]
    rows = []; vectors = []
    original_seconds = base['config']['preparation_seconds']
    for factor in [1, 2, 4]:
        variant = copy.deepcopy(base)
        variant['config']['preparation_seconds'] = original_seconds * factor
        _, _, groups, data, prior = build(variant, graph, training, packed['configuration'])
        value, gradient, metrics = evaluate(theta, groups, data, prior)
        if not np.isfinite(float(value)):
            raise ValueError('nonfinite objective')
        if factor == 1 and abs(metrics['mse'] - saved['metrics']['mse']) > 1e-10:
            raise ValueError('checkpoint score does not reproduce')
        vector, _ = ravel_pytree(gradient)
        vector = np.asarray(vector)[mask]
        summary = summarize(gradient, active, names)
        row = {'preparation_seconds': original_seconds * factor, 'metrics': metrics,
               'gradient': summary}
        if vectors:
            row['versus_original'] = compare_vectors(vectors[0], vector)
            row['versus_previous'] = compare_vectors(vectors[-1], vector)
        vectors.append(vector); rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'gradient'}), flush=True)
    result = {'schema_version': 1, 'input_sha256': {k: digest(getattr(a,k)) for k in ['checkpoint','manifest','training','graph']},
              'script_sha256': digest(__file__), 'jax': jax.__version__, 'variants': rows,
              'scope': 'Three frozen-parameter training objective/gradient evaluations. Only preparation duration changes by factors 1, 2, 4. Active raw-coordinate gradients; no fitting, held-out scoring, or independent AD verification.'}
    with Path(a.output).open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
