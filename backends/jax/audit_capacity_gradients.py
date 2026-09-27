"""Check refined-step gradients at a reproduced failed capacity iterate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from extensions import restore
from objective import build, evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['failure', 'manifest', 'graph', 'training', 'output']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    failure, manifest, graph, training = [json.loads(Path(getattr(args, key)).read_text())
        for key in ['failure', 'manifest', 'graph', 'training']]
    if failure['format'] != 'wormsim-capacity-failure-replay' or failure['reference_manifest_sha256'] != digest(args.manifest):
        raise ValueError('failure lineage mismatch')
    for key in ['graph', 'training']:
        if manifest['input_sha256'][key] != digest(getattr(args, key)):
            raise ValueError('input hash mismatch: ' + key)
    packed = failure['model']
    if set(packed['configuration']['extensions']) != {'observation'} or packed['configuration']['solver'] is not None:
        raise ValueError('requires observation-only Euler model')
    training['groups'] = [g for g in training['groups'] if training['names'][g['target']] in manifest['targets']]
    trials = sorted(t for g in training['groups'] for t in g['training_trials'])
    if trials != packed['base_model']['training_trials'] or trials != manifest['training_trials']:
        raise ValueError('subset lineage mismatch')
    training['training_trials'] = trials
    training['classification_pairs'] = 0
    for group in training['groups']:
        group['labels'] = []
    rows, gradients = [], []
    for divisor in [2, 4]:
        candidate = copy.deepcopy(packed)
        candidate['base_model']['config']['dt'] /= divisor
        _, theta, active = restore(candidate, graph, training['groups'][0]['recording']['times'])
        _, _, groups, data, prior = build(candidate['base_model'], graph, training, candidate['configuration'])
        value, gradient, metrics = evaluate(theta, groups, data, prior)
        flat = np.concatenate([np.asarray(g).ravel() for g in jax.tree.leaves(gradient)])
        if not np.isfinite(float(value)) or not np.isfinite(flat).all():
            raise ValueError('nonfinite refined objective or gradient')
        gradients.append(flat)
        norm = float(np.linalg.norm(flat))
        if norm == 0:
            raise ValueError('zero gradient cannot define a diagnostic direction')
        direction = jax.tree.map(lambda g, mask: jnp.where(mask, g / norm, 0.), gradient, active)
        analytic = float(sum(jnp.sum(g*d) for g, d in zip(jax.tree.leaves(gradient), jax.tree.leaves(direction))))
        checks = []
        for h in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                perturbed = jax.tree.map(lambda p, d: p + sign*h*d, theta, direction)
                loss, _, _ = evaluate(perturbed, groups, data, prior)
                values.append(float(loss))
            finite_difference = (values[1]-values[0])/(2*h)
            checks.append(dict(step=h, analytic=analytic, central_difference=finite_difference,
                absolute_error=abs(finite_difference-analytic), relative_error=abs(finite_difference-analytic)/norm))
        row = dict(dt=candidate['base_model']['config']['dt'], metrics=metrics,
            gradient_norm=norm, finite_gradient_coordinates=len(flat), directional_checks=checks)
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    a, b = gradients
    receipt = dict(schema_version=1, input_sha256={key:digest(getattr(args,key)) for key in ['failure','manifest','graph','training']},
        backend_source_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        epoch=failure['epoch'], refinements=rows,
        gradient_cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),
        gradient_relative_step_difference=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
        scope='Frozen failing parameters; training subset only. Directional checks cover the gradient direction at two step sizes, not every coordinate or optimization trajectory. No refitting or held-out comparison.')
    with Path(args.output).open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
