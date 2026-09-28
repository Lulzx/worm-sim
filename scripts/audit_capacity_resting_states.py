#!/usr/bin/env python3
"""Frozen cross-initialization of two prepared states; no fitting or selection."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from audit_capacity_fit import digest
from replay_level0_atlas import Replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint', 'probe', 'excursion', 'graph', 'training', 'output']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('output exists')
    saved, probe, receipt, graph, training = [json.loads(Path(getattr(args, k)).read_text())
        for k in ['checkpoint', 'probe', 'excursion', 'graph', 'training']]
    for key in ['checkpoint', 'graph', 'training']:
        if digest(getattr(args, key)) != receipt['input_sha256'][key]:
            raise ValueError('input hash differs: ' + key)
    if digest(args.probe) != receipt['probe_sha256']:
        raise ValueError('probe hash differs')
    if digest(Path(__file__).with_name('replay_level0_atlas.py')) != receipt['numerics']['replay_script_sha256']:
        raise ValueError('replay source changed')
    if saved['targets'] != probe['targets'] or saved['format'] != probe['format']:
        raise ValueError('checkpoint metadata differs')
    models = {'endpoint': saved['model'], 'probe': probe['model']}
    for model in models.values():
        if set(model['configuration']['extensions']) != {'observation'} or model['configuration']['solver'] is not None:
            raise ValueError('unsupported model')
        if model['base_model']['selection_trials']:
            raise ValueError('held-out selection present')
    groups = [g for g in training['groups'] if training['names'][g['target']] in saved['targets']]
    trials = sorted(t for g in groups for t in g['training_trials'])
    if any(m['base_model']['training_trials'] != trials for m in models.values()):
        raise ValueError('training subset differs')
    total = sum(g['sample_weight'] for g in groups)
    originals = {k: Replay(m['base_model'], graph) for k, m in models.items()}
    seeds = {k: r.state.copy() for k, r in originals.items()}
    rows = []
    for name, packed in models.items():
        for dt_factor in [1., .5]:
            for seed_name, seed in seeds.items():
                model = copy.deepcopy(packed['base_model'])
                model['config']['dt'] *= dt_factor
                model['config']['preparation_seconds'] = 480.
                model['initial'] = seed.tolist()
                replay = Replay(model, graph)
                gains = np.exp(packed['extension_parameters']['observation']['log_gain']['values'])
                mse = 0.
                for group in groups:
                    pred = replay.response(training['names'][group['target']], group['recording']['times']) * gains / replay.gain
                    traces = group['recording']['traces']
                    weights = np.array([t['provenance']['id_confidence'] for t in traces])
                    mean = np.array([t['values'] for t in traces]).T
                    observed = pred[:, [replay.index[t['neuron']] for t in traces]]
                    mse += group['sample_weight']/total * (float(np.sum((observed-mean)**2*weights)/(weights.sum()*len(pred))) + group['irreducible_mse'])
                row = {'parameters': name, 'seed_state': seed_name, 'dt': replay.dt,
                    'relaxation_seconds': 480., 'training_mse': mse,
                    'rhs_linf': float(np.max(np.abs(replay.rhs(replay.state, None, 0.)))),
                    'distance_to_original_states_linf': {k: float(np.max(np.abs(replay.state-s))) for k, s in seeds.items()},
                    'state': replay.state.tolist()}
                if not np.isfinite(replay.state).all() or not np.isfinite(mse):
                    raise ValueError('nonfinite cross-initialization')
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k != 'state'}), flush=True)
    result = {'schema_version': 1,
        'input_sha256': {k: digest(getattr(args,k)) for k in ['checkpoint','probe','excursion','graph','training']},
        'script_sha256': digest(__file__),
        'replay_script_sha256': digest(Path(__file__).with_name('replay_level0_atlas.py')),
        'original_states': {k:v.tolist() for k,v in seeds.items()}, 'variants': rows,
        'scope': 'Two frozen parameter sets, two prepared-state seeds, two integration steps, 480-second unforced relaxation. Training-only response diagnostic, not fitting, branch selection, or a formal stability/bifurcation proof.'}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
