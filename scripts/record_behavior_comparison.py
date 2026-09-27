#!/usr/bin/env python3
"""Bind the three validation-selected behavior-assisted fits and their common inputs."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='runs/behavior-comparison.json')
args = parser.parse_args()


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


models = {}
receipts = {}
common = None
for name in ['lds', 'gru', 'level0']:
    path = f'runs/{name}-behavior-fit.json'
    model = load(path)
    receipt = load(f'runs/{name}-behavior-receipt.json')
    if digest(path) != receipt['model_sha256']:
        raise ValueError(f'{name}: model hash mismatch')
    behavior = dict(model['behavior'])
    behavior.pop('source_commit')  # Metadata differs; coefficients and lineage must match exactly.
    if common is None:
        common = behavior
    elif behavior != common:
        raise ValueError(f'{name}: behavior coefficients or lineage differ')
    for partition in ['validation', 'test']:
        for key in ['dataset_hash', 'split_hash', 'graph_hash']:
            if receipt[partition][key] != model[key] or model[key] != behavior[key]:
                raise ValueError(f'{name}: {key} mismatch')
    if model['training_trials'] != behavior['training_trials']:
        raise ValueError(f'{name}: neural/behavior training cohorts differ')
    if models and model['selection_trials'] != next(iter(models.values()))['selection_trials']:
        raise ValueError(f'{name}: validation cohorts differ')
    selection = receipt['fit' if name == 'level0' else 'selection']
    for key in ['source_commit', 'dataset_hash', 'split_hash']:
        if selection[key] != model[key]:
            raise ValueError(f'{name}: selection {key} mismatch')
    candidates = selection['epochs' if name == 'level0' else 'candidates']
    if name == 'lds':
        selected = [c for c in candidates if c['rank'] == model['gaussian']['dim'] and c['iteration'] == model['iteration']]
    else:
        epoch = model['selected_epoch' if name == 'level0' else 'epoch']
        selected = [c for c in candidates if c['epoch'] == epoch]
    if len(selected) != 1 or selected[0]['validation_criterion'] != max(c['validation_criterion'] for c in candidates):
        raise ValueError(f'{name}: model is not the validation-selected candidate')
    # Recorded selection must agree with independently generated validation predictions.
    horizons = receipt['validation']['animal_bootstrap']['horizons']
    for value, horizon in zip(selected[0]['validation_horizon_r2'], horizons):
        if abs(value - horizon['confidence_weighted']['point']) > 1e-12:
            raise ValueError(f'{name}: validation prediction/selection mismatch')
    models[name] = model
    receipts[name] = receipt

legacy_path = 'runs/level0-filter-compat-predictions.json'
legacy = load(legacy_path)
archived = load('runs/level0-filter-test-predictions.json')
if [(t['id'], t['fluorescence']) for t in legacy['trials']] != [(t['id'], t['fluorescence']) for t in archived['trials']]:
    raise ValueError('Legacy Level 0 numerical predictions changed')

summary = {}
for name, receipt in receipts.items():
    summary[name] = {
        'model_sha256': receipt['model_sha256'],
        'training_source_commit': models[name]['source_commit'],
        'behavior_source_commit': models[name]['behavior']['source_commit'],
        'receipt_path': f'docs/{name}-behavior-receipt.json',
        'receipt_sha256': digest(f'runs/{name}-behavior-receipt.json'),
        'selected': ({'epoch': models[name]['selected_epoch']} if name == 'level0' else
                     {'epoch': models[name]['epoch']} if name == 'gru' else
                     {'rank': models[name]['gaussian']['dim'], 'iteration': models[name]['iteration']}),
        'validation_horizons': receipt['validation']['animal_bootstrap']['horizons'],
        'test_horizons': receipt['test']['animal_bootstrap']['horizons'],
        'reported_free_parameters': receipt['test']['free_parameters'],
        'additional_calibration_statistics': receipt['fit']['calibration_statistics'] if name == 'level0' else 0,
        'preprocessing_assessment': receipt['test']['preprocessing_assessment'],
    }
output = {
    'schema_version': 1,
    'scope': 'Exploratory comparison on a fixed retrospective processed-signal benchmark. All artifacts selected on validation before this test scoring.',
    'common_behavior_parameters_and_lineage_equal_excluding_source_commit': True,
    'common_behavior_forecast': common,
    'input_protocol': 'All three families call the same Rust BehaviorModel::inputs implementation. Equal coefficients, channel order, calibration, sample interval and trial data determine identical eight-dimensional covariates. Actual future behavior is excluded. Source revision metadata differs and is retained separately.',
    'behavior_implementation_sha256': digest('src/bench/behavior.rs'),
    'models': summary,
    'legacy_level0_compatibility': {
        'equal_prediction_trials': len(legacy['trials']),
        'archived_predictions_sha256': digest('runs/level0-filter-test-predictions.json'),
        'current_predictions_sha256': digest(legacy_path),
        'scope': 'Exact numerical fluorescence-array equality; metadata revisions differ.',
    },
    'limitations': [
        'Only three test animals, repeatedly inspected; animal-bootstrap intervals have limited resolution.',
        'Marginal model intervals do not establish significance of differences; no paired-difference claim.',
        'Additional capacity, different neural training objectives and inference methods prevent causal attribution to behavior inputs alone.',
        'Level 0 reports calibration separately; LDS and GRU include it in reported scalar counts.',
        'LDS and GRU use persistence for unseen identities; Level 0 uses default readouts.',
        'Behavior trajectories are plug-in forecasts, with no propagated behavior uncertainty.',
        'Upstream retrospective preprocessing prevents prospective forecasting claims.',
    ],
}
Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
print(args.output)
