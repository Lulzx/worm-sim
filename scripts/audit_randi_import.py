#!/usr/bin/env python3
"""Independently check every emitted atlas trace against source arrays and audit the split."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/randi-source/exported_data'


def load(name):
    return json.loads((ROOT / name).read_text())


def digest(name):
    with (ROOT / name).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


data = load('runs/randi-data.json')
report = load('runs/randi-import.json')
split = load('runs/randi-split.json')
assert report['dataset_hash'] == split['dataset_hash']
assert report['manifest_sha256'] == digest('data/randi-source-manifest.json')
assert data['graph_hash'] == split['graph_hash']
trials = {t['id']: t for t in data['trials']}
assert len(trials) == len(data['trials']) == len(report['events'])
assert set(trials) == {e['trial'] for e in report['events']}
partitions = {p: set(split[p]) for p in ['train', 'validation', 'test']}
assert set.union(*partitions.values()) == set(trials)
assert sum(map(len, partitions.values())) == len(trials)
groups = {p: {trials[t]['stimulated_neuron'] for t in ids} for p, ids in partitions.items()}
assert sum(map(len, groups.values())) == len(set.union(*groups.values()))
assert all(t['forecast_origin'] is None and not t['response_labels'] for t in trials.values())
by_recording = {}
for event in report['events']:
    by_recording.setdefault(event['source_recording'], []).append(event)
max_error = 0.0
samples = 0
missing = 0
trace_windows = 0
for events in by_recording.values():
    events.sort(key=lambda e: e['source_frames'][0])
    assert all(a['source_frames'][1] <= b['source_frames'][0] for a, b in zip(events, events[1:]))
    index = int(events[0]['trial'].split('-')[1])
    matrix = np.loadtxt(SOURCE / f'{index}_gcamp.txt')
    times = np.loadtxt(SOURCE / f'{index}_t.txt')
    labels = [label.strip() for label in (SOURCE / f'{index}_labels.txt').read_text().splitlines()]
    stim_frames = np.atleast_1d(np.loadtxt(SOURCE / f'{index}_stim_volume_i.txt', dtype=int))
    stim_rois = np.atleast_1d(np.loadtxt(SOURCE / f'{index}_stim_neurons.txt', dtype=int))
    for event in events:
        trial = trials[event['trial']]
        frame = event['source_stimulation_frame']
        start, stop = event['source_frames']
        assert frame == stim_frames[event['event_index']]
        assert trial['stimulated_neuron'] == labels[stim_rois[event['event_index']]]
        assert event['source_stimulation_seconds'] == times[frame]
        assert start == np.searchsorted(times, times[frame] - report['config']['baseline_seconds'])
        assert stop == np.searchsorted(times, times[frame] + report['config']['response_seconds'])
        assert np.array_equal(trial['recording']['times'], times[frame:stop] - times[frame])
        assert len(trial['recording']['traces']) == event['observed_traces']
        for trace in trial['recording']['traces']:
            assert labels.count(trace['neuron']) == 1
            column = labels.index(trace['neuron'])
            baseline = matrix[start:frame, column]
            baseline = baseline[np.isfinite(baseline)]
            assert len(baseline) / (frame-start) >= report['config']['minimum_baseline_fraction']
            mean = np.mean(baseline)
            assert mean > 0
            expected = (matrix[frame:stop, column] - mean) / mean
            actual = np.array([np.nan if v is None else v for v in trace['values']])
            finite = np.isfinite(expected)
            assert np.array_equal(finite, np.isfinite(actual))
            assert np.allclose(actual[finite], expected[finite], rtol=1e-10, atol=1e-10)
            max_error = max(max_error, float(np.max(np.abs(actual[finite] - expected[finite]))))
            samples += int(finite.sum())
            missing += int((~finite).sum())
            trace_windows += 1

receipt = {
    'schema_version': 1,
    'importer_source_commit': report['source_commit'],
    'audit_script_sha256': digest('scripts/audit_randi_import.py'),
    'dataset_hash': report['dataset_hash'],
    'dataset_file_sha256': digest('runs/randi-data.json'),
    'split_file_sha256': digest('runs/randi-split.json'),
    'manifest_sha256': report['manifest_sha256'],
    'import_report_sha256': digest('runs/randi-import.json'),
    'source_recordings': report['source_recordings'],
    'represented_recordings': len(by_recording),
    'source_events': report['source_events'],
    'trailing_blank_labels': report['trailing_blank_labels'],
    'reordered_event_recordings': report['reordered_event_recordings'],
    'included_events': len(trials),
    'excluded_events': report['excluded_events'],
    'excluded_trace_windows': report['excluded_trace_windows'],
    'excluded_labels': report['excluded_labels'],
    'trace_windows_checked': trace_windows,
    'finite_response_samples_checked': samples,
    'missing_response_samples_checked': missing,
    'maximum_absolute_numpy_delta_f_over_f_error': max_error,
    'partitions': {p: {'trials': len(ids), 'stimulated_neurons': sorted(groups[p]), 'recordings': len({trials[t]['recording']['animal_id'] for t in ids})} for p, ids in partitions.items()},
    'stimulated_neuron_partitions_disjoint': True,
    'retained_source_sample_ranges_disjoint_including_baselines': True,
    'response_label_count': 0,
    'limitations': report['limitations'] + ['No atlas model fitted or scored by this audit; no comparison to a retrained held-out-neuron Creamer baseline.'],
}
path = ROOT / 'docs/randi-import-audit.json'
path.write_text(json.dumps(receipt, indent=2) + '\n')
print(path)
print(f'{len(trials)} events, {trace_windows} traces, {samples} finite samples; max error {max_error:.3g}')
