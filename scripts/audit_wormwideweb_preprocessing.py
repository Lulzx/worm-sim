#!/usr/bin/env python3
"""Verify source hashes and retrospectively normalized HDF5 traces; execute no upstream code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_arrays(original, published, times):
    original = np.asarray(original, dtype=np.float64)
    published = np.asarray(published, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if original.ndim != 2 or original.shape != published.shape or original.shape[0] != len(times):
        raise ValueError('Expected time-by-neuron arrays matching timestamps')
    if len(times) < 4 or not np.isfinite(original).all() or not np.isfinite(published).all() or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Invalid numerical inputs')
    scale = original.std(axis=0, ddof=1)
    if not (scale > 0).all():
        raise ValueError('Cannot audit constant original traces')
    normalized = (original - original.mean(axis=0)) / scale
    error = float(np.max(np.abs(normalized - published)))
    # Synthetic intervention in the later half only, not an observed biological effect.
    # Recompute the source's normalization to demonstrate its temporal dependency.
    boundary = len(times)//2
    changed = original.copy()
    changed[boundary:] += scale
    transformed = (changed - changed.mean(axis=0)) / changed.std(axis=0, ddof=1)
    prefix = int(np.searchsorted(times, times[0]+10.0, side='right'))
    prefix = min(prefix, boundary)
    return {
        'frames': len(times), 'neurons': original.shape[1],
        'max_abs_published_mean': float(np.max(np.abs(published.mean(axis=0)))),
        'max_abs_published_sample_std_minus_one': float(np.max(np.abs(published.std(axis=0, ddof=1)-1))),
        'max_abs_whole_recording_zscore_error': error,
        'matches_whole_recording_zscore_at_1e_minus_10': error < 1e-10,
        'future_only_counterfactual': {
            'description': 'Add one original full-recording sample SD per neuron to the later half, recompute whole-recording z-score; original prefix unchanged. Demonstration of normalization dependence, not a measured performance effect.',
            'changed_from_source_frame_zero_based': boundary,
            'prefix_frames_compared': prefix,
            'original_prefix_max_change': float(np.max(np.abs(changed[:prefix]-original[:prefix]))),
            'normalized_prefix_max_change': float(np.max(np.abs(transformed[:prefix]-normalized[:prefix]))),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='runs/wormwideweb-source/baseline')
    parser.add_argument('--fetch-receipt', default='docs/wormwideweb-fetch-receipt.json')
    parser.add_argument('--import-receipt', default='runs/wormwideweb-benchmark.json.import.json')
    parser.add_argument('--sources', default='data/wormwideweb-preprocessing-sources.json')
    parser.add_argument('--source-cache', default='runs/wormwideweb-preprocessing-source')
    parser.add_argument('--output', default='runs/wormwideweb-preprocessing-audit.json')
    args = parser.parse_args()
    fetch = json.loads(Path(args.fetch_receipt).read_text())
    imported = json.loads(Path(args.import_receipt).read_text())
    sources = json.loads(Path(args.sources).read_text())
    verified = []
    for source in sources:
        url = 'https://raw.githubusercontent.com/flavell-lab/{repo}/{commit}/{path}'.format(**source)
        path = Path(args.source_cache)/source['repo']/source['commit']/source['path']
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            content = urllib.request.urlopen(url, timeout=30).read()
            if hashlib.sha256(content).hexdigest() != source['sha256']:
                raise ValueError('Source hash mismatch: '+url)
            path.write_bytes(content)
        if digest(path) != source['sha256']:
            raise ValueError('Cached source hash mismatch: '+str(path))
        verified.append({**source, 'url': url})
    animals = []
    expected = {a['animal']: a['source_sha256'] for a in imported['animals']}
    for animal in fetch['animals']:
        path = Path(args.directory)/animal['file']
        sha = digest(path)
        if sha != animal['sha256'] or expected.get(animal['animal_id']) != sha:
            raise ValueError('HDF5/import source mismatch: '+str(path))
        with h5py.File(path) as f:
            result = audit_arrays(f['gcamp/trace_array_original'][:], f['gcamp/trace_array'][:], f['timing/timestamp_confocal'][:])
            result['behavior_channels'] = sorted(f['behavior'].keys())
        animals.append({'animal': animal['animal_id'], 'sha256': sha, **result})
    if len(animals) != len(expected) or len({a['animal'] for a in animals}) != len(expected):
        raise ValueError('Audit animal coverage mismatch')
    receipt = {
        'schema_version': 1,
        'dataset_hash': imported['dataset_hash'],
        'graph_hash': imported['graph_hash'],
        'import_receipt_sha256': digest(args.import_receipt),
        'source_manifest_sha256': digest(args.sources),
        'python_dependencies': {'numpy': np.__version__, 'h5py': h5py.__version__},
        'sources': verified,
        'observed_status': 'retrospective_whole_recording_standardization' if all(a['matches_whole_recording_zscore_at_1e_minus_10'] for a in animals) else 'normalization_hypothesis_not_confirmed_for_all_animals',
        'animals': animals,
        'interpretation': [
            'All input hashes are checked against fetch and import receipts. Numerical normalization checks cover every exported ROI, not only selected NeuroPAL labels.',
            'The present benchmark is prediction of retrospectively processed traces; code-level exclusion of future targets does not establish end-to-end temporal causality.',
            'Whole-recording mean/scale include future frames of the same animal. This is distinct from train/test animal identity leakage.',
            'Reference notebook cell 179 enables interpolation, marker division and bleach correction before original and z-scored exports. Exact per-file processing invocations are not attested by this source audit.',
            'Reference behavior code contains centered angular-velocity/pumping filtering, imputation and velocity artifact interpolation; notebook cell 9 sets lag 150 at 20 Hz. Exact effective lookahead for each exported behavior sample is not recovered.',
            'Choosing trace_array_original or adding prefix-only normalization removes neither previously applied interpolation nor global bleaching correction.',
            'Existing scores remain reproducible on the hashed retrospective benchmark; their bias relative to a prospectively processed benchmark is unmeasured.',
        ],
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2)+'\n')
    print(receipt['observed_status'], len(animals), 'animals', sum(a['neurons'] for a in animals), 'exported ROI traces')
    print(args.output)

if __name__ == '__main__':
    main()
