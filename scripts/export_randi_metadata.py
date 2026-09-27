#!/usr/bin/env python3
"""Export inert numeric metadata from the hash-pinned full atlas; never import upstream code."""
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/randi-source'
ARCHIVE = SOURCE / 'exported_data_full.tar.gz'
FULL_SHA = 'f59e8f1f74cc468559a230a3b44832ebe394680be0b73ee871633510a4df9165'
FULL_URL = 'https://osf.io/download/34m5v/?version=1'
ATLAS_SHA = '53a99055667b853e1d3d6be573ec2613d38c9f6989f302ec38ecd38dd50c7975'
ATLAS_URL = 'https://raw.githubusercontent.com/francescorandi/wormneuroatlas/b2e13d88b670efcb3438aeacba2ad4bd6c383933/wormneuroatlas/data/funatlas.h5'


class NumericContainer:
    pass


class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in [('pumpprobe.Fconn', 'Fconn'), ('wormdatamodel.data.recording', 'recording')]:
            return NumericContainer
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(np._core.multiarray, name)
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        raise ValueError(f'Unsupported pickle global {module}.{name}')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(path, url, expected):
    if not path.exists() or digest(path) != expected:
        temp = path.with_suffix(path.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=90) as response, temp.open('wb') as f:
            while block := response.read(1024 * 1024):
                f.write(block)
        if digest(temp) != expected:
            raise ValueError('Downloaded source hash mismatch')
        temp.replace(path)


def numeric(value):
    return value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value


def main():
    SOURCE.mkdir(exist_ok=True)
    fetch(ARCHIVE, FULL_URL, FULL_SHA)
    fetch(SOURCE / 'funatlas.h5', ATLAS_URL, ATLAS_SHA)
    manifest_path = ROOT / 'data/randi-source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = {f['path']: f for f in manifest['files']}

    def text(index, suffix):
        name = f'{index}_{suffix}.txt'
        path = SOURCE / 'exported_data' / name
        if path.stat().st_size != files[name]['bytes'] or digest(path) != files[name]['sha256']:
            raise ValueError('Text source mismatch: ' + name)
        return path.read_text()

    records = {}
    for entry in manifest['files']:
        if not entry['path'].endswith('_ds_name.txt'):
            continue
        index = int(entry['path'].split('_')[0])
        name = Path(text(index, 'ds_name').strip()).name
        if name in records:
            raise ValueError('Duplicate recording basename')
        records[name] = {'index': index, 'source_recording': name}
    source_entries = []
    with tarfile.open(ARCHIVE, 'r|gz') as archive:
        for entry in archive:
            parts = Path(entry.name).parts
            if len(parts) != 3 or parts[0] != 'exported_data_full' or parts[1] not in records or parts[2] not in ['fconn.pickle', 'recording.pickle']:
                continue
            if not entry.isfile():
                raise ValueError('Metadata member is not a regular file')
            raw = archive.extractfile(entry).read()
            obj = NumericUnpickler(io.BytesIO(raw)).load()
            rec = records[parts[1]]
            section = 'detector' if parts[2] == 'fconn.pickle' else 'acquisition'
            if section in rec:
                raise ValueError('Duplicate archive metadata member')
            source_entries.append({'path': entry.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            if section == 'detector':
                index = rec['index']
                frames = np.atleast_1d(np.loadtxt(io.StringIO(text(index, 'stim_volume_i')), dtype=int))
                targets = np.atleast_1d(np.loadtxt(io.StringIO(text(index, 'stim_neurons')), dtype=int))
                if not np.array_equal(frames, obj.i0s + obj.shift_vols) or not np.array_equal(targets, obj.stim_neurons):
                    raise ValueError('Full/text stimulation metadata differs')
                if len(obj.resp_neurons_by_stim) != len(frames):
                    raise ValueError('Response annotation/event count mismatch')
                responses = [np.asarray(r, dtype=int).tolist() for r in obj.resp_neurons_by_stim]
                if any(i < 0 or i >= obj.n_neurons for row in responses for i in row):
                    raise ValueError('Response ROI out of bounds')
                rec[section] = {
                    'n_neurons': int(obj.n_neurons), 'sample_dt': float(obj.Dt),
                    'stimulus_frames': frames.tolist(), 'target_rois': targets.tolist(),
                    'response_rois_by_event': responses,
                    'targeted_neuron_hit': numeric(obj.targeted_neuron_hit),
                    'detector_window_start_frames': numeric(obj.i0s),
                    'detector_window_stop_frames': numeric(obj.i1s),
                    'settings': {k: numeric(getattr(obj, k, None)) for k in ['nan_thresh', 'deriv_thresh', 'ampl_thresh', 'deriv_min_time', 'ampl_min_time']},
                    'negative_response_eligibility': None,
                }
            else:
                rec[section] = {k: numeric(getattr(obj, k, None)) for k in ['optogeneticsType', 'optogeneticsN', 'optogeneticsFrameCount', 'optogeneticsNPulses', 'optogeneticsRepRateDivider', 'optogeneticsNTrains', 'optogeneticsTimeBtwTrains']}
    if any('detector' not in r or 'acquisition' not in r for r in records.values()):
        raise ValueError('Missing full metadata for a text recording')
    records = sorted(records.values(), key=lambda r: r['index'])
    output = SOURCE / 'annotations.json'
    output.write_text(json.dumps({'schema_version': 1, 'full_archive_sha256': FULL_SHA, 'text_manifest_sha256': digest(manifest_path), 'records': records}, allow_nan=False) + '\n')
    receipt = {
        'schema_version': 1, 'export_script_sha256': digest(Path(__file__)),
        'full_archive_url': FULL_URL, 'full_archive_sha256': FULL_SHA, 'full_archive_bytes': ARCHIVE.stat().st_size,
        'pair_statistics_url': ATLAS_URL, 'pair_statistics_sha256': ATLAS_SHA,
        'text_manifest_sha256': digest(manifest_path), 'annotation_export_sha256': digest(output),
        'verified_recordings': len(records),
        'stimulation_entries_matched_exactly': sum(len(r['detector']['stimulus_frames']) for r in records),
        'positive_detector_roi_entries': sum(len(row) for r in records for row in r['detector']['response_rois_by_event']),
        'acquisition_detector_count_mismatches': [r['index'] for r in records if r['acquisition']['optogeneticsN'] != len(r['detector']['stimulus_frames'])],
        'metadata_files': source_entries,
        'limitations': [
            'Per-event positive detector lists are preserved; absent list membership is not converted into a negative response label.',
            'Per-ROI detector eligibility masks were not preserved in Fconn objects; reconstructed raw signal quality would require a separate audited pipeline.',
            'Pulse counts/dividers/train counts are raw acquisition fields, not calibrated membrane currents or a verified duration/power conversion.',
            'Acquisition metadata is not joined to individual detector events merely by position; count mismatches are retained.',
            'Pair-level detection q and equivalence q are separate statistical tests and may both be significant.',
        ],
    }
    path = ROOT / 'docs/randi-metadata-audit.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(path)
    print(receipt['verified_recordings'], receipt['stimulation_entries_matched_exactly'], receipt['positive_detector_roi_entries'])


if __name__ == '__main__':
    main()
