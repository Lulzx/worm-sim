#!/usr/bin/env python3
"""Nominal light-spot geometry audit; decode only allowlisted spatial arrays."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from inspect_dunn_identity_metadata import identity_labels
from inspect_dunn_pickle_structure import Opaque, load_structure

EXPRESSING = {'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR', 'RIVL', 'RIVR'}


def coordinates(state, key):
    if key not in {'x', 'y'}:
        raise ValueError('only x and y coordinate arrays may be decoded')
    value = state[key]
    if not isinstance(value, Opaque) or value.pickle_global not in {
        ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct')
    }:
        raise ValueError('expected inert coordinate array')
    version, shape, dtype, fortran, raw = value.state
    if (version != 1 or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)
            or not isinstance(dtype, Opaque) or dtype.pickle_global != ('numpy', 'dtype')
            or dtype.arguments[0] != 'f8' or dtype.state[1] not in ('<', '>')
            or type(fortran) is not bool or not isinstance(raw, bytes)
            or len(raw) != math.prod(shape) * 8):
        raise ValueError('unsupported coordinate encoding')
    result = np.frombuffer(raw, dtype=dtype.state[1] + 'f8').reshape(shape, order='F' if fortran else 'C')
    if not np.isfinite(result).all():
        raise ValueError('nonfinite coordinates')
    return result


def spot_members(x, y, cx, cy, diameter):
    if not all(math.isfinite(v) for v in (cx, cy, diameter)) or diameter <= 0:
        raise ValueError('invalid circle')
    return np.flatnonzero((x - cx)**2 + (y - cy)**2 <= (diameter / 2)**2).tolist()


def audit(state):
    if state['quant_method'] != 'gcamp-extractor':
        raise ValueError('only the audited gcamp-extractor convention is supported')
    metadata = state['md']
    if metadata['gooey_args']['subject_strain'] != 'FC121':
        raise ValueError('expression annotation is specific to FC121')
    zsize = metadata['gooey_args']['zsize']
    if type(zsize) is not int or zsize <= 0:
        raise ValueError('invalid frame-to-volume factor')
    total, _, _ = identity_labels(state['ID1'])
    labels = state['ID1'].state[4]
    x, y = coordinates(state, 'x'), coordinates(state, 'y')
    if x.shape != y.shape or x.shape[1] != total:
        raise ValueError('coordinate/identity dimensions differ')
    moco = metadata['postprocessing']['moco']
    if moco['registration_method'] != 'manual_rigid_xy':
        raise ValueError('unsupported registration method')
    offsets = moco['registration_global_offset']
    if any(len(offsets[k]) != len(x) for k in ('x', 'y')):
        raise ValueError('offset lengths differ')
    raw = metadata['stim_metadata']['stim_param_list']
    processed = state['stim_param_list']
    if len(raw) != len(processed):
        raise ValueError('delivered and processed event counts differ')
    rows = []
    maximum_error = 0.
    for index, (r, p) in enumerate(zip(raw, processed)):
        if (r['stim_on'], r['stim_off']) != (p['stim_on'], p['stim_off']):
            raise ValueError('event alignment differs')
        onset = p['stim_on']
        if type(onset) is not int or onset % zsize:
            raise ValueError('onset is not on a volume boundary')
        t = onset // zsize
        if not 0 <= t < len(x):
            raise ValueError('onset outside coordinate grid')
        re, pe = r['event'], p['event']
        if re['event_type'] != 'circle-button' or pe['event_type'] != 'circle-button':
            raise ValueError('unsupported event type')
        if re['stim_diameter'] != pe['stim_diameter']:
            raise ValueError('diameter differs')
        center = [pe[k] for k in ('x', 'y')]
        for k, c in zip(('x', 'y'), center):
            error = abs(c - (re[k] - offsets[k][t]))
            if not math.isfinite(error) or error > 1e-8:
                raise ValueError('saved registration correction differs')
            maximum_error = max(maximum_error, error)
        inside = spot_members(x[t], y[t], *center, pe['stim_diameter'])
        named = [labels[i] for i in inside if isinstance(labels[i], str) and labels[i]]
        expressing = sorted(n for n in named if n in EXPRESSING)
        unknown = len(inside) - len(named)
        rows.append({'event_index': index, 'onset_volume_index': t,
                     'inside_segmented_count': len(inside), 'inside_missing_identity_count': unknown,
                     'inside_named_labels': sorted(named), 'inside_named_expressing_labels': expressing})
    return {'segmented_neurons': total, 'events': rows,
            'registration_max_absolute_error_pixels': maximum_error,
            'events_by_named_expressing_count': dict(sorted(Counter(len(r['inside_named_expressing_labels']) for r in rows).items())),
            'events_with_unlabeled_somata_inside': sum(r['inside_missing_identity_count'] > 0 for r in rows),
            'named_expressing_event_counts': dict(sorted(Counter(n for r in rows for n in r['inside_named_expressing_labels']).items()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('pickle', 'download-receipt', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError('output already exists')
    raw = Path(a.pickle).read_bytes()
    receipt_raw = Path(a.download_receipt).read_bytes()
    receipt = json.loads(receipt_raw)
    sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != receipt['bytes'] or sha != receipt['sha256']:
        raise ValueError('download receipt differs')
    root, _ = load_structure(raw)
    result = audit(root.state)
    result.update({'schema_version': 1, 'pickle_sha256': sha,
                   'download_receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
                   'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ['audit_dunn_spatial_metadata.py', 'inspect_dunn_pickle_structure.py', 'inspect_dunn_identity_metadata.py']},
                   'rule': 'gcamp-extractor x/y at stim_on // zsize; processed circle center equals delivered center minus saved XY offset; radius=diameter/2; inclusive Euclidean XY disk.',
                   'scope': 'Identity and XY coordinate metadata only. Response arrays remain opaque. Nominal projected geometry and class-level opsin annotation do not establish effective illumination, single-neuron targeting, label confidence or animal independence.'})
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k != 'events'}, indent=2))


if __name__ == '__main__':
    main()
