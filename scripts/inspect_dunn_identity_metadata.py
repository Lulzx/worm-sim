#!/usr/bin/env python3
"""Inspect only identity labels and explicit subject metadata in an inert pickle."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from inspect_dunn_pickle_structure import Opaque, load_structure


def identity_labels(value):
    if not isinstance(value, Opaque) or value.pickle_global not in {
        ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct')
    }:
        raise ValueError('expected inert NumPy identity array')
    state = getattr(value, 'state', None)
    if not isinstance(state, tuple) or len(state) != 5:
        raise ValueError('unsupported identity array state')
    version, shape, dtype, _, labels = state
    if (version != 1 or not isinstance(shape, tuple) or len(shape) != 1
            or not isinstance(labels, list) or shape != (len(labels),)
            or not isinstance(dtype, Opaque) or dtype.pickle_global != ('numpy', 'dtype')
            or not dtype.arguments or dtype.arguments[0] not in ('O', 'O8', 'O4')):
        raise ValueError('expected one-dimensional object identity array')
    names = []
    missing = 0
    for label in labels:
        if label is None or isinstance(label, float) and math.isnan(label) or label == '':
            missing += 1
        elif isinstance(label, str):
            names.append(label)
        else:
            raise ValueError('unsupported identity label type')
    return len(labels), missing, Counter(names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('pickle', 'download-receipt', 'graph', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError('output already exists')
    raw = Path(a.pickle).read_bytes()
    receipt_raw = Path(a.download_receipt).read_bytes()
    receipt = json.loads(receipt_raw)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != receipt['bytes'] or digest != receipt['sha256']:
        raise ValueError('download receipt mismatch')
    root, _ = load_structure(raw)
    total, missing, labels = identity_labels(root.state.get('ID1'))
    graph_raw = Path(a.graph).read_bytes()
    canonical = {n['id'] for n in json.loads(graph_raw)['neurons']}
    metadata = root.state['md']
    subject = {}
    fields = ('animal_id', 'worm_id', 'subject_id', 'animal_identifier', 'subject_identifier', 'subject_strain')
    for prefix, container in [('md', metadata), ('md.gooey_args', metadata.get('gooey_args', {}))]:
        for field in fields:
            if field in container:
                value = container[field]
                if not isinstance(value, (str, int, bool)) and value is not None:
                    raise ValueError('unsupported subject metadata scalar')
                subject[prefix + '.' + field] = value
    result = {
        'schema_version': 1, 'pickle_sha256': digest,
        'download_receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
        'graph_sha256': hashlib.sha256(graph_raw).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inert_loader_sha256': hashlib.sha256(Path(__file__).with_name('inspect_dunn_pickle_structure.py').read_bytes()).hexdigest(),
        'segmented_neurons': total, 'missing_labels': missing,
        'canonical_label_counts': {k: v for k, v in sorted(labels.items()) if k in canonical},
        'unmatched_label_counts': {k: v for k, v in sorted(labels.items()) if k not in canonical},
        'explicit_subject_fields': subject,
        'checked_subject_field_names': list(fields),
        'scope': 'Only ID1 identity strings and the explicit subject-field allowlist inspected. NumPy arrays and upstream classes remain inert; neural response values not reported. Canonical name matches do not establish labeling confidence, target uniqueness, opsin expression or animal independence.',
    }
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
