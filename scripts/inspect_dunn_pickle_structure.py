#!/usr/bin/env python3
"""Inspect processed-recording field names without constructing scientific arrays.

All pickle globals become inert placeholders. No upstream classes or NumPy
constructors are imported. This is a metadata locator, not a recording loader.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import pickle
import pickletools
from pathlib import Path


class Opaque:
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.arguments = args
        obj.keyword_arguments = kwargs
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


class StructureUnpickler(pickle.Unpickler):
    def __init__(self, stream):
        super().__init__(stream)
        self.globals = {}

    def find_class(self, module, name):
        key = (module, name)
        if key not in self.globals:
            self.globals[key] = type('OpaqueGlobal', (Opaque,), {'pickle_global': key})
        return self.globals[key]

    def persistent_load(self, pid):
        raise ValueError('persistent references are not supported')


def load_structure(raw):
    # Extension opcodes can bypass find_class via the process-global cache.
    for opcode, _, _ in pickletools.genops(raw):
        if opcode.name in {'EXT1', 'EXT2', 'EXT4', 'PERSID', 'BINPERSID'}:
            raise ValueError('unsupported reference opcode: ' + opcode.name)
    loader = StructureUnpickler(io.BytesIO(raw))
    root = loader.load()
    if not isinstance(root, Opaque) or not isinstance(getattr(root, 'state', None), dict):
        raise ValueError('expected an object with dictionary state')
    return root, loader


def inspect(raw):
    root, loader = load_structure(raw)
    state = root.state
    metadata = state.get('md')
    if not isinstance(metadata, dict):
        raise ValueError('expected md dictionary')

    def structure(mapping):
        if any(not isinstance(k, str) for k in mapping):
            raise ValueError('field names must be strings')
        return {key: ('opaque:' + '.'.join(value.pickle_global)
                      if isinstance(value, Opaque) else type(value).__name__)
                for key, value in sorted(mapping.items())}

    events = metadata.get('stim_metadata', {}).get('stim_param_list')
    event_schema = None
    if isinstance(events, list):
        fields = {}
        event_types = Counter()
        for item in events:
            if not isinstance(item, dict):
                raise ValueError('expected delivered event dictionaries')
            for key, kind in structure(item).items():
                fields.setdefault(key, set()).add(kind)
            event = item.get('event')
            if isinstance(event, dict):
                for key, kind in structure(event).items():
                    fields.setdefault('event.' + key, set()).add(kind)
                event_type = event.get('event_type')
                if isinstance(event_type, str):
                    event_types[event_type] += 1
        event_schema = {
            'count': len(events),
            'field_types': {k: sorted(v) for k, v in sorted(fields.items())},
            'event_type_counts': dict(sorted(event_types.items())),
        }

    stimulus_lengths = {}
    for prefix, container in [('object', state),
                              ('md.stim_metadata', metadata.get('stim_metadata', {})),
                              ('md.alg_metadata', metadata.get('alg_metadata', {}))]:
        if isinstance(container, dict):
            for key, value in container.items():
                if key.startswith('stim_') and isinstance(value, list):
                    stimulus_lengths[prefix + '.' + key] = len(value)

    # Only structure, event counts and event-type labels leave the loader.
    return {
        'object_global': list(root.pickle_global),
        'object_fields': structure(state),
        'metadata_fields': structure(metadata),
        'nested_metadata_fields': {k: structure(v) for k, v in sorted(metadata.items()) if isinstance(v, dict)},
        'delivered_event_schema': event_schema,
        'stimulus_list_lengths': dict(sorted(stimulus_lengths.items())),
        'replaced_globals': [list(k) for k in sorted(loader.globals)],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pickle', required=True)
    p.add_argument('--download-receipt', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError('output already exists')
    receipt_raw = Path(a.download_receipt).read_bytes()
    download = json.loads(receipt_raw)
    raw = Path(a.pickle).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != download['bytes'] or sha != download['sha256']:
        raise ValueError('download receipt mismatch')
    result = inspect(raw)
    result.update({
        'schema_version': 1, 'pickle_sha256': sha,
        'download_receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'One candidate processed recording: field names/types and delivered-event count/type categories only. All pickle globals replaced by inert placeholders; numerical arrays not constructed, response values not reported. Does not establish stimulus eligibility or animal independence.',
    })
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
