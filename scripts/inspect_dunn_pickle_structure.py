#!/usr/bin/env python3
"""Inspect processed-recording field names without constructing scientific arrays.

All pickle globals become inert placeholders. No upstream classes or NumPy
constructors are imported. This is a metadata locator, not a recording loader.
"""
import argparse
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


def inspect(raw):
    # Extension opcodes can bypass find_class via the process-global cache.
    for opcode, _, _ in pickletools.genops(raw):
        if opcode.name in {'EXT1', 'EXT2', 'EXT4', 'PERSID', 'BINPERSID'}:
            raise ValueError('unsupported reference opcode: ' + opcode.name)
    loader = StructureUnpickler(io.BytesIO(raw))
    root = loader.load()
    if not isinstance(root, Opaque) or not isinstance(getattr(root, 'state', None), dict):
        raise ValueError('expected an object with dictionary state')
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

    # Only keys and type names leave the loader. Values are deliberately omitted.
    return {
        'object_global': list(root.pickle_global),
        'object_fields': structure(state),
        'metadata_fields': structure(metadata),
        'nested_metadata_fields': {k: structure(v) for k, v in sorted(metadata.items()) if isinstance(v, dict)},
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
        'scope': 'One candidate processed recording, field names and types only. All pickle globals replaced by inert placeholders; numerical arrays not constructed, values not reported. Does not establish stimulus eligibility or animal independence.',
    })
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
