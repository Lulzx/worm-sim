#!/usr/bin/env python3
"""Fetch the pinned wild-type atlas text export and hash every extracted source file."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/randi-source'
NAME = 'exported_data.tar.gz'
SHA256 = 'd6e7b3d93175b40b7ae17bde2182835e9c2144388142c522ee9be3832f6ce836'
URL = 'https://osf.io/download/9mecf/?version=1'
SIZE = 523093816


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
archive = OUT / NAME
if not archive.exists() or digest(archive) != SHA256:
    temporary = OUT / (NAME + '.partial')
    with urllib.request.urlopen(URL, timeout=90) as response, temporary.open('wb') as f:
        while block := response.read(1024 * 1024):
            f.write(block)
    if temporary.stat().st_size != SIZE or digest(temporary) != SHA256:
        raise ValueError('Atlas archive size/hash mismatch')
    temporary.replace(archive)
files = []
with tarfile.open(archive) as source:
    for entry in source:
        if entry.isdir() and entry.name == 'exported_data':
            continue
        parts = Path(entry.name).parts
        if not entry.isfile() or len(parts) != 2 or parts[0] != 'exported_data' or not parts[1].endswith('.txt'):
            raise ValueError('Unexpected archive member: ' + entry.name)
        content = source.extractfile(entry).read()
        path = OUT / entry.name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        files.append({'path': parts[1], 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
receipt = {
    'schema_version': 1,
    'source': 'Randi et al. Nature 2023, OSF e2syt, wild-type processed text export, file version 1',
    'url': URL,
    'archive_sha256': SHA256,
    'archive_bytes': SIZE,
    'files': sorted(files, key=lambda f: f['path']),
}
path = ROOT / 'data/randi-source-manifest.json'
path.write_text(json.dumps(receipt, indent=2) + '\n')
print(f'Verified {len(files)} files; wrote {path}')
