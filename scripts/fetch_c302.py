#!/usr/bin/env python3
"""Fetch small source files at the audited commit; never execute upstream Python.
The core importer and simulator are Rust. This stdlib-only script handles download.
"""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
lock = json.loads((ROOT / 'docs/upstream-lock.json').read_text())
repo = next(r for r in lock['repositories'] if r['repository'] == 'openworm/c302')
base = f"https://raw.githubusercontent.com/openworm/c302/{repo['head']}/"
output = ROOT / 'runs/upstream'
output.mkdir(parents=True, exist_ok=True)
receipts = []
for source, destination in [
    ('c302/ConnectomeReader.py', output / 'ConnectomeReader.py'),
    ('c302/data/herm_full_edgelist.csv', output / 'herm_full_edgelist.csv'),
    ('LICENSE', ROOT / 'licenses/c302-MIT.txt'),
]:
    url = base + source
    data = urllib.request.urlopen(url, timeout=30).read()
    destination.write_bytes(data)
    receipts.append({'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
tree = ast.parse((output / 'ConnectomeReader.py').read_text())
names = next(ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
             t.id == 'PREFERRED_NEURON_NAMES' for t in node.targets))
assert len(names) == len(set(names)) == 302
(ROOT / 'data/c302-neuron-ids.json').write_text(json.dumps(sorted(names), separators=(',', ':'))+'\n')
(ROOT / 'docs/c302-fetch-receipt.json').write_text(json.dumps(receipts, indent=2)+'\n')
print('Fetched pinned c302 sources; 302 canonical IDs. Raw sources: runs/upstream/')
