#!/usr/bin/env python3
"""Fetch pinned Atanas/Kim processed recordings; never execute upstream code.

Only baseline animals with NeuroPAL labels are extracted. Large data stay under
ignored runs/. Archive and per-animal checksums must match published metadata.
"""
import bz2
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path('runs/wormwideweb-source')
COMMIT = '0422546192ef58d0de4740a0aa91fd38de22da2e'
RECORD = '19388374'
GIT_FILES = {
    'activity/papers.json': '3b181eff9d929f7e2689b38b2a8243ca557b5870',
    'activity/raw/atanas_kim_2023.csv': '39c8f0fcdd113080dd2f21014a0189940e9422d8',
}
ARCHIVES = {
    'processed_h5.tar.bz2': (568776589, '4689839382dd8855e3d6f64aadf89403'),
    'neuropal_label.json.bz2': (16169, '1b8b0edf02badd6d8d028e2f6c30d24d'),
}


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            md5.update(chunk); sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def fetch_archive(name, size, md5):
    path = ROOT/name
    url = f'https://zenodo.org/api/records/{RECORD}/files/{name}/content'
    if not path.exists():
        temporary = path.with_suffix(path.suffix+'.partial')
        total = 0
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
            while chunk := response.read(1024*1024):
                total += len(chunk)
                if total > size:raise ValueError(f'{name}: exceeds pinned size')
                out.write(chunk)
                if total % (64*1024*1024) == 0:print(f'{name}: {total:,}/{size:,} bytes',flush=True)
        if total != size or digests(temporary)[0] != md5:
            raise ValueError(f'{name}: size/checksum mismatch')
        temporary.replace(path)
    actual_md5, sha = digests(path)
    if path.stat().st_size != size or actual_md5 != md5:
        raise ValueError(f'{name}: cached file checksum mismatch')
    return {'url':url,'bytes':size,'md5':md5,'sha256':sha}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    sources = []
    for relative, blob in GIT_FILES.items():
        url = f'https://raw.githubusercontent.com/flavell-lab/WormWideWeb-data/{COMMIT}/{relative}'
        data = urllib.request.urlopen(url,timeout=30).read()
        actual = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if actual != blob:raise ValueError(f'Git blob mismatch: {relative}')
        (ROOT/Path(relative).name).write_bytes(data)
        sources.append({'url':url,'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest()})
    for name,(size,md5) in ARCHIVES.items():sources.append(fetch_archive(name,size,md5))
    labels = bz2.decompress((ROOT/'neuropal_label.json.bz2').read_bytes())
    json.loads(labels)
    (ROOT/'neuropal_label.json').write_bytes(labels)
    rows = list(csv.DictReader(io.StringIO((ROOT/'atanas_kim_2023.csv').read_text())))
    selected = {r['filename']:r for r in rows if r['label']=='true' and 'baseline' in r['type'].split(',')}
    output = ROOT/'baseline'; output.mkdir(exist_ok=True)
    extracted = []
    seen = set()
    with tarfile.open(ROOT/'processed_h5.tar.bz2',mode='r|bz2') as archive:
        for member in archive:
            name = Path(member.name).name
            if name not in selected:continue
            if not member.isfile() or name in seen or member.size > 512*1024*1024:
                raise ValueError(f'Invalid/repeated archive member: {name}')
            seen.add(name)
            temporary = output/(name+'.partial')
            sha = hashlib.sha256()
            with archive.extractfile(member) as source, temporary.open('wb') as target:
                for chunk in iter(lambda:source.read(1024*1024),b''):
                    sha.update(chunk);target.write(chunk)
            if sha.hexdigest() != selected[name]['checksum']:
                raise ValueError(f'Animal checksum mismatch: {name}')
            temporary.replace(output/name)
            extracted.append({'file':name,'animal_id':selected[name]['uid'],'sha256':sha.hexdigest(),'bytes':member.size})
    if seen != set(selected):raise ValueError('Archive missing selected animals')
    receipt={'source_commit':COMMIT,'zenodo_record':RECORD,
             'paper':'https://doi.org/10.1016/j.cell.2023.07.035',
             'selection':'baseline and NeuroPAL labels; no neural-outcome-based selection',
             'license_metadata':'No license field in the inspected Zenodo metadata; raw data are not redistributed in this repository.',
             'sources':sources,'animals':sorted(extracted,key=lambda x:x['animal_id'])}
    Path('docs/wormwideweb-fetch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'Verified and extracted {len(extracted)} labeled baseline animals.',flush=True)


if __name__ == '__main__':main()
