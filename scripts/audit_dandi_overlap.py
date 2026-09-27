#!/usr/bin/env python3
"""Metadata-only overlap audit: DANDI atlas assets versus pinned OSF recording IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assets(path):
    page=json.loads(Path(path).read_text())
    if page['next'] is not None or len(page['results'])!=page['count']:
        raise ValueError('requires a complete, single-page asset inventory')
    rows={}
    for row in page['results']:
        if row['path'] in rows:
            raise ValueError('duplicate asset path')
        rows[row['path']]={k:row[k] for k in ['asset_id','blob','zarr','size']}
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['published-assets','draft-assets','source-manifest','source-directory','output']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args()
    manifest=json.loads(Path(a.source_manifest).read_text())
    recordings={}
    for row in manifest['files']:
        match=re.fullmatch(r'(\d+)_ds_name\.txt',row['path'])
        if not match:continue
        path=Path(a.source_directory)/row['path']
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise ValueError('recording-name file differs from pinned source')
        stamp=re.search(r'/pumpprobe_(\d{8})_(\d{6})/?\s*$',path.read_text())
        if stamp is None:raise ValueError('unrecognized source recording identifier')
        recordings[match[1]]={'session':stamp[1]+'-'+stamp[2],'source_name_sha256':row['sha256']}
    if not recordings:raise ValueError('no pinned recording identifiers')
    published=assets(a.published_assets);draft=assets(a.draft_assets)
    subject_sessions={};unknown=[];conflicts=[]
    for path in draft:
        m=re.fullmatch(r'sub-(\d+)/sub-\1_ses-(\d{8}-\d{6})_.+\.nwb',path)
        if m is None:raise ValueError('unrecognized DANDI asset path: '+path)
        subject,session=m[1],m[2]
        subject_sessions.setdefault(subject,set()).add(session)
        if subject not in recordings:unknown.append(path)
        elif recordings[subject]['session']!=session:conflicts.append(path)
    matches=[{'subject':subject,'session':row['session'],'source_name_sha256':row['source_name_sha256'],
              'asset_count':sum(path.startswith(f'sub-{subject}/') for path in draft)}
             for subject,row in sorted(recordings.items(),key=lambda v:int(v[0]))
             if subject_sessions.get(subject)=={row['session']}]
    receipt={'schema_version':1,'dandiset':'001075','published_version':'0.240930.1859',
        'checked_version':'draft','script_sha256':digest(__file__),
        'source_manifest_sha256':digest(a.source_manifest),
        'published_inventory_sha256':digest(a.published_assets),'draft_inventory_sha256':digest(a.draft_assets),
        'published_assets_url':'https://api.dandiarchive.org/api/dandisets/001075/versions/0.240930.1859/assets/?page_size=1000',
        'draft_assets_url':'https://api.dandiarchive.org/api/dandisets/001075/versions/draft/assets/?page_size=1000',
        'published_asset_count':len(published),'draft_asset_count':len(draft),
        'identical_asset_paths_ids_blobs_and_sizes':published==draft,
        'pinned_source_recordings':len(recordings),'draft_subject_count':len(subject_sessions),
        'matched_subject_sessions':len(matches),'matches':matches,
        'unknown_subject_assets':unknown,'conflicting_session_assets':conflicts,
        'unrepresented_source_recordings':sorted(set(recordings)-set(subject_sessions)),
        'scope':'Metadata only: complete DANDI asset inventories and content-hash-verified OSF recording-name files. No NWB objects, imaging, fluorescence, or held-out outcome values downloaded or read. Matching names are provenance evidence, not a biological animal-identity verification.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in receipt.items() if k!='matches'},indent=2))

if __name__=='__main__':main()
