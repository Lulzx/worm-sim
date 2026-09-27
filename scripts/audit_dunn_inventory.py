#!/usr/bin/env python3
"""Metadata-only inventory of a candidate independent perturbation cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',required=True)
    parser.add_argument('--source-directory',required=True)
    parser.add_argument('--output-directory',required=True)
    args=parser.parse_args()
    out=Path(args.output_directory);out.mkdir(exist_ok=False)
    sources={}
    def fetch(name,url):
        with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
        (out/name).write_bytes(raw)
        sources[name]={'url':url,'sha256':digest(raw),'bytes':len(raw)}
        return raw
    root='https://api.dandiarchive.org/api/dandisets/001623/versions/0.251015.0312/'
    meta=json.loads(fetch('dandi-metadata.json',root))
    page=json.loads(fetch('dandi-assets.json',root+'assets/?page_size=1000'))
    zenodo=json.loads(fetch('zenodo-metadata.json','https://zenodo.org/api/records/17353307'))
    commit='98016334b89ef087bdf39de938dad52a2cc8a47a'
    github='https://raw.githubusercontent.com/focolab/2025-dunn-et-al-curr-biol/'+commit+'/'
    fetch('upstream-readme.md',github+'README.md')
    fetch('upstream-data-schema.py',github+'lib/wbliveDataClass.py')
    if page['next'] is not None or len(page['results'])!=page['count']:
        raise ValueError('incomplete asset inventory')
    files={f['key']:{k:f[k] for k in ['size','checksum']} for f in zenodo['files']}
    if len(files)!=len(zenodo['files']):raise ValueError('duplicate Zenodo filenames')
    raw=Path(args.source_manifest).read_bytes();manifest=json.loads(raw);old=set()
    for row in manifest['files']:
        if not re.fullmatch(r'\d+_ds_name\.txt',row['path']):continue
        data=(Path(args.source_directory)/row['path']).read_bytes()
        if len(data)!=row['bytes'] or digest(data)!=row['sha256']:
            raise ValueError('source recording-name hash mismatch')
        stamp=re.search(r'/pumpprobe_(\d{8})_(\d{6})/?\s*$',data.decode())
        if stamp is None:raise ValueError('unrecognized source recording name')
        old.add(stamp[1]+stamp[2])
    if not old:raise ValueError('no source recording names')
    records=[];seen=set()
    for asset in sorted(page['results'],key=lambda x:x['path']):
        match=re.fullmatch(r'sub-(\d{8}-\d{2}-\d{2}-\d{2})/sub-\1_ses-(\d{8}T\d{6})\.nwb',asset['path'])
        if match is None:raise ValueError('unrecognized candidate path')
        rec=match[1];stamp=rec.replace('-','')
        if stamp!=match[2].replace('T','') or rec in seen:raise ValueError('conflicting/duplicate recording identifier')
        seen.add(rec)
        records.append({'recording':rec,'path':asset['path'],'asset_id':asset['asset_id'],'bytes':asset['size'],
            'timestamp_matches_randi':stamp in old,'processed_pickle_metadata':files.get(rec+'.pkl')})
    receipt={'schema_version':1,'dandiset':'001623','version':'0.251015.0312','zenodo_record':17353307,
        'upstream_commit':commit,'source_manifest_sha256':digest(raw),'script_sha256':digest(Path(__file__).read_bytes()),
        'sources':sources,'candidate_assets':len(records),'randi_recording_names_checked':len(old),
        'matching_randi_timestamps':sum(r['timestamp_matches_randi'] for r in records),
        'matching_processed_pickle_names':sum(r['processed_pickle_metadata'] is not None for r in records),
        'zenodo_file_count':len(files),'other_zenodo_files':sorted(set(files)-{r['recording']+'.pkl' for r in records}),
        'records':records,
        'scope':'Only repository source text and JSON metadata inventories read. No NWB, pickle, CSV, archive, calcium or image payload read. Recording-name nonoverlap does not prove animal independence or compatible per-neuron stimuli. No confirmatory holdout secured.'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['records','sources']},indent=2))


if __name__=='__main__':main()
