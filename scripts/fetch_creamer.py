#!/usr/bin/env python3
"""Fetch the audited upstream model/data assets, without executing upstream code."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = next(r for r in json.loads((ROOT/'docs/upstream-lock.json').read_text())['repositories']
            if r['repository']=='Nondairy-Creamer/Creamer_LDS_2026')
OUT = ROOT/'runs/creamer-source'
PATHS = ['models/'+name+'.pkl' for name in
         ('connectome_constrained','fully_connected','shuffled_constrained')]
PATHS += ['data/measured_stams.pkl','data/measured_corr.pkl','ssm_classes.py',
          'lgssm_utilities.py','metrics.py','quick_start_examples/predict_stams.py','LICENSE']

def fetch(path):
    url=f"https://raw.githubusercontent.com/{REPO['repository']}/{REPO['head']}/{path}"
    target=OUT/path
    expected=ENTRIES[path]['sha']
    data=target.read_bytes() if target.exists() else b''
    def git_hash(data):
        return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if git_hash(data)!=expected:
        data=urllib.request.urlopen(url,timeout=90).read()
        if git_hash(data)!=expected:
            raise ValueError('upstream Git blob mismatch: '+path)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    return {'path':path,'url':url,'git_blob':expected,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}

if __name__=='__main__':
    req=urllib.request.Request(f"https://api.github.com/repos/{REPO['repository']}/git/trees/{REPO['head']}?recursive=1",headers={'User-Agent':'wormsim'})
    ENTRIES={e['path']:e for e in json.load(urllib.request.urlopen(req))['tree'] if e['type']=='blob'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipt=list(pool.map(fetch,PATHS))
    (ROOT/'docs/creamer-fetch-receipt.json').write_text(json.dumps({'commit':REPO['head'],'files':receipt},indent=2)+'\n')
    (ROOT/'licenses/creamer-MIT.txt').write_bytes((OUT/'LICENSE').read_bytes())
    print(f'Fetched and verified {len(receipt)} files at {REPO["head"]}')
