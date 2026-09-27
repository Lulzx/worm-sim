#!/usr/bin/env python3
"""Compare a declared longer fit's repeated prefix with a hashed reference run."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def complete_rows(raw, field):
    # A live writer may be midway through its final line. Ignore only that line.
    lines=raw.split(b'\n')[:-1]
    rows=[json.loads(line) for line in lines]
    start=1 if field=='evaluation' else 0
    if [row[field] for row in rows]!=list(range(start,start+len(rows))):
        raise ValueError('noncontiguous '+field+' history')
    return rows


def compare_value(left,right,path='root'):
    if isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise ValueError('keys differ: '+path)
        for key in left:compare_value(left[key],right[key],path+'.'+key)
    elif isinstance(left,list):
        if not isinstance(right,list) or len(left)!=len(right):raise ValueError('list differs: '+path)
        for i,(a,b) in enumerate(zip(left,right)):compare_value(a,b,path+f'[{i}]')
    elif isinstance(left,float):
        if type(right) not in (int,float) or not math.isfinite(left) or not math.isfinite(right) or abs(left-right)>1e-10:
            raise ValueError('numeric value differs: '+path)
    elif type(left)!=type(right) or left!=right:
        raise ValueError('value differs: '+path)


def check(reference,candidate,declaration):
    for key in ['input_sha256','targets','training_trials','warm_start','configuration','fit_config','bounds','backend_source_sha256','jax','scipy']:
        if reference[key]!=candidate[key]:raise ValueError('manifest mismatch: '+key)
    a=reference['fitting_optimizer'];b=candidate['fitting_optimizer']
    if a.keys()!=b.keys():raise ValueError('optimizer fields differ')
    for key in a:
        if key in ('max_evaluations','max_iterations'):
            if b[key]!=declaration[key] or b[key]<a[key]:raise ValueError('invalid extended budget')
        elif a[key]!=b[key]:raise ValueError('optimizer differs: '+key)
    for key in ['maxls','maxcor','ftol','gtol']:
        if b[key]!=declaration[key]:raise ValueError('optimizer differs from declaration')
    if candidate['input_sha256']['warm_start']!=declaration['parent_sha256'] or candidate['fit_config']['dt']!=declaration['dt'] or candidate['targets']!=declaration['targets']:
        raise ValueError('candidate differs from declaration')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['reference','candidate','declaration','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--allow-partial',action='store_true',help='Emit a clearly incomplete progress receipt')
    a=p.parse_args();ref=Path(a.reference);candidate=Path(a.candidate)
    declaration_raw=Path(a.declaration).read_bytes();declaration=json.loads(declaration_raw)
    names={'manifest':'manifest.json','evaluations':'evaluations.jsonl','accepted':'progress.jsonl'}
    old={k:(ref/name).read_bytes() for k,name in names.items()}
    new={k:(candidate/name).read_bytes() for k,name in names.items()}
    for key,expected in declaration['prefix_reference_sha256'].items():
        if digest(old[key])!=expected:raise ValueError('reference hash mismatch: '+key)
    check(json.loads(old['manifest']),json.loads(new['manifest']),declaration)
    report={};complete=True
    for key,field in [('evaluations','evaluation'),('accepted','epoch')]:
        if not old[key].endswith(b'\n'):raise ValueError('reference has incomplete final row')
        left=complete_rows(old[key],field);right=complete_rows(new[key],field)
        if not left:raise ValueError('empty reference history')
        count=min(len(left),len(right));complete &= len(right)>=len(left)
        for i in range(count):
            expected={k:v for k,v in left[i].items() if k!='elapsed_seconds'}
            actual={k:v for k,v in right[i].items() if k!='elapsed_seconds'}
            compare_value(expected,actual,key+f'[{i}]')
        report[key]={'matched':count,'required':len(left),'candidate_complete_rows':len(right)}
    if not complete and not a.allow_partial:raise ValueError('prefix incomplete; do not interpret extension yet')
    receipt={'schema_version':1,'complete_reference_prefix_verified':complete,'comparisons':report,
        'declaration_sha256':digest(declaration_raw),'reference_sha256':{k:digest(v) for k,v in old.items()},
        'candidate_snapshot_sha256':{k:digest(v) for k,v in new.items()},'script_sha256':digest(Path(__file__).read_bytes()),
        'float_absolute_tolerance':1e-10,'excluded_comparison_fields':['elapsed_seconds'],
        'scope':'Declared input/configuration/source agreement and trial/accepted scalar history only. No claim of bitwise parameter or optimizer-state equality, convergence, capacity acceptance or held-out performance.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
