#!/usr/bin/env python3
"""Restricted numeric export + independent NumPy reference for the pinned baseline.
Requires NumPy. Never imports or executes upstream Python modules.
Equations follow Creamer_LDS_2026 (MIT; see licenses/creamer-MIT.txt).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/creamer-source'
NAMES=('connectome_constrained','fully_connected','shuffled_constrained')
GROUPS=('dynamics_weights','dynamics_input_weights','dynamics_cov',
        'emissions_weights','emissions_input_weights','emissions_cov')

class NumericModel:
    """Inert attribute container: no upstream methods or constructors."""

class NumericUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('ssm_classes','Lgssm'):
            return NumericModel
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
            return getattr(np._core.multiarray,name)
        if module=='numpy' and name in ('ndarray','dtype'):
            return getattr(np,name)
        raise ValueError(f'Unsupported pickle global {module}.{name}')

def read_asset(path,receipt):
    data=(SOURCE/path).read_bytes()
    expected=next(x for x in receipt['files'] if x['path']==path)
    if hashlib.sha256(data).hexdigest()!=expected['sha256']:
        raise ValueError('Source hash mismatch: '+path)
    import io
    return NumericUnpickler(io.BytesIO(data)).load()

def operator(array):
    a=np.asarray(array,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all():
        raise ValueError('Expected finite numeric matrix')
    rows,cols=a.shape
    # Signed zero is preserved by falling back to dense representation.
    if np.any((a==0)&np.signbit(a)):
        return {'kind':'dense','rows':rows,'cols':cols,'values':a.ravel().tolist()}
    if not np.any(a):
        return {'kind':'zero','rows':rows,'cols':cols}
    if rows==cols and np.array_equal(a,np.eye(rows)):
        return {'kind':'identity','size':rows}
    if cols%rows==0:
        lags=cols//rows
        values=np.stack([np.diag(a[:,lag*rows:(lag+1)*rows]) for lag in range(lags)])
        reconstruction=np.zeros_like(a)
        for lag in range(lags):
            reconstruction[np.arange(rows),lag*rows+np.arange(rows)]=values[lag]
        if np.array_equal(a,reconstruction):
            return {'kind':'lagged_diagonal','size':rows,'lags':lags,'values':values.ravel().tolist()}
    if np.count_nonzero(a)<a.size*0.4:
        rr,cc=np.nonzero(a)
        return {'kind':'csr','rows':rows,'cols':cols,
                'offsets':np.r_[0,np.cumsum(np.bincount(rr,minlength=rows))].tolist(),
                'columns':cc.tolist(),'values':a[rr,cc].tolist()}
    return {'kind':'dense','rows':rows,'cols':cols,'values':a.ravel().tolist()}

def nullable(a):
    a=np.asarray(a)
    if np.isinf(a).any():raise ValueError('Infinite measurement')
    return [None if np.isnan(x) else float(x) for x in a.ravel()]

def pearson(a,b):
    a=np.asarray(a).copy();b=np.asarray(b).copy()
    np.fill_diagonal(a,np.nan);np.fill_diagonal(b,np.nan)
    mask=np.isfinite(a)&np.isfinite(b)
    x=a[mask];y=b[mask];x=x-x.mean();y=y-y.mean()
    return {'correlation':float(np.mean(x*y)/np.std(x)/np.std(y)),'pairs':int(mask.sum())}

def matmul(a,b):
    # Explicit NumPy contraction avoids platform BLAS floating-status warnings;
    # use a separate arithmetic path from Rust's contiguous sparse-row kernel.
    return np.einsum('ik,kj->ij',a,b,optimize=False)

def references(model):
    n=model.dynamics_dim
    if model.dynamics_lags!=1 or model.emissions_dim!=n or model.input_dim!=n:
        raise ValueError('Exporter currently supports equal dimensions and one state lag')
    if model.sample_rate!=2:raise ValueError('Expected upstream 2 Hz timebase')
    w,h,c,d=(getattr(model,k) for k in ('dynamics_weights','dynamics_input_weights','emissions_weights','emissions_input_weights'))
    x=np.zeros((n,n));responses=[]
    for t in range(60):
        x=matmul(w,x) if t else np.zeros_like(x)
        if t<model.dynamics_input_lags:x=x+h[:,t*n:(t+1)*n]
        y=matmul(c,x)
        if t<model.emissions_input_lags:y=y+d[:,t*n:(t+1)*n]
        responses.append(y)
    responses=np.stack(responses)
    # Literal upstream padding/slice convention; extra selected zeros have no effect.
    padded=np.concatenate((np.zeros((30,n,n)),responses),axis=0)
    stams=padded[15:].sum(axis=0)/model.sample_rate
    q=model.dynamics_cov
    covariance=matmul(w,w.T)+q
    for _ in range(100):covariance=matmul(matmul(w,covariance),w.T)+q
    std=np.sqrt(covariance.diagonal())
    correlation=covariance/(std[:,None]*std[None,:])
    pairs=[(0,0),(0,1),(1,0),(n//2,n//3),(n-1,n-1),(n-1,0)]
    probes=[{'responding':r,'stimulated':s,'values':responses[:,r,s].tolist()} for r,s in pairs]
    return stams,correlation,probes

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    receipt=json.loads((ROOT/'docs/creamer-fetch-receipt.json').read_text())
    start=time.perf_counter()
    measured={name:read_asset('data/measured_'+name+'.pkl',receipt) for name in ('stams','corr')}
    bundle={'schema_version':1,'source':{'repository':'Nondairy-Creamer/Creamer_LDS_2026','commit':receipt['commit'],
        'artifacts':[{k:v for k,v in r.items() if k in ('path','sha256')} for r in receipt['files']],
        'split':'Upstream train/test recording split; not a stimulated-neuron-held-out split.'},
        'pre_seconds':15,'post_seconds':30,'covariance_iterations':100,'models':[],
        'measured':{metric+'_'+split:nullable(measured[metric][split]) for metric in ('stams','corr') for split in ('train','test')}}
    summary=[]
    for name in NAMES:
        model=read_asset('models/'+name+'.pkl',receipt)
        stams,corr,probes=references(model)
        learned={}
        for group in GROUPS:
            if not model.param_props['update'][group]:continue
            mask=model.param_props['mask'][group]
            shape=model.param_props['shape'][group]
            count=int(np.count_nonzero(mask)) if mask is not None else (model.dynamics_dim if shape=='diag' else getattr(model,group).size)
            learned[group]=count
        item={'name':name,'neurons':[str(x) for x in model.cell_ids],'sample_rate':float(model.sample_rate),
              'input_lags':int(model.dynamics_input_lags),'emission_input_lags':int(model.emissions_input_lags),
              'learned_parameters':learned,**{g:operator(getattr(model,g)) for g in GROUPS},
              'reference':{'stams':stams.ravel().tolist(),'correlation':corr.ravel().tolist(),'probes':probes,
                           'stams_test_score':pearson(stams,measured['stams']['test']),
                           'corr_test_score':pearson(corr,measured['corr']['test'])}}
        bundle['models'].append(item)
        summary.append({'name':name,'representations':{g:item[g]['kind'] for g in GROUPS},
                        'learned_parameters':learned,'stams_test_score':item['reference']['stams_test_score'],
                        'corr_test_score':item['reference']['corr_test_score']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as f:json.dump(bundle,f,separators=(',',':'),allow_nan=False)
    summary={'numpy_version':np.__version__,'reference_seconds':time.perf_counter()-start,'json_bytes':args.output.stat().st_size,'models':summary}
    (ROOT/'docs/creamer-numpy-reference.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
