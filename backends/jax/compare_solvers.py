"""Compare library adaptive solvers on one frozen full-state atlas response."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from level0 import Level0, parameters
from solvers import Adaptive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','graph','target','output']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--frames',type=int,default=40)
    p.add_argument('--rtol',type=float,default=1e-7)
    p.add_argument('--atol',type=float,default=1e-9)
    a=p.parse_args()
    if a.frames<2:raise ValueError('at least two frames are required')
    out=Path(a.output);out.mkdir(exist_ok=False)
    model=json.loads(Path(a.model).read_text());graph=json.loads(Path(a.graph).read_text())
    names=sorted(n['id'] for n in graph['neurons']);target=jnp.asarray(names.index(a.target))
    times=np.arange(a.frames)*model['sample_dt'];theta=parameters(model)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    runner=eqx.filter_jit(lambda engine,p,i:engine.solve(p,i))
    results={};states={}
    for method in ['tsit5','kvaerno5']:
        settings=Adaptive(method=method,rtol=a.rtol,atol=a.atol,dt0=model['config']['dt'])
        engine=Level0(model,graph,times,settings)
        start=time.perf_counter();solution=runner(engine,theta,target)
        states[method]=np.asarray(solution.ys)
        if not np.isfinite(states[method]).all():raise ValueError('nonfinite adaptive state')
        seconds=time.perf_counter()-start
        results[method]={'settings':asdict(settings),'seconds_including_compile':seconds,'stats':{k:int(v) for k,v in solution.stats.items()}}
        np.save(out/f'{method}-states.npy',states[method])
        print(f'{method}: {seconds:.3f}s, {results[method]["stats"]}',flush=True)
    error=float(np.max(np.abs(states['tsit5']-states['kvaerno5'])))
    report={'schema_version':1,'source_commit':source,'source_worktree_dirty':dirty,
            'input_sha256':{k:hashlib.sha256(Path(getattr(a,k)).read_bytes()).hexdigest() for k in ['model','graph']},
            'target':a.target,'neurons':len(names),'frames':a.frames,'preparation_seconds':model['config'].get('preparation_seconds',0.),
            'sample_dt':model['sample_dt'],'devices':[str(d) for d in jax.devices()],
            'methods':results,'maximum_state_difference':error,
            'scope':'Frozen parameter numerical comparison, not a fit or biological benchmark. Full states include prepared voltage/calcium/gates. Timings include compilation and no gradient.'}
    (out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(f'maximum state difference: {error:.3g}',flush=True)

if __name__=='__main__':main()
