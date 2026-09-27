#!/usr/bin/env python3
"""Compare Taichi forward/reverse results against a Rust-generated fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
import taichi as ti
from level0 import Level0

def main():
    p=argparse.ArgumentParser()
    p.add_argument('fixture',type=Path);p.add_argument('--arch',choices=['cpu','metal'],default='metal')
    p.add_argument('--batch',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--validate',action='store_true')
    args=p.parse_args()
    if args.repeats<1:p.error('--repeats must be positive')
    fixture=json.loads(args.fixture.read_text())
    dtype=ti.f64 if args.arch=='cpu' else ti.f32
    ti.init(arch=ti.cpu if args.arch=='cpu' else ti.metal,enable_fallback=False,
            default_fp=dtype,fast_math=False,debug=args.validate,cpu_max_num_threads=1)
    model=Level0(fixture,batch=args.batch,dtype=dtype)
    start=time.perf_counter();value,gradient=model.value_and_grad(validation=args.validate)
    compile_and_first_seconds=time.perf_counter()-start
    forward=[];backward=[]
    for _ in range(args.repeats):
        if not args.validate:
            start=time.perf_counter();model.forward();forward.append(time.perf_counter()-start)
        start=time.perf_counter();value,gradient=model.value_and_grad(validation=args.validate);backward.append(time.perf_counter()-start)
    reference=fixture['reference'];expected=np.asarray(reference['gradient'])
    indices=np.asarray(fixture.get('gradient_indices',list(range(len(expected)))),dtype=int)
    actual=gradient[indices]
    voltage=model.voltage.to_numpy()
    fluorescence=model.calcium.to_numpy()*model.prepared.to_numpy()[5*model.n:6*model.n][None,None,:]
    atol,rtol=(1e-11,1e-8) if args.arch=='cpu' else (2e-8,2e-3)
    passed=bool(np.allclose(actual,expected,atol=atol,rtol=rtol)
                and np.allclose(voltage,np.asarray(reference['voltage'])[:,None,:],atol=atol*100,rtol=rtol)
                and np.allclose(fluorescence,np.asarray(reference['fluorescence'])[:,None,:],atol=atol*100,rtol=rtol)
                and np.isclose(value,reference['loss'],atol=atol,rtol=rtol))
    report={'backend':'taichi','taichi_version':ti.__version__,'arch_requested':args.arch,
            'arch_actual':str(ti.lang.impl.current_cfg().arch),'fallback_allowed':False,
            'dtype':'f64' if dtype==ti.f64 else 'f32','machine':platform.machine(),
            'fixture_sha256':hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            'fixture':fixture['fixture'],'neurons':model.n,'parameters':model.p,
            'gradient_parameters_checked':len(indices),'batch':args.batch,'steps':model.steps,
            'compile_and_first_seconds':compile_and_first_seconds,
            'forward_seconds':forward,'forward_reverse_seconds':backward,
            'median_forward_reverse_seconds':statistics.median(backward),
            'loss':value,'reference_loss':reference['loss'],
            'max_voltage_error':float(np.max(np.abs(voltage-np.asarray(reference['voltage'])[:,None,:]))),
            'max_fluorescence_error':float(np.max(np.abs(fluorescence-np.asarray(reference['fluorescence'])[:,None,:]))),
            'max_gradient_error':float(np.max(np.abs(actual-expected))),
            'gradient_tolerance':{'absolute':atol,'relative':rtol},
            'gradients':actual.tolist(),'reference_gradients':expected.tolist(),
            'autodiff_validation':args.validate,'parity_passed':passed,
            'state_and_adjoint_bytes':3*(model.steps+1)*model.batch*model.n*(8 if dtype==ti.f64 else 4)*2}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if 'gradients' not in k},indent=2))
    if not passed:raise SystemExit('Taichi/Rust parity failed')

if __name__=='__main__':main()
