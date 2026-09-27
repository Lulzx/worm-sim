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
from checkpoint import CheckpointLevel0

def main():
    p=argparse.ArgumentParser()
    p.add_argument('fixture',type=Path);p.add_argument('--arch',choices=['cpu','metal'],default='cpu')
    p.add_argument('--batch',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--validate',action='store_true')
    p.add_argument('--checkpoint-steps',type=int,default=0,help='Replay windows of this size; 0 retains the full tape')
    p.add_argument('--ad-stack-size',type=int,default=256,help='Explicit CPU AD stack capacity; automatic sizing crashes on c302 with Taichi 1.7.4')
    args=p.parse_args()
    if args.checkpoint_steps<0:p.error('--checkpoint-steps must be nonnegative')
    if args.repeats<1:p.error('--repeats must be positive')
    if args.ad_stack_size<1:p.error('--ad-stack-size must be positive')
    fixture_bytes=args.fixture.read_bytes()
    fixture=json.loads(fixture_bytes)
    dtype=ti.f64 if args.arch=='cpu' else ti.f32
    ti.init(arch=ti.cpu if args.arch=='cpu' else ti.metal,enable_fallback=False,
            default_fp=dtype,fast_math=False,debug=args.validate,cpu_max_num_threads=1,offline_cache=False,ad_stack_size=args.ad_stack_size)
    model=(CheckpointLevel0(fixture,args.checkpoint_steps,batch=args.batch,dtype=dtype)
           if args.checkpoint_steps else Level0(fixture,batch=args.batch,dtype=dtype))
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
    atol,rtol=(1e-11,1e-8) if args.arch=='cpu' else (2e-8,2e-3)
    if args.checkpoint_steps:
        state_passed,max_voltage_error,max_fluorescence_error=model.audit_states(reference,atol*100,rtol)
    else:
        voltage=model.voltage.to_numpy()
        fluorescence=model.calcium.to_numpy()*model.prepared.to_numpy()[5*model.n:6*model.n][None,None,:]
        expected_v=np.asarray(reference['voltage'])[:,None,:]
        expected_f=np.asarray(reference['fluorescence'])[:,None,:]
        state_passed=bool(np.allclose(voltage,expected_v,atol=atol*100,rtol=rtol)
                          and np.allclose(fluorescence,expected_f,atol=atol*100,rtol=rtol))
        max_voltage_error=float(np.max(np.abs(voltage-expected_v)))
        max_fluorescence_error=float(np.max(np.abs(fluorescence-expected_f)))
    passed=bool(np.allclose(actual,expected,atol=atol,rtol=rtol) and state_passed
                and np.isclose(value,reference['loss'],atol=atol,rtol=rtol))
    report={'backend':'taichi','taichi_version':ti.__version__,'arch_requested':args.arch,
            'arch_actual':str(ti.lang.impl.current_cfg().arch),'fallback_allowed':False,
            'dtype':'f64' if dtype==ti.f64 else 'f32','machine':platform.machine(),
            'fixture_sha256':hashlib.sha256(fixture_bytes).hexdigest(),
            'fixture':fixture['fixture'],'graph_hash':fixture.get('graph_hash'),
            'neurons':model.n,'parameters':model.p,
            'chemical_edges':model.m,'gap_edges':model.g,
            'neighbor_iteration':'static_slots' if model.static_neighbors else 'dynamic_csr',
            'max_chemical_degree':int(model.max_chemical_degree),
            'max_gap_degree':int(model.max_gap_degree),
            'gradient_indices':indices.tolist(),
            'reference_forward_ad_seconds':reference.get('forward_ad_seconds'),
            'checked_reference_gradients_above_absolute_tolerance':int(np.count_nonzero(np.abs(expected)>atol)),
            'gradient_parameters_checked':len(indices),'batch':args.batch,'steps':model.steps,
            'compile_and_first_seconds':compile_and_first_seconds,
            'forward_seconds':forward,'forward_reverse_seconds':backward,
            'median_forward_reverse_seconds':statistics.median(backward),
            'loss':value,'reference_loss':reference['loss'],
            'max_voltage_error':max_voltage_error,
            'max_fluorescence_error':max_fluorescence_error,
            'max_gradient_error':float(np.max(np.abs(actual-expected))),
            'gradient_tolerance':{'absolute':atol,'relative':rtol},
            'gradients':actual.tolist(),'reference_gradients':expected.tolist(),
            'autodiff_validation':args.validate,'ad_stack_size':args.ad_stack_size,
            'offline_cache':False,'parity_passed':passed,
            'state_and_adjoint_bytes':3*(model.state_steps+1)*model.batch*model.n*(8 if dtype==ti.f64 else 4)*2}
    report['checkpoint_steps']=model.state_steps if args.checkpoint_steps else 0
    if args.checkpoint_steps:
        report['memory_accounting']=model.memory_accounting()
    # Full-network receipts retain diagnostics rather than duplicating all vectors.
    if len(indices)>128:
        errors=np.abs(actual-expected)
        worst=np.argsort(errors)[-16:][::-1]
        report['largest_gradient_errors']=[{'index':int(indices[j]),'actual':float(actual[j]),
            'reference':float(expected[j]),'absolute_error':float(errors[j])} for j in worst]
        report['gradient_mismatches']=int(np.count_nonzero(~np.isclose(actual,expected,atol=atol,rtol=rtol)))
        report['all_parameters_checked']=bool(np.array_equal(indices,np.arange(model.p)))
        report['gradient_indices_sha256']=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
        report['gradients_sha256']=hashlib.sha256(actual.astype('<f8').tobytes()).hexdigest()
        report['reference_gradients_sha256']=hashlib.sha256(expected.astype('<f8').tobytes()).hexdigest()
        for key in ['gradient_indices','gradients','reference_gradients']:del report[key]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if 'gradients' not in k and k!='gradient_indices'},indent=2))
    if not passed:raise SystemExit('Taichi/Rust parity failed')

if __name__=='__main__':main()
