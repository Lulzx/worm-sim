"""Training-only quasi-Newton capacity comparison with a hard evaluation budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
import scipy
from extensions import pack
from fit import checkpoint
from objective import build, evaluate
from overfit import prepare, warm_parameters, bounds, atomic_best, failure_artifact
from quasi_newton import optimize_active


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['model','graph','training','warm-start','output']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--max-evaluations',type=int,default=201)
    parser.add_argument('--max-iterations',type=int,default=200)
    parser.add_argument('--maxcor',type=int,default=20,help='L-BFGS curvature correction pairs retained')
    parser.add_argument("--coordinate-scaling", choices=["identity", "curvature-v1"], default="identity")
    parser.add_argument('--dt',type=float,default=.005)
    parser.add_argument('--preparation-seconds',type=float,default=None)
    parser.add_argument('--fixed-seed-audit', help='Explicit frozen-seed audit receipt for this warm parent')
    a=parser.parse_args()
    if a.max_evaluations<1 or a.max_iterations<1 or a.maxcor<1 or not np.isfinite(a.dt) or a.dt<=0:
        raise ValueError('invalid budgets or step')
    paths={k:Path(getattr(a,k)) for k in ['model','graph','training','warm_start']}
    original,graph,training,parent=[json.loads(paths[k].read_text()) for k in paths]
    if digest(paths['model'])!=training['model_sha256']:
        raise ValueError('training export belongs to another checkpoint')
    base=parent['model']['base_model'];targets=parent['targets']
    # Fresh overfit.py parents record warm_start as null; their rest is the saved initial state.
    rest=(parent.get('warm_start') or {}).get('initial_rest_parameter',base['initial'][0])
    model,training,config=prepare(original,training,targets,base['config']['epochs'],base['config']['learning_rate'],rest,base['config']['preparation_seconds'])
    model['initial']=list(base['initial'])
    seed_override=None
    if a.fixed_seed_audit:
        paths['fixed_seed_audit']=Path(a.fixed_seed_audit)
        seed_audit=json.loads(paths['fixed_seed_audit'].read_text())
        for key,path in [('checkpoint',paths['warm_start']),('training',paths['training']),('graph',paths['graph'])]:
            if seed_audit['input_sha256'][key]!=digest(path):
                raise ValueError('fixed-seed audit lineage differs: '+key)
        audit_script=Path(__file__).resolve().parents[2]/'scripts'/'audit_capacity_fixed_seed.py'
        if seed_audit['script_sha256']!=digest(audit_script):
            raise ValueError('fixed-seed audit source differs')
        seed_override=seed_audit['fixed_initial_state']
        model['initial']=seed_override
    model['config']['dt']=a.dt
    if a.preparation_seconds is not None:
        model['config']['preparation_seconds']=a.preparation_seconds
    theta=warm_parameters(parent,model,graph,training,config,targets,allow_step_change=True,
        allow_preparation_change=a.preparation_seconds is not None, initial_override=seed_override)
    _,active,groups,data,prior=build(model,graph,training,config)
    scaling = {'native/threshold': 0.13, 'native/rest': 0.16} if a.coordinate_scaling == 'curvature-v1' else {}
    coordinate_scale = jax.tree.map(lambda v: np.ones(v.shape), theta)
    coordinate_scale['groups'] = np.array([scaling.get('native/' + g['name'].split('/')[0], 1.)
                                         for g in model['parameters']['groups']])
    reference=bounds(groups);out=Path(a.output);out.mkdir(exist_ok=False)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    warm={'initial_rest_parameter':rest,'checkpoint_sha256':digest(paths['warm_start']),'source_epoch':base['epoch'],
        'parent_dt':base['config']['dt'],'run_dt':a.dt,
        'initial_state_policy':'audited fixed prepared seed' if a.fixed_seed_audit else 'inherited parent seed',
        'fixed_seed_audit_sha256':digest(paths['fixed_seed_audit']) if a.fixed_seed_audit else None,
        'parent_preparation_seconds':base['config']['preparation_seconds'],
        'run_preparation_seconds':model['config']['preparation_seconds'],
        'parent_training_mse':parent['metrics']['mse'],'optimizer_state':'fresh L-BFGS history; no Adam moments',
        'epoch_convention':'accepted L-BFGS iterations; trial evaluations separately logged'}
    def write(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    manifest={'format':'wormsim-training-capacity-diagnostic','source_commit':source,
        'source_worktree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'input_sha256':{k:digest(v) for k,v in paths.items()},'targets':targets,'training_trials':training['training_trials'],
        'warm_start':warm,'process_id':os.getpid(),'configuration':config,'fit_config':model['config'],
        'bounds':reference,'backend_source_sha256':{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'jax':jax.__version__,'scipy':scipy.__version__,'devices':[str(d) for d in jax.devices()],
        'fitting_optimizer':{'method':'SciPy L-BFGS-B','bounds':None,'max_evaluations':a.max_evaluations,
            'coordinate_scaling':a.coordinate_scaling,'family_scales':scaling,
            'coordinate_transform':'theta = initial + scale * z; frozen coordinates unchanged',
            'gtol_coordinates':'scaled optimizer coordinates z',
            'max_iterations':a.max_iterations,'maxls':20,'maxcor':a.maxcor,'ftol':1e-12,'gtol':1e-9},
        'selection':'minimum training MSE among accepted iterates, earliest tie; line-search trials are not selectable',
        'scope':'Training subset only. Native fit_config retained for parameter/checkpoint compatibility; fitting_optimizer controls this diagnostic. No priors or dynamical changes.'}
    write('manifest.json',manifest)
    start=time.perf_counter();best=float('inf');best_epoch=None;final=None
    with (out/'evaluations.jsonl').open('x',buffering=1) as trials,(out/'progress.jsonl').open('x',buffering=1) as progress:
        def on_evaluation(n,p,value,gradient,metrics,finite):
            trials.write(json.dumps({'evaluation':n,'finite':finite,
                'metrics':{k:v if np.isfinite(v) else None for k,v in metrics.items()},
                'elapsed_seconds':time.perf_counter()-start},allow_nan=False)+'\n')
            if not finite:
                failed=failure_artifact(model,p,config,n,source,value,gradient,metrics,targets,warm)
                failed.update(evaluation=n,epoch_convention='evaluation index, not accepted iteration',reference_manifest_sha256=digest(out/'manifest.json'))
                write('failure.json',failed)
        def on_accepted(epoch,n,p,metrics):
            nonlocal best,best_epoch,final
            if epoch==0 and not a.fixed_seed_audit and a.dt==base['config']['dt'] and model['config']['preparation_seconds']==base['config']['preparation_seconds'] and abs(metrics['mse']-parent['metrics']['mse'])>1e-10:
                raise ValueError('initial score differs from parent')
            if epoch==0:
                write('initial-objective.json', {'parent_mse':parent['metrics']['mse'],
                    'run_initial_mse':metrics['mse'], 'difference':metrics['mse']-parent['metrics']['mse'],
                    'parent_dt':base['config']['dt'], 'run_dt':a.dt,
                    'parent_preparation_seconds':base['config']['preparation_seconds'],
                    'run_preparation_seconds':model['config']['preparation_seconds']})
            gains=np.exp(np.asarray(p['observation']['log_gain']))
            final=dict(epoch=epoch,evaluations=n,**metrics,
                captured_start_zero_energy=(reference['zero_response_mse']-metrics['mse'])/(reference['zero_response_mse']-reference['start_zero_mean_response_bound']),
                gain_min=float(gains.min()),gain_max=float(gains.max()),elapsed_seconds=time.perf_counter()-start)
            artifact={'format':'wormsim-training-capacity-diagnostic','targets':targets,
                'model':pack(checkpoint(model,p,epoch,source),p,config),'metrics':final,'warm_start':warm}
            progress.write(json.dumps(final,allow_nan=False)+'\n')
            if metrics['mse']<best:
                atomic_best(out/'best.json',artifact);best=metrics['mse'];best_epoch=epoch
            atomic_best(out/'last-accepted.json',artifact)
            print(json.dumps(final,allow_nan=False),flush=True)
        outcome=optimize_active(theta,active,lambda p:evaluate(p,groups,data,prior),on_accepted,on_evaluation,
            max_evaluations=a.max_evaluations,max_iterations=a.max_iterations,max_corrections=a.maxcor,coordinate_scale=coordinate_scale)
    outcome.pop('final_theta');outcome.pop('final_metrics')
    eligible_termination=outcome['status']=='evaluation_budget_exhausted' or outcome.get('optimizer_status') in (0,1)
    write('result.json',dict(**outcome,final=final,bounds=reference,best_epoch=best_epoch,
        best_training_mse=None if best_epoch is None else best,
        best_checkpoint_sha256=None if best_epoch is None else digest(out/'best.json'),
        capacity_gate=bool(final and eligible_termination and final['captured_start_zero_energy']>=.9),
        capacity_gate_definition='last accepted iterate captures at least 90%; nonfinite termination cannot pass; optimizer convergence is reported separately'))
    if outcome['status']=='nonfinite_evaluation':raise SystemExit(1)


if __name__=='__main__':main()
