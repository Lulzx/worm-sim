#!/usr/bin/env python3
"""Independent step refinement and local stability audit of a captured failed fit."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from audit_capacity_fit import digest
from replay_level0_atlas import Replay


def voltage_gate_jacobian(replay):
    """Calcium is downstream only; its eigenvalues are separately -1/tau_c."""
    n=replay.n;v=replay.state[:n];s=replay.state[2*n:]
    release=1/(1+np.exp(-(v-replay.threshold)*replay.slope))
    derivative=release*(1-release)*replay.slope
    matrix=np.zeros((2*n,2*n));i=np.arange(n)
    matrix[i,i]=-replay.inv_tau
    np.add.at(matrix,(replay.post,replay.post),-replay.weight*s[replay.pre]*replay.inv_tau[replay.post])
    np.add.at(matrix,(replay.post,n+replay.pre),replay.weight*(replay.reversal-v[replay.post])*replay.inv_tau[replay.post])
    for a,b in [(replay.ga,replay.gb),(replay.gb,replay.ga)]:
        np.add.at(matrix,(a,a),-replay.gap*replay.inv_tau[a])
        np.add.at(matrix,(a,b),replay.gap*replay.inv_tau[a])
    matrix[n+i,i]=derivative*(1-s)*replay.inv_synapse_tau
    matrix[n+i,n+i]=-(release+1)*replay.inv_synapse_tau
    return matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['failure','manifest','graph','training','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args()
    failed,manifest,graph,training=[json.loads(Path(getattr(a,k)).read_text()) for k in ['failure','manifest','graph','training']]
    assert failed['format']=='wormsim-capacity-failure-replay'
    assert failed['reference_manifest_sha256']==digest(a.manifest)
    assert manifest['input_sha256']['graph']==digest(a.graph)
    assert manifest['input_sha256']['training']==digest(a.training)
    packed=failed['model'];base=packed['base_model']
    assert set(packed['configuration']['extensions'])=={'observation'} and packed['configuration']['solver'] is None
    targets=manifest['targets'];groups=[g for g in training['groups'] if training['names'][g['target']] in targets]
    assert sorted(t for g in groups for t in g['training_trials'])==base['training_trials']==manifest['training_trials']
    gains=np.exp(packed['extension_parameters']['observation']['log_gain']['values'])
    total=sum(g['sample_weight'] for g in groups);rows=[];finest=None
    for dt in [base['config']['dt'],base['config']['dt']/2,base['config']['dt']/4]:
        model=copy.deepcopy(base);model['config']['dt']=dt
        row={'dt':dt,'finite_preparation':False,'finite_responses':False,'training_mse':None}
        with np.errstate(over='ignore',invalid='ignore'):
            try:
                replay=Replay(model,graph);row['finite_preparation']=True
                row['prepared_derivative_max']=float(np.max(np.abs(replay.rhs(replay.state,None,0.))))
                mse=0.
                for g in groups:
                    pred=replay.response(training['names'][g['target']],g['recording']['times'])*gains/replay.gain
                    assert np.isfinite(pred).all()
                    traces=g['recording']['traces'];weights=np.array([t['provenance']['id_confidence'] for t in traces])
                    mean=np.array([t['values'] for t in traces]).T
                    observed=pred[:,[replay.index[t['neuron']] for t in traces]]
                    mse+=g['sample_weight']/total*(float(np.sum((observed-mean)**2*weights)/(weights.sum()*len(pred)))+g['irreducible_mse'])
                assert np.isfinite(mse)
                row.update(finite_responses=True,training_mse=mse);finest=replay
            except (AssertionError,FloatingPointError) as error:row['failure_type']=type(error).__name__
        rows.append(row)
    if finest is None:raise ValueError('no finite refined trajectory; local equilibrium audit unavailable')
    matrix=voltage_gate_jacobian(finest);n=finest.n;rng=np.random.default_rng(13);errors=[]
    for _ in range(4):
        direction=rng.normal(size=2*n);full=np.r_[direction[:n],np.zeros(n),direction[n:]];h=1e-6
        finite=(finest.rhs(finest.state+h*full,None,0.)-finest.rhs(finest.state-h*full,None,0.))/(2*h)
        actual=np.r_[finite[:n],finite[2*n:]];analytic=matrix@direction
        np.testing.assert_allclose(analytic,actual,rtol=1e-6,atol=2e-7)
        errors.append(float(np.max(np.abs(actual-analytic))))
    eigenvalues=np.r_[np.linalg.eigvals(matrix),-finest.inv_calcium_tau]
    stable=eigenvalues.real<0
    limit=float(np.min(-2*eigenvalues.real[stable]/np.abs(eigenvalues[stable])**2))
    for row in rows:row['local_euler_amplification_radius']=float(np.max(np.abs(1+row['dt']*eigenvalues)))
    receipt={'schema_version':1,'failure_sha256':digest(a.failure),'manifest_sha256':digest(a.manifest),
        'graph_sha256':digest(a.graph),'training_sha256':digest(a.training),'script_sha256':digest(__file__),
        'replay_script_sha256':digest(Path(__file__).with_name('replay_level0_atlas.py')),
        'epoch':failed['epoch'],'matched_finite_replay_epochs':failed['matched_finite_epochs'],
        'failure_observation':failed['history'][-1],'step_refinement':rows,
        'linearization':{'state_source_dt':finest.dt,'prepared_derivative_max':float(np.max(np.abs(finest.rhs(finest.state,None,0.)))),
            'eigenvalue_count':len(eigenvalues),'max_real_eigenvalue':float(eigenvalues.real.max()),
            'min_real_eigenvalue':float(eigenvalues.real.min()),'negative_mode_euler_step_limit':limit,
            'directional_finite_difference_max_error':max(errors)},
        'scope':'Frozen failing parameters, independent NumPy trajectories and local Jacobian at the finest-step prepared state. Euler amplification is local, not a global nonlinear stability guarantee; smaller-step replay is not a retrained model or a successful gradient check.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
