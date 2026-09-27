#!/usr/bin/env python3
"""Audit the initial unforced transient of captured failing capacity parameters."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from audit_capacity_fit import digest
from audit_capacity_stability import voltage_gate_jacobian
from replay_level0_atlas import Replay


def spectrum(replay, state, dt):
    replay.state = state
    eigenvalues = np.r_[np.linalg.eigvals(voltage_gate_jacobian(replay)), -replay.inv_calcium_tau]
    return {'max_real_eigenvalue':float(eigenvalues.real.max()),
        'min_real_eigenvalue':float(eigenvalues.real.min()),
        'euler_amplification_radius':float(np.abs(1+dt*eigenvalues).max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['failure','manifest','graph','output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    failed, manifest, graph = [json.loads(Path(getattr(a,k)).read_text()) for k in ['failure','manifest','graph']]
    if failed['format'] != 'wormsim-capacity-failure-replay' or failed['reference_manifest_sha256'] != digest(a.manifest):
        raise ValueError('failure lineage mismatch')
    if manifest['input_sha256']['graph'] != digest(a.graph):
        raise ValueError('graph hash mismatch')
    packed = failed['model']
    if set(packed['configuration']['extensions']) != {'observation'} or packed['configuration']['solver'] is not None:
        raise ValueError('requires observation-only Euler model')
    base = packed['base_model']; model = copy.deepcopy(base)
    model['config']['preparation_seconds'] = 0.
    replay = Replay(model,graph);n = replay.n
    if not (np.all(replay.weight >= 0) and np.all(replay.gap >= 0) and np.all(replay.inv_tau > 0)
            and np.all((replay.seed[2*n:] >= 0) & (replay.seed[2*n:] <= 1))):
        raise ValueError('conductance invariant assumptions fail')
    # Unforced voltage RHS is a positive weighted sum of differences to rest,
    # reversal potentials and neighboring voltages. Its continuous flow cannot
    # leave this interval while synaptic gates remain in [0,1].
    endpoints = np.r_[replay.seed[:n], replay.rest, replay.reversal]
    lower, upper = float(endpoints.min()), float(endpoints.max())
    rows=[]
    for divisor in [1,2,4]:
        dt = base['config']['dt']/divisor;state = replay.seed.copy()
        row = {'dt':dt,'initial_local_spectrum':spectrum(replay,state.copy(),dt),
            'first_voltage_bound_exit':None,'first_gate_bound_exit':None}
        max_voltage = float(np.abs(state[:n]).max())
        steps = round(.5/dt)
        if not np.isclose(steps*dt,.5):raise ValueError('step must divide diagnostic interval')
        with np.errstate(over='ignore',invalid='ignore'):
            for step in range(1,steps+1):
                state = state + dt*replay.rhs(state,None,0.)
                if not np.isfinite(state).all():raise ValueError('nonfinite within short diagnostic interval')
                max_voltage = max(max_voltage,float(np.abs(state[:n]).max()))
                if row['first_voltage_bound_exit'] is None and np.any((state[:n]<lower-1e-12)|(state[:n]>upper+1e-12)):
                    row['first_voltage_bound_exit'] = {'time':step*dt,'voltage_min':float(state[:n].min()),
                        'voltage_max':float(state[:n].max())}
                if row['first_gate_bound_exit'] is None and np.any((state[2*n:]<0)|(state[2*n:]>1)):
                    row['first_gate_bound_exit'] = step*dt
        row.update(end_time=steps*dt,max_absolute_voltage=max_voltage,
            final_voltage_min=float(state[:n].min()),final_voltage_max=float(state[:n].max()))
        rows.append(row)
    receipt = {'schema_version':1,'input_sha256':{k:digest(getattr(a,k)) for k in ['failure','manifest','graph']},
        'source_sha256':{name:digest(Path(__file__).with_name(name)) for name in ['audit_capacity_preparation.py','audit_capacity_stability.py','replay_level0_atlas.py']},
        'unforced_continuous_voltage_interval':[lower,upper], 'transients':rows,
        'scope':'Frozen failing parameters; first 0.5 seconds of unforced preparation. The positive-conductance continuous voltage interval assumes gates in [0,1]. Local eigenvalues diagnose the seed, not global stability. No refit or held-out outcomes.'}
    with Path(a.output).open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
