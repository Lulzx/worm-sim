#!/usr/bin/env python3
"""Independent NumPy audit of local chemical gate sensitivity at preparation."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_connectome_fit import load, digest
from replay_level0_atlas import Replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--native',required=True)
    parser.add_argument('--graph',default='runs/c302-audit.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    model,native=load(args.model),load(args.native)
    assert native['model_sha256']==digest(args.model)
    assert native['model_source_commit']==model['source_commit']
    assert native['graph_hash']==model['graph_hash']
    assert native['epoch']==model['epoch']
    replay=Replay(model,load(args.graph))
    assert native['dt']==replay.dt and native['preparation_seconds']==replay.preparation
    force=replay.reversal-replay.state[replay.post]
    sensitivity=replay.inv_tau[replay.post]*replay.weight*force
    indexed={(e['pre'],e['post']):e for e in native['edges']}
    expected={(replay.names[a],replay.names[b]) for a,b in zip(replay.pre,replay.post)}
    assert len(indexed)==len(native['edges'])==native['chemical_edges']==len(sensitivity)
    assert set(indexed)==expected
    error=0.
    for i,(a,b) in enumerate(zip(replay.pre,replay.post)):
        edge=indexed[replay.names[a],replay.names[b]]
        error=max(error,abs(sensitivity[i]-edge['d_voltage_rate_d_gate']))
        assert abs(force[i]-edge['driving_force'])<1e-10
    assert error<1e-10
    counts={'exact_zero_gate_sensitivity_edges':int(np.count_nonzero(sensitivity==0)),
            'positive_gate_sensitivity_edges':int(np.count_nonzero(sensitivity>0)),
            'negative_gate_sensitivity_edges':int(np.count_nonzero(sensitivity<0))}
    for name,value in counts.items():assert value==native[name]
    norm=float(np.linalg.norm(sensitivity))
    assert abs(norm-native['gate_sensitivity_l2'])<1e-10
    receipt={'schema_version':1,'native_source_commit':native['source_commit'],
             'model_source_commit':model['source_commit'],'model_sha256':digest(args.model),
             'graph_file_sha256':digest(args.graph),'native_sha256':digest(args.native),
             'audit_script_sha256':digest(__file__),
             'replay_script_sha256':digest(Path(__file__).with_name('replay_level0_atlas.py')),
             'epoch':model['epoch'],'preparation_seconds':replay.preparation,'dt':replay.dt,
             'chemical_edges':len(sensitivity),**counts,'gate_sensitivity_l2':norm,
             'max_gate_sensitivity_error':float(error),
             'scope':'Independent prepared-state dynamics and local chemical gate Jacobian. No optimization replay, biological sign inference or held-out prediction score.'}
    Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
