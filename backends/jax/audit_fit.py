"""Compare completed JAX/Rust training trajectories without selecting on test data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def load(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();run=Path(a.run);reference=Path(a.reference)
    reports=load(run/'selection.json');expected=load(reference/'selection.json')
    manifest=load(run/'manifest.json')
    selected=load(run/'selected.json')
    epochs=selected['config']['epochs']
    assert [r['epoch'] for r in reports]==[r['epoch'] for r in expected]==list(range(epochs+1))
    assert selected['epoch']==min(reports,key=lambda r:(r['validation_mse'],r['epoch']))['epoch']
    assert selected==load(run/f"epoch-{selected['epoch']}.json")
    comparisons=[];hashes={}
    for r,e in zip(reports,expected,strict=True):
        epoch=r['epoch'];model=load(run/f'epoch-{epoch}.json');original=load(reference/f'epoch-{epoch}.json')
        for field in ['schema_version','graph_hash','dataset_hash','split_hash','config','initial','kernel_prior','training_trials','selection_trials','sample_dt','classification_evidence_hash']:
            assert model[field]==original[field],field
        assert model['epoch']==original['epoch']==epoch
        assert model['source_commit']==manifest['source_commit']
        assert (model.get('observation_log_gain') is None)==(original.get('observation_log_gain') is None)
        assert model['parameters']['raw_to_group']==original['parameters']['raw_to_group']
        for g,h in zip(model['parameters']['groups'],original['parameters']['groups'],strict=True):
            assert {k:v for k,v in g.items() if k!='value'}=={k:v for k,v in h.items() if k!='value'}
            if not g['trainable']:
                assert g['value']==h['value']
        errors={'groups':float(np.max(np.abs(np.array([g['value'] for g in model['parameters']['groups']])-np.array([g['value'] for g in original['parameters']['groups']])))),
                'kernel':float(np.max(np.abs(np.array(model['kernel_raw'])-np.array(original['kernel_raw']))))}
        assert (model.get('classifier') is None)==(original.get('classifier') is None)
        if model.get('classifier'):
            assert {k:v for k,v in model['classifier'].items() if k not in ['bias','raw_slope']}=={k:v for k,v in original['classifier'].items() if k not in ['bias','raw_slope']}
            errors['classifier']=max(abs(model['classifier'][k]-original['classifier'][k]) for k in ['bias','raw_slope'])
        if model.get('observation_log_gain') is not None:
            errors['log_gain']=abs(model['observation_log_gain']-original['observation_log_gain'])
        assert max(errors.values())<1e-8,errors
        native=load(run/f'validation-{epoch}'/'selection.json')
        assert native['partition']=='validation' and native['epoch']==epoch and native['validation_mse']==r['validation_mse']
        validation_error=abs(r['validation_mse']-e['validation_mse'])
        assert validation_error<1e-10
        loss_errors={}
        if epoch:
            for key,refkey in [('mse','preceding_training_mse'),('bce','preceding_training_classification_bce'),('prior','preceding_penalty')]:
                if e.get(refkey) is not None:
                    loss_errors[key]=abs(r['preceding_training_components'][key]-e[refkey])
                    assert loss_errors[key]<1e-10
        comparisons.append({'epoch':epoch,'maximum_parameter_errors':errors,'validation_mse_error':validation_error,'training_component_errors':loss_errors})
        for directory in [run,reference]:
            path=directory/f'epoch-{epoch}.json';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report={'schema_version':1,'audit_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'jax_fit_source_commit':manifest['source_commit'],'jax_fit_source_worktree_dirty':manifest['source_worktree_dirty'],'selected_epoch':selected['epoch'],'reference_selected_epoch':load(reference/'selected.json')['epoch'],'epochs':comparisons,'checkpoint_sha256':hashes,'scope':'Completed training trajectory and validation-only selection parity. Checks identities, frozen parameters, objective components and parameter updates. Does not recompute gradients, audit source-data processing or inspect test results.'}
    with open(a.output,'x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(f"verified {len(comparisons)} checkpoints; selected epoch {selected['epoch']}")

if __name__=='__main__':
    main()
