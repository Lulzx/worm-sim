import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_objective import example

class AuditTests(unittest.TestCase):
    def test_complete_trajectory_and_tampered_checkpoint(self):
        model,_,_=example()
        model.update(schema_version=1,source_commit='jax-source',classification_evidence_hash='evidence',selection_trials=['validation'])
        model['config']['epochs']=1
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'jax';ref=root/'rust';run.mkdir();ref.mkdir()
            def write(path,value):path.write_text(json.dumps(value))
            write(run/'manifest.json',{'source_commit':'jax-source','source_worktree_dirty':False})
            reports=[];original_reports=[]
            for epoch in range(2):
                checkpoint=copy.deepcopy(model);checkpoint['epoch']=epoch
                checkpoint['parameters']['groups'][0]['value']+=epoch*.001
                write(run/f'epoch-{epoch}.json',checkpoint)
                original=copy.deepcopy(checkpoint);original['source_commit']='rust-source'
                write(ref/f'epoch-{epoch}.json',original)
                evaluation=run/f'validation-{epoch}';evaluation.mkdir()
                write(evaluation/'selection.json',{'partition':'validation','epoch':epoch,'validation_mse':1.-epoch*.1})
                reports.append({'epoch':epoch,'validation_mse':1.-epoch*.1,'preceding_training_components':None if epoch==0 else {'mse':.1,'bce':.2,'prior':.3}})
                original_reports.append({'epoch':epoch,'validation_mse':1.-epoch*.1,'preceding_training_mse':None if epoch==0 else .1,'preceding_training_classification_bce':None if epoch==0 else .2,'preceding_penalty':None if epoch==0 else .3})
            write(run/'selection.json',reports);write(ref/'selection.json',original_reports)
            write(run/'selected.json',checkpoint);write(ref/'selected.json',original)
            command=[sys.executable,str(Path(__file__).with_name('audit_fit.py')),'--run',str(run),'--reference',str(ref)]
            passed=subprocess.run(command+['--output',str(root/'pass.json')],capture_output=True,text=True)
            self.assertEqual(passed.returncode,0,passed.stderr)
            bad=json.loads((run/'epoch-0.json').read_text())
            bad['parameters']['groups'][0]['value']+=.01
            write(run/'epoch-0.json',bad)
            failed=subprocess.run(command+['--output',str(root/'fail.json')],capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertFalse((root/'fail.json').exists())

if __name__=='__main__':
    unittest.main()
