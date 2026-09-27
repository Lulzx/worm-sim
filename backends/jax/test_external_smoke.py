"""Cross-language pipeline check. CI builds the Rust executables before this test."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BIN=ROOT/'target/release/examples'


class ExternalSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required=['external_atlas','export_external_atlas_fixture','export_atlas_training']
        if not all((BIN/n).exists() for n in required):
            raise unittest.SkipTest('build --release --example external_atlas --example export_external_atlas_fixture --example export_atlas_training to run Rust/JAX integration')

    def test_fit_reload_score_and_checkpoint_binding(self):
        self.check_pipeline('checkpoint')

    def test_continuous_adjoint_fit_reload_score(self):
        self.check_pipeline('continuous')

    def check_pipeline(self,adjoint):
        def run(*args, good=True):
            result=subprocess.run([str(a) for a in args],cwd=ROOT,text=True,capture_output=True)
            if good:self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            else:self.assertNotEqual(result.returncode,0)
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);inputs=folder/'inputs'
            run(BIN/'export_external_atlas_fixture',inputs)
            common=[inputs/n for n in ['graph.wsc','data.json','split.json']]
            run(BIN/'export_atlas_training',*common,inputs/'model.json',folder/'training.json')
            config=json.loads((ROOT/'backends/jax/examples/extensions-synthetic.json').read_text())
            config['solver']['adjoint']=adjoint
            if adjoint=='checkpoint':
                config['solver']['checkpoints']=2
            else:
                config['solver'].update(adjoint_rtol=1e-8,adjoint_atol=1e-10,adjoint_max_steps=10000)
            (folder/'configuration.json').write_text(json.dumps(config))
            out=folder/'fit'
            run(sys.executable,'backends/jax/fit_extensions.py','--model',inputs/'model.json','--configuration',folder/'configuration.json','--graph-json',inputs/'graph.json','--graph',common[0],'--data',common[1],'--split',common[2],'--training',folder/'training.json','--scorer',BIN/'external_atlas','--output',out)
            selected=json.loads((out/'selected.json').read_text());epoch=selected['base_model']['epoch']
            reports=json.loads((out/'selection.json').read_text());self.assertEqual(len(reports),3)
            self.assertEqual(epoch,min(range(3),key=lambda i:reports[i]['validation_mse']))
            self.assertEqual((out/'selected.json').read_bytes(),(out/f'epoch-{epoch}.json').read_bytes())
            start=json.loads((out/'epoch-0.json').read_text());end=json.loads((out/'epoch-2.json').read_text())
            self.assertNotEqual(start['extension_parameters']['dark_edges'],end['extension_parameters']['dark_edges'])
            run(sys.executable,'backends/jax/predict_extensions.py','--checkpoint',out/'selected.json','--graph',inputs/'graph.json','--plan',out/'validation-plan.json','--output',folder/'replayed.json')
            original=json.loads((out/f'validation-{epoch}-predictions.json').read_text())
            self.assertEqual(json.loads((folder/'replayed.json').read_text()),original)
            run(BIN/'external_atlas','score',*common,out/'selected.json','validation',folder/'replayed.json',folder/'rescored')
            score=json.loads((folder/'rescored/selection.json').read_text())
            observed={t['id']:t for t in json.loads(common[1].read_text())['trials']}
            squares=[]
            for trial in original['trials']:
                for trace in observed[trial['id']]['recording']['traces']:
                    squares.extend((np.asarray(trace['values'])-trial['fluorescence'][trace['neuron']])**2)
            self.assertAlmostEqual(score['mse'],float(np.mean(squares)),places=13)
            self.assertEqual(score['checkpoint_sha256'],hashlib.sha256((out/'selected.json').read_bytes()).hexdigest())
            # Rebinding predictions to a changed parameter file must fail.
            selected['extension_parameters']['dark_edges']['raw_strength']['values'][0]+=.1
            (folder/'tampered.json').write_text(json.dumps(selected))
            run(BIN/'external_atlas','score',*common,folder/'tampered.json','validation',folder/'replayed.json',folder/'bad',good=False)
            # A native entry point must not ignore the JAX extension envelope.
            run(BIN/'external_atlas','plan',*common,out/'selected.json','validation',folder/'bad-plan.json',good=False)

if __name__=='__main__':unittest.main()
