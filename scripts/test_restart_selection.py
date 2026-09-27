import copy
import json
from pathlib import Path
import tempfile
import unittest
from audit_connectome_fit import digest, load
from select_atlas_restart import select, normalized_config


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.write('data.json',{'graph_hash':'graph','trials':[{'id':'validation',
            'recording':{'times':[0.,1.],'traces':[{'neuron':'A','values':[0.,1.],
                         'provenance':{'id_confidence':1.}}]}}]})
        self.write('split.json',{'train':['training'],'validation':['validation'],'test':['test'],
                                 'dataset_hash':'data'})
        self.manifest={'schema_version':1,'data':'data.json','split':'split.json',
            'data_file_sha256':digest(self.root/'data.json'),'split_file_sha256':digest(self.root/'split.json'),
            'dataset_hash':'data','split_hash':'split','graph_hash':'graph','fit_source_commit':'a'*40,'runs':[]}
        for seed in (1,2):self.add_run(seed,float(seed-1))

    def write(self,path,data):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data))

    def add_run(self,seed,offset):
        directory=f'run{seed}'
        config=normalized_config({'epochs':1,'sign_initialization':{'seed':seed,'reversal_magnitude':.5}})
        self.write(f'config{seed}.json',config)
        entry={'seed':seed,'config':f'config{seed}.json','config_sha256':digest(self.root/f'config{seed}.json'),'directory':directory}
        self.manifest['runs']=[e for e in self.manifest['runs'] if e['seed']!=seed]+[entry]
        self.write(f'{directory}/config.json',config)
        base={'config':config,'dataset_hash':'data','split_hash':'split','graph_hash':'graph',
              'source_commit':'a'*40,'training_trials':['training'],'selection_trials':['validation']}
        records=[{'epoch':0,'validation_mse':2.},{'epoch':1,'validation_mse':offset**2}]
        for record in records:
            self.write(f"{directory}/epoch-{record['epoch']}.report.json",record)
            self.write(f"{directory}/epoch-{record['epoch']}.json",dict(base,epoch=record['epoch']))
        self.write(f'{directory}/selected.json',dict(base,epoch=1))
        self.write(f'{directory}/selection.json',records)
        self.write(f'{directory}/validation-predictions.json',dict(base,seed=seed,trials=[
            {'id':'validation','times':[0.,1.],'fluorescence':{'A':[offset,1+offset]}}]))
        self.write(f'{directory}/validation-report.json',{'pooled_trace_scores':{'mse':offset**2}})

    def test_selection_needs_no_test_files_and_ties_use_smaller_seed(self):
        self.assertEqual(select(self.manifest,self.root)['selected_seed'],1)
        self.assertFalse(list(self.root.rglob('test*')))
        self.add_run(2,0.)
        self.manifest['runs'].reverse()
        self.assertEqual(select(self.manifest,self.root)['selected_seed'],1)

    def test_incomplete_or_duplicate_cohort_is_rejected(self):
        duplicate=copy.deepcopy(self.manifest)
        duplicate['runs'].append(duplicate['runs'][0])
        with self.assertRaises(AssertionError):select(duplicate,self.root)
        records=load(self.root/'run2/selection.json')
        self.write('run2/selection.json',records[:-1])
        with self.assertRaises(AssertionError):select(self.manifest,self.root)

    def test_changed_validation_prediction_is_detected_by_rescoring(self):
        pred=load(self.root/'run1/validation-predictions.json')
        pred['trials'][0]['fluorescence']['A'][0]=9.
        self.write('run1/validation-predictions.json',pred)
        with self.assertRaises(AssertionError):select(self.manifest,self.root)

    def test_changed_declared_config_is_rejected(self):
        config=load(self.root/'config1.json');config['epochs']=2
        self.write('config1.json',config)
        with self.assertRaises(AssertionError):select(self.manifest,self.root)


if __name__=='__main__':unittest.main()
