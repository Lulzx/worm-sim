import copy
import json
import unittest
from pathlib import Path
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
from test_objective import example
from objective import build, evaluate
from extensions import initialize, pack, restore
from fit import checkpoint
from overfit import prepare, bounds, warm_parameters, atomic_best, failure_artifact


class OverfitTests(unittest.TestCase):
    def fixture(self):
        m,g,t=example()
        m.update(epoch=0,selection_trials=['validation'])
        m['config']['sign_initialization']={'seed':1,'reversal_magnitude':.5}
        for i,group in enumerate(m['parameters']['groups']):
            group['name']=f'toy/{i}'
        t['groups'][0]['training_trials']=['train']
        return m,g,t

    def test_subset_rejects_unknown_targets_and_keeps_source_untouched(self):
        m,g,t=self.fixture();original=copy.deepcopy((m,t))
        out,data,config=prepare(m,t,['A'],10,.01,-.2)
        self.assertEqual((m,t),original)
        self.assertEqual(out['selection_trials'],[])
        self.assertEqual(out['training_trials'],['train'])
        self.assertIsNone(out['classifier'])
        self.assertEqual(data['classification_pairs'],0)
        for targets in [[],['A','A'],['B'],['test']]:
            with self.assertRaises(ValueError):prepare(m,t,targets,10,.01,-.2)
        longer,_,_=prepare(m,t,['A'],10,.01,-.2,120.)
        self.assertEqual(longer['config']['preparation_seconds'],120.)
        self.assertEqual((m,t),original)
        for duration in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):prepare(m,t,['A'],10,.01,-.2,duration)
        t['groups'][0]['training_trials']=['validation']
        with self.assertRaises(ValueError):prepare(m,t,['A'],10,.01,-.2)

    def test_warm_start_preserves_predictions_and_rejects_incompatible_parents(self):
        m,g,t=self.fixture();m,t,c=prepare(m,t,['A'],10,.01,-.2)
        engine,theta,active=initialize(m,g,[0.,.05,.1],c)
        theta=jax.tree.map(lambda value,mask:jnp.where(mask,value+.03,value),theta,active)
        saved={'format':'wormsim-training-capacity-diagnostic','targets':['A'],
            'model':pack(checkpoint(m,theta,7,'parent'),theta,c),'metrics':{'epoch':7,'mse':.03}}
        new=copy.deepcopy(m);new['config'].update(epochs=3,learning_rate=.001)
        q=warm_parameters(saved,new,g,t,c,['A'])
        for left,right in zip(jax.tree.leaves(theta),jax.tree.leaves(q),strict=True):
            np.testing.assert_array_equal(left,right)
        np.testing.assert_allclose(engine.response(theta,jnp.asarray(0)),engine.response(q,jnp.asarray(0)),atol=0.)
        refined=copy.deepcopy(new);refined['config']['dt']/=2
        with self.assertRaises(ValueError):warm_parameters(saved,refined,g,t,c,['A'])
        q=warm_parameters(saved,refined,g,t,c,['A'],allow_step_change=True)
        for left,right in zip(jax.tree.leaves(theta),jax.tree.leaves(q),strict=True):
            np.testing.assert_array_equal(left,right)
        longer=copy.deepcopy(new);longer['config']['preparation_seconds']+=1.
        with self.assertRaises(ValueError):warm_parameters(saved,longer,g,t,c,['A'])
        q=warm_parameters(saved,longer,g,t,c,['A'],allow_preparation_change=True)
        for left,right in zip(jax.tree.leaves(theta),jax.tree.leaves(q),strict=True):
            np.testing.assert_array_equal(left,right)
        for duration in [0.,-1.,float('nan'),new['config']['preparation_seconds']/2]:
            invalid=copy.deepcopy(new);invalid['config']['preparation_seconds']=duration
            with self.assertRaises(ValueError):warm_parameters(saved,invalid,g,t,c,['A'],allow_preparation_change=True)
        invalid=copy.deepcopy(longer);invalid['initial'][0]=99.
        with self.assertRaises(ValueError):warm_parameters(saved,invalid,g,t,c,['A'],allow_preparation_change=True)
        for invalid_step in [0.,float('nan'),new['config']['dt']*2]:
            invalid=copy.deepcopy(refined);invalid['config']['dt']=invalid_step
            with self.assertRaises(ValueError):warm_parameters(saved,invalid,g,t,c,['A'],allow_step_change=True)
        invalid=copy.deepcopy(refined);invalid['initial'][0]=99.
        with self.assertRaises(ValueError):warm_parameters(saved,invalid,g,t,c,['A'],allow_step_change=True)
        for mutate in [lambda x:x.update(targets=['B']),
                       lambda x:x['model']['base_model'].update(training_trials=['test']),
                       lambda x:x['model']['base_model']['initial'].__setitem__(0,99.),
                       lambda x:x['model']['base_model']['config'].update(preparation_seconds=5.),
                       lambda x:x['model']['base_model']['parameters']['groups'][2].update(value=99.),
                       lambda x:x['model']['base_model']['parameters']['groups'][0].update(name='wrong'),
                       lambda x:x['metrics'].update(epoch=8)]:
            invalid=copy.deepcopy(saved);mutate(invalid)
            with self.assertRaises(ValueError):warm_parameters(invalid,new,g,t,c,['A'])

    def test_best_checkpoint_replacement_keeps_previous_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'best.json'
            atomic_best(path,{'epoch':0,'mse':1.})
            atomic_best(path,{'epoch':1,'mse':.5})
            self.assertEqual(json.loads(path.read_text()),{'epoch':1,'mse':.5})
            with self.assertRaises(ValueError):atomic_best(path,{'epoch':2,'mse':float('nan')})
            self.assertEqual(json.loads(path.read_text())['epoch'],1)
            self.assertFalse(path.with_name('best.json.tmp').exists())

    def test_failure_artifact_retains_finite_parameters_and_nulls_nonfinite_metrics(self):
        m,g,t=self.fixture();m,t,c=prepare(m,t,['A'],10,.01,-.2)
        theta,_,_,_,_=build(m,g,t,c)
        gradient=jax.tree.map(jnp.zeros_like,theta)
        gradient['groups']=gradient['groups'].at[0].set(jnp.nan)
        artifact=failure_artifact(m,theta,c,3,'test',jnp.nan,gradient,{'mse':float('nan')},['A'],None)
        encoded=json.loads(json.dumps(artifact,allow_nan=False))
        self.assertEqual(encoded['nonfinite_gradient_coordinates'],1)
        self.assertIsNone(encoded['metrics']['mse'])
        _,restored,_=restore(encoded['model'],g,t['groups'][0]['recording']['times'])
        for a,b in zip(jax.tree.leaves(theta),jax.tree.leaves(restored),strict=True):
            np.testing.assert_array_equal(a,b)
        theta['groups']=theta['groups'].at[0].set(jnp.inf)
        artifact=failure_artifact(m,theta,c,4,'test',jnp.nan,gradient,{'mse':float('inf')},['A'],None)
        self.assertIsNone(artifact['model'])
        self.assertEqual(artifact['nonfinite_parameter_coordinates'],1)
        json.dumps(artifact,allow_nan=False)

    def test_per_neuron_gains_gradient_roundtrip_and_bounds(self):
        m,g,t=self.fixture();m,t,c=prepare(m,t,['A'],10,.01,-.2)
        theta,active,groups,data,prior=build(m,g,t,c)
        self.assertFalse(bool(active['log_gain']))
        reference=bounds(groups)
        mean=np.array([[0.,0.],[.01,.02],[.03,.02]])
        self.assertAlmostEqual(reference['mean_response_bound'],.02)
        self.assertAlmostEqual(reference['zero_response_mse'],.02+np.sum(mean**2*[1.,.5])/4.5)
        _,grad,_=evaluate(theta,groups,data,prior)
        for i in range(2):
            plus=copy.deepcopy(theta);minus=copy.deepcopy(theta)
            plus['observation']['log_gain']=plus['observation']['log_gain'].at[i].add(1e-5)
            minus['observation']['log_gain']=minus['observation']['log_gain'].at[i].add(-1e-5)
            finite=(evaluate(plus,groups,data,prior)[0]-evaluate(minus,groups,data,prior)[0])/2e-5
            np.testing.assert_allclose(grad['observation']['log_gain'][i],finite,rtol=1e-5,atol=1e-10)
        engine,_,_=initialize(m,g,[0.,.05,.1],c)
        baseline=np.asarray(engine.response(theta,jnp.asarray(0)))
        changed=copy.deepcopy(theta)
        changed['observation']['log_gain']=theta['observation']['log_gain'].at[0].add(np.log(2))
        np.testing.assert_allclose(engine.response(changed,jnp.asarray(0)),baseline*[2.,1.],atol=1e-14)
        saved=pack(checkpoint(m,changed,1,'test'),changed,c)
        restored,q,_=restore(json.loads(json.dumps(saved)),g,[0.,.05,.1])
        np.testing.assert_allclose(restored.response(q,jnp.asarray(0)),baseline*[2.,1.],atol=1e-14)
        invalid=copy.deepcopy(saved);invalid['extension_parameters']['observation']['log_gain']['shape']=[3]
        with self.assertRaises(ValueError):restore(invalid,g,[0.,.05,.1])
        m['parameters']['groups'][0].update(name='calcium_scale/A',trainable=True)
        with self.assertRaises(ValueError):initialize(m,g,[0.,.05,.1],c)

if __name__=='__main__':unittest.main()
