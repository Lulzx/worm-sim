import copy
import json
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from test_objective import example
from objective import build, evaluate
from extensions import initialize, pack, restore
from fit import checkpoint
from overfit import prepare, bounds


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
