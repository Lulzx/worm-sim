import unittest
import copy
import jax
import jax.numpy as jnp
import numpy as np
import optax
from test_objective import example
from fit import fit, make_optimizer, checkpoint
from objective import build

class FitTests(unittest.TestCase):
    def test_selection_ties_and_frozen_coordinates(self):
        model,graph,training=example()
        model['epoch']=0
        model['config'].update(epochs=2,learning_rate=.001,optimizer={'kind':'adamw','weight_decay':.1})
        seen=[]
        def score(candidate):
            seen.append(copy.deepcopy(candidate))
            return [2.,1.,1.][candidate['epoch']]
        selected,reports=fit(model,graph,training,'test-source',score)
        self.assertEqual(selected['epoch'],1)
        self.assertEqual([x['epoch'] for x in reports],[0,1,2])
        self.assertNotEqual(seen[0]['kernel_raw'],seen[2]['kernel_raw'])
        for m in seen:
            self.assertEqual(m['parameters']['groups'][2],model['parameters']['groups'][2])
            self.assertEqual(m['training_trials'],model['training_trials'])
            self.assertEqual(m['kernel_prior'],model['kernel_prior'])
            self.assertEqual(m['initial'],model['initial'])
        model['epoch']=1
        with self.assertRaises(ValueError):
            fit(model,graph,training,'source',score)

    def test_library_cosine_endpoints(self):
        config={'epochs':3,'learning_rate':.01,'learning_rate_schedule':{'kind':'cosine','minimum_fraction':.2}}
        p={'x':jnp.asarray(1.)}
        opt=make_optimizer(config,{'x':True});state=opt.init(p)
        for rate in [.01,.006,.002]:
            update,state=opt.update({'x':jnp.asarray(1.)},state,p)
            self.assertAlmostEqual(float(update['x']),-rate/(1.+1e-8),places=12)
            p=optax.apply_updates(p,update)

if __name__=='__main__':
    unittest.main()
