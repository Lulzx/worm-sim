import unittest
import copy
import numpy as np
import jax
import jax.numpy as jnp
from test_level0 import fixture
from level0 import Level0, response
from objective import build, evaluate


def example():
    model,graph,times=fixture()
    for key in ['graph_hash','dataset_hash','split_hash']:
        model[key]=key
    model['training_trials']=['train']
    model['sample_dt']=.05
    model['kernel_prior']=list(model['kernel_raw'])
    for i,g in enumerate(model['parameters']['groups']):
        g.update(trainable=i!=2,prior_mean=g['value']-.1)
    model['classifier']={'bias':-.7,'raw_slope':.3,'epsilon':.01,'area_scale':.05}
    model['config'].update(prior_strength=.02,sign_prior_strength=.03,kernel_prior_strength=.04,
                          classification={'weight':.1},correlation={'weight':.2,'epsilon':.03},
                          observation_gain={'initial_gain':1.5,'prior_strength':.1})
    training={k:model[k] for k in ['graph_hash','dataset_hash','split_hash','training_trials']}
    training.update(names=['A','B'],sign_probabilities=[.7],classification_pairs=1,groups=[{
        'target':0,'sample_weight':6.,'irreducible_mse':.02,'labels':[[1,True]],
        'recording':{'times':times,'traces':[
            {'neuron':'A','values':[0.,.01,.03],'provenance':{'id_confidence':1.}},
            {'neuron':'B','values':[0.,.02,.02],'provenance':{'id_confidence':.5}}]}}])
    return model,graph,training


class ObjectiveTests(unittest.TestCase):
    def test_composed_loss_matches_numpy_and_finite_differences(self):
        model,graph,training=example()
        theta,active,groups,data,prior=build(model,graph,training)
        value,gradient,metrics=evaluate(theta,groups,data,prior)
        p=np.asarray(response(Level0(model,graph,[0.,.05,.1]),theta,jnp.asarray(0)))
        y=np.array([[0.,0.],[.01,.02],[.03,.02]])
        expected_mse=np.sum((p-y)**2*np.array([1.,.5]))/4.5+.02
        area=.05*np.sum(p[:,1]**2/(np.hypot(p[:,1],.01)+.01))
        logit=-.7+np.logaddexp(0,.3)*np.log1p(area/.05)
        bce=np.logaddexp(0,-logit)
        pc=p-p.mean(0);yc=y-y.mean(0)
        corr=np.mean(1-np.mean(pc*yc,0)/np.sqrt((np.mean(pc**2,0)+.03**2)*(np.mean(yc**2,0)+.03**2)))
        self.assertAlmostEqual(metrics['mse'],expected_mse,places=13)
        self.assertAlmostEqual(metrics['bce'],bce,places=13)
        self.assertAlmostEqual(metrics['correlation'],corr,places=13)
        self.assertAlmostEqual(float(value),expected_mse+.1*bce+.2*corr+metrics['prior'],places=13)
        self.assertEqual(float(gradient['groups'][2]),0.)
        self.assertFalse(bool(active['groups'][2]))
        for key,i in [('groups',13),('kernel',0),('classifier',0),('classifier',1),('log_gain',None)]:
            plus,minus=dict(theta),dict(theta)
            if i is None:
                plus[key]+=1e-5;minus[key]-=1e-5;actual=gradient[key]
            else:
                plus[key]=theta[key].at[i].add(1e-5)
                minus[key]=theta[key].at[i].add(-1e-5)
                actual=gradient[key][i]
            finite=(evaluate(plus,groups,data,prior)[0]-evaluate(minus,groups,data,prior)[0])/2e-5
            np.testing.assert_allclose(actual,finite,atol=1e-9,rtol=2e-5)

    def test_lineage_mismatch_fails(self):
        model,graph,training=example()
        changed=copy.deepcopy(training)
        changed['training_trials']=['test']
        with self.assertRaises(ValueError):
            build(model,graph,changed)

if __name__=='__main__':
    unittest.main()
