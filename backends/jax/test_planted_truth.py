import copy
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import test_overfit
from objective import build, evaluate
from extensions import initialize, pack
from fit import checkpoint, fit
from overfit import prepare, bounds
from planted_truth import (noise_ceiling, make, metrics, predict, score, trial_residuals,
    noise_variance, energies, FORMAT)


def fixture():
    m,g,t=test_overfit.OverfitTests.fixture(None)
    m['training_trials']=t['training_trials']=['t1','t2']
    t['groups'][0]['training_trials']=['t1','t2']
    trace=lambda n,v,w:{'neuron':n,'values':v,'provenance':{'id_confidence':w}}
    data={'trials':[
        {'id':'t1','stimulated_neuron':'A','recording':{'traces':[trace('A',[0.,0.,.02],1.),trace('B',[0.,.03,.01],.5)]}},
        {'id':'t2','stimulated_neuron':'A','recording':{'traces':[trace('A',[0.,.02,.04],1.),trace('B',[0.,.01,.03],.5)]}}]}
    return m,g,t,data


def truth(m,g,t,shift=.03):
    q,_,c=prepare(m,t,['A'],10,.01,-.2)
    engine,theta,active=initialize(q,g,[0.,.05,.1],c)
    theta=jax.tree.map(lambda v,a:jnp.where(a,v+shift,v),theta,active)
    return {'format':'wormsim-training-capacity-diagnostic','targets':['A'],
        'model':pack(checkpoint(q,theta,7,'truth'),theta,c),'metrics':{}},theta


class PlantedTruthTests(unittest.TestCase):
    def test_residuals_reproduce_mean_and_reject_foreign_trials(self):
        m,g,t,data=fixture()
        res=trial_residuals(t['groups'][0],{x['id']:x for x in data['trials']},t['names'])
        np.testing.assert_allclose(res['A'][1],[[0.,-.01,-.01],[0.,.01,.01]],atol=1e-15)
        bad=copy.deepcopy(data);bad['trials'][0]['recording']['traces'][0]['values'][1]=.5
        with self.assertRaises(ValueError):
            trial_residuals(t['groups'][0],{x['id']:x for x in bad['trials']},t['names'])
        bad=copy.deepcopy(data);bad['trials'][1]['stimulated_neuron']='B'
        with self.assertRaises(ValueError):
            trial_residuals(t['groups'][0],{x['id']:x for x in bad['trials']},t['names'])

    def test_noise_variance_is_unbiased_for_the_weighted_mean(self):
        rng=np.random.default_rng(3);w=np.array([1.,.5,.8,.3]);W=w.sum()
        draws=rng.normal(size=(200000,4,2))
        means=(w[None,:,None]*draws).sum(1)/W
        estimates=np.stack([noise_variance(w,d-mu[None]) for d,mu in zip(draws[:2000],means[:2000])])
        np.testing.assert_allclose(estimates.mean(0),means.var(0),rtol=.05)
        self.assertIsNone(noise_variance(np.array([1.]),np.zeros((1,3))))

    def test_metrics_match_objective_and_bounds(self):
        m,g,t,data=fixture()
        q,u,c=prepare(m,t,['A'],10,.01,-.2)
        theta,active,groups,dg,pg=build(q,g,u,c)
        _,_,result=evaluate(theta,groups,dg,pg)
        engine,_,_=initialize(q,g,[0.,.05,.1],c)
        p=metrics(u['groups'],predict(engine,theta,u['groups'],u['names']))
        self.assertAlmostEqual(p['mse'],result['mse'],places=14)
        b=bounds(groups);zero,start=energies(u['groups'])
        self.assertAlmostEqual(zero,b['zero_response_mse'],places=14)
        self.assertAlmostEqual(start,b['start_zero_mean_response_bound'],places=14)

    def test_noise_free_truth_is_recovered_exactly_and_fit_accepts_export(self):
        m,g,t,data=fixture();saved,theta=truth(m,g,t)
        out,block=make(m,g,t,data,['A'],None,saved,None,'none',0)
        out['synthetic']=dict(format=FORMAT,**block)
        self.assertAlmostEqual(block['oracle']['captured_start_zero_energy'],1.,places=12)
        self.assertAlmostEqual(block['oracle']['signal_recovery'],1.,places=12)
        q,u,c=prepare(m,out,['A'],10,.01,-.2)
        th,_,groups,dg,pg=build(q,g,u,c)
        th=jax.tree.map(lambda a,b:b,th,theta)
        _,_,result=evaluate(th,groups,dg,pg)
        self.assertAlmostEqual(result['mse'],block['oracle']['mse'],places=12)
        scored=score(g,out,saved,saved)
        self.assertAlmostEqual(scored['signal_recovery'],1.,places=12)
        self.assertIsNone(scored['chemical_sign_agreement'])
        with self.assertRaises(ValueError):
            fit(q,g,out,'x',lambda c:0.)

    def test_residual_noise_has_real_sampling_variance_and_lowers_oracle_capture(self):
        m,g,t,data=fixture();saved,_=truth(m,g,t)
        energies_=[];captures=[]
        for seed in range(400):
            out,block=make(m,g,t,data,['A'],None,saved,None,'residual',seed)
            energies_.append(block['oracle']['noise_energy'])
            captures.append(block['oracle']['captured_start_zero_energy'])
        # Two equal-weight trials have opposite residuals, so some sign draws cancel.
        self.assertLessEqual(max(captures),1.+1e-12)
        self.assertLess(np.mean(captures),1.)
        expected=noise_ceiling(t,data,['A'])['noise_energy']
        self.assertAlmostEqual(np.mean(energies_)/expected,1.,delta=.1)

    def test_truth_must_share_fit_configuration(self):
        m,g,t,data=fixture();saved,_=truth(m,g,t)
        with self.assertRaises(ValueError):
            make(m,g,t,data,['A'],.08,saved,None,'none',0)
        bad=copy.deepcopy(saved);bad['model']['configuration']['extensions']['observation']['initial_gain']=3.
        with self.assertRaises(ValueError):
            make(m,g,t,data,['A'],None,bad,None,'none',0)


if __name__=='__main__':
    unittest.main()
