import copy
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from test_level0 import fixture
from level0 import Level0, parameters, response
from modulation import Modulation
from solvers import Adaptive, integrate


def specification(beta=0.):
    return {'schema_version':1,'source':'synthetic mechanism test; no measured biological map',
        'channels':[{'id':'test/head','species':'test','compartment':'head','tau_seconds':2.}],
        'release':[{'neuron':name,'channel':'test/head','group':'shared-release','alpha':2.} for name in ['A','B']],
        'receptors':[{'neuron':'B','channel':'test/head','effect':effect,'group':effect,'kd':.5,'beta':beta} for effect in ['gain','leak','synapse']]}


class ModulationTests(unittest.TestCase):
    def test_concentration_matches_exact_constant_release_and_bath_solution(self):
        spec=specification();spec['channels'][0]['bath']=.3
        mod=Modulation(['A','B'],spec);p=mod.parameters()
        self.assertEqual(p['raw_release'].shape,(1,))
        def rhs(t,c,p):return mod.derivative(c,jnp.asarray([.2,.3]),p)
        times=jnp.asarray([0.,.1,.5,1.])
        solve=eqx.filter_jit(lambda p:integrate(rhs,jnp.zeros(1),p,times,Adaptive(rtol=1e-10,atol=1e-12)).ys)
        actual=np.asarray(solve(p))[:,0]
        expected=1.3*(1.-np.exp(-np.asarray(times)/2.))
        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-7)

    def test_disabled_sensitivities_preserve_fast_network_exactly(self):
        model,graph,times=fixture();mod=Modulation(['A','B'],specification())
        baseline=Level0(model,graph,times)
        extended=Level0(model,graph,times,modulation=mod)
        p=parameters(model,mod)
        before=np.asarray(response(baseline,parameters(model),jnp.asarray(0)))
        after=np.asarray(response(extended,p,jnp.asarray(0)))
        np.testing.assert_allclose(before,after,atol=1e-14,rtol=1e-13)
        states=eqx.filter_jit(lambda p:extended.solve(p,jnp.asarray(0)).ys)(p)
        self.assertGreater(float(states[-1,-1]),0.)

    def test_sparse_receptor_effects_are_local_positive_and_bounded(self):
        mod=Modulation(['A','B'],specification(2.));p=mod.parameters()
        effect=np.asarray(mod.multipliers(jnp.asarray([.5]),p))
        np.testing.assert_array_equal(effect[:,0],1.)
        np.testing.assert_allclose(effect[:,1],np.exp(3.*np.tanh(1./3.)),atol=1e-14)
        for beta in [-1e9,1e9]:
            q={**p,'sensitivity':jnp.full_like(p['sensitivity'],beta)}
            v=np.asarray(mod.multipliers(jnp.asarray([1.]),q))
            self.assertTrue(np.all(v>=np.exp(-3.)) and np.all(v<=np.exp(3.)))
        np.testing.assert_array_equal(mod.multipliers(jnp.asarray([-.1]),p),1.)

    def test_coupled_reverse_gradients_include_slow_parameters(self):
        model,graph,_=fixture();times=[0.,.1,.2,.3]
        model['config']['preparation_seconds']=.1
        mod=Modulation(['A','B'],specification(.7))
        engine=Level0(model,graph,times,Adaptive(method='tsit5',rtol=1e-10,atol=1e-12),mod)
        p=parameters(model,mod)
        def loss(p):return jnp.sum(engine.response(p,jnp.asarray(0))**2)
        evaluate=eqx.filter_jit(loss);vg=eqx.filter_jit(eqx.filter_value_and_grad(loss))
        _,grad=vg(p)
        for key in ['raw_tau','raw_release','raw_kd','sensitivity']:
            plus,minus=dict(p),dict(p)
            plus['modulation']=dict(p['modulation']);minus['modulation']=dict(p['modulation'])
            plus['modulation'][key]=p['modulation'][key].at[0].add(1e-4)
            minus['modulation'][key]=p['modulation'][key].at[0].add(-1e-4)
            finite=(evaluate(plus)-evaluate(minus))/2e-4
            actual=grad['modulation'][key][0]
            self.assertGreater(abs(float(actual)),1e-9,key)
            np.testing.assert_allclose(actual,finite,atol=1e-8,rtol=5e-4)

    def test_compartments_and_missing_release_maps_do_not_cross_talk(self):
        spec=specification(.7)
        spec['channels'].append({'id':'test/body','species':'test','compartment':'body','tau_seconds':5.,'bath':.3})
        spec['receptors'].append({'neuron':'A','channel':'test/body','effect':'gain','group':'body-gain','kd':.5,'beta':2.})
        mod=Modulation(['A','B'],spec);p=mod.parameters()
        np.testing.assert_allclose(mod.derivative(jnp.zeros(2),jnp.asarray([.2,.3]),p),[.5,.06],atol=1e-14)
        effect=np.asarray(mod.multipliers(jnp.asarray([1.,0.]),p))
        np.testing.assert_array_equal(effect[:,0],1.)
        self.assertTrue(np.all(effect[:,1]>1.))
        spec['release']=[];spec['receptors']=[]
        absent=Modulation(['A','B'],spec)
        np.testing.assert_allclose(absent.derivative(jnp.zeros(2),jnp.ones(2),absent.parameters()),[0.,.06],atol=1e-14)
        np.testing.assert_array_equal(absent.multipliers(jnp.ones(2),absent.parameters()),1.)

    def test_invalid_maps_and_conflicting_ties_fail(self):
        for mutate in [lambda s:s.update(unknown=True),lambda s:s.update(source=''),lambda s:s['channels'][0].update(tau_seconds=0.),lambda s:s['channels'][0].update(initial=-1.),lambda s:s['release'][0].update(neuron='unknown'),lambda s:s['receptors'][0].update(effect='unknown'),lambda s:s['release'][0].update(alpha=3.),lambda s:s['release'].append(copy.deepcopy(s['release'][0]))]:
            spec=specification();mutate(spec)
            with self.assertRaises(ValueError):Modulation(['A','B'],spec)
        model,graph,times=fixture();mod=Modulation(['B','A'],specification())
        with self.assertRaises(ValueError):Level0(model,graph,times,modulation=mod)

if __name__=='__main__':unittest.main()
