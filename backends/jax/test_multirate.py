import copy
import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from test_level0 import fixture
from test_objective import example
from test_modulation import specification
from test_plasticity import specification as plasticity_spec
from modulation import Modulation
from plasticity import Plasticity
from level0 import Level0, parameters
from solvers import Adaptive
from multirate import Multirate
from extensions import initialize, pack, restore


class MultirateTests(unittest.TestCase):
    def test_constant_bath_exact_and_zero_feedback_fast_parity(self):
        model,graph,_=fixture();model['config']['preparation_seconds']=.07
        times=[0.,.05,.19];spec=specification();spec['release']=[]
        spec['channels'][0]['bath']=.4
        mod=Modulation(['A','B'],spec);p=parameters(model,mod)
        adaptive=Adaptive(method='tsit5',rtol=1e-10,atol=1e-12)
        coupled=Level0(model,graph,times,adaptive=adaptive,modulation=mod)
        split=Level0(model,graph,times,adaptive=adaptive,modulation=mod,multirate=Multirate(.04))
        solve=eqx.filter_jit(lambda engine:engine.solve(p,jnp.asarray(0)))
        actual=solve(split);reference=solve(coupled)
        np.testing.assert_allclose(actual.ys[:,:6],reference.ys[:,:6],atol=1e-9,rtol=1e-8)
        np.testing.assert_allclose(actual.ys[:,6],.4*(1-np.exp(-(.07+np.asarray(times))/2)),atol=1e-14,rtol=1e-13)
        for end in .07+np.asarray(times):self.assertIn(end,np.asarray(split.slow_boundaries))
        self.assertLessEqual(float(np.diff(split.slow_boundaries).max()),.04+1e-14)
        self.assertEqual(int(actual.stats['slow_windows']),len(split.slow_boundaries)-1)

    def test_refinement_converges_to_fully_coupled_dynamics(self):
        model,graph,_=fixture();model['config']['preparation_seconds']=.2
        spec=specification(1.2);spec['channels'][0]['tau_seconds']=.4
        mod=Modulation(['A','B'],spec);p=parameters(model,mod)
        adaptive=Adaptive(method='tsit5',rtol=1e-10,atol=1e-12)
        times=[0.,.2,.4,.6];solve=eqx.filter_jit(lambda engine:engine.solve(p,jnp.asarray(0)).ys)
        reference=np.asarray(solve(Level0(model,graph,times,adaptive=adaptive,modulation=mod)))
        errors=[]
        for step in [.1,.05,.025]:
            split=Level0(model,graph,times,adaptive=adaptive,modulation=mod,multirate=Multirate(step))
            values=np.asarray(solve(split));errors.append(float(np.max(np.abs(values-reference))))
            self.assertTrue(np.all(values[:,6:]>=0))
        self.assertGreater(errors[0],1e-5)
        self.assertLess(errors[1],.8*errors[0]);self.assertLess(errors[2],.8*errors[1])
        self.assertLess(errors[-1],.02)

    def test_reverse_gradients_include_accumulation_coarse_update_and_preparation(self):
        model,graph,_=fixture();model['config']['preparation_seconds']=.07
        mod=Modulation(['A','B'],specification(.7))
        plasticity=Plasticity(graph,plasticity_spec())
        engine=Level0(model,graph,[0.,.1,.2],adaptive=Adaptive(method='tsit5',rtol=1e-10,atol=1e-12),modulation=mod,plasticity=plasticity,multirate=Multirate(.04))
        p=parameters(model,modulation=mod,plasticity=plasticity)
        def loss(p):return jnp.sum(engine.response(p,jnp.asarray(0))**2)
        value=eqx.filter_jit(loss);_,gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))(p)
        for family,key,index in [('modulation','raw_tau',0),('modulation','raw_release',0),('modulation','raw_kd',0),('modulation','sensitivity',0),('plasticity','raw',(0,0)),(None,'kernel',0)]:
            a,b=copy.deepcopy(p),copy.deepcopy(p)
            ap,bp,g=(a,b,gradient) if family is None else (a[family],b[family],gradient[family])
            ap[key]=ap[key].at[index].add(1e-4);bp[key]=bp[key].at[index].add(-1e-4)
            actual=g[key][index]
            self.assertGreater(abs(float(actual)),1e-10)
            np.testing.assert_allclose(actual,(value(a)-value(b))/2e-4,atol=1e-9,rtol=2e-4)

    def test_checkpoint_settings_and_invalid_configuration(self):
        model,graph,training=example();times=training['groups'][0]['recording']['times']
        config={'schema_version':1,'solver':{'method':'kvaerno5','rtol':1e-9,'atol':1e-11},'extensions':{'modulation':specification(.2)},'multirate':{'slow_dt':.03,'max_windows':50}}
        engine,p,_=initialize(model,graph,times,config)
        new,q,_=restore(pack(model,p,config),graph,times)
        solve=eqx.filter_jit(lambda engine,p:engine.solve(p,jnp.asarray(0)).ys)
        np.testing.assert_array_equal(solve(engine,p),solve(new,q))
        # A zero-preparation run must retain the initial response sample.
        model['config']['preparation_seconds']=0.
        zero,p,_=initialize(model,graph,times,config)
        np.testing.assert_array_equal(solve(zero,p)[0],zero.initial_state(p))
        for args in [{'slow_dt':0.},{'slow_dt':float('nan')},{'slow_dt':.1,'max_windows':0}]:
            with self.assertRaises(ValueError):Multirate(**args)
        bad=copy.deepcopy(config);bad['multirate']['max_windows']=1
        with self.assertRaises(ValueError):initialize(model,graph,times,bad)
        for key in ['solver','extensions']:
            bad=copy.deepcopy(config);bad[key]=None if key=='solver' else {}
            with self.assertRaises(ValueError):initialize(model,graph,times,bad)

if __name__=='__main__':unittest.main()
