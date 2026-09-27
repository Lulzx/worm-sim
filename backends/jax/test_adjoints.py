import copy
import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from solvers import Adaptive, integrate
from test_solvers import stiff_fixture
from test_objective import example
from test_modulation import specification
from test_plasticity import specification as plasticity_spec
from level0 import Level0, parameters
from extensions import initialize, pack, restore


class AdjointTests(unittest.TestCase):
    def test_analytic_decay_initial_state_and_parameter_gradients(self):
        times=jnp.asarray([0.,.1,.4,.8])
        def rhs(t,y,p):return -p*y
        for method in ['tsit5','kvaerno5']:
            for mode in ['checkpoint','continuous']:
                settings=Adaptive(method=method,rtol=1e-12,atol=1e-14,adjoint=mode,
                                  checkpoints=2 if mode=='checkpoint' else None)
                def loss(x):return jnp.sum(integrate(rhs,x[0],x[1],times,settings).ys**2)
                x=jnp.asarray([.7,.9]);value,gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))(x)
                decay=np.exp(-2*.9*np.asarray(times))
                expected=[2*.7*decay.sum(),-2*.7**2*np.sum(np.asarray(times)*decay)]
                np.testing.assert_allclose(gradient,expected,atol=1e-8,rtol=1e-7)
                self.assertAlmostEqual(float(value),.7**2*decay.sum(),places=8)

    def test_preparation_and_discontinuous_currents_match_checkpoint_gradients(self):
        model,graph,times=stiff_fixture()
        model['parameters']['groups'][0]['value']=float(np.log(np.expm1(.02)))
        p=parameters(model);results=[]
        for mode in ['checkpoint','continuous']:
            engine=Level0(model,graph,times,adaptive=Adaptive(method='tsit5',rtol=1e-10,atol=1e-12,adjoint=mode))
            def loss(p):return jnp.sum(engine.response(p,jnp.asarray(0))**2)
            evaluate=eqx.filter_jit(loss)
            value,gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))(p)
            results.append((value,gradient))
            for key,i in [('groups',0),('kernel',0)]:
                plus,minus=copy.deepcopy(p),copy.deepcopy(p)
                plus[key]=plus[key].at[i].add(1e-4);minus[key]=minus[key].at[i].add(-1e-4)
                np.testing.assert_allclose(gradient[key][i],(evaluate(plus)-evaluate(minus))/2e-4,atol=2e-9,rtol=2e-4)
        np.testing.assert_allclose(results[0][0],results[1][0],atol=1e-14)
        for a,b in zip(jax.tree.leaves(results[0][1]),jax.tree.leaves(results[1][1]),strict=True):
            np.testing.assert_allclose(a,b,atol=2e-9,rtol=2e-4)

    def test_continuous_adjoint_composes_with_multirate_and_parameterized_initial_state(self):
        model,graph,training=example();times=training['groups'][0]['recording']['times']
        config={'schema_version':1,'solver':{'method':'tsit5','rtol':1e-10,'atol':1e-12,'adjoint':'continuous','adjoint_rtol':1e-10,'adjoint_atol':1e-12,'adjoint_max_steps':10000},'multirate':{'slow_dt':.03},'extensions':{'modulation':specification(.7),'plasticity':plasticity_spec()}}
        engine,p,_=initialize(model,graph,times,config)
        restored,q,_=restore(pack(model,p,config),graph,times)
        self.assertEqual(restored.adaptive,engine.adaptive)
        def loss(p):return jnp.sum(restored.response(p,jnp.asarray(0))**2)
        evaluate=eqx.filter_jit(loss);_,gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))(q)
        for family,key,i in [('modulation','raw_tau',0),('modulation','raw_release',0),('modulation','sensitivity',0),('plasticity','raw',(0,0))]:
            a,b=copy.deepcopy(p),copy.deepcopy(p)
            a[family][key]=a[family][key].at[i].add(1e-4);b[family][key]=b[family][key].at[i].add(-1e-4)
            actual=gradient[family][key][i]
            self.assertGreater(abs(float(actual)),1e-11)
            np.testing.assert_allclose(actual,(evaluate(a)-evaluate(b))/2e-4,atol=1e-9,rtol=3e-4)

    def test_backward_step_budget_failure_is_not_suppressed(self):
        settings=Adaptive(method='tsit5',rtol=1e-10,atol=1e-12,adjoint='continuous',adjoint_max_steps=1)
        def rhs(t,y,p):return -p*y
        def loss(p):return integrate(rhs,jnp.asarray(1.),p,jnp.asarray([1.]),settings).ys[0]
        gradient=eqx.filter_jit(eqx.filter_grad(loss))
        with self.assertRaisesRegex(Exception,'maximum number of solver steps'):
            gradient(jnp.asarray(.9)).block_until_ready()

    def test_invalid_or_ignored_adjoint_options_fail(self):
        for settings in [dict(adjoint='unknown'),dict(checkpoints=0),dict(checkpoints=True),dict(adjoint='continuous',checkpoints=2),dict(adjoint_atol=1e-9),dict(adjoint='continuous',adjoint_rtol=float('nan')),dict(adjoint='continuous',adjoint_max_steps=0)]:
            with self.assertRaises(ValueError):Adaptive(**settings)

if __name__=='__main__':unittest.main()
