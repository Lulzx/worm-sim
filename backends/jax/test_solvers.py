import unittest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from level0 import Level0, parameters
from solvers import Adaptive


def stiff_fixture():
    def inverse(x):return float(np.log(np.expm1(x)))
    raw=[inverse(.001),0.,0.,inverse(1.),inverse(1.),inverse(1.),inverse(.02)]
    graph={'neurons':[{'id':'A'}],'chemical':[],'gaps':[]}
    times=[0.,.007,.023,.025,.063]
    model={'parameters':{'groups':[{'value':v} for v in raw],'raw_to_group':list(range(7))},
           'initial':[.3,0.,0.],'kernel_raw':[inverse(1.),inverse(1.),inverse(.1),inverse(.1)],
           'config':{'dt':.01,'preparation_seconds':.017}}
    return model,graph,times


class SolverTests(unittest.TestCase):
    def test_stiff_membrane_and_forcing_jumps_match_analytic_solution(self):
        model,graph,times=stiff_fixture();theta=parameters(model)
        tau=float(jax.nn.softplus(theta['groups'][0]))+1e-9
        expected=[.3*np.exp(-.017/tau)]
        for a,b,current in zip(times,times[1:],[1.,1.,.1,.1]):
            expected.append(current+(expected[-1]-current)*np.exp(-(b-a)/tau))
        for method in ['tsit5','kvaerno5']:
            engine=Level0(model,graph,times,Adaptive(method=method,rtol=1e-9,atol=1e-11,dt0=.01))
            solution=eqx.filter_jit(lambda p:engine.solve(p,jnp.asarray(0)))(theta)
            np.testing.assert_allclose(np.asarray(solution.ys)[:,0],expected,atol=2e-8,rtol=2e-7)
            self.assertGreater(int(solution.stats['num_steps']),0)
            # The adaptive route does not allocate the reference's fixed grid.
            self.assertEqual(engine.steps.shape,(2,))

    def test_implicit_reverse_gradient_includes_preparation_and_input(self):
        model,graph,times=stiff_fixture()
        # Less fast membrane retains dependence on the unforced preparation.
        model['parameters']['groups'][0]['value']=float(np.log(np.expm1(.02)))
        theta=parameters(model)
        engine=Level0(model,graph,times,Adaptive(rtol=1e-10,atol=1e-12))
        def loss(p):return jnp.sum(engine.response(p,jnp.asarray(0))**2)
        value_gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))
        evaluate=eqx.filter_jit(loss)
        _,gradient=value_gradient(theta)
        for key,i in [('groups',0),('kernel',0)]:
            plus,minus=dict(theta),dict(theta)
            plus[key]=theta[key].at[i].add(1e-4)
            minus[key]=theta[key].at[i].add(-1e-4)
            finite=(evaluate(plus)-evaluate(minus))/2e-4
            self.assertGreater(abs(float(gradient[key][i])),1e-8)
            np.testing.assert_allclose(gradient[key][i],finite,atol=2e-9,rtol=2e-4)

    def test_invalid_configuration_and_exhausted_budget_fail(self):
        for settings in [dict(method='unknown'),dict(rtol=0.),dict(atol=float('nan')),dict(dt0=-1.),dict(dtmax=0.),dict(max_steps=0),dict(max_steps=True)]:
            with self.assertRaises(ValueError):Adaptive(**settings)
        model,graph,times=stiff_fixture()
        engine=Level0(model,graph,times,Adaptive(method='tsit5',max_steps=1))
        with self.assertRaisesRegex(Exception,'maximum number of solver steps'):
            # Diffrax must not return an apparently valid truncated trajectory.
            engine.response(parameters(model),jnp.asarray(0)).block_until_ready()

if __name__=='__main__':
    unittest.main()
