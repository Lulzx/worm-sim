"""Small independent forward and reverse-mode checks; no experimental data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from level0 import Level0, parameters, response
from replay_level0_atlas import Replay


def fixture():
    graph={'neurons':[{'id':'A'},{'id':'B'}],
           'chemical':[{'pre':'A','post':'B','synapse_count':2.}],
           'gaps':[{'a':'A','b':'B','size':1.}]}
    raw=[-1.,-.8, 0.1,-.1, -.2,.2, 1.,1.2, -.2,-.1, .3,.4, -2.,.8,-3.,-2.]
    model={'parameters':{'groups':[{'value':v} for v in raw], 'raw_to_group':list(range(len(raw)))},
           'initial':[.1,-.1,.4,.3,.2,.3], 'kernel_raw':[-.2,.1],
           'config':{'dt':.005,'preparation_seconds':.04,'observation_gain':{'initial_gain':2.}},
           'observation_log_gain':float(np.log(2.))}
    return model,graph,[0.,.05,.1]


class Checks(unittest.TestCase):
    def test_forward_numpy_and_batched_targets(self):
        model,graph,times=fixture()
        engine=Level0(model,graph,times)
        theta=parameters(model)
        batch=eqx.filter_jit(lambda p:jax.vmap(lambda target:engine.response(p,target))(jnp.arange(2)))(theta)
        for i,name in enumerate(['A','B']):
            expected=Replay(model,graph).response(name,times)
            np.testing.assert_allclose(batch[i],expected,atol=1e-13,rtol=1e-12)
        np.testing.assert_array_equal(np.asarray(batch)[:,0],0.)

    def test_reverse_gradient_and_optax_step(self):
        model,graph,times=fixture()
        engine=Level0(model,graph,times)
        theta=parameters(model)
        def loss(p):
            result=engine.response(p,jnp.asarray(0))
            # A newly composed readout loss needs no hand-derived adjoint.
            return jnp.mean((result-.03)**2)+.02*jnp.mean(result**4)
        value_gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))
        value,gradient=value_gradient(theta)
        evaluate=eqx.filter_jit(loss)
        for key,indices in [('groups',[0,2,4,6,8,10,12,13,14,15]),('kernel',[0,1]),('log_gain',[None])]:
            for i in indices:
                plus,minus=dict(theta),dict(theta)
                if i is None:
                    plus[key]=theta[key]+1e-5
                    minus[key]=theta[key]-1e-5
                    actual=gradient[key]
                else:
                    plus[key]=theta[key].at[i].add(1e-5)
                    minus[key]=theta[key].at[i].add(-1e-5)
                    actual=gradient[key][i]
                finite=(evaluate(plus)-evaluate(minus))/2e-5
                np.testing.assert_allclose(actual,finite,atol=1e-9,rtol=2e-5)
        optimizer=optax.adam(1e-3)
        updates,_=optimizer.update(gradient,optimizer.init(theta),theta)
        updated=optax.apply_updates(theta,updates)
        self.assertLess(float(evaluate(updated)),float(value))

    def test_invalid_grid(self):
        model,graph,_=fixture()
        for times in [[0.],[.1,.2],[0.,0.],[0.,float('nan')]]:
            with self.assertRaises(ValueError):
                Level0(model,graph,times)


if __name__=='__main__':
    unittest.main()
