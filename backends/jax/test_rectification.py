import copy
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from test_level0 import fixture
from level0 import Level0,parameters,response
from rectification import GapRectification


def specification(beta):
    return {'schema_version':1,'source':'synthetic mechanism check, not an innexin assignment',
            'edges':[{'a':'A','b':'B','group':'example','asymmetry':beta,'voltage_scale':.2}]}


class RectificationTests(unittest.TestCase):
    def test_current_conservation_and_dissipation(self):
        model,graph,times=fixture();graph['chemical']=[]
        model['parameters']['groups']=model['parameters']['groups'][:12]+model['parameters']['groups'][14:]
        model['parameters']['raw_to_group']=list(range(14))
        rect=GapRectification(graph,specification(1.2))
        engine=Level0(model,graph,times,rectification=rect);p=parameters(model,rectification=rect)
        tau=jax.nn.softplus(p['groups'][:2])+1e-9
        magnitude=[]
        for delta in [-.5,0.,.5]:
            state=jnp.asarray([0.,delta,0.,0.,0.,0.])
            derivative=engine.rhs_current(state,p,jnp.asarray(0),0.)
            currents=derivative[:2]*tau+(state[:2]-p['groups'][2:4])
            np.testing.assert_allclose(jnp.sum(currents),0.,atol=1e-14)
            self.assertGreaterEqual(float(currents[0]*delta),-1e-14)
            magnitude.append(abs(float(currents[0])))
        self.assertGreater(magnitude[2],magnitude[0])

    def test_zero_asymmetry_parity_and_automatic_gradients(self):
        model,graph,times=fixture()
        rect=GapRectification(graph,specification(0.))
        engine=Level0(model,graph,times,rectification=rect)
        p=parameters(model,rectification=rect)
        expected=response(Level0(model,graph,times),parameters(model),jnp.asarray(0))
        np.testing.assert_allclose(response(engine,p,jnp.asarray(0)),expected,atol=1e-14,rtol=1e-13)
        p['rectification']['asymmetry']=jnp.asarray([.4])
        def loss(p):return jnp.sum(engine.response(p,jnp.asarray(0))**2)
        evaluate=eqx.filter_jit(loss);_,gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))(p)
        a,b=copy.deepcopy(p),copy.deepcopy(p)
        a['rectification']['asymmetry']+=1e-4;b['rectification']['asymmetry']-=1e-4
        finite=(evaluate(a)-evaluate(b))/2e-4
        actual=gradient['rectification']['asymmetry'][0]
        self.assertGreater(abs(float(actual)),1e-9)
        np.testing.assert_allclose(actual,finite,atol=1e-10,rtol=1e-5)

    def test_sparse_selection_tying_and_invalid_maps(self):
        _,graph,_=fixture()
        extended=copy.deepcopy(graph);extended['neurons'].append({'id':'C'});extended['gaps'].append({'a':'B','b':'C','size':2.})
        spec=specification(.7)
        rect=GapRectification(extended,spec)
        factor=rect.multiplier(jnp.ones(2),rect.parameters())
        self.assertEqual(float(factor[1]),1.)
        spec['edges'].append({**spec['edges'][0],'a':'B','b':'C'})
        tied=GapRectification(extended,spec)
        self.assertEqual(tied.parameters()['asymmetry'].shape,(1,))
        for mutate in [lambda s:s.update(source=''),lambda s:s['edges'][0].update(a='B',b='A'),lambda s:s['edges'][0].update(voltage_scale=0.),lambda s:s['edges'][0].update(asymmetry=float('nan')),lambda s:s['edges'].append(copy.deepcopy(s['edges'][0]))]:
            invalid=specification(.2);mutate(invalid)
            with self.assertRaises(ValueError):GapRectification(graph,invalid)

if __name__=='__main__':unittest.main()
