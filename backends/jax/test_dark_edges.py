import copy
import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from test_level0 import fixture
from level0 import Level0,parameters,response
from dark_edges import DarkEdges


def spec():
    return {'schema_version':1,'source':'synthetic extra-connection test, not evidence',
            'max_edges':1,'l1_strength':.03,'edges':[{'pre':'B','post':'A','group':'extra','strength':.2,'reversal':.5}]}

class DarkEdgeTests(unittest.TestCase):
    def test_declared_edge_current_and_physical_l1_penalty(self):
        _,graph,_=fixture();before=copy.deepcopy(graph);module=DarkEdges(graph,spec());p=module.parameters()
        np.testing.assert_allclose(module.current(jnp.asarray([.1,.3]),jnp.asarray([.2,.4]),p),[.032,0.],atol=1e-14)
        self.assertAlmostEqual(float(module.penalty(p)),.006,places=14)
        self.assertEqual(graph,before)
        np.testing.assert_allclose(module.current(jnp.asarray([.1,.3]),jnp.asarray([.2,.4]),p,jnp.asarray([2.,3.])),[.064,0.],atol=1e-14)

    def test_empty_mode_is_baseline_and_coupled_gradients_match(self):
        model,graph,times=fixture();empty=spec();empty['edges']=[]
        absent=DarkEdges(graph,empty);base=Level0(model,graph,times)
        engine=Level0(model,graph,times,dark_edges=absent)
        np.testing.assert_allclose(response(engine,parameters(model,dark_edges=absent),jnp.asarray(0)),response(base,parameters(model),jnp.asarray(0)),atol=1e-14,rtol=1e-13)
        module=DarkEdges(graph,spec());engine=Level0(model,graph,times,dark_edges=module)
        p=parameters(model,dark_edges=module)
        def loss(p):return jnp.sum(engine.response(p,jnp.asarray(0))**2)+engine.extension_penalty(p)
        value=eqx.filter_jit(loss);_,gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))(p)
        optimizer=optax.adam(1e-3)
        updates,_=optimizer.update(gradient,optimizer.init(p),p)
        self.assertLess(float(value(optax.apply_updates(p,updates))),float(value(p)))
        for key in ['raw_strength','raw_sign']:
            a,b=copy.deepcopy(p),copy.deepcopy(p)
            a['dark_edges'][key]+=1e-4;b['dark_edges'][key]-=1e-4
            finite=(value(a)-value(b))/2e-4
            actual=gradient['dark_edges'][key][0]
            self.assertGreater(abs(float(actual)),1e-8)
            np.testing.assert_allclose(actual,finite,atol=1e-9,rtol=1e-5)

    def test_topology_binding_and_duplicate_rejection(self):
        model,graph,times=fixture();declared=spec()
        declared['max_edges']=2;declared['edges']*=2
        with self.assertRaises(ValueError):DarkEdges(graph,declared)
        module=DarkEdges(graph,spec());different=copy.deepcopy(graph)
        different['chemical']=[]
        with self.assertRaises(ValueError):Level0(model,different,times,dark_edges=module)
        tied=spec();tied['max_edges']=2
        extended=copy.deepcopy(graph);extended['neurons'].append({'id':'C'})
        tied['edges'].append({**tied['edges'][0],'pre':'C','strength':.3})
        with self.assertRaises(ValueError):DarkEdges(extended,tied)

    def test_tied_penalty_counts_edges_and_invalid_masks_fail(self):
        _,graph,_=fixture();extended=copy.deepcopy(graph);extended['neurons'].append({'id':'C'})
        declared=spec();declared['max_edges']=2
        declared['edges'].append({**declared['edges'][0],'pre':'C'})
        module=DarkEdges(extended,declared)
        self.assertEqual(module.parameters()['raw_strength'].shape,(1,))
        self.assertAlmostEqual(float(module.penalty(module.parameters())),.012,places=14)
        for mutate in [lambda s:s.update(max_edges=0),lambda s:s.update(source=''),lambda s:s.update(l1_strength=-1.),lambda s:s['edges'][0].update(pre='A',post='B'),lambda s:s['edges'][0].update(pre='A',post='A'),lambda s:s['edges'][0].update(pre='unknown'),lambda s:s['edges'][0].update(reversal=1.)]:
            invalid=spec();mutate(invalid)
            with self.assertRaises(ValueError):DarkEdges(graph,invalid)

if __name__=='__main__':unittest.main()
