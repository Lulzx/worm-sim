import copy
import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from test_level0 import fixture
from test_modulation import specification as modulation_spec
from test_dark_edges import spec as dark_spec
from modulation import Modulation
from dark_edges import DarkEdges
from plasticity import Plasticity
from level0 import Level0, parameters, response
from solvers import Adaptive, integrate


def specification(mode='both'):
    return {'schema_version':1,'source':'synthetic plasticity test; no biological assignments',
            'types':[{'id':'synthetic','mode':mode,'utilization':.3,
                      'tau_depression':.4,'tau_facilitation':.2,'rate_scale':3.}],
            'edges':[{'pre':'A','post':'B','type':'synthetic'}]}


class PlasticityTests(unittest.TestCase):
    def test_constant_drive_matches_analytic_depression_and_facilitation(self):
        _,graph,_=fixture();times=jnp.asarray([0.,.1,.5,1.])
        for mode in ('depression','facilitation'):
            module=Plasticity(graph,specification(mode));p=module.parameters()
            def rhs(t,state,p):return module.derivative(state,jnp.asarray([.7,.1]),p)
            solve=eqx.filter_jit(lambda p:integrate(rhs,module.initial_state(p),p,times,Adaptive(method='tsit5',rtol=1e-10,atol=1e-12)).ys)
            actual=np.asarray(solve(p));rate=2.1;u=.3
            if mode=='depression':
                equilibrium=1/(1+.4*u*rate)
                x=equilibrium+(1-equilibrium)*np.exp(-(1/.4+u*rate)*np.asarray(times))
                expected=np.stack((x,np.full(4,u)),axis=1)
                self.assertLess(actual[-1,0]*actual[-1,1],u)
            else:
                equilibrium=(u/.2+u*rate)/(1/.2+u*rate)
                utilization=equilibrium+(u-equilibrium)*np.exp(-(1/.2+u*rate)*np.asarray(times))
                expected=np.stack((np.ones(4),utilization),axis=1)
                self.assertGreater(actual[-1,0]*actual[-1,1],u)
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-8)

    def test_zero_drive_recovery_and_inward_boundary_derivatives(self):
        _,graph,_=fixture();module=Plasticity(graph,specification());p=module.parameters()
        initial=jnp.asarray([.2,.8]);times=jnp.asarray([0.,.2,1.])
        def rhs(t,state,p):return module.derivative(state,jnp.zeros(2),p)
        actual=eqx.filter_jit(lambda p:integrate(rhs,initial,p,times,Adaptive(method='tsit5',rtol=1e-10,atol=1e-12)).ys)(p)
        np.testing.assert_allclose(actual,np.stack((1-.8*np.exp(-np.asarray(times)/.4),.3+.5*np.exp(-np.asarray(times)/.2)),axis=1),atol=1e-9,rtol=1e-8)
        for rate in [0.,1.]:
            for state in [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]:
                d=np.asarray(module.derivative(jnp.asarray(state),jnp.full(2,rate),p))
                for v,s in zip(d,state):self.assertGreaterEqual(v if s==0 else -v,0.)

    def test_source_type_sharing_matches_expanded_edges(self):
        _,graph,_=fixture();graph=copy.deepcopy(graph)
        graph['neurons'].extend([{'id':'C'},{'id':'D'}])
        graph['chemical'].extend([{'pre':'A','post':'C'},{'pre':'D','post':'B'}])
        spec=specification();spec['edges'] += [{'pre':'A','post':'C','type':'synthetic'},{'pre':'D','post':'B','type':'synthetic'}]
        module=Plasticity(graph,spec);self.assertEqual(len(module.slots),2)
        p=module.parameters();state=jnp.asarray([.6,.8,.4,.5]);release=jnp.asarray([.2,.3,.4,.7])
        # Independent per-edge equations: A's two outgoing edges must agree;
        # D shares parameters, but has its own resource/utilization state.
        factors=np.asarray(module.multiplier(state));np.testing.assert_allclose(factors,[.24,.24,.4])
        derivative=np.asarray(module.derivative(state,release,p))
        for edge,slot in [(0,0),(1,0),(2,1)]:
            x,u=float(state[slot]),float(state[2+slot]);rate=3*float(release[0 if edge<2 else 3])
            np.testing.assert_allclose(derivative[[slot,2+slot]],[(1-x)/.4-u*x*rate,(.3-u)/.2+.3*(1-u)*rate])
        spec['types'].append({**spec['types'][0],'id':'different'})
        spec['edges'][1]['type']='different'
        self.assertEqual(len(Plasticity(graph,spec).slots),3)

    def test_composed_extensions_and_gradients_through_initial_state(self):
        model,graph,_=fixture();model['config']['preparation_seconds']=.2
        dark=DarkEdges(graph,dark_spec());spec=specification()
        spec['edges'].append({'pre':'B','post':'A','type':'synthetic'})
        plasticity=Plasticity(graph,spec,dark);mod=Modulation(['A','B'],modulation_spec(.4))
        engine=Level0(model,graph,[0.,.1,.2,.3],adaptive=Adaptive(method='tsit5',rtol=1e-10,atol=1e-12),modulation=mod,dark_edges=dark,plasticity=plasticity)
        p=parameters(model,modulation=mod,dark_edges=dark,plasticity=plasticity)
        def loss(p):return jnp.sum(engine.response(p,jnp.asarray(0))**2)
        evaluate=eqx.filter_jit(loss);_,gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss))(p)
        for i in range(4):
            plus,minus=copy.deepcopy(p),copy.deepcopy(p)
            plus['plasticity']['raw']=p['plasticity']['raw'].at[0,i].add(1e-4)
            minus['plasticity']['raw']=p['plasticity']['raw'].at[0,i].add(-1e-4)
            actual=gradient['plasticity']['raw'][0,i]
            self.assertGreater(abs(float(actual)),1e-9)
            np.testing.assert_allclose(actual,(evaluate(plus)-evaluate(minus))/2e-4,atol=1e-9,rtol=2e-4)
        self.assertEqual(engine.initial_state(p).shape,(11,))
        # Independently verify both anatomical and extra chemical multipliers.
        plain=Level0(model,graph,[0.,.1],modulation=mod,dark_edges=dark)
        state=engine.initial_state(p);changed=state.at[engine.plasticity_start:].set(jnp.asarray([.5,.8,.4,.6]))
        observed=engine.rhs_current(changed,p,jnp.asarray(0),0.)[:2]-plain.rhs_current(state[:7],p,jnp.asarray(0),0.)[:2]
        raw=p['groups'][engine.mapping];v,s=state[:2],state[4:6]
        multiplier=plasticity.multiplier(changed[7:]);synapse=mod.multipliers(state[6:7],p['modulation'])[2]
        chemical=(jax.nn.softplus(raw[12])+1e-9)*2*s[0]*(2*jax.nn.sigmoid(raw[13])-1-v[1])*synapse[1]
        extra=dark.current(v,s,p['dark_edges'],synapse)
        expected=extra*(multiplier[1]-1)
        expected=expected.at[1].add(chemical*(multiplier[0]-1))
        np.testing.assert_allclose(observed,expected/(jax.nn.softplus(raw[:2])+1e-9),atol=1e-14)

    def test_fixed_step_converges_to_implicit_solution(self):
        model,graph,times=fixture();plasticity=Plasticity(graph,specification())
        p=parameters(model,plasticity=plasticity)
        implicit=Level0(model,graph,times,adaptive=Adaptive(rtol=1e-10,atol=1e-12),plasticity=plasticity)
        expected=np.asarray(eqx.filter_jit(lambda p:implicit.solve(p,jnp.asarray(0)).ys)(p))
        errors=[]
        for dt in [.002,.001]:
            fixed=copy.deepcopy(model);fixed['config']['dt']=dt
            engine=Level0(fixed,graph,times,plasticity=plasticity)
            actual=np.asarray(eqx.filter_jit(lambda p:engine.solve(p,jnp.asarray(0)).ys)(p))
            errors.append(np.max(np.abs(actual-expected)))
            self.assertTrue(np.all((actual[:,6:]>=0)&(actual[:,6:]<=1)))
        self.assertLess(errors[1],.55*errors[0])
        self.assertLess(errors[1],.001)

    def test_empty_mode_and_invalid_declarations(self):
        model,graph,times=fixture();spec=specification();spec['edges']=[];spec['types']=[]
        empty=Plasticity(graph,spec)
        np.testing.assert_allclose(response(Level0(model,graph,times,plasticity=empty),parameters(model,plasticity=empty),jnp.asarray(0)),response(Level0(model,graph,times),parameters(model),jnp.asarray(0)),atol=1e-14)
        for mutate in [lambda s:s.update(source=''),lambda s:s['types'][0].update(utilization=0.),lambda s:s['types'][0].update(tau_depression=0.),lambda s:s['types'][0].update(mode='unknown'),lambda s:s['edges'][0].update(pre='B',post='A'),lambda s:s['edges'][0].update(type='missing'),lambda s:s['edges'].append(copy.deepcopy(s['edges'][0]))]:
            invalid=specification();mutate(invalid)
            with self.assertRaises(ValueError):Plasticity(graph,invalid)
        dark=DarkEdges(graph,dark_spec());module=Plasticity(graph,specification(),dark)
        with self.assertRaises(ValueError):Level0(model,graph,times,plasticity=module)

if __name__=='__main__':unittest.main()
