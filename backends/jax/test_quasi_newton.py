import unittest
import jax.numpy as jnp
import numpy as np
from quasi_newton import optimize_active


class QuasiNewtonTests(unittest.TestCase):
    def test_quadratic_and_frozen_coordinates(self):
        accepted=[];evaluations=[]
        theta={'x':jnp.array([5.,7.])};active={'x':jnp.array([True,False])}
        def objective(p):
            x=p['x'];loss=(x[0]-2.)**2+(x[1]-3.)**2
            return loss,{'x':2*(x-jnp.array([2.,3.]))},{'mse':float(loss)}
        result=optimize_active(theta,active,objective,
            lambda i,n,p,m:accepted.append((i,np.asarray(p['x']))),
            lambda n,p,v,g,m,f:evaluations.append(np.asarray(p['x'])))
        self.assertTrue(result['optimizer_success'])
        np.testing.assert_allclose(result['final_theta']['x'],[2.,7.],atol=1e-10)
        self.assertTrue(all(row[1]==7. for row in evaluations))
        self.assertEqual(result['active_coordinates'],1)
        self.assertEqual(result['options']['maxcor'],20)
        self.assertEqual([r[0] for r in accepted],list(range(len(accepted))))

    def test_hard_budget_and_nonfinite_trial_keep_last_accepted(self):
        for budget,nonfinite,status in [(1,False,'evaluation_budget_exhausted'),(10,True,'nonfinite_evaluation')]:
            accepted=[];evaluated=[]
            def objective(p):
                x=float(p['x'][0]);v=float('nan') if nonfinite and x!=5 else (x-2)**2
                return v,{'x':jnp.array([2*(x-2)])},{'mse':v}
            result=optimize_active({'x':jnp.array([5.])},{'x':jnp.array([True])},objective,
                lambda i,n,p,m:accepted.append(i),lambda n,p,v,g,m,f:evaluated.append(f),max_evaluations=budget)
            self.assertEqual(result['status'],status)
            self.assertEqual(accepted,[0])
            self.assertEqual(float(result['final_theta']['x'][0]),5.)
            self.assertLessEqual(result['evaluations'],budget)
            if nonfinite:self.assertFalse(evaluated[-1])

    def test_real_objective_adapter(self):
        import jax
        from test_overfit import OverfitTests
        from overfit import prepare
        from objective import build, evaluate
        model,graph,training=OverfitTests().fixture()
        model,training,config=prepare(model,training,['A'],3,.001,-.2)
        theta,active,groups,data,prior=build(model,graph,training,config)
        initial=float(evaluate(theta,groups,data,prior)[0]);accepted=[]
        result=optimize_active(theta,active,lambda p:evaluate(p,groups,data,prior),
            lambda i,n,p,m:accepted.append(m['mse']),lambda *args:None,max_evaluations=5,max_iterations=2)
        self.assertLessEqual(result['final_metrics']['mse'],initial)
        self.assertGreater(len(accepted),1)
        for before,after,mask in zip(jax.tree.leaves(theta),jax.tree.leaves(result['final_theta']),jax.tree.leaves(active),strict=True):
            np.testing.assert_array_equal(np.asarray(before)[~np.asarray(mask)],np.asarray(after)[~np.asarray(mask)])

    def test_scaled_chain_rule(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        def objective(p):
            x=p['x']; loss=jnp.sum(x*x)
            return loss, {'x':2*x}, {'mse':float(loss)}
        def inspect(fun,x0,**kwargs):
            x=np.array([.3,-.2]); value,gradient=fun(x)
            h=1e-5
            fd=np.array([(fun(x+h*np.eye(2)[i])[0]-fun(x-h*np.eye(2)[i])[0])/(2*h) for i in range(2)])
            np.testing.assert_allclose(gradient,fd,rtol=1e-7,atol=1e-8)
            return SimpleNamespace(x=x0,success=True,status=0,message='test')
        with patch('quasi_newton.minimize',side_effect=inspect):
            optimize_active({'x':jnp.array([5.,7.])},{'x':jnp.array([True,True])},objective,
                lambda *args:None,lambda *args:None,coordinate_scale={'x':jnp.array([.13,.16])})

    def test_scaled_quadratic_and_invalid_scales(self):
        theta={'x':jnp.array([5.,7.,9.])}; active={'x':jnp.array([True,True,False])}
        seen=[]
        def objective(p):
            x=p['x']; weights=jnp.array([100.,1.,3.])
            loss=jnp.sum(weights*(x-2.)**2)
            return loss, {'x':2*weights*(x-2.)}, {'mse':float(loss)}
        result=optimize_active(theta,active,objective,lambda *args:None,
            lambda n,p,*args:seen.append(np.asarray(p['x'])),
            coordinate_scale={'x':jnp.array([.1,1.,4.])},max_evaluations=30)
        self.assertTrue(result['optimizer_success'])
        np.testing.assert_array_equal(seen[0],[5.,7.,9.])
        np.testing.assert_allclose(result['final_theta']['x'],[2.,2.,9.],atol=1e-8)
        self.assertTrue(all(x[2]==9. for x in seen))
        for scale in [jnp.array([0.,1.,1.]),jnp.array([float('nan'),1.,1.]),jnp.ones(2)]:
            with self.assertRaisesRegex(ValueError,'scale'):
                optimize_active(theta,active,objective,lambda *args:None,lambda *args:None,
                                coordinate_scale={'x':scale})

    def test_mask_validation(self):
        with self.assertRaises(ValueError):
            optimize_active({'x':jnp.ones(2)},{'x':jnp.ones(2)},None,None,None)

    def test_curvature_history_option_and_validation(self):
        def objective(p):
            loss=jnp.sum((p['x']-2.)**2)
            return loss,{'x':2*(p['x']-2.)},{'mse':float(loss)}
        theta={'x':jnp.array([5.,7.])};active={'x':jnp.array([True,True])}
        result=optimize_active(theta,active,objective,lambda *args:None,lambda *args:None,
            max_evaluations=10,max_corrections=100)
        self.assertTrue(result['optimizer_success'])
        self.assertEqual(result['options']['maxcor'],100)
        np.testing.assert_allclose(result['final_theta']['x'],[2.,2.],atol=1e-10)
        for invalid in [0,-1,True,1.5]:
            with self.assertRaisesRegex(ValueError,'curvature history'):
                optimize_active(theta,active,objective,lambda *args:None,lambda *args:None,
                    max_corrections=invalid)


if __name__=='__main__':unittest.main()
