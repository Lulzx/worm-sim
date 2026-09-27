import copy
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from test_objective import example
from test_extensions import configuration
from extensions import initialize,restore
from optimization import rate_multipliers
from fit import fit,make_optimizer
from objective import build,evaluate


def named_example():
    model,graph,training=example()
    for i,g in enumerate(model['parameters']['groups']):g['name']=('tau/' if i<2 else 'other/')+str(i)
    model['epoch']=0;model['config'].update(epochs=2,learning_rate=.001,optimizer={'kind':'adamw','weight_decay':.1})
    return model,graph,training


class OptimizationTests(unittest.TestCase):
    def test_exact_adamw_coordinate_rates_and_cosine_schedule(self):
        model,graph,training=named_example();config=configuration()
        config['optimization']={'learning_rate_multipliers':{'base_types':{'tau':.2},'base_groups':{'tau/0':.7},'parameters':{'kernel':.5,'modulation.raw_tau':2.}}}
        _,p,active=initialize(model,graph,[0.,.05,.1],config)
        rates=rate_multipliers(model,p,config)
        np.testing.assert_array_equal(rates['groups'][:2],[.7,.2])
        self.assertEqual(float(rates['modulation']['raw_tau'][0]),2.)
        # Compare different rates against separate library AdamW instances over
        # several updates, including decoupled decay and the cosine endpoint.
        opt_config={'epochs':3,'learning_rate':.01,'optimizer':{'kind':'adamw','weight_decay':.2},'learning_rate_schedule':{'kind':'cosine','minimum_fraction':.2}}
        p={'a':jnp.asarray(.7),'b':jnp.asarray(-.4)};scales={'a':jnp.asarray(.25),'b':jnp.asarray(2.)}
        shared=make_optimizer(opt_config,{'a':jnp.asarray(True),'b':jnp.asarray(True)},scales);state=shared.init(p)
        separate={k:optax.adamw(optax.cosine_decay_schedule(.01*float(scales[k]),2,alpha=.2),weight_decay=.2) for k in p}
        states={k:opt.init(p[k]) for k,opt in separate.items()};expected=dict(p)
        for g in [{'a':.1,'b':-.3},{'a':-.2,'b':.5},{'a':.3,'b':.1}]:
            grads=jax.tree.map(jnp.asarray,g);updates,state=shared.update(grads,state,p)
            updates=jax.tree.map(lambda u,s:u*s,updates,scales);p=optax.apply_updates(p,updates)
            for k,opt in separate.items():
                update,states[k]=opt.update(grads[k],states[k],expected[k]);expected[k]=optax.apply_updates(expected[k],update)
                self.assertAlmostEqual(float(p[k]),float(expected[k]),places=14)

    def test_zero_rate_preserves_objective_and_freezes_updates_and_decay(self):
        model,graph,training=named_example();config=configuration()
        config['optimization']={'learning_rate_multipliers':{'base_groups':{'tau/0':0.},'parameters':{'modulation.raw_tau':0.,'kernel':0.}}}
        p,active,g,d,pr=build(model,graph,training,config)
        loss,gradient,_=evaluate(p,g,d,pr)
        q,_,h,e,prior=build(model,graph,training,configuration())
        self.assertAlmostEqual(float(loss),float(evaluate(q,h,e,prior)[0]),places=13)
        self.assertEqual(float(gradient['groups'][0]),0.)
        np.testing.assert_array_equal(gradient['kernel'],0.)
        np.testing.assert_array_equal(gradient['modulation']['raw_tau'],0.)
        self.assertFalse(bool(active['groups'][0]))
        seen=[]
        def score(candidate):
            seen.append(candidate);restore(candidate,graph,[0.,.05,.1]);return 1.
        selected,_=fit(model,graph,training,'test',score,configuration=config)
        self.assertEqual(selected['base_model']['epoch'],0)
        for candidate in seen:
            self.assertEqual(candidate['base_model']['parameters']['groups'][0]['value'],model['parameters']['groups'][0]['value'])
            self.assertEqual(candidate['base_model']['kernel_raw'],model['kernel_raw'])
            self.assertEqual(candidate['extension_parameters']['modulation']['raw_tau'],seen[0]['extension_parameters']['modulation']['raw_tau'])
        self.assertNotEqual(seen[0]['extension_parameters']['dark_edges'],seen[-1]['extension_parameters']['dark_edges'])

    def test_invalid_selectors_and_effective_decay_fail(self):
        model,graph,_=named_example()
        for selectors in [{'base_types':{'missing':.1}},{'base_groups':{'missing':.1}},{'parameters':{'missing':.1}},{'parameters':{'kernel':-1.}},{'parameters':{'kernel':float('nan')}},{'parameters':{'kernel':True}},{'unknown':{}}]:
            config=configuration();config['optimization']={'learning_rate_multipliers':selectors}
            with self.assertRaises(ValueError):initialize(model,graph,[0.,.05,.1],config)
        with self.assertRaises(ValueError):make_optimizer({'epochs':2,'learning_rate':.1,'optimizer':{'kind':'adamw','weight_decay':1.}},{'x':jnp.asarray(True)},{'x':jnp.asarray(20.)})

if __name__=='__main__':unittest.main()
