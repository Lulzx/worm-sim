import copy
import json
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from test_objective import example
from test_modulation import specification as modulation_spec
from test_dark_edges import spec as dark_spec
from test_plasticity import specification as plasticity_spec
from extensions import initialize, pack, restore
from objective import build, evaluate
from fit import fit, checkpoint
from level0 import response
from predict_extensions import predict


def configuration():
    return {'schema_version':1,'solver':None,'extensions':{
        'modulation':modulation_spec(.2), 'dark_edges':dark_spec(),
        'plasticity':plasticity_spec('depression'),
        'rectification':{'schema_version':1,'source':'synthetic test','edges':[
            {'a':'A','b':'B','group':'gap','asymmetry':.1,'voltage_scale':.3}]}}}


class ExtensionTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_fitted_extension_parameters(self):
        model,graph,training=example();config=configuration();times=training['groups'][0]['recording']['times']
        engine,p,active=initialize(model,graph,times,config)
        p=jax.tree.map(lambda v,a:jnp.where(a,v+.01,v),p,active)
        saved=pack(checkpoint(model,p,1,'test'),p,config)
        new,q,new_active=restore(json.loads(json.dumps(saved)),graph,times)
        for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(q),strict=True):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(response(engine,p,jnp.asarray(0)),response(new,q,jnp.asarray(0)),atol=1e-14)
        self.assertEqual(jax.tree.structure(active),jax.tree.structure(new_active))
        for mutation in [lambda s:s['extension_parameters'].pop('modulation'),lambda s:s['extension_parameters']['dark_edges']['raw_strength'].update(shape=[2]),lambda s:s['extension_parameters']['dark_edges']['raw_strength'].update(values=[float('nan')]),lambda s:s['configuration'].update(unknown=True),lambda s:s.update(schema_version=2),lambda s:s['extension_parameters']['plasticity']['raw']['values'].__setitem__(2,99.)]:
            invalid=copy.deepcopy(saved);mutation(invalid)
            with self.assertRaises(ValueError):restore(invalid,graph,times)
        config={'schema_version':1,'solver':None,'extensions':{'plasticity':{'schema_version':1,'source':'empty test','types':[],'edges':[]}}}
        _,p,_=initialize(model,graph,times,config)
        restore(json.loads(json.dumps(pack(model,p,config))),graph,times)

    def test_l1_is_added_once_and_unused_parameters_are_frozen_during_fit(self):
        model,graph,training=example();config=configuration()
        p,active,groups,data,prior=build(model,graph,training,config)
        self.assertFalse(bool(active['plasticity']['raw'][0,2]))
        no_penalty=copy.deepcopy(config);no_penalty['extensions']['dark_edges']['l1_strength']=0.
        q,_,other_groups,other_data,other_prior=build(model,graph,training,no_penalty)
        value,grad,_=evaluate(p,groups,data,prior)
        other,other_grad,_=evaluate(q,other_groups,other_data,other_prior)
        self.assertAlmostEqual(float(value-other),.006,places=12)
        self.assertAlmostEqual(float(grad['dark_edges']['raw_strength'][0]-other_grad['dark_edges']['raw_strength'][0]),.03*float(jax.nn.sigmoid(p['dark_edges']['raw_strength'][0])),places=12)
        twice=copy.deepcopy(training);twice['groups']*=2;twice['classification_pairs']*=2
        r,_,g,d,pr=build(model,graph,twice,config)
        self.assertAlmostEqual(float(evaluate(r,g,d,pr)[0]),float(value),places=12)
        model['epoch']=0;model['config'].update(epochs=2,learning_rate=.001,optimizer={'kind':'adamw','weight_decay':.1})
        seen=[]
        def score(candidate):
            seen.append(copy.deepcopy(candidate));restore(candidate,graph,[0.,.05,.1])
            return [2.,1.,1.][candidate['base_model']['epoch']]
        selected,reports=fit(model,graph,training,'test',score,configuration=config)
        self.assertEqual(selected['base_model']['epoch'],1)
        self.assertNotEqual(seen[0]['extension_parameters']['dark_edges'],seen[-1]['extension_parameters']['dark_edges'])
        self.assertEqual(seen[0]['extension_parameters']['plasticity']['raw']['values'][2],seen[-1]['extension_parameters']['plasticity']['raw']['values'][2])

    def test_plan_prediction_requires_authoritative_topology_and_lineage(self):
        model,graph,training=example();model.update(epoch=0,source_commit='test',selection_trials=['validation'])
        config=configuration();times=[0.,.05,.1]
        _,p,_=initialize(model,graph,times,config);saved=pack(model,p,config)
        plan={k:model[k] for k in ['graph_hash','dataset_hash','split_hash','training_trials','selection_trials','sample_dt']}
        plan.update(schema_version=1,partition='validation',names=['A','B'],chemical_topology=[[0,1,2.]],gap_topology=[[0,1,1.]],trials=[{'id':'validation','times':times,'target':1,'neurons':['A','B'],'response_neurons':['A']}])
        result=predict(saved,graph,plan,'a'*64)
        self.assertEqual(result['model'],'jax-atlas:'+'a'*64)
        self.assertEqual(result['trials'][0]['response_scores']['A'],.05*sum(abs(x) for x in result['trials'][0]['fluorescence']['A']))
        for mutate in [lambda s:s.update(training_trials=['test']),lambda s:s['chemical_topology'][0].__setitem__(2,99.),lambda s:s['trials'].append(copy.deepcopy(s['trials'][0]))]:
            invalid=copy.deepcopy(plan);mutate(invalid)
            with self.assertRaises(ValueError):predict(saved,graph,invalid,'a'*64)

if __name__=='__main__':unittest.main()
