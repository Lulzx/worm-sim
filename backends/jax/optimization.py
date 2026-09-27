"""Declared learning-rate multipliers; Optax retains all optimizer state/math."""
import math
import jax
import jax.numpy as jnp
from modulation import fields


def rate_multipliers(model, theta, configuration):
    scales=jax.tree.map(jnp.ones_like,theta)
    options={} if configuration is None else configuration.get('optimization',{})
    fields(options,[],['learning_rate_multipliers'])
    declared=options.get('learning_rate_multipliers',{})
    fields(declared,[],['base_types','base_groups','parameters'])
    selectors={key:declared.get(key,{}) for key in ['base_types','base_groups','parameters']}
    for values in selectors.values():
        if not isinstance(values,dict) or any(not isinstance(k,str) or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for k,v in values.items()):
            raise ValueError('learning-rate multipliers must be named finite nonnegative numbers')
    # Exact tied-group overrides take precedence over their type defaults.
    if selectors['base_types'] or selectors['base_groups']:
        names=[g.get('name') for g in model['parameters']['groups']]
        if any(not isinstance(name,str) or not name for name in names) or len(set(names))!=len(names):
            raise ValueError('group-rate selectors require unique named base groups')
        kinds=[name.split('/',1)[0] for name in names]
        if not set(selectors['base_types'])<=set(kinds) or not set(selectors['base_groups'])<=set(names):
            raise ValueError('unknown base type or tied group in learning-rate configuration')
        scales['groups']=jnp.asarray([selectors['base_groups'].get(name,selectors['base_types'].get(kind,1.)) for name,kind in zip(names,kinds)],dtype=float)
    leaves={}
    for family,values in theta.items():
        if family=='groups':continue
        if isinstance(values,dict):
            for key in values:leaves[family+'.'+key]=(family,key)
        else:leaves[family]=(family,None)
    for name,rate in selectors['parameters'].items():
        if name not in leaves:
            raise ValueError(f'unknown parameter family in learning-rate configuration: {name}')
        family,key=leaves[name]
        if key is None:scales[family]=jnp.full_like(theta[family],rate)
        else:scales[family][key]=jnp.full_like(theta[family][key],rate)
    return scales
