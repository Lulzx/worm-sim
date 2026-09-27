"""Optional conservative gap rectification on explicitly selected anatomical pairs."""
import math
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import equinox as eqx
from modulation import fields


class GapRectification(eqx.Module):
    edge: jax.Array
    group: jax.Array
    scale: jax.Array
    initial_asymmetry: jax.Array
    pairs: tuple = eqx.field(static=True)
    groups: tuple = eqx.field(static=True)
    source: str = eqx.field(static=True)

    def __init__(self, graph, spec):
        fields(spec,['schema_version','source','edges'])
        if spec['schema_version']!=1 or not isinstance(spec['source'],str) or not spec['source'].strip():
            raise ValueError('rectification requires version 1 and explicit provenance')
        self.source=spec['source']
        self.pairs=tuple(sorted((e['a'],e['b']) for e in graph['gaps']))
        names={n['id'] for n in graph['neurons']}
        if any(a not in names or b not in names for a,b in self.pairs):
            raise ValueError('gap endpoint is absent from neuron identities')
        if len(set(self.pairs))!=len(self.pairs) or any(a>=b for a,b in self.pairs):
            raise ValueError('gap pairs must be unique and canonically oriented a < b')
        rows=spec['edges'];groups={};seen=set();indices=[]
        for row in rows:
            fields(row,['a','b','group','asymmetry','voltage_scale'])
            pair=(row['a'],row['b'])
            if pair not in self.pairs or pair in seen:
                raise ValueError('rectification pair is duplicate, reversed or absent from anatomy')
            seen.add(pair);indices.append(self.pairs.index(pair))
            name=row['group'];value=row['asymmetry'];scale=row['voltage_scale']
            if not isinstance(name,str) or not name or not math.isfinite(value) or not math.isfinite(scale) or scale<=1e-9:
                raise ValueError('invalid rectification group, asymmetry or voltage scale')
            if name in groups and groups[name]!=(value,scale):
                raise ValueError('conflicting initial values for tied rectification group')
            groups[name]=(value,scale)
        self.groups=tuple(sorted(groups))
        self.edge=jnp.asarray(indices,dtype=jnp.int32)
        self.group=jnp.asarray([self.groups.index(r['group']) for r in rows],dtype=jnp.int32)
        self.initial_asymmetry=jnp.asarray([groups[k][0] for k in self.groups],dtype=float)
        self.scale=jnp.asarray([groups[k][1] for k in self.groups],dtype=float)

    def parameters(self):
        return {'asymmetry':self.initial_asymmetry}

    def multiplier(self, delta_voltage, params):
        rho=jnp.tanh(params['asymmetry'][self.group])
        factor=1.+rho*jnp.tanh(delta_voltage[self.edge]/self.scale[self.group])
        return jnp.ones(len(self.pairs)).at[self.edge].set(factor)
