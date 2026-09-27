"""Explicit sparse additions to the chemical mask, with physical-strength L1 cost."""
import math
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import equinox as eqx
from modulation import fields,positive_raw


class DarkEdges(eqx.Module):
    pre: jax.Array
    post: jax.Array
    group: jax.Array
    initial_strength: jax.Array
    initial_sign: jax.Array
    names: tuple = eqx.field(static=True)
    anatomy: tuple = eqx.field(static=True)
    groups: tuple = eqx.field(static=True)
    source: str = eqx.field(static=True)
    max_edges: int = eqx.field(static=True)
    l1_strength: float = eqx.field(static=True)

    def __init__(self, graph, spec):
        fields(spec,['schema_version','source','max_edges','l1_strength','edges'])
        if spec['schema_version']!=1 or not isinstance(spec['source'],str) or not spec['source'].strip():
            raise ValueError('dark edges require version 1 and explicit provenance')
        self.source=spec['source'];self.names=tuple(sorted(n['id'] for n in graph['neurons']))
        if not self.names or len(set(self.names))!=len(self.names):
            raise ValueError('invalid neuron identities')
        self.anatomy=tuple(sorted((e['pre'],e['post']) for e in graph['chemical']))
        self.max_edges=spec['max_edges'];self.l1_strength=float(spec['l1_strength'])
        if type(self.max_edges) is not int or not 0<=self.max_edges<=len(self.names)*(len(self.names)-1):
            raise ValueError('invalid declared dark-edge budget')
        if not math.isfinite(self.l1_strength) or self.l1_strength<0:
            raise ValueError('L1 strength must be finite and nonnegative')
        rows=spec['edges']
        if len(rows)>self.max_edges:raise ValueError('dark-edge budget exceeded')
        index={n:i for i,n in enumerate(self.names)};seen=set();groups={}
        for row in rows:
            fields(row,['pre','post','group','strength','reversal'])
            pair=(row['pre'],row['post'])
            if any(n not in index for n in pair) or pair[0]==pair[1] or pair in self.anatomy or pair in seen:
                raise ValueError('dark edge is unknown, self, duplicate or already anatomical')
            seen.add(pair)
            name=row['group'];strength=row['strength'];reversal=row['reversal']
            raw=positive_raw(strength)
            if not isinstance(name,str) or not name or not math.isfinite(reversal) or not -1<reversal<1:
                raise ValueError('invalid dark-edge group or reversal')
            values=(raw,math.log1p(reversal)-math.log1p(-reversal))
            if name in groups and groups[name]!=values:
                raise ValueError('conflicting tied dark-edge initialization')
            groups[name]=values
        self.groups=tuple(sorted(groups))
        self.pre=jnp.asarray([index[r['pre']] for r in rows],dtype=jnp.int32)
        self.post=jnp.asarray([index[r['post']] for r in rows],dtype=jnp.int32)
        self.group=jnp.asarray([self.groups.index(r['group']) for r in rows],dtype=jnp.int32)
        self.initial_strength=jnp.asarray([groups[k][0] for k in self.groups],dtype=float)
        self.initial_sign=jnp.asarray([groups[k][1] for k in self.groups],dtype=float)

    def parameters(self):
        return {'raw_strength':self.initial_strength,'raw_sign':self.initial_sign}

    def strengths(self, params):
        return (jax.nn.softplus(params['raw_strength'])+1e-9)[self.group]

    def current(self, voltage, gate, params, receptor_multiplier=None):
        weight=self.strengths(params)
        if receptor_multiplier is not None:
            weight=weight*receptor_multiplier[self.post]
        reversal=(2*jax.nn.sigmoid(params['raw_sign'])-1)[self.group]
        current=weight*gate[self.pre]*(reversal-voltage[self.post])
        return jnp.zeros(len(self.names)).at[self.post].add(current)

    def penalty(self, params):
        # Sum over actual edges, including multiplicity of a tied group.
        return self.l1_strength*jnp.sum(self.strengths(params))
