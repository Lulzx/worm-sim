"""Rate-adapted Tsodyks-Markram plasticity, shared by source and edge type."""
import math
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import equinox as eqx
from modulation import fields, positive_raw


class Plasticity(eqx.Module):
    names: tuple = eqx.field(static=True)
    pairs: tuple = eqx.field(static=True)
    types: tuple = eqx.field(static=True)
    source: str = eqx.field(static=True)
    slots: tuple = eqx.field(static=True)
    edge: jax.Array
    edge_slot: jax.Array
    pre: jax.Array
    kind: jax.Array
    depression: jax.Array
    facilitation: jax.Array
    initial_raw: jax.Array

    def __init__(self, graph, spec, dark_edges=None):
        fields(spec, ['schema_version', 'source', 'types', 'edges'])
        if spec['schema_version'] != 1 or not isinstance(spec['source'], str) or not spec['source'].strip():
            raise ValueError('plasticity requires version 1 and explicit provenance')
        self.source = spec['source']
        self.names = tuple(sorted(n['id'] for n in graph['neurons']))
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('invalid neuron identities')
        chemical = tuple(sorted((e['pre'], e['post']) for e in graph['chemical']))
        extra = () if dark_edges is None else tuple(
            (self.names[int(a)], self.names[int(b)])
            for a, b in zip(np.asarray(dark_edges.pre), np.asarray(dark_edges.post)))
        if dark_edges is not None and (dark_edges.names != self.names or dark_edges.anatomy != chemical):
            raise ValueError('dark-edge anatomy differs from plasticity graph')
        self.pairs = chemical + extra
        if len(set(self.pairs)) != len(self.pairs) or any(a not in self.names or b not in self.names for a, b in self.pairs):
            raise ValueError('invalid chemical topology')
        kinds = {}
        for row in spec['types']:
            fields(row, ['id', 'mode', 'utilization', 'tau_depression', 'tau_facilitation', 'rate_scale'])
            name, mode, u = row['id'], row['mode'], row['utilization']
            if not isinstance(name, str) or not name.strip() or name in kinds:
                raise ValueError('invalid or duplicate plasticity type')
            if mode not in ('depression', 'facilitation', 'both') or not math.isfinite(u) or not 0 < u < 1:
                raise ValueError('invalid plasticity mode or utilization')
            raw = [math.log(u) - math.log1p(-u)] + [positive_raw(row[k]) for k in
                ('tau_depression', 'tau_facilitation', 'rate_scale')]
            kinds[name] = (mode, raw)
        self.types = tuple(sorted(kinds))
        rows = spec['edges']; seen = set(); selected = []
        for row in rows:
            fields(row, ['pre', 'post', 'type'])
            pair = (row['pre'], row['post'])
            if pair not in self.pairs or pair in seen or row['type'] not in kinds:
                raise ValueError('plasticity edge is absent, duplicate or has unknown type')
            seen.add(pair); selected.append((pair[0], row['type']))
        if {row['type'] for row in rows} != set(kinds):
            raise ValueError('plasticity types must be used by at least one edge')
        self.slots = tuple(sorted(set(selected)))
        self.edge = jnp.asarray([self.pairs.index((r['pre'], r['post'])) for r in rows], dtype=jnp.int32)
        self.edge_slot = jnp.asarray([self.slots.index(s) for s in selected], dtype=jnp.int32)
        self.pre = jnp.asarray([self.names.index(s[0]) for s in self.slots], dtype=jnp.int32)
        self.kind = jnp.asarray([self.types.index(s[1]) for s in self.slots], dtype=jnp.int32)
        self.depression = jnp.asarray([kinds[s[1]][0] != 'facilitation' for s in self.slots])
        self.facilitation = jnp.asarray([kinds[s[1]][0] != 'depression' for s in self.slots])
        self.initial_raw = jnp.asarray([kinds[k][1] for k in self.types], dtype=float).reshape((-1, 4))

    def parameters(self):
        return {'raw': self.initial_raw}

    def physical(self, params):
        raw = params['raw'][self.kind]
        return jax.nn.sigmoid(raw[:, 0]), jax.nn.softplus(raw[:, 1:]) + 1e-9

    def initial_state(self, params):
        u, _ = self.physical(params)
        return jnp.concatenate((jnp.ones(len(self.slots)), u))

    def derivative(self, state, release, params):
        count = len(self.slots)
        x, u = state[:count], state[count:]
        baseline, positive = self.physical(params)
        td, tf, scale = positive[:, 0], positive[:, 1], positive[:, 2]
        rate = scale * release[self.pre]
        dx = jnp.where(self.depression, (1-x)/td-u*x*rate, 0.)
        du = jnp.where(self.facilitation, (baseline-u)/tf+baseline*(1-u)*rate, 0.)
        return jnp.concatenate((dx, du))

    def multiplier(self, state):
        count = len(self.slots)
        efficacy = state[:count] * state[count:]
        return jnp.ones(len(self.pairs)).at[self.edge].set(efficacy[self.edge_slot])
