"""Sparse, compartmental peptide/monoamine layer from specification section 5.

Maps and initial values must be supplied explicitly with provenance. No receptor
or release map is inferred from neuron names or missing expression measurements.
"""
import math
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import equinox as eqx


def positive_raw(value):
    if not math.isfinite(value) or value<=1e-9:
        raise ValueError('positive physical values must exceed 1e-9')
    x=value-1e-9
    return x+math.log(-math.expm1(-x))


def fields(record, required, optional=()):
    if not isinstance(record,dict) or not set(required)<=record.keys() or not record.keys()<=set(required)|set(optional):
        raise ValueError('missing or unknown modulation configuration field')


class Modulation(eqx.Module):
    sender: jax.Array
    release_channel: jax.Array
    release_group: jax.Array
    receiver_channel: jax.Array
    receiver_group: jax.Array
    effect_index: jax.Array
    initial: jax.Array
    bath: jax.Array
    initial_tau: jax.Array
    initial_release: jax.Array
    initial_kd: jax.Array
    initial_beta: jax.Array
    names: tuple = eqx.field(static=True)
    channels: tuple = eqx.field(static=True)
    release_groups: tuple = eqx.field(static=True)
    receptor_groups: tuple = eqx.field(static=True)
    source: str = eqx.field(static=True)
    max_log_effect: float = eqx.field(static=True)

    def __init__(self, names, spec):
        fields(spec,['schema_version','source','channels','release','receptors'],['max_log_effect'])
        for c in spec['channels']:
            fields(c,['id','species','compartment','tau_seconds'],['initial','bath'])
        for row in spec['release']:
            fields(row,['neuron','channel','group','alpha'])
        for row in spec['receptors']:
            fields(row,['neuron','channel','effect','group','kd','beta'])
        if spec.get('schema_version')!=1 or not isinstance(spec.get('source'),str) or not spec['source'].strip():
            raise ValueError('modulation needs schema version 1 and explicit source provenance')
        if not names or len(set(names))!=len(names):
            raise ValueError('neuron names must be unique and nonempty')
        self.names=tuple(names);self.source=spec['source']
        self.max_log_effect=float(spec.get('max_log_effect',3.))
        if not math.isfinite(self.max_log_effect) or not 0<self.max_log_effect<=30:
            raise ValueError('max_log_effect must lie in (0,30]')
        channels=spec['channels']
        self.channels=tuple(c['id'] for c in channels)
        if not channels or len(set(self.channels))!=len(channels) or any(not isinstance(c,str) or not c for c in self.channels):
            raise ValueError('channel identifiers must be unique and nonempty')
        for c in channels:
            if not all(isinstance(c.get(k),str) and c[k] for k in ['species','compartment']):
                raise ValueError('each channel must declare species and compartment')
        if len({(c['species'],c['compartment']) for c in channels})!=len(channels):
            raise ValueError('duplicate species/compartment channel')
        self.initial_tau=jnp.asarray([positive_raw(c['tau_seconds']) for c in channels])
        self.initial=jnp.asarray([c.get('initial',0.) for c in channels],dtype=float)
        self.bath=jnp.asarray([c.get('bath',0.) for c in channels],dtype=float)
        if not all(np.isfinite(np.asarray(x)).all() and np.all(np.asarray(x)>=0) for x in [self.initial,self.bath]):
            raise ValueError('initial concentrations and bath drives must be finite and nonnegative')
        ni={name:i for i,name in enumerate(names)};ci={name:i for i,name in enumerate(self.channels)}
        def ties(records,fields):
            groups={}
            for r in records:
                group=r['group']
                if not isinstance(group,str) or not group:
                    raise ValueError('each sparse map row needs a named parameter group')
                values=tuple(float(r[k]) for k in fields)
                if not all(math.isfinite(v) for v in values):
                    raise ValueError('nonfinite modulation parameter')
                if group in groups and groups[group]!=values:
                    raise ValueError('tied modulation parameters have conflicting initial values')
                groups[group]=values
            order=tuple(sorted(groups))
            return order,[groups[k] for k in order],[order.index(r['group']) for r in records]
        release=spec['release'];receptors=spec['receptors']
        if len({(r['neuron'],r['channel']) for r in release})!=len(release):
            raise ValueError('duplicate release map row')
        if len({(r['neuron'],r['channel'],r['effect'],r['group']) for r in receptors})!=len(receptors):
            raise ValueError('duplicate receptor map row')
        try:
            self.sender=jnp.asarray([ni[r['neuron']] for r in release],dtype=jnp.int32)
            self.release_channel=jnp.asarray([ci[r['channel']] for r in release],dtype=jnp.int32)
            self.receiver_channel=jnp.asarray([ci[r['channel']] for r in receptors],dtype=jnp.int32)
            effects={'gain':0,'leak':1,'synapse':2}
            self.effect_index=jnp.asarray([effects[r['effect']]*len(names)+ni[r['neuron']] for r in receptors],dtype=jnp.int32)
        except KeyError as e:
            raise ValueError(f'unknown modulation neuron, channel or effect: {e}') from e
        self.release_groups,values,mapping=ties(release,['alpha'])
        self.initial_release=jnp.asarray([positive_raw(v[0]) for v in values],dtype=float)
        self.release_group=jnp.asarray(mapping,dtype=jnp.int32)
        self.receptor_groups,values,mapping=ties(receptors,['kd','beta'])
        self.initial_kd=jnp.asarray([positive_raw(v[0]) for v in values],dtype=float)
        self.initial_beta=jnp.asarray([v[1] for v in values],dtype=float)
        self.receiver_group=jnp.asarray(mapping,dtype=jnp.int32)

    def parameters(self):
        return {'raw_tau':self.initial_tau,'raw_release':self.initial_release,
                'raw_kd':self.initial_kd,'sensitivity':self.initial_beta}

    def drive(self, release, params):
        alpha=(jax.nn.softplus(params['raw_release'])+1e-9)[self.release_group]
        return jnp.zeros(len(self.channels)).at[self.release_channel].add(alpha*release[self.sender])

    def derivative(self, concentration, release, params):
        return (self.drive(release,params)+self.bath-concentration)/(jax.nn.softplus(params['raw_tau'])+1e-9)

    def advance_mean_drive(self, concentration, mean_drive, duration, params):
        exponent=-duration/(jax.nn.softplus(params['raw_tau'])+1e-9)
        return jnp.exp(exponent)*concentration-jnp.expm1(exponent)*(mean_drive+self.bath)

    def multipliers(self, concentration, params):
        # Solvers can undershoot zero slightly; receptor activation has no
        # negative-concentration interpretation. The ODE itself is unmodified.
        c=jnp.maximum(concentration[self.receiver_channel],0.)
        kd=(jax.nn.softplus(params['raw_kd'])+1e-9)[self.receiver_group]
        occupancy=c/(kd+c)
        changes=params['sensitivity'][self.receiver_group]*occupancy
        log_effect=jnp.zeros(3*len(self.names)).at[self.effect_index].add(changes)
        limit=self.max_log_effect
        return jnp.exp(limit*jnp.tanh(log_effect/limit)).reshape((3,len(self.names)))
