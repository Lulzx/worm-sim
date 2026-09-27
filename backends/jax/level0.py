"""JAX/Equinox Level 0 dynamics with Diffrax integration and reverse AD.

The compatibility solver uses the frozen Rust Euler time grid. No Rust/NumPy
RHS, gradients, optimizers or saved prediction values are called by this module.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import equinox as eqx
import diffrax
import numpy as np
from solvers import Adaptive, integrate
from typing import NamedTuple
from modulation import Modulation
from rectification import GapRectification
from dark_edges import DarkEdges


class AdaptiveSolution(NamedTuple):
    ys: jax.Array
    stats: dict


class Level0(eqx.Module):
    mapping: jax.Array
    pre: jax.Array
    post: jax.Array
    counts: jax.Array
    ga: jax.Array
    gb: jax.Array
    sizes: jax.Array
    initial: jax.Array
    steps: jax.Array
    save_times: jax.Array
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    adaptive: Adaptive | None = eqx.field(static=True)
    preparation: float = eqx.field(static=True)
    modulation: Modulation | None
    rectification: GapRectification | None
    dark_edges: DarkEdges | None

    def __init__(self, model, graph, times, adaptive=None, modulation=None, rectification=None, dark_edges=None):
        if adaptive is not None and not isinstance(adaptive,Adaptive):
            raise TypeError('adaptive settings must be an Adaptive instance')
        self.adaptive=adaptive
        self.modulation=modulation
        self.rectification=rectification
        self.dark_edges=dark_edges
        names = sorted(n['id'] for n in graph['neurons'])
        index = {name:i for i,name in enumerate(names)}
        self.n = len(names)
        if modulation is not None and (not isinstance(modulation,Modulation) or modulation.names!=tuple(names)):
            raise ValueError('modulation neuron order differs from the model')
        chemical = sorted(graph['chemical'], key=lambda e:(e['pre'],e['post']))
        gaps = sorted(graph['gaps'], key=lambda e:(e['a'],e['b']))
        if dark_edges is not None and (not isinstance(dark_edges,DarkEdges) or dark_edges.names!=tuple(names) or dark_edges.anatomy!=tuple((e['pre'],e['post']) for e in chemical)):
            raise ValueError('dark-edge anatomy differs from model')
        if rectification is not None and (not isinstance(rectification,GapRectification) or rectification.pairs!=tuple((e['a'],e['b']) for e in gaps)):
            raise ValueError('rectification topology differs from model')
        self.m = len(chemical)
        self.mapping = jnp.asarray(model['parameters']['raw_to_group'], dtype=jnp.int32)
        if len(self.mapping) != 6*self.n+2*self.m+len(gaps)+1:
            raise ValueError('parameter layout does not match graph')
        self.pre = jnp.asarray([index[e['pre']] for e in chemical], dtype=jnp.int32)
        self.post = jnp.asarray([index[e['post']] for e in chemical], dtype=jnp.int32)
        self.counts = jnp.asarray([e['synapse_count'] for e in chemical])
        self.ga = jnp.asarray([index[e['a']] for e in gaps], dtype=jnp.int32)
        self.gb = jnp.asarray([index[e['b']] for e in gaps], dtype=jnp.int32)
        self.sizes = jnp.asarray([e['size'] for e in gaps])
        self.initial = jnp.asarray(model['initial'])
        if self.initial.shape != (3*self.n,):
            raise ValueError('initial state size mismatch')
        if modulation is not None:
            self.initial=jnp.concatenate((self.initial,modulation.initial))
        times = np.asarray(times, dtype=float)
        dt = model['config']['dt']
        prep = model['config'].get('preparation_seconds', 0.)
        if (len(times)<2 or times[0]!=0 or not np.isfinite(times).all()
                or np.any(np.diff(times)<=0) or not np.isfinite(dt) or dt<=0
                or not np.isfinite(prep) or prep<0):
            raise ValueError('invalid time grid')
        self.preparation=prep
        self.save_times = jnp.asarray(prep+times)
        if adaptive is not None:
            self.steps=jnp.asarray([0.,float(prep+times[-1])])
            self.max_steps=adaptive.max_steps
            return
        # Reproduce the reference's floating-point step accumulation, including
        # final fractional steps, instead of silently changing the integrator.
        steps = [0.]
        for end in prep+times:
            if (end-steps[-1])/dt > 1e6:
                raise ValueError('interval exceeds reference step limit')
            while steps[-1] < end:
                next_time = min(steps[-1]+dt, float(end))
                if next_time <= steps[-1]:
                    raise ValueError('time step cannot advance')
                steps.append(next_time)
        self.steps = jnp.asarray(steps)
        self.max_steps = len(steps)-1

    def rhs(self, t, state, args):
        theta,target=args
        interval=jnp.searchsorted(self.save_times,t,side='right')-1
        kernel=jax.nn.softplus(theta['kernel'])
        current=jnp.where((interval>=0)&(interval<len(kernel)),kernel[jnp.clip(interval,0,len(kernel)-1)],0.)
        return self.rhs_current(state,theta,target,current)

    def rhs_current(self, state, theta, target, current):
        raw = theta['groups'][self.mapping]
        positive = jax.nn.softplus(raw)+1e-9
        n,m = self.n,self.m
        v,c,s = state[:n],state[n:2*n],state[2*n:3*n]
        if self.modulation is None:
            release = jax.nn.sigmoid((v-raw[2*n:3*n])*positive[3*n:4*n])
            dv = (-(v-raw[n:2*n])).at[target].add(current)
        else:
            gain,leak,synapse=self.modulation.multipliers(state[3*n:],theta['modulation'])
            release=jax.nn.sigmoid((v-raw[2*n:3*n])*positive[3*n:4*n]*gain)
            dv=(-(v-raw[n:2*n])*leak).at[target].add(current)
        weight = positive[6*n:6*n+m]*self.counts
        if self.modulation is not None:
            weight=weight*synapse[self.post]
        reversal = 2*jax.nn.sigmoid(raw[6*n+m:6*n+2*m])-1
        dv = dv.at[self.post].add(weight*s[self.pre]*(reversal-v[self.post]))
        if self.dark_edges is not None:
            dv=dv+self.dark_edges.current(v,s,theta['dark_edges'],None if self.modulation is None else synapse)
        delta=v[self.gb]-v[self.ga]
        gap = positive[6*n+2*m:-1]*self.sizes*delta
        if self.rectification is not None:
            gap=gap*self.rectification.multiplier(delta,theta['rectification'])
        dv = dv.at[self.ga].add(gap).at[self.gb].add(-gap)
        fast=jnp.concatenate((dv/positive[:n], (release-c)/positive[4*n:5*n],
                                (release*(1-s)-s)/positive[-1]))
        if self.modulation is None:
            return fast
        return jnp.concatenate((fast,self.modulation.derivative(state[3*n:],release,theta['modulation'])))

    def extension_penalty(self, theta):
        """Add once to a training objective, independently of batch/trace count."""
        return jnp.asarray(0.) if self.dark_edges is None else self.dark_edges.penalty(theta['dark_edges'])

    def solve(self, theta, target):
        if self.adaptive is not None:
            # Implicit RK stages may evaluate beyond an interval's end.
            # Keep the input constant within each solve so those stages cannot
            # sample the next stimulus; carry state and gradients across solves.
            kernel=jax.nn.softplus(theta['kernel'])
            indices=jnp.arange(len(self.save_times)-1)
            currents=jnp.where(indices<len(kernel),kernel[jnp.clip(indices,0,len(kernel)-1)],0.)
            if self.preparation>0:
                boundaries=jnp.concatenate((jnp.zeros(1),self.save_times))
                currents=jnp.concatenate((jnp.zeros(1),currents))
            else:
                boundaries=self.save_times
            def rhs(t,state,args):
                p,cell,current=args
                return self.rhs_current(state,p,cell,current)
            def advance(state,interval):
                start,end,current=interval
                solution=integrate(rhs,state,(theta,target,current),jnp.asarray([end]),self.adaptive,t0=start)
                state=solution.ys[0]
                counts=jnp.stack([solution.stats[k] for k in ['num_steps','num_accepted_steps','num_rejected_steps']])
                return state,(state,counts)
            _,(states,counts)=jax.lax.scan(advance,self.initial,(boundaries[:-1],boundaries[1:],currents))
            if self.preparation==0:
                states=jnp.concatenate((self.initial[None,:],states),axis=0)
            totals=jnp.sum(counts,axis=0)
            return AdaptiveSolution(states,dict(zip(['num_steps','num_accepted_steps','num_rejected_steps'],totals)))
        return diffrax.diffeqsolve(
            diffrax.ODETerm(self.rhs), diffrax.Euler(),
            t0=self.steps[0], t1=self.steps[-1], dt0=None,
            y0=self.initial, args=(theta,target),
            stepsize_controller=diffrax.StepTo(ts=self.steps),
            saveat=diffrax.SaveAt(ts=self.save_times),
            adjoint=diffrax.RecursiveCheckpointAdjoint(), max_steps=self.max_steps)
    def response(self, theta, target):
        solution=self.solve(theta,target)
        calcium = solution.ys[:,self.n:2*self.n]
        raw = theta['groups'][self.mapping]
        scale = jax.nn.softplus(raw[5*self.n:6*self.n])+1e-9
        return jnp.exp(theta['log_gain'])*scale*(calcium-calcium[0])


def parameters(model, modulation=None, rectification=None, dark_edges=None):
    out = {'groups':jnp.asarray([g['value'] for g in model['parameters']['groups']]),
            'kernel':jnp.asarray(model['kernel_raw']),
            'log_gain':jnp.asarray(model.get('observation_log_gain') or 0.)}
    if modulation is not None:
        out['modulation']=modulation.parameters()
    if rectification is not None:
        out['rectification']=rectification.parameters()
    if dark_edges is not None:
        out['dark_edges']=dark_edges.parameters()
    return out


response = eqx.filter_jit(lambda engine,theta,target: engine.response(theta,target))
