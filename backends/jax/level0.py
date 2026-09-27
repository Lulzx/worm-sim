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

    def __init__(self, model, graph, times):
        names = sorted(n['id'] for n in graph['neurons'])
        index = {name:i for i,name in enumerate(names)}
        self.n = len(names)
        chemical = sorted(graph['chemical'], key=lambda e:(e['pre'],e['post']))
        gaps = sorted(graph['gaps'], key=lambda e:(e['a'],e['b']))
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
        times = np.asarray(times, dtype=float)
        dt = model['config']['dt']
        prep = model['config'].get('preparation_seconds', 0.)
        if (len(times)<2 or times[0]!=0 or not np.isfinite(times).all()
                or np.any(np.diff(times)<=0) or not np.isfinite(dt) or dt<=0
                or not np.isfinite(prep) or prep<0):
            raise ValueError('invalid time grid')
        self.save_times = jnp.asarray(prep+times)
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
        theta, target = args
        raw = theta['groups'][self.mapping]
        positive = jax.nn.softplus(raw)+1e-9
        n,m = self.n,self.m
        v,c,s = state[:n],state[n:2*n],state[2*n:]
        release = jax.nn.sigmoid((v-raw[2*n:3*n])*positive[3*n:4*n])
        interval = jnp.searchsorted(self.save_times, t, side='right')-1
        kernel = jax.nn.softplus(theta['kernel'])
        current = jnp.where((interval>=0)&(interval<len(kernel)),kernel[jnp.clip(interval,0,len(kernel)-1)],0.)
        dv = (-(v-raw[n:2*n])).at[target].add(current)
        weight = positive[6*n:6*n+m]*self.counts
        reversal = 2*jax.nn.sigmoid(raw[6*n+m:6*n+2*m])-1
        dv = dv.at[self.post].add(weight*s[self.pre]*(reversal-v[self.post]))
        gap = positive[6*n+2*m:-1]*self.sizes*(v[self.gb]-v[self.ga])
        dv = dv.at[self.ga].add(gap).at[self.gb].add(-gap)
        return jnp.concatenate((dv/positive[:n], (release-c)/positive[4*n:5*n],
                                (release*(1-s)-s)/positive[-1]))

    def response(self, theta, target):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.rhs), diffrax.Euler(),
            t0=self.steps[0], t1=self.steps[-1], dt0=None,
            y0=self.initial, args=(theta,target),
            stepsize_controller=diffrax.StepTo(ts=self.steps),
            saveat=diffrax.SaveAt(ts=self.save_times),
            adjoint=diffrax.RecursiveCheckpointAdjoint(), max_steps=self.max_steps)
        calcium = solution.ys[:,self.n:2*self.n]
        raw = theta['groups'][self.mapping]
        scale = jax.nn.softplus(raw[5*self.n:6*self.n])+1e-9
        return jnp.exp(theta['log_gain'])*scale*(calcium-calcium[0])


def parameters(model):
    return {'groups':jnp.asarray([g['value'] for g in model['parameters']['groups']]),
            'kernel':jnp.asarray(model['kernel_raw']),
            'log_gain':jnp.asarray(model.get('observation_log_gain') or 0.)}


response = eqx.filter_jit(lambda engine,theta,target: engine.response(theta,target))
