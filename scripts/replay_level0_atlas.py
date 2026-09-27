#!/usr/bin/env python3
"""Independent NumPy Euler replay of frozen Level 0 atlas response predictions.

Uses canonical graph JSON exported by `wormsim unpack`; does not import Rust or
Taichi dynamics. No fitted state or outcome value is used beyond the saved model.
"""
import numpy as np


class Replay:
    def __init__(self, model, graph):
        self.model = model
        self.names = sorted(n['id'] for n in graph['neurons'])
        self.index = {n:i for i,n in enumerate(self.names)}
        self.n = n = len(self.names)
        chemical = sorted(graph['chemical'], key=lambda e:(e['pre'],e['post']))
        gaps = sorted(graph['gaps'], key=lambda e:(e['a'],e['b']))
        m = len(chemical)
        tied = model['parameters']
        raw = np.asarray([tied['groups'][i]['value'] for i in tied['raw_to_group']])
        assert len(raw) == 6*n+2*m+len(gaps)+1
        positive = np.logaddexp(0,raw)+1e-9
        self.inv_tau = 1/positive[:n]
        self.rest = raw[n:2*n]
        self.threshold = raw[2*n:3*n]
        self.slope = positive[3*n:4*n]
        self.inv_calcium_tau = 1/positive[4*n:5*n]
        self.scale = positive[5*n:6*n]
        self.pre = np.array([self.index[e['pre']] for e in chemical], dtype=int)
        self.post = np.array([self.index[e['post']] for e in chemical], dtype=int)
        self.weight = positive[6*n:6*n+m]*[e['synapse_count'] for e in chemical]
        self.reversal = 2/(1+np.exp(-raw[6*n+m:6*n+2*m]))-1
        self.ga = np.array([self.index[e['a']] for e in gaps], dtype=int)
        self.gb = np.array([self.index[e['b']] for e in gaps], dtype=int)
        self.gap = positive[6*n+2*m:-1]*[e['size'] for e in gaps]
        self.inv_synapse_tau = 1/positive[-1]
        self.kernel = np.logaddexp(0,np.asarray(model['kernel_raw']))
        self.dt = model['config']['dt']
        self.preparation = model['config'].get('preparation_seconds',0.)
        assert self.dt > 0 and self.preparation >= 0
        self.seed = np.asarray(model['initial'])
        assert len(self.seed) == 3*n
        self.state, self.origin = self.advance(self.seed.copy(),0.,self.preparation,None,0.)

    def rhs(self, state, target, current):
        n = self.n
        v,c,s = state[:n],state[n:2*n],state[2*n:]
        release = 1/(1+np.exp(-(v-self.threshold)*self.slope))
        dv = -(v-self.rest)
        if target is not None:
            dv[target] += current
        chemical = self.weight*s[self.pre]*(self.reversal-v[self.post])
        dv += np.bincount(self.post,weights=chemical,minlength=n)
        gap = self.gap*(v[self.gb]-v[self.ga])
        dv += np.bincount(self.ga,weights=gap,minlength=n)
        dv -= np.bincount(self.gb,weights=gap,minlength=n)
        return np.concatenate([dv*self.inv_tau,(release-c)*self.inv_calcium_tau,(release*(1-s)-s)*self.inv_synapse_tau])

    def advance(self, state, time, end, target, current):
        assert (end-time)/self.dt <= 1e6
        while time < end:
            next_time = min(time+self.dt,end)
            assert next_time > time
            state = state+(next_time-time)*self.rhs(state,target,current)
            time = next_time
        assert np.isfinite(state).all()
        return state,time

    def response(self, target, times):
        assert times[0] == 0
        state,time = self.state.copy(),self.origin
        baseline = state[self.n:2*self.n].copy()
        output = [np.zeros(self.n)]
        for t,end in enumerate(times[1:]):
            current = self.kernel[t] if t < len(self.kernel) else 0.
            state,time = self.advance(state,time,self.preparation+end,self.index[target],current)
            output.append(self.scale*(state[self.n:2*self.n]-baseline))
        return np.asarray(output)
