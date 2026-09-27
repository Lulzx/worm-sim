"""First-order held-concentration splitting, with Diffrax for fast windows."""
from dataclasses import dataclass
import math
import numpy as np
import jax
import jax.numpy as jnp
from solvers import integrate


@dataclass(frozen=True)
class Multirate:
    slow_dt: float
    max_windows: int = 100_000

    def __post_init__(self):
        if not math.isfinite(self.slow_dt) or self.slow_dt <= 0:
            raise ValueError('slow_dt must be finite and positive')
        if type(self.max_windows) is not int or self.max_windows < 1:
            raise ValueError('max_windows must be a positive integer')

    def grid(self, save_times):
        """Always split at preparation, stimulus, and requested output boundaries."""
        ends = np.asarray(save_times, dtype=float)
        boundaries = [0.]
        for end in ends:
            duration = float(end) - boundaries[-1]
            if duration == 0:
                continue
            count_float = duration / self.slow_dt
            if not math.isfinite(count_float) or count_float > self.max_windows:
                raise ValueError('multirate window budget exceeded')
            count = max(1, math.ceil(count_float))
            if len(boundaries) - 1 + count > self.max_windows:
                raise ValueError('multirate window budget exceeded')
            boundaries.extend(np.linspace(boundaries[-1], end, count + 1)[1:].tolist())
        boundaries = np.asarray(boundaries)
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError('multirate grid cannot advance')
        indices = np.searchsorted(boundaries, ends)
        if not np.array_equal(boundaries[indices], ends):
            raise ValueError('multirate grid omitted a requested output')
        return boundaries, indices


def solve(engine, theta, target, initial):
    n = engine.n
    stop = engine.plasticity_start
    size = stop - 3*n
    kernel = jax.nn.softplus(theta['kernel'])
    intervals = jnp.searchsorted(engine.save_times, engine.slow_boundaries[:-1], side='right') - 1
    currents = jnp.where((intervals >= 0) & (intervals < len(kernel)), kernel[jnp.clip(intervals, 0, len(kernel)-1)], 0.)

    def rhs(t, augmented, args):
        p, cell, current, concentration = args
        fast, _ = augmented
        state = jnp.concatenate((fast[:3*n], concentration, fast[3*n:]))
        derivative, release = engine.rhs_current(state, p, cell, current, return_release=True)
        fast_derivative = jnp.concatenate((derivative[:3*n], derivative[stop:]))
        return fast_derivative, engine.modulation.drive(release, p['modulation'])

    def advance(carry, interval):
        state, saved, totals = carry
        start, end, current, output_index = interval
        concentration = state[3*n:stop]
        fast = jnp.concatenate((state[:3*n], state[stop:]))
        solution = integrate(rhs, (fast, jnp.zeros(size)), (theta, target, current, concentration),
                             jnp.asarray([end]), engine.adaptive, t0=start)
        fast, integrated_drive = (part[0] for part in solution.ys)
        concentration = engine.modulation.advance_mean_drive(
            concentration, integrated_drive / (end-start), end-start, theta['modulation'])
        state = jnp.concatenate((fast[:3*n], concentration, fast[3*n:]))
        counts = jnp.stack([solution.stats[k] for k in ['num_steps', 'num_accepted_steps', 'num_rejected_steps']])
        saved = jax.lax.cond(output_index >= 0, lambda values: values.at[output_index].set(state), lambda values: values, saved)
        return (state, saved, totals+counts), None

    saved = jnp.zeros((len(engine.save_times), len(initial)))
    if engine.preparation == 0:
        saved = saved.at[0].set(initial)
    (_, states, totals), _ = jax.lax.scan(advance, (initial, saved, jnp.zeros(3,dtype=jnp.int64)),
        (engine.slow_boundaries[:-1], engine.slow_boundaries[1:], currents, engine.slow_output_indices))
    stats = dict(zip(['num_steps', 'num_accepted_steps', 'num_rejected_steps'], totals))
    stats['slow_windows'] = len(engine.slow_boundaries)-1
    return states, stats
