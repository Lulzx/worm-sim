"""Positive per-neuron fluorescence calibration, independent of neural dynamics."""
import math
import jax
import jax.numpy as jnp
import equinox as eqx
from modulation import fields


class Observation(eqx.Module):
    names: tuple = eqx.field(static=True)
    initial_gain: float = eqx.field(static=True)
    prior_strength: float = eqx.field(static=True)

    def __init__(self, names, spec):
        fields(spec, ['initial_gain', 'prior_strength'])
        self.names = tuple(names)
        self.initial_gain = float(spec['initial_gain'])
        self.prior_strength = float(spec['prior_strength'])
        if not math.isfinite(self.initial_gain) or self.initial_gain <= 0:
            raise ValueError('observation gain must be finite and positive')
        if not math.isfinite(self.prior_strength) or self.prior_strength < 0:
            raise ValueError('observation prior must be finite and nonnegative')

    def parameters(self):
        return {'log_gain': jnp.full(len(self.names), math.log(self.initial_gain))}

    def penalty(self, params):
        return self.prior_strength * jnp.mean((params['log_gain'] - math.log(self.initial_gain))**2)
