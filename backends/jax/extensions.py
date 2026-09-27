"""Strict, portable JAX atlas envelope; never masquerades as a native checkpoint."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
from observation import Observation
from modulation import Modulation, fields
from rectification import GapRectification
from dark_edges import DarkEdges
from plasticity import Plasticity
from solvers import Adaptive
from multirate import Multirate
from optimization import rate_multipliers
from level0 import Level0, parameters

FORMAT = 'wormsim-jax-atlas'


def initialize(model, graph, times, configuration):
    if configuration is None:
        modules = {}
    else:
        fields(configuration, ['schema_version', 'extensions', 'solver'], ['multirate', 'optimization'])
        if configuration['schema_version'] != 1:
            raise ValueError('unsupported JAX configuration version')
        specs = configuration['extensions']
        fields(specs, [], ['modulation', 'rectification', 'dark_edges', 'plasticity', 'observation'])
        modules = {}
        if 'observation' in specs:
            modules['observation'] = Observation(sorted(n['id'] for n in graph['neurons']), specs['observation'])
        if 'modulation' in specs:
            modules['modulation'] = Modulation(sorted(n['id'] for n in graph['neurons']), specs['modulation'])
        if 'rectification' in specs:
            modules['rectification'] = GapRectification(graph, specs['rectification'])
        if 'dark_edges' in specs:
            modules['dark_edges'] = DarkEdges(graph, specs['dark_edges'])
        if 'plasticity' in specs:
            modules['plasticity'] = Plasticity(graph, specs['plasticity'], modules.get('dark_edges'))
        if configuration['solver'] is not None:
            fields(configuration['solver'], [], ['method', 'rtol', 'atol', 'dt0', 'dtmax', 'max_steps', 'adjoint', 'checkpoints', 'adjoint_rtol', 'adjoint_atol', 'adjoint_max_steps'])
            modules['adaptive'] = Adaptive(**configuration['solver'])
        if configuration.get('multirate') is not None:
            fields(configuration['multirate'], ['slow_dt'], ['max_windows'])
            modules['multirate'] = Multirate(**configuration['multirate'])
    engine = Level0(model, graph, times, **modules)
    theta = parameters(model, **{k: v for k, v in modules.items() if k not in ('adaptive', 'multirate')})
    if model.get('classifier'):
        theta['classifier'] = jnp.asarray([model['classifier']['bias'], model['classifier']['raw_slope']])
    active = jax.tree.map(lambda v: jnp.ones_like(v, dtype=bool), theta)
    active['groups'] = jnp.asarray([g['trainable'] for g in model['parameters']['groups']])
    active['log_gain'] = jnp.asarray(model.get('observation_log_gain') is not None)
    if 'observation' in modules:
        # Replace the global gain; keep the native calcium scale fixed to avoid
        # two independently trainable amplitude factors for each neuron.
        active['log_gain'] = jnp.asarray(False)
        if any(g['trainable'] and g['name'].startswith('calcium_scale/') for g in model['parameters']['groups']):
            raise ValueError('per-neuron observation gains require frozen native calcium scales')
    if configuration is not None and 'plasticity' in modules:
        by_type = {t['id']: t for t in configuration['extensions']['plasticity']['types']}
        modes = [by_type[t]['mode'] for t in modules['plasticity'].types]
        active['plasticity']['raw'] = jnp.asarray(
            [[True, m != 'facilitation', m != 'depression', True] for m in modes], dtype=bool).reshape((-1, 4))
    rates=rate_multipliers(model,theta,configuration)
    active=jax.tree.map(lambda mask,rate:mask & (rate>0),active,rates)
    return engine, theta, active


def pack(base_model, theta, configuration):
    keys = set(configuration['extensions'])
    expected = {'groups', 'kernel', 'log_gain'} | keys
    if base_model.get('classifier'):
        expected.add('classifier')
    if set(theta) != expected:
        raise ValueError('unexpected or missing fitted parameter family')
    extension_parameters = jax.tree.map(lambda v: {'shape': list(v.shape), 'values': np.asarray(v).reshape(-1).tolist()}, {k: theta[k] for k in sorted(keys)})
    return {'format': FORMAT, 'schema_version': 1, 'base_model': copy.deepcopy(base_model),
            'configuration': copy.deepcopy(configuration), 'extension_parameters': extension_parameters}


def restore(checkpoint, graph, times):
    fields(checkpoint, ['format', 'schema_version', 'base_model', 'configuration', 'extension_parameters'])
    if checkpoint['format'] != FORMAT or checkpoint['schema_version'] != 1:
        raise ValueError('unsupported JAX atlas checkpoint')
    engine, theta, active = initialize(checkpoint['base_model'], graph, times, checkpoint['configuration'])
    saved = checkpoint['extension_parameters']
    if not isinstance(saved, dict) or set(saved) != set(checkpoint['configuration']['extensions']):
        raise ValueError('checkpoint extension families mismatch')
    for family, values in saved.items():
        if not isinstance(values, dict) or set(values) != set(theta[family]):
            raise ValueError('checkpoint extension parameter keys mismatch')
        for key, values_array in values.items():
            fields(values_array, ['shape', 'values'])
            expected_shape=theta[family][key].shape
            array = np.asarray(values_array['values'], dtype=float)
            if values_array['shape'] != list(expected_shape) or array.ndim != 1 or array.size != theta[family][key].size or not np.isfinite(array).all():
                raise ValueError('invalid checkpoint extension parameter shape or value')
            array=array.reshape(expected_shape)
            if np.any(array[~np.asarray(active[family][key])] != np.asarray(theta[family][key])[~np.asarray(active[family][key])]):
                raise ValueError('inactive extension parameter changed')
            theta[family][key] = jnp.asarray(array)
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(theta)):
        raise ValueError('nonfinite checkpoint parameter')
    return engine, theta, active
