"""SciPy L-BFGS-B adapter over exactly the active JAX parameter coordinates."""
import jax
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize


class EvaluationLimit(Exception):
    pass


class NonfiniteEvaluation(Exception):
    pass


def optimize_active(theta, active, objective, accepted, evaluated, *, max_evaluations=201, max_iterations=200):
    if type(max_evaluations) is not int or max_evaluations < 1 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('positive integer budgets required')
    flat, unravel = ravel_pytree(theta)
    mask, _ = ravel_pytree(active)
    if jax.tree.structure(theta) != jax.tree.structure(active) or mask.shape != flat.shape:
        raise ValueError('active parameter layout differs')
    for parameter, flag in zip(jax.tree.leaves(theta),jax.tree.leaves(active),strict=True):
        if parameter.shape != flag.shape or np.asarray(flag).dtype != bool:
            raise ValueError('active mask must have matching boolean leaves')
    indices = np.flatnonzero(np.asarray(mask))
    if not len(indices) or not np.isfinite(np.asarray(flat)).all():
        raise ValueError('requires finite parameters and active coordinates')
    def unpack(x):
        return unravel(flat.at[indices].set(x))
    calls=0;iterations=0;cache=None;last=None
    def fun(x):
        nonlocal calls,cache
        if cache is not None and np.array_equal(x,cache[0]):
            return cache[1],cache[2]
        if calls >= max_evaluations:
            raise EvaluationLimit()
        candidate=unpack(x);calls+=1
        value,gradient,metrics=objective(candidate)
        g,_=ravel_pytree(gradient)
        finite=bool(np.isfinite(float(value)) and np.isfinite(np.asarray(g)).all())
        evaluated(calls,candidate,value,gradient,metrics,finite)
        if not finite:raise NonfiniteEvaluation('nonfinite line-search evaluation')
        cache=(np.array(x,copy=True),float(value),np.asarray(g)[indices].copy(),candidate,metrics)
        return cache[1],cache[2]
    def record(x,iteration):
        nonlocal last
        fun(x)
        last=(np.array(x,copy=True),cache[1],cache[3],cache[4])
        accepted(iteration,calls,cache[3],cache[4])
    def callback(x):
        nonlocal iterations
        record(x,iterations+1);iterations+=1
    options=dict(maxiter=max_iterations,maxfun=max_evaluations,maxls=20,maxcor=20,ftol=1e-12,gtol=1e-9)
    x0=np.asarray(flat)[indices].copy()
    try:
        record(x0,0)
        result=minimize(fun,x0,jac=True,method='L-BFGS-B',callback=callback,options=options)
        status={'status':'optimizer_terminated','optimizer_success':bool(result.success),
            'optimizer_status':int(result.status),'message':str(result.message)}
        if last is None or not np.array_equal(result.x,last[0]):
            raise ValueError('optimizer returned an unrecorded accepted iterate')
    except EvaluationLimit:
        status={'status':'evaluation_budget_exhausted','optimizer_success':False}
    except NonfiniteEvaluation:
        status={'status':'nonfinite_evaluation','optimizer_success':False}
    return dict(**status,evaluations=calls,accepted_iterations=iterations,options=options,
        active_coordinates=len(indices),final_theta=None if last is None else last[2],
        final_metrics=None if last is None else last[3])
