"""Library-backed adaptive integration; settings are static JIT configuration."""
from dataclasses import dataclass
import math
import diffrax


@dataclass(frozen=True)
class Adaptive:
    method: str = 'kvaerno5'
    rtol: float = 1e-7
    atol: float = 1e-9
    dt0: float = 1e-3
    dtmax: float | None = None
    max_steps: int = 100_000  # Per continuous-input interval.
    adjoint: str = 'checkpoint'
    checkpoints: int | None = None
    adjoint_rtol: float | None = None
    adjoint_atol: float | None = None
    adjoint_max_steps: int | None = None

    def __post_init__(self):
        if self.method not in ('tsit5','kvaerno5'):
            raise ValueError('adaptive method must be tsit5 or kvaerno5')
        if any(not math.isfinite(v) or v<=0 for v in (self.rtol,self.atol,self.dt0)):
            raise ValueError('adaptive tolerances and initial step must be finite and positive')
        if self.dtmax is not None and (not math.isfinite(self.dtmax) or self.dtmax<=0):
            raise ValueError('maximum step must be finite and positive')
        if type(self.max_steps) is not int or self.max_steps<1:
            raise ValueError('maximum steps must be a positive integer')
        if self.adjoint not in ('checkpoint','continuous'):
            raise ValueError('adjoint must be checkpoint or continuous')
        if self.checkpoints is not None and (type(self.checkpoints) is not int or self.checkpoints<1):
            raise ValueError('checkpoints must be a positive integer or null')
        if any(v is not None and (not math.isfinite(v) or v<=0) for v in (self.adjoint_rtol,self.adjoint_atol)):
            raise ValueError('adjoint tolerances must be finite and positive')
        if self.adjoint_max_steps is not None and (type(self.adjoint_max_steps) is not int or self.adjoint_max_steps<1):
            raise ValueError('adjoint maximum steps must be a positive integer')
        if self.adjoint=='checkpoint' and any(v is not None for v in (self.adjoint_rtol,self.adjoint_atol,self.adjoint_max_steps)):
            raise ValueError('backward solver settings require the continuous adjoint')
        if self.adjoint=='continuous' and self.checkpoints is not None:
            raise ValueError('checkpoint budget does not apply to continuous adjoints')


def make_adjoint(settings, solver):
    if settings.adjoint=='checkpoint':
        return diffrax.RecursiveCheckpointAdjoint(checkpoints=settings.checkpoints)
    return diffrax.BacksolveAdjoint(solver=solver,
        stepsize_controller=diffrax.PIDController(
            rtol=settings.adjoint_rtol if settings.adjoint_rtol is not None else settings.rtol,
            atol=settings.adjoint_atol if settings.adjoint_atol is not None else settings.atol,
            dtmax=settings.dtmax),
        max_steps=settings.adjoint_max_steps if settings.adjoint_max_steps is not None else settings.max_steps,
        throw=True)


def integrate(rhs, y0, args, times, settings, *, t0=0.):
    """Solve one continuous-input interval and fail on solver error.

    Diffrax supplies nonlinear solves, PID adaptation and the selected adjoint.
    No custom integrator or gradient is used. Continuous gradients approximate
    the underlying ODE sensitivity, not the discrete numerical-program gradient.
    """
    solver=diffrax.Tsit5() if settings.method=='tsit5' else diffrax.Kvaerno5()
    controller=diffrax.PIDController(rtol=settings.rtol,atol=settings.atol,dtmax=settings.dtmax)
    return diffrax.diffeqsolve(diffrax.ODETerm(rhs),solver,t0=t0,t1=times[-1],
        dt0=settings.dt0,y0=y0,args=args,saveat=diffrax.SaveAt(ts=times),
        stepsize_controller=controller,adjoint=make_adjoint(settings,solver),
        max_steps=settings.max_steps,throw=True)
