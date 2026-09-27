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

    def __post_init__(self):
        if self.method not in ('tsit5','kvaerno5'):
            raise ValueError('adaptive method must be tsit5 or kvaerno5')
        if any(not math.isfinite(v) or v<=0 for v in (self.rtol,self.atol,self.dt0)):
            raise ValueError('adaptive tolerances and initial step must be finite and positive')
        if self.dtmax is not None and (not math.isfinite(self.dtmax) or self.dtmax<=0):
            raise ValueError('maximum step must be finite and positive')
        if type(self.max_steps) is not int or self.max_steps<1:
            raise ValueError('maximum steps must be a positive integer')


def integrate(rhs, y0, args, times, settings, *, t0=0.):
    """Solve one continuous-input interval and fail on solver error.

    Diffrax supplies nonlinear solves, PID adaptation and the checkpointed
    discretize-then-optimize adjoint. No custom integrator or gradient is used.
    """
    solver=diffrax.Tsit5() if settings.method=='tsit5' else diffrax.Kvaerno5()
    controller=diffrax.PIDController(rtol=settings.rtol,atol=settings.atol,dtmax=settings.dtmax)
    return diffrax.diffeqsolve(diffrax.ODETerm(rhs),solver,t0=t0,t1=times[-1],
        dt0=settings.dt0,y0=y0,args=args,saveat=diffrax.SaveAt(ts=times),
        stepsize_controller=controller,adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=settings.max_steps,throw=True)
