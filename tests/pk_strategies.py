"""Hypothesis strategies, tolerances and small helpers shared by the tests.

All parameter values are synthetic and only span plausible magnitudes; they do not
describe any particular drug.

Tolerances for comparisons with the references, ``|engine - ref| <= rtol * |ref| +
atol * scale`` with ``scale = largest dose / v1`` (a natural concentration scale):

* matrix exponential (well-conditioned cases only): ``rtol = 1e-10``,
  ``atol = 1e-13``. SciPy's ``expm`` is accurate to about 1e-15 relative per
  propagation step there, and errors grow at most linearly over the (few dozen)
  steps; the observed agreement is about 1e-14.
* ODE integration at ``rtol = 1e-10``: ``rtol = 1e-7``, ``atol = 1e-9``. Adaptive
  integrators control only local errors, and global errors of 1e-10 to 1e-9 are
  normal at this setting (the largest observed for LSODA over 200 random histories
  was 4e-10 of the scale; Radau was better than 1e-12).

Both are many orders of magnitude tighter than any formula error would produce.
"""

import math

import numpy as np
from hypothesis import strategies as st

import pkcore as pk
from pkcore import Dose, Route
from reference import Parameters

EXPM_RTOL = 1e-10
EXPM_ATOL = 1e-13
ODE_RTOL = 1e-7
ODE_ATOL = 1e-9


def log_uniform(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Floats spread evenly on a log scale over [lo, hi]."""
    return st.floats(math.log(lo), math.log(hi)).map(math.exp)


def times_in(lo: float, hi: float) -> st.SearchStrategy[float]:
    return st.floats(lo, hi, allow_subnormal=False)


@st.composite
def parameters(draw: st.DrawFn, *, two_compartment: bool | None = None) -> Parameters:
    """Raw parameters of a one- or two-compartment model with oral absorption."""
    two = draw(st.booleans()) if two_compartment is None else two_compartment
    cl = draw(log_uniform(0.05, 50.0))
    v1 = draw(log_uniform(1.0, 500.0))
    q, v2 = (draw(log_uniform(0.05, 50.0)), draw(log_uniform(1.0, 1000.0))) if two else (0.0, 1.0)
    ka = draw(log_uniform(0.05, 10.0))
    f = draw(st.floats(0.0, 1.0, allow_subnormal=False))
    tlag = draw(times_in(0.0, 2.0))
    return Parameters(cl=cl, v1=v1, q=q, v2=v2, ka=ka, f=f, tlag=tlag)


@st.composite
def doses(
    draw: st.DrawFn,
    *,
    max_doses: int = 8,
    horizon: float = 72.0,
    routes: tuple[Route, ...] = tuple(Route),
) -> list[Dose]:
    """An irregular dosing history: unsorted times, mixed routes, overlapping infusions."""
    history = []
    for _ in range(draw(st.integers(0, max_doses))):
        route = draw(st.sampled_from(routes))
        duration = draw(log_uniform(0.01, 24.0)) if route is Route.IV_INFUSION else None
        time = draw(times_in(0.0, horizon))
        amount = draw(st.floats(0.0, 1000.0, allow_subnormal=False))
        history.append(Dose(time, amount, route, duration))
    return history


def observation_times(horizon: float = 96.0) -> st.SearchStrategy[list[float]]:
    return st.lists(times_in(0.0, horizon), min_size=1, max_size=20)


def model_of(p: Parameters) -> pk.Model:
    """The pkcore model with the same raw parameters as the reference."""
    if p.q == 0.0:
        return pk.OneCompartment(p.cl, p.v1, ka=p.ka, f=p.f, tlag=p.tlag)
    return pk.TwoCompartment(p.cl, p.v1, p.q, p.v2, ka=p.ka, f=p.f, tlag=p.tlag)


def concentration_scale(p: Parameters, history: list[Dose]) -> float:
    """A natural concentration scale for absolute tolerances: largest dose / v1 (mg/L)."""
    return max([1.0] + [d.amount for d in history]) / p.v1


def administered(p: Parameters, history: list[Dose], t: float) -> float:
    """Input received by time ``t``: boluses, infused amounts and f * oral after the lag.

    An oral dose counts once ``(t - time) - tlag >= 0``, the convention (including the
    order of the floating-point operations) of pkcore's right-continuous evaluation.
    """
    total = 0.0
    for d in history:
        if d.route is Route.IV_BOLUS and d.time <= t:
            total += d.amount
        elif d.route is Route.ORAL and (t - d.time) - p.tlag >= 0.0:
            total += p.f * d.amount
        elif d.route is Route.IV_INFUSION and d.duration is not None:
            total += d.amount / d.duration * float(np.clip(t - d.time, 0.0, d.duration))
    return total
