"""One-sided derivatives of C at event times (used by Cmax/Cmin).

The metrics decide where extrema lie from the signs of the left and right derivatives
at grid points, including event times where the derivative jumps. These tests check
those one-sided derivatives directly, at event times that are not exactly
representable (``0.7 + 0.1``), against closed forms and against the right-hand side of
the ODE evaluated on exact matrix-exponential states.
"""

import math

import numpy as np
import pytest
from hypothesis import assume, given

from pk_strategies import doses, model_of, parameters
from pkcore import Dose, OneCompartment, Route
from pkcore._engine import Side, central_slope, prepare
from reference import Parameters, concentration_slopes, well_conditioned

CL, V, KA, F, TLAG = 2.0, 20.0, 1.3, 0.7, 0.1
K = CL / V
T0 = 0.7
DOSE = 100.0


def slopes(history: list[Dose], t: float) -> tuple[float, float]:
    """(right, left) derivative of the central amount at ``t``."""
    kin, table = prepare(OneCompartment(CL, V, ka=KA, f=F, tlag=TLAG), history)
    at = np.full((1, 1, 1), t)
    with np.errstate(all="ignore"):
        right = float(central_slope(kin, table, at, Side.RIGHT)[0, 0])
        left = float(central_slope(kin, table, at, Side.LEFT)[0, 0])
    return right, left


def test_bolus() -> None:
    right, left = slopes([Dose(T0, DOSE, Route.IV_BOLUS)], T0)
    assert left == 0.0
    np.testing.assert_allclose(right, -K * DOSE, rtol=1e-15)


@pytest.mark.parametrize("duration", [0.1, 2.0])
def test_infusion_start_and_end(duration: float) -> None:
    history = [Dose(T0, DOSE, Route.IV_INFUSION, duration=duration)]
    rate = DOSE / duration
    right, left = slopes(history, T0)
    assert left == 0.0
    assert right == pytest.approx(rate, rel=1e-15)
    end = T0 + duration  # the engine's end time, fl(time + duration)
    right, left = slopes(history, end)
    np.testing.assert_allclose(left, rate * math.exp(-K * duration), rtol=1e-12)
    np.testing.assert_allclose(right, -rate * -math.expm1(-K * duration), rtol=1e-12)


def test_oral_start_after_the_lag() -> None:
    history = [Dose(T0, DOSE, Route.ORAL)]
    start = T0 + TLAG  # 0.7999999999999999, not 0.8
    right, left = slopes(history, start)
    assert left == 0.0
    np.testing.assert_allclose(right, F * DOSE * KA, rtol=1e-15)
    assert slopes(history, math.nextafter(start, 0.0)) == (0.0, 0.0)


@given(p=parameters(), history=doses())
def test_one_sided_slopes_at_every_event_match_the_ode(p: Parameters, history: list[Dose]) -> None:
    assume(well_conditioned(p) and history)
    events = {d.time for d in history}
    events |= {d.time + p.tlag for d in history if d.route is Route.ORAL}
    events |= {d.time + d.duration for d in history if d.duration is not None}
    t = np.array(sorted(events))
    kin, table = prepare(model_of(p), history)
    with np.errstate(all="ignore"):
        right = central_slope(kin, table, t.reshape(1, 1, -1), Side.RIGHT)[0] / p.v1
        left = central_slope(kin, table, t.reshape(1, 1, -1), Side.LEFT)[0] / p.v1
    expected_right, expected_left, scale = concentration_slopes(p, history, t)
    # the derivative is a sum of terms of size `scale`; the states are exact to ~1e-14
    atol = 1e-10 * float(np.max(scale))
    np.testing.assert_allclose(right, expected_right, rtol=0.0, atol=atol)
    np.testing.assert_allclose(left, expected_left, rtol=0.0, atol=atol)
