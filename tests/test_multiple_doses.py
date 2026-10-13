"""Multiple and irregular dosing histories: the engine against independent solutions.

Fixed scenarios are checked against Radau integration and the matrix exponential;
property tests draw random parameters and random dosing histories (irregular,
unsorted times, mixed routes, overlapping infusions) and compare with the matrix
exponential (well-conditioned draws only) and with LSODA integration (all draws).
Tolerances are explained in ``pk_strategies``.
"""

import numpy as np
import pytest
from hypothesis import assume, given

import pkcore as pk
from pk_strategies import (
    EXPM_ATOL,
    EXPM_RTOL,
    ODE_ATOL,
    ODE_RTOL,
    concentration_scale,
    doses,
    model_of,
    observation_times,
    parameters,
)
from pkcore import Dose, Route
from reference import Parameters, expm_solution, ode_solution, well_conditioned

ONE = Parameters(cl=2.0, v1=20.0, ka=1.3, f=0.7, tlag=0.25)
TWO = Parameters(cl=3.0, v1=15.0, q=5.0, v2=40.0, ka=0.9, f=0.8, tlag=0.5)

IRREGULAR = [
    Dose(13.7, 30.0, Route.IV_BOLUS),
    Dose(0.0, 100.0, Route.IV_BOLUS),
    Dose(-6.0, 50.0, Route.ORAL),
    Dose(2.5, 60.0, Route.IV_INFUSION, duration=5.0),
    Dose(4.0, 40.0, Route.IV_INFUSION, duration=1.5),  # overlaps the previous infusion
    Dose(1.0, 80.0, Route.ORAL),
    Dose(10.0, 50.0, Route.ORAL),
    Dose(10.0, 25.0, Route.IV_BOLUS),  # two routes at the same time
    Dose(30.25, 70.0, Route.IV_INFUSION, duration=0.5),
]
IRREGULAR_TIMES = np.concatenate(
    [
        [-6.0, -5.0, 0.0, 0.5, 1.0, 1.25, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5, 6.0, 7.5, 10.0, 10.5],
        [12.0, 13.7, 15.0, 24.0, 30.25, 30.5, 30.75, 36.0, 48.0, 72.0],
    ]
)


@pytest.mark.parametrize("p", [ONE, TWO], ids=["one-compartment", "two-compartment"])
def test_regular_oral_regimen(p: Parameters) -> None:
    history = [Dose(12.0 * i, 100.0, Route.ORAL) for i in range(10)]
    times = np.concatenate([12.0 * np.arange(11), 12.0 * np.arange(10) + 1.5, [150.0, 200.0]])
    check_against_references(p, history, times)


@pytest.mark.parametrize("p", [ONE, TWO], ids=["one-compartment", "two-compartment"])
def test_irregular_mixed_history(p: Parameters) -> None:
    check_against_references(p, IRREGULAR, IRREGULAR_TIMES)


def check_against_references(p: Parameters, history: list[Dose], times: np.ndarray) -> None:
    model = model_of(p)
    scale = concentration_scale(p, history)
    exact = expm_solution(p, history, times)
    ode = ode_solution(p, history, times)
    c = pk.concentration(model, history, times)
    np.testing.assert_allclose(c, ode.concentration, rtol=ODE_RTOL, atol=ODE_ATOL * scale)
    np.testing.assert_allclose(c, exact.concentration, rtol=EXPM_RTOL, atol=EXPM_ATOL * scale)
    trough = pk.trough(model, history, times)
    np.testing.assert_allclose(trough, exact.trough, rtol=EXPM_RTOL, atol=EXPM_ATOL * scale)
    np.testing.assert_allclose(trough, ode.trough, rtol=ODE_RTOL, atol=ODE_ATOL * scale)


def test_dose_order_does_not_matter() -> None:
    model = model_of(TWO)
    forward = pk.concentration(model, IRREGULAR, IRREGULAR_TIMES)
    backward = pk.concentration(model, IRREGULAR[::-1], IRREGULAR_TIMES)
    np.testing.assert_allclose(forward, backward, rtol=1e-14, atol=0.0)


def test_repeated_dose_equals_double_dose() -> None:
    model = model_of(TWO)
    twice = [Dose(1.0, 40.0, Route.ORAL), Dose(1.0, 40.0, Route.ORAL)]
    once = [Dose(1.0, 80.0, Route.ORAL)]
    np.testing.assert_allclose(
        pk.concentration(model, twice, IRREGULAR_TIMES),
        pk.concentration(model, once, IRREGULAR_TIMES),
        rtol=1e-15,
    )


@given(p=parameters(), history=doses(), times=observation_times())
def test_random_histories_match_matrix_exponential(
    p: Parameters, history: list[Dose], times: list[float]
) -> None:
    assume(well_conditioned(p))
    t = np.array([*times, *(d.time for d in history)])  # include the dose times
    model = model_of(p)
    scale = concentration_scale(p, history)
    exact = expm_solution(p, history, t)
    np.testing.assert_allclose(
        pk.concentration(model, history, t),
        exact.concentration,
        rtol=EXPM_RTOL,
        atol=EXPM_ATOL * scale,
    )
    np.testing.assert_allclose(
        pk.trough(model, history, t), exact.trough, rtol=EXPM_RTOL, atol=EXPM_ATOL * scale
    )


@given(p=parameters(), history=doses(), times=observation_times())
def test_random_histories_match_ode_integration(
    p: Parameters, history: list[Dose], times: list[float]
) -> None:
    t = np.array(times)
    model = model_of(p)
    scale = concentration_scale(p, history)
    ode = ode_solution(p, history, t, method="LSODA")
    np.testing.assert_allclose(
        pk.concentration(model, history, t),
        ode.concentration,
        rtol=ODE_RTOL,
        atol=ODE_ATOL * scale,
    )
    auc = np.array([float(pk.auc(model, history, -np.inf, x)) for x in t])
    np.testing.assert_allclose(
        auc, ode.right["auc"], rtol=ODE_RTOL, atol=ODE_ATOL * scale * max(1.0, float(t.max()))
    )
