"""Extreme parameter values: results stay finite and reach the right limits.

Expected values are limiting forms of the model (for example no elimination,
instantaneous equilibrium or instantaneous absorption); each tolerance is the size of
the neglected term, stated next to the check.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import pkcore as pk
from pk_strategies import log_uniform, times_in
from pkcore import Dose, Route

DOSE = 100.0
T = np.array([0.0, 1e-6, 1e-3, 0.5, 1.0, 10.0, 100.0, 1e3, 1e4])


def test_negligible_elimination() -> None:
    cl, v = 1e-9, 1e3  # k = 1e-12 per hour
    k = cl / v
    model = pk.OneCompartment(cl, v, ka=1.0, f=0.5)
    bolus = pk.concentration(model, [Dose(0.0, DOSE, Route.IV_BOLUS)], T)
    np.testing.assert_allclose(bolus, DOSE / v, rtol=k * T.max())  # neglects k t
    infusion = pk.concentration(model, [Dose(0.0, DOSE, Route.IV_INFUSION, 1e5)], T)
    np.testing.assert_allclose(infusion, DOSE / 1e5 * T / v, rtol=k * T.max())
    oral = pk.concentration(model, [Dose(0.0, DOSE, Route.ORAL)], T)
    np.testing.assert_allclose(oral, 0.5 * DOSE / v * -np.expm1(-T), rtol=2 * k * T.max())
    auc = float(pk.auc(model, [Dose(0.0, DOSE, Route.IV_BOLUS)], 0.0, 1e4))
    np.testing.assert_allclose(auc, DOSE / v * 1e4, rtol=k * 1e4)
    total = float(pk.auc(model, [Dose(0.0, DOSE, Route.IV_BOLUS)], 0.0, math.inf))
    np.testing.assert_allclose(total, DOSE / cl, rtol=1e-14)


def test_very_fast_elimination() -> None:
    cl, v = 1e8, 1.0  # k = 1e8 per hour
    model = pk.OneCompartment(cl, v, ka=1.0, f=0.5)
    bolus = pk.concentration(model, [Dose(0.0, DOSE, Route.IV_BOLUS)], T)
    assert bolus[0] == DOSE / v
    assert np.all(bolus[1:] < 1e-40)  # exp(-100) and smaller
    rate = DOSE / 1e5
    infusion = pk.concentration(model, [Dose(0.0, DOSE, Route.IV_INFUSION, 1e5)], T[2:])
    np.testing.assert_allclose(infusion, rate / cl, rtol=1e-14)  # steady state reached at once
    # Formation-rate limited: C = f D ka exp(-ka t) / cl, neglecting ka / k = 1e-8.
    oral = pk.concentration(model, [Dose(0.0, DOSE, Route.ORAL)], T[2:])
    np.testing.assert_allclose(oral, 0.5 * DOSE * np.exp(-T[2:]) / cl, rtol=2e-8)
    total = float(pk.auc(model, [Dose(0.0, DOSE, Route.ORAL)], 0.0, math.inf))
    np.testing.assert_allclose(total, 0.5 * DOSE / cl, rtol=1e-14)


@pytest.mark.parametrize("volume", [2.0**-30, 2.0**-10, 2.0**10, 2.0**30])
def test_concentration_scales_inversely_with_volume(volume: float) -> None:
    # Powers of two keep cl / v = k exact, so C * V must not change at all.
    k = 0.2
    history = [
        Dose(0.0, DOSE, Route.IV_BOLUS),
        Dose(1.0, DOSE, Route.IV_INFUSION, duration=2.0),
        Dose(5.0, DOSE, Route.ORAL),
    ]
    unit = pk.concentration(pk.OneCompartment(k, 1.0, ka=0.7), history, T)
    scaled = pk.concentration(pk.OneCompartment(k * volume, volume, ka=0.7), history, T)
    np.testing.assert_allclose(scaled * volume, unit, rtol=1e-15, atol=1e-300)


def test_instantaneous_absorption_approaches_an_iv_bolus() -> None:
    cl, v, tlag = 2.0, 20.0, 0.5
    fast = pk.OneCompartment(cl, v, ka=1e9, f=0.8, tlag=tlag)
    t = tlag + np.array([1e-5, 0.1, 1.0, 10.0, 100.0])
    oral = pk.concentration(fast, [Dose(0.0, DOSE, Route.ORAL)], t)
    bolus = 0.8 * DOSE / v * np.exp(-cl / v * (t - tlag))
    np.testing.assert_allclose(oral, bolus, rtol=1e-9)  # neglects k / ka = 1e-10


def test_negligible_absorption() -> None:
    cl, v, ka = 2.0, 20.0, 1e-9
    slow = pk.OneCompartment(cl, v, ka=ka, f=0.8)
    t = np.array([0.5, 1.0, 10.0, 100.0])
    oral = pk.concentration(slow, [Dose(0.0, DOSE, Route.ORAL)], t)
    # Input rate f D ka (depot depletion ka t <= 1e-7 neglected).
    expected = 0.8 * DOSE * ka / cl * -np.expm1(-cl / v * t)
    np.testing.assert_allclose(oral, expected, rtol=2e-7)
    assert float(pk.apparent_oral_terminal_rate(slow)) == ka  # flip-flop


def one_compartment_bolus_and_infusion(cl: float, v: float, t: np.ndarray) -> np.ndarray:
    """Textbook one-compartment curve: bolus DOSE at 0, infusion of DOSE over [3, 5] h."""
    k = cl / v
    bolus = DOSE / v * np.exp(-k * t)
    rate, start, duration = DOSE / 2.0, 3.0, 2.0
    running = np.clip(t - start, 0.0, duration)
    after = np.maximum(t - start - duration, 0.0)
    infusion = rate / cl * -np.expm1(-k * running) * np.exp(-k * after)
    return bolus + infusion


def test_two_compartments_with_negligible_distribution() -> None:
    q, cl, v1 = 1e-9, 2.0, 15.0
    two = pk.TwoCompartment(cl, v1, q, 40.0)
    history = [Dose(0.0, DOSE, Route.IV_BOLUS), Dose(3.0, DOSE, Route.IV_INFUSION, duration=2.0)]
    t = T[T <= 100.0]
    np.testing.assert_allclose(
        pk.concentration(two, history, t),
        one_compartment_bolus_and_infusion(cl, v1, t),
        rtol=q / v1 * 100.0 * 2,  # neglects k12 t
    )
    assert np.all(pk.amounts(two, history, t).peripheral <= DOSE * 2 * q / v1 * 100.0)


def test_two_compartments_with_instantaneous_distribution() -> None:
    q, cl, v1, v2 = 1e9, 2.0, 15.0, 40.0
    two = pk.TwoCompartment(cl, v1, q, v2)
    t = np.array([1e-3, 0.5, 1.0, 10.0, 100.0])
    np.testing.assert_allclose(
        pk.concentration(two, [Dose(0.0, DOSE, Route.IV_BOLUS)], t),
        DOSE / (v1 + v2) * np.exp(-cl / (v1 + v2) * t),
        rtol=1e-7,  # neglects terms of order cl / q and exp(-q t / v)
    )
    np.testing.assert_allclose(float(pk.terminal_rate(two)), cl / (v1 + v2), rtol=1e-7)


def test_very_late_times_underflow_to_zero() -> None:
    model = pk.TwoCompartment(3.0, 15.0, 5.0, 40.0, ka=0.9)
    history = [
        Dose(0.0, DOSE, route, 1.0 if route is Route.IV_INFUSION else None) for route in Route
    ]
    c = pk.concentration(model, history, [1e6, 1e8])
    assert np.all(c == 0.0)
    np.testing.assert_allclose(float(pk.auc(model, history, 0.0, 1e8)), 3 * DOSE / 3.0, rtol=1e-13)


extreme = log_uniform(1e-6, 1e6)


@st.composite
def extreme_models(draw: st.DrawFn) -> pk.Model:
    ka, f, tlag = draw(extreme), draw(st.floats(0.0, 1.0)), draw(times_in(0.0, 10.0))
    if draw(st.booleans()):
        return pk.OneCompartment(draw(extreme), draw(extreme), ka=ka, f=f, tlag=tlag)
    cl, v1, q, v2 = draw(extreme), draw(extreme), draw(extreme), draw(extreme)
    return pk.TwoCompartment(cl, v1, q, v2, ka=ka, f=f, tlag=tlag)


@given(
    model=extreme_models(),
    route=st.sampled_from(list(Route)),
    amount=st.floats(0.0, 1e4),
    duration=log_uniform(1e-4, 1e3),
    times=st.lists(times_in(0.0, 1e4), min_size=1, max_size=10),
)
def test_results_are_finite_and_non_negative_for_extreme_parameters(
    model: pk.Model, route: Route, amount: float, duration: float, times: list[float]
) -> None:
    history = [Dose(0.0, amount, route, duration if route is Route.IV_INFUSION else None)]
    c = pk.concentration(model, history, times)
    amounts = pk.amounts(model, history, times)
    values = [c, pk.trough(model, history, times), *vars(amounts).values()]
    for value in values:
        assert np.all(np.isfinite(value))
        assert np.all(value >= 0.0)
    total = float(pk.auc(model, history, 0.0, math.inf))
    f = float(model.f) if route is Route.ORAL else 1.0
    np.testing.assert_allclose(total, f * amount / float(model.cl), rtol=1e-12, atol=1e-300)
