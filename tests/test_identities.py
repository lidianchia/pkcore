"""Analytic identities that any correct linear PK engine must satisfy.

Expected values are mathematical identities (single-dose AUC0-inf = F*D/CL, the
geometric-series accumulation ratio, mass balance, linearity, superposition and
time-shift invariance), never values produced by pkcore itself. Tolerances are a few
hundred units in the last place at most: these identities hold up to rounding.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import pkcore as pk
from pk_strategies import (
    administered,
    concentration_scale,
    doses,
    log_uniform,
    model_of,
    observation_times,
    parameters,
    times_in,
)
from pkcore import Dose, Route
from reference import Parameters


def bioavailable(p: Parameters, dose: Dose) -> float:
    return (p.f if dose.route is Route.ORAL else 1.0) * dose.amount


@given(
    p=parameters(),
    route=st.sampled_from(list(Route)),
    time=times_in(-50.0, 50.0),
    amount=st.floats(0.1, 1000.0),
    duration=log_uniform(0.01, 24.0),
)
def test_single_dose_auc_to_infinity_is_bioavailable_dose_over_clearance(
    p: Parameters, route: Route, time: float, amount: float, duration: float
) -> None:
    dose = Dose(time, amount, route, duration if route is Route.IV_INFUSION else None)
    model = model_of(p)
    expected = bioavailable(p, dose) / p.cl
    np.testing.assert_allclose(float(pk.auc(model, [dose], time, math.inf)), expected, rtol=1e-13)
    np.testing.assert_allclose(
        float(pk.auc(model, [dose], -math.inf, math.inf)), expected, rtol=1e-13
    )


@given(
    p=parameters(),
    history=doses(),
    cuts=st.lists(times_in(-10.0, 120.0), min_size=3, max_size=3).map(sorted),
)
def test_auc_is_additive_over_adjacent_intervals(
    p: Parameters, history: list[Dose], cuts: list[float]
) -> None:
    a, b, c = cuts
    model = model_of(p)
    whole = float(pk.auc(model, history, a, c))
    parts = float(pk.auc(model, history, a, b)) + float(pk.auc(model, history, b, c))
    atol = 1e-15 * concentration_scale(p, history) * (c - a)
    np.testing.assert_allclose(parts, whole, rtol=1e-13, atol=atol)


@given(p=parameters(), history=doses(), times=observation_times())
def test_mass_balance(p: Parameters, history: list[Dose], times: list[float]) -> None:
    """Input received = depot + central + peripheral + eliminated, at every time."""
    model = model_of(p)
    amounts = pk.amounts(model, history, times)
    remaining = amounts.depot + amounts.central + amounts.peripheral
    total_input = np.array([administered(p, history, t) for t in times])
    scale = max([1.0] + [d.amount for d in history])
    np.testing.assert_allclose(
        remaining + amounts.eliminated, total_input, rtol=1e-12, atol=1e-13 * scale
    )
    np.testing.assert_allclose(
        pk.concentration(model, history, times) * p.v1, amounts.central, rtol=1e-15, atol=1e-300
    )


@given(
    p=parameters(),
    history=doses(),
    times=observation_times(),
    power=st.integers(-10, 10),
    factor=st.floats(0.01, 100.0),
)
def test_linearity_in_dose(
    p: Parameters, history: list[Dose], times: list[float], power: int, factor: float
) -> None:
    model = model_of(p)
    base = pk.concentration(model, history, times)

    def scaled(k: float) -> list[Dose]:
        return [Dose(d.time, d.amount * k, d.route, d.duration) for d in history]

    # Scaling by a power of two is exact in floating point, so the result must be
    # exactly scaled (outside the subnormal range).
    exact = pk.concentration(model, scaled(2.0**power), times)
    np.testing.assert_allclose(exact, base * 2.0**power, rtol=0.0, atol=1e-300)
    general = pk.concentration(model, scaled(factor), times)
    np.testing.assert_allclose(general, base * factor, rtol=2e-15, atol=1e-300)


@given(p=parameters(), first=doses(), second=doses(), times=observation_times())
def test_superposition(
    p: Parameters, first: list[Dose], second: list[Dose], times: list[float]
) -> None:
    model = model_of(p)
    together = pk.concentration(model, first + second, times)
    apart = pk.concentration(model, first, times) + pk.concentration(model, second, times)
    np.testing.assert_allclose(together, apart, rtol=1e-14, atol=1e-300)


dyadic_times = st.integers(0, 96 * 64).map(lambda k: k / 64)


@st.composite
def dyadic_histories(draw: st.DrawFn) -> list[Dose]:
    history = []
    for _ in range(draw(st.integers(0, 6))):
        route = draw(st.sampled_from(list(Route)))
        duration = draw(st.integers(1, 24 * 64)) / 64 if route is Route.IV_INFUSION else None
        history.append(Dose(draw(dyadic_times), draw(st.floats(0.0, 1000.0)), route, duration))
    return history


@given(
    p=parameters(),
    lag=st.integers(0, 128),
    history=dyadic_histories(),
    times=st.lists(dyadic_times, min_size=1, max_size=20),
    shift=st.integers(-500, 500),
)
def test_time_shift_invariance(
    p: Parameters, lag: int, history: list[Dose], times: list[float], shift: int
) -> None:
    """Shifting doses and observations together changes nothing.

    With times on a 1/64 h grid and integer shifts every time difference is computed
    exactly, so the results must be bit-identical.
    """
    p = Parameters(p.cl, p.v1, p.q, p.v2, p.ka, p.f, tlag=lag / 64)
    model = model_of(p)
    moved = [Dose(d.time + shift, d.amount, d.route, d.duration) for d in history]
    t = np.array(times)
    np.testing.assert_array_equal(
        pk.concentration(model, moved, t + shift), pk.concentration(model, history, t)
    )
    np.testing.assert_array_equal(pk.trough(model, moved, t + shift), pk.trough(model, history, t))


@given(
    k=log_uniform(0.005, 2.0),
    interval=log_uniform(0.5, 48.0),
    count=st.integers(1, 200),
)
def test_iv_bolus_accumulation_ratio(k: float, interval: float, count: int) -> None:
    """After n doses every tau, C(t_n) / C1(0) = (1 - exp(-n k tau)) / (1 - exp(-k tau))."""
    model = pk.OneCompartment(cl=k * 10.0, v=10.0)
    history = [Dose(i * interval, 50.0, Route.IV_BOLUS) for i in range(count)]
    last = (count - 1) * interval
    ratio = float(pk.concentration(model, history, [last])[0]) / (50.0 / 10.0)
    expected = math.expm1(-count * k * interval) / math.expm1(-k * interval)
    np.testing.assert_allclose(ratio, expected, rtol=1e-12)


def test_iv_bolus_accumulation_ratio_at_steady_state() -> None:
    """After many doses the peak ratio is 1 / (1 - exp(-k tau)) to double precision."""
    k, interval, count = 0.1, 12.0, 40  # exp(-count k tau) = exp(-48) ~ 1e-21
    model = pk.OneCompartment(cl=k * 25.0, v=25.0)
    history = [Dose(i * interval, 100.0, Route.IV_BOLUS) for i in range(count + 1)]
    last = (count - 1) * interval
    single_peak = 100.0 / 25.0
    expected_peak = single_peak / -math.expm1(-k * interval)
    peak = float(pk.concentration(model, history[:count], [last])[0])
    np.testing.assert_allclose(peak, expected_peak, rtol=1e-13)
    # the pre-dose trough of the next dose is the left limit at its own time
    trough = float(pk.trough(model, history, [last + interval])[0])
    np.testing.assert_allclose(trough, expected_peak * math.exp(-k * interval), rtol=1e-13)


def test_auc_over_a_steady_state_interval_equals_single_dose_auc() -> None:
    """At steady state, AUC over one dosing interval = single-dose AUC0-inf = F*D/CL."""
    interval, count = 12.0, 60
    models = {
        "one": Parameters(cl=2.0, v1=20.0, ka=1.3, f=0.7, tlag=0.25),
        "two": Parameters(cl=6.0, v1=15.0, q=3.0, v2=20.0, ka=0.9, f=0.8, tlag=0.5),
    }
    for p in models.values():
        model = model_of(p)
        for route in Route:
            duration = 2.0 if route is Route.IV_INFUSION else None
            history = [Dose(i * interval, 100.0, route, duration) for i in range(count)]
            start = (count - 1) * interval
            value = float(pk.auc(model, history, start, start + interval))
            expected = bioavailable(p, history[0]) / p.cl
            np.testing.assert_allclose(value, expected, rtol=1e-12)
