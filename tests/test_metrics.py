"""Metrics: Cmax/Tmax, Cmin, trough, AUC windows, terminal rates.

Expected values come from literature closed forms (single-dose and steady-state
Tmax of the one-compartment oral model, Garrett's limit for ka == k), from root
finding with ``scipy.optimize.brentq`` on the textbook derivative, from LAPACK
eigenvalues, and from the matrix-exponential reference evaluated on dense grids.

Tolerances: Tmax ``rtol = 1e-12`` (root finding narrows the bracket to a few units in
the last place; what remains is the band in which rounding makes the sign of the
computed derivative unreliable, observed up to about 1e-15 relative); Cmax
``rtol = 1e-13`` (an error d in Tmax changes C only by O(d^2)).
"""

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy.optimize import brentq

import pkcore as pk
from pk_strategies import (
    EXPM_ATOL,
    EXPM_RTOL,
    concentration_scale,
    doses,
    log_uniform,
    model_of,
    parameters,
)
from pkcore import Dose, Route
from reference import (
    Parameters,
    concentration_derivatives,
    concentration_slopes,
    expm_solution,
    well_conditioned,
)

DOSE = 100.0
TMAX_RTOL = 1e-12
CMAX_RTOL = 1e-13


@pytest.mark.parametrize(
    ("ka", "k"), [(1.3, 0.1), (0.05, 0.4), (0.125, 0.125)], ids=["usual", "flip-flop", "ka==k"]
)
def test_single_oral_dose_peak_matches_closed_form(ka: float, k: float) -> None:
    v, f, tlag = 20.0, 0.7, 0.75
    model = pk.OneCompartment(k * v, v, ka=ka, f=f, tlag=tlag)
    peak = pk.cmax(model, [Dose(0.0, DOSE, Route.ORAL)], 0.0, 96.0)
    # Tmax - tlag = ln(ka / k) / (ka - k), and 1 / k when ka == k (Garrett 1994).
    t_peak = 1.0 / k if ka == k else math.log(ka / k) / (ka - k)
    np.testing.assert_allclose(float(peak.time), tlag + t_peak, rtol=TMAX_RTOL)
    np.testing.assert_allclose(
        float(peak.value), f * DOSE / v * math.exp(-k * t_peak), rtol=CMAX_RTOL
    )


@pytest.mark.parametrize(
    ("dose_time", "ka", "end"), [(0.7, 5.0, 2000.0), (0.7, 8.0, 72.0), (8.7, 8.0, 72.0)]
)
def test_peak_after_a_lag_that_is_not_exactly_representable(
    dose_time: float, ka: float, end: float
) -> None:
    """Absorption starts at fl(time + tlag) = 0.7999999999999999 for time 0.7, tlag 0.1."""
    k, v, tlag = 0.1, 20.0, 0.1
    model = pk.OneCompartment(k * v, v, ka=ka, tlag=tlag)
    peak = pk.cmax(model, [Dose(dose_time, DOSE, Route.ORAL)], 0.0, end)
    t_peak = math.log(ka / k) / (ka - k)
    np.testing.assert_allclose(float(peak.time), dose_time + tlag + t_peak, rtol=TMAX_RTOL)
    np.testing.assert_allclose(float(peak.value), DOSE / v * math.exp(-k * t_peak), rtol=CMAX_RTOL)


@pytest.mark.parametrize(
    "case",
    [
        # (cl, v, ka, f, tlag, dose time, window end)
        (20.0, 20.0, 2.0, 1.0, 0.0, 0.0, 1000.0),
        (380.0, 20.0, 20.0, 0.7, 2.0, 100.0, 1e5),
        # flip-flop: C' underflows to exactly zero before C'' + k C' does
        (100.0, 20.0, 0.5, 1.0, 0.0, 0.0, 5000.0),
    ],
)
def test_peak_in_a_window_much_longer_than_the_curve(
    case: tuple[float, float, float, float, float, float, float],
) -> None:
    """Long before the window ends C and C' underflow to exactly zero.

    Regression: a zero derivative at the far end of an interval was not treated as
    a sign change, so Cmax came out as 0.
    """
    cl, v, ka, f, tlag, dose_time, end = case
    k = cl / v
    model = pk.OneCompartment(cl, v, ka=ka, f=f, tlag=tlag)
    peak = pk.cmax(model, [Dose(dose_time, DOSE, Route.ORAL)], 0.0, end)
    t_peak = math.log(ka / k) / (ka - k)
    np.testing.assert_allclose(float(peak.time), dose_time + tlag + t_peak, rtol=TMAX_RTOL)
    expected = f * DOSE / v * math.exp(-k * t_peak)
    np.testing.assert_allclose(float(peak.value), expected, rtol=CMAX_RTOL)


def test_two_extrema_between_the_same_two_events() -> None:
    """A small bolus during absorption: C first dips, then rises to the oral peak.

    Both extrema lie between the bolus and the (distant) window end. Regression: with a
    sampling grid, the peak was missed (15.55 instead of 15.92). Expected values: roots
    of the reference derivative (matrix exponential and the ODE right-hand side).
    """
    p = Parameters(cl=1.0, v1=5.0, q=50.0, v2=50.0, ka=0.5)
    history = [Dose(0.0, 1000.0, Route.ORAL), Dose(3.5996, 0.2436, Route.IV_BOLUS)]
    model = model_of(p)

    def reference_slope(t: float) -> float:
        return float(concentration_derivatives(p, history, [t])["slope_right"][0])

    t_dip = brentq(reference_slope, 3.5996 + 1e-9, 4.5, xtol=1e-15)
    t_peak = brentq(reference_slope, 4.5, 8.0, xtol=1e-15)
    assert 3.5996 < t_dip < t_peak
    peak = pk.cmax(model, history, 3.5986, 5003.5986)
    np.testing.assert_allclose(float(peak.time), t_peak, rtol=TMAX_RTOL)
    reference = expm_solution(p, history, [t_dip, t_peak]).concentration
    np.testing.assert_allclose(float(peak.value), reference[1], rtol=1e-12)
    low = pk.cmin(model, history, 3.5996, 5.0)
    np.testing.assert_allclose(float(low.time), t_dip, rtol=TMAX_RTOL)
    np.testing.assert_allclose(float(low.value), reference[0], rtol=1e-12)


def test_peak_of_an_infusion_and_of_a_bolus() -> None:
    cl, v, duration = 2.0, 20.0, 3.0
    model = pk.OneCompartment(cl, v)
    infusion = pk.cmax(model, [Dose(1.0, DOSE, Route.IV_INFUSION, duration)], 0.0, 24.0)
    assert float(infusion.time) == 1.0 + duration
    expected = DOSE / duration / cl * -math.expm1(-cl / v * duration)
    np.testing.assert_allclose(float(infusion.value), expected, rtol=1e-14)
    bolus = pk.cmax(model, [Dose(2.0, DOSE, Route.IV_BOLUS)], 0.0, 24.0)
    assert float(bolus.time) == 2.0
    np.testing.assert_allclose(float(bolus.value), DOSE / v, rtol=1e-15)


def test_steady_state_oral_peak_matches_closed_form() -> None:
    ka, k, v, f, tlag, tau, count = 1.1, 0.15, 30.0, 0.9, 0.5, 12.0, 30
    model = pk.OneCompartment(k * v, v, ka=ka, f=f, tlag=tlag)
    history = [Dose(i * tau, DOSE, Route.ORAL) for i in range(count)]
    start = (count - 1) * tau
    peak = pk.cmax(model, history, start, start + tau)
    # Steady-state Tmax after the dose (Gibaldi & Perrier form):
    # ln(ka (1 - exp(-k tau)) / (k (1 - exp(-ka tau)))) / (ka - k).
    t_ss = math.log(ka * -math.expm1(-k * tau) / (k * -math.expm1(-ka * tau))) / (ka - k)
    np.testing.assert_allclose(float(peak.time), start + tlag + t_ss, rtol=TMAX_RTOL)
    c_ss = f * DOSE / v * math.exp(-k * t_ss) / -math.expm1(-k * tau)
    np.testing.assert_allclose(float(peak.value), c_ss, rtol=CMAX_RTOL)


def textbook_two_compartment_oral(p: Parameters, t: float, derivative: bool = False) -> float:
    """Tri-exponential single-dose oral solution (float64, well-separated rates)."""
    k10, k12, k21 = p.cl / p.v1, p.q / p.v1, p.q / p.v2
    s = k10 + k12 + k21
    root = math.sqrt(s * s - 4.0 * k10 * k21)
    lam1, lam2, ka = (s - root) / 2.0, (s + root) / 2.0, p.ka
    tau = t - p.tlag
    terms = [
        ((k21 - lam1) / ((ka - lam1) * (lam2 - lam1)), lam1),
        ((k21 - lam2) / ((ka - lam2) * (lam1 - lam2)), lam2),
        ((k21 - ka) / ((lam1 - ka) * (lam2 - ka)), ka),
    ]
    scale = p.f * DOSE * ka / p.v1
    if derivative:
        return scale * sum(-rate * c * math.exp(-rate * tau) for c, rate in terms)
    return scale * sum(c * math.exp(-rate * tau) for c, rate in terms)


def test_two_compartment_peak_matches_root_of_textbook_derivative() -> None:
    p = Parameters(cl=3.0, v1=15.0, q=5.0, v2=40.0, ka=0.9, f=0.8, tlag=0.5)
    peak = pk.cmax(model_of(p), [Dose(0.0, DOSE, Route.ORAL)], 0.0, 48.0)
    t_ref = brentq(lambda t: textbook_two_compartment_oral(p, t, True), 0.6, 20.0, xtol=1e-15)
    np.testing.assert_allclose(float(peak.time), t_ref, rtol=TMAX_RTOL)
    np.testing.assert_allclose(
        float(peak.value), textbook_two_compartment_oral(p, t_ref), rtol=CMAX_RTOL
    )


def test_turning_point_just_before_a_dose_at_the_window_end() -> None:
    """At a dose at the window end the derivative's left limit is used.

    The second dose (at 9 h, the window end) starts absorption; the peak of the first
    dose lies just before it.
    """
    k, ka, v = 0.1, 1.0, 20.0
    model = pk.OneCompartment(k * v, v, ka=ka)
    history = [Dose(6.0, DOSE, Route.ORAL), Dose(9.0, DOSE, Route.ORAL)]
    t_peak = math.log(ka / k) / (ka - k)
    assert 8.0 < 6.0 + t_peak < 9.0
    peak = pk.cmax(model, history, 0.0, 9.0)
    np.testing.assert_allclose(float(peak.time), 6.0 + t_peak, rtol=TMAX_RTOL)
    np.testing.assert_allclose(float(peak.value), DOSE / v * math.exp(-k * t_peak), rtol=CMAX_RTOL)


def test_cmin_over_a_dosing_interval_is_the_pre_dose_trough() -> None:
    cl, v, tau, count = 2.5, 25.0, 12.0, 6
    k = cl / v
    model = pk.OneCompartment(cl, v)
    history = [Dose(i * tau, DOSE, Route.IV_BOLUS) for i in range(count + 1)]
    start, end = (count - 1) * tau, count * tau
    low = pk.cmin(model, history, start, end)
    expected = DOSE / v * math.exp(-k * tau) * math.expm1(-count * k * tau) / math.expm1(-k * tau)
    assert float(low.time) == end  # attained as the left limit at the next dose
    np.testing.assert_allclose(float(low.value), expected, rtol=1e-13)
    np.testing.assert_allclose(float(pk.trough(model, history, end)), expected, rtol=1e-13)
    # The window is closed and C is right-continuous: the bolus at `end` is part of
    # C(end), which is therefore the maximum over [start, end].
    peak = pk.cmax(model, history, start, end)
    assert float(peak.time) == end
    expected_peak = DOSE / v * math.expm1(-(count + 1) * k * tau) / math.expm1(-k * tau)
    np.testing.assert_allclose(float(peak.value), expected_peak, rtol=1e-13)


def test_interior_minimum_matches_closed_form() -> None:
    """Two compartments: a bolus declining while an infusion builds up.

    With ``C1(t) = c1 exp(-l1 t) + c2 exp(-l2 t)`` (per unit amount in v1), a bolus D
    and an infusion at rate R from t = 0 give
    ``dC/dt = sum_i c_i exp(-l_i t) (R - D l_i) / v1``, which for ``D l1 < R < D l2``
    vanishes once, at ``t* = ln(c2 (D l2 - R) / (c1 (R - D l1))) / (l2 - l1)``.
    """
    cl, v1, q, v2, dose, rate = 3.0, 15.0, 5.0, 40.0, 100.0, 20.0
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    root = math.sqrt(s * s - 4.0 * k10 * k21)
    l1, l2 = (s - root) / 2.0, (s + root) / 2.0
    c1, c2 = (k21 - l1) / (l2 - l1), (l2 - k21) / (l2 - l1)
    assert dose * l1 < rate < dose * l2
    t_star = math.log(c2 * (dose * l2 - rate) / (c1 * (rate - dose * l1))) / (l2 - l1)

    def textbook(t: float) -> float:
        bolus = dose * (c1 * math.exp(-l1 * t) + c2 * math.exp(-l2 * t))
        infusion = rate * (c1 * -math.expm1(-l1 * t) / l1 + c2 * -math.expm1(-l2 * t) / l2)
        return (bolus + infusion) / v1

    model = pk.TwoCompartment(cl, v1, q, v2)
    assert 4.0 < t_star < 5.0
    # With an infusion that ends at 5 h, just after the minimum, the window [0, 5] has
    # no event inside: the minimum is found only if the left derivative at the
    # infusion end (the window end) is (correctly) positive.
    for duration in (24.0, 5.0):
        history = [
            Dose(0.0, dose, Route.IV_BOLUS),
            Dose(0.0, rate * duration, Route.IV_INFUSION, duration),
        ]
        low = pk.cmin(model, history, 0.0, 5.0)
        np.testing.assert_allclose(float(low.time), t_star, rtol=TMAX_RTOL)
        np.testing.assert_allclose(float(low.value), textbook(t_star), rtol=CMAX_RTOL)


def test_left_limit_of_the_reduced_derivative_at_an_event() -> None:
    """A dip and a peak between a bolus (window start) and an infusion end (window end).

    The split between the two extrema is found only if the reduced derivative is taken
    on the left of the infusion end. Expected values: roots of the reference derivative.
    """
    p = Parameters(cl=1.0, v1=5.0, q=50.0, v2=50.0, ka=0.5)
    history = [
        Dose(0.0, 1000.0, Route.ORAL),
        Dose(3.5996, 0.26, Route.IV_BOLUS),
        Dose(-1.0, 0.18 * 8.0, Route.IV_INFUSION, duration=8.0),
    ]

    def reference_slope(t: float) -> float:
        return float(concentration_derivatives(p, history, [t])["slope_right"][0])

    t_dip = brentq(reference_slope, 3.6, 5.0, xtol=1e-15)
    t_peak = brentq(reference_slope, 5.0, 6.5, xtol=1e-15)
    peak = pk.cmax(model_of(p), history, 3.5996, 7.0)
    np.testing.assert_allclose(float(peak.time), t_peak, rtol=TMAX_RTOL)
    reference = expm_solution(p, history, [t_dip, t_peak]).concentration
    np.testing.assert_allclose(float(peak.value), reference[1], rtol=1e-12)
    assert reference[0] < reference[1]


def test_a_slope_of_exactly_zero_at_the_end_of_a_piece_counts_as_a_turn() -> None:
    from pkcore.metrics import _refine, _turning  # noqa: PLC0415 (private helpers)

    g_lo, g_hi = np.array([1.0, 1.0, -1.0]), np.array([0.0, -0.0, 0.0])
    lo, hi = np.zeros(3), np.ones(3)
    np.testing.assert_array_equal(_turning(g_lo, g_hi, lo, hi), [True, True, False])

    # g is positive before 0.3 and exactly zero after it (as after an underflow).
    def g(x: np.ndarray, _index: np.ndarray) -> np.ndarray:
        return np.where(x < 0.3, 1.0, 0.0)

    located = _refine(g, (np.array([0.0]), np.array([1.0])), (np.array([1.0]), np.array([0.0])))
    np.testing.assert_allclose(located, 0.3, rtol=1e-14)


@given(p=parameters(), history=doses(max_doses=5, horizon=36.0))
def test_extrema_bound_the_reference_curve(p: Parameters, history: list[Dose]) -> None:
    """Cmax and Cmin bound the reference curve, are attained, and are stationary.

    The reference (matrix exponential) is sampled at 2000 points plus every event
    time; Cmax must not be below any sample and Cmin not above any sample or left
    limit, both must be values of the reference curve, and an interior Tmax away from
    events must be a zero of the reference derivative.
    """
    assume(well_conditioned(p))
    start, end = 0.0, 48.0
    model = model_of(p)
    scale = concentration_scale(p, history)
    events = [d.time for d in history] + [d.time + p.tlag for d in history]
    events += [d.time + d.duration for d in history if d.duration is not None]
    grid = np.unique(np.concatenate([np.linspace(start, end, 2000), events]))
    grid = grid[(grid >= start) & (grid <= end)]
    reference = expm_solution(p, history, grid)
    tolerance = EXPM_ATOL * scale + EXPM_RTOL * float(np.max(reference.concentration))
    peak, low = pk.cmax(model, history, start, end), pk.cmin(model, history, start, end)
    assert float(peak.value) >= np.max(reference.concentration) - tolerance
    lowest = min(np.min(reference.concentration), np.min(reference.trough[grid > start]))
    assert float(low.value) <= lowest + tolerance
    at = expm_solution(p, history, [float(peak.time), float(low.time)])
    np.testing.assert_allclose(at.concentration[0], float(peak.value), rtol=1e-9, atol=tolerance)
    # the infimum is a value or (inside the window) a left limit of the reference curve
    attained = [at.concentration[1]] + ([at.trough[1]] if float(low.time) > start else [])
    assert min(abs(x - float(low.value)) for x in attained) <= 1e-9 * float(low.value) + tolerance
    t_peak = float(peak.time)
    if start < t_peak < end and min(abs(t_peak - e) for e in [*events, start]) > 1e-6:
        right, _, size = concentration_slopes(p, history, [t_peak])
        assert abs(right[0]) <= 1e-6 * size[0] + tolerance


@given(
    p=parameters(two_compartment=True),
    bolus_time=st.floats(0.05, 6.0),
    bolus_amount=log_uniform(1e-3, 50.0),
)
def test_extrema_with_a_bolus_during_absorption(
    p: Parameters, bolus_time: float, bolus_amount: float
) -> None:
    """Cmax/Cmin bound a dense sample of C even when two extrema follow one event.

    Consistency check (the dense sample is pkcore's verified concentration) plus an
    independent one: at an interior Tmax away from events the reference derivative
    vanishes.
    """
    history = [Dose(0.0, 1000.0, Route.ORAL), Dose(bolus_time, bolus_amount, Route.IV_BOLUS)]
    start, end = bolus_time, bolus_time + 2000.0
    model = model_of(p)
    t = np.concatenate([np.linspace(start, start + 50.0, 50001), np.linspace(start, end, 2001)])
    c = pk.concentration(model, history, t)
    tolerance = 1e-12 * float(np.max(c)) + 1e-300
    peak, low = pk.cmax(model, history, start, end), pk.cmin(model, history, start, end)
    assert float(peak.value) >= float(np.max(c)) - tolerance
    assert float(low.value) <= float(np.min(c)) + tolerance
    t_peak = float(peak.time)
    events = [start, end, p.tlag]
    if well_conditioned(p) and min(abs(t_peak - e) for e in events) > 1e-6:
        derivatives = concentration_derivatives(p, history, [t_peak])
        assert abs(derivatives["slope_right"][0]) <= 1e-6 * derivatives["slope_scale"][0]


def test_terminal_rates_and_half_lives() -> None:
    one = pk.OneCompartment(2.0, 20.0, ka=1.3)
    np.testing.assert_allclose(float(pk.terminal_rate(one)), 0.1, rtol=1e-15)
    np.testing.assert_allclose(float(pk.terminal_half_life(one)), math.log(2.0) / 0.1, rtol=1e-15)

    p = Parameters(cl=3.0, v1=15.0, q=5.0, v2=40.0, ka=0.01)
    two = model_of(p)
    k10, k12, k21 = p.cl / p.v1, p.q / p.v1, p.q / p.v2
    eigenvalues = np.linalg.eigvals(np.array([[-(k10 + k12), k21], [k12, -k21]]))
    slowest = float(np.min(-eigenvalues.real))
    np.testing.assert_allclose(float(pk.terminal_rate(two)), slowest, rtol=1e-13)
    np.testing.assert_allclose(
        float(pk.terminal_half_life(two)), math.log(2.0) / slowest, rtol=1e-13
    )
    # ka = 0.01 < lambda_z: flip-flop, the oral terminal phase follows ka.
    assert float(pk.apparent_oral_terminal_rate(two)) == 0.01
    np.testing.assert_allclose(
        float(pk.apparent_oral_terminal_half_life(two)), math.log(2.0) / 0.01, rtol=1e-15
    )


@pytest.mark.parametrize(
    ("ka", "t1"), [(0.002, 800.0), (2.0, 400.0)], ids=["flip-flop", "absorption-fast"]
)
def test_terminal_log_slope_of_the_reference_curve(ka: float, t1: float) -> None:
    """The late log-slope of an independent oral curve equals min(ka, lambda_z).

    lambda_z is about 0.04 per hour; at the chosen times every other exponential has
    decayed by more than 1e-13 relative to the terminal one.
    """
    p = Parameters(cl=3.0, v1=15.0, q=5.0, v2=40.0, ka=ka)
    t2 = t1 + 100.0
    curve = expm_solution(p, [Dose(0.0, DOSE, Route.ORAL)], [t1, t2]).concentration
    slope = math.log(curve[0] / curve[1]) / (t2 - t1)
    np.testing.assert_allclose(float(pk.apparent_oral_terminal_rate(model_of(p))), slope, rtol=1e-6)


def test_metrics_for_draws_match_scalar_models() -> None:
    draws = pk.OneCompartment(cl=[1.0, 2.0, 4.0], v=20.0, ka=[0.5, 1.3, 3.0], tlag=[0.0, 0.5, 1.0])
    history = [Dose(0.0, DOSE, Route.ORAL), Dose(12.0, DOSE, Route.ORAL)]
    peak = pk.cmax(draws, history, 0.0, 24.0)
    low = pk.cmin(draws, history, 6.0, 24.0)
    area = pk.auc(draws, history, 0.0, 24.0)
    assert peak.value.shape == peak.time.shape == low.value.shape == area.shape == (3,)
    for i, (cl, ka, tlag) in enumerate([(1.0, 0.5, 0.0), (2.0, 1.3, 0.5), (4.0, 3.0, 1.0)]):
        scalar = pk.OneCompartment(cl, 20.0, ka=ka, tlag=tlag)
        single = pk.cmax(scalar, history, 0.0, 24.0)
        np.testing.assert_allclose(peak.value[i], float(single.value), rtol=CMAX_RTOL)
        np.testing.assert_allclose(peak.time[i], float(single.time), rtol=TMAX_RTOL)
        np.testing.assert_allclose(
            low.value[i], float(pk.cmin(scalar, history, 6.0, 24.0).value), rtol=CMAX_RTOL
        )
        np.testing.assert_allclose(area[i], float(pk.auc(scalar, history, 0.0, 24.0)), rtol=1e-14)
