"""Accuracy of the numerical kernels against 120-digit decimal arithmetic.

The references are the textbook (cancellation-prone) formulas, evaluated with
``decimal`` at 120 significant digits so that their cancellation is harmless. The
inputs are the exact binary values of the float64 arguments. Times are drawn from
``{0} U [1e-9, 1e4]``: for even smaller times the references would need still more
digits (the kernels' behaviour there is covered by explicit tests below).

Tolerances: ``e1`` and ``e2`` are well conditioned (relative condition number about
1), so they must agree to a few units in the last place (``1e-15`` and ``4e-15``). Any
value containing ``exp(-x)`` inherits that function's condition number ``x``: the
rounding of ``x = rate * t`` alone changes the result by up to ``x * 1.1e-16``
relative. Those comparisons therefore use ``rtol = 4e-16 * (1 + x)``.
"""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pkcore._kernels import conv3, dd, e1, e2

EPS = float(np.finfo(np.float64).eps)
_PRECISION = 120


def _d(x: float) -> Decimal:
    return Decimal(x)


def ref_e1(rate: float, t: float) -> float:
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        r, tt = _d(rate), _d(t)
        return float(tt if r == 0 else (1 - (-r * tt).exp()) / r)


def ref_dd(a: float, b: float, t: float) -> float:
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        da, db, dt = _d(a), _d(b), _d(t)
        if da == db:
            return float(dt * (-da * dt).exp())
        return float(((-da * dt).exp() - (-db * dt).exp()) / (db - da))


def _e1_dec(r: Decimal, t: Decimal) -> Decimal:
    return t if r == 0 else (1 - (-r * t).exp()) / r


def ref_e2(a: float, b: float, t: float) -> float:
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        da, db, dt = _d(a), _d(b), _d(t)
        if da == db:
            if da == 0:
                return float(dt * dt / 2)
            return float((1 - (-da * dt).exp() * (1 + da * dt)) / (da * da))
        return float((_e1_dec(da, dt) - _e1_dec(db, dt)) / (db - da))


def ref_conv3(a: float, b: float, c: float, t: float) -> float:
    """Second divided difference of exp(-z t) at {a, b, c} (confluent cases included)."""
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        lo, mid, hi = sorted((_d(a), _d(b), _d(c)))
        dt = _d(t)

        def h(z: Decimal) -> Decimal:
            return (-z * dt).exp()

        def first(x: Decimal, y: Decimal) -> Decimal:
            return -dt * h(x) if x == y else (h(y) - h(x)) / (y - x)

        if lo == hi:
            return float(dt * dt / 2 * h(lo))
        if lo == mid:
            return float((first(lo, hi) - first(lo, lo)) / (hi - lo))
        return float((first(mid, hi) - first(lo, mid)) / (hi - lo))


def _arr(*xs: float) -> tuple[np.ndarray, ...]:
    return tuple(np.array(x, dtype=float) for x in xs)


rates = st.one_of(st.just(0.0), st.floats(1e-9, 1e9, allow_subnormal=False))
positive_rates = st.floats(1e-6, 1e6, allow_subnormal=False)
durations = st.one_of(st.just(0.0), st.floats(1e-9, 1e4))
relative_gaps = st.sampled_from([0.0, 1e-16, 1e-15, 1e-14, 1e-12, 1e-9, 1e-6, 1e-3, 0.1, 1.0])


@given(rate=rates, t=durations)
def test_e1_matches_decimal(rate: float, t: float) -> None:
    value = float(e1(*_arr(rate, t)))
    np.testing.assert_allclose(value, ref_e1(rate, t), rtol=1e-15, atol=0.0)


@given(a=positive_rates, gap=relative_gaps, sign=st.sampled_from([-1.0, 1.0]), t=durations)
def test_dd_matches_decimal_including_nearly_equal_rates(
    a: float, gap: float, sign: float, t: float
) -> None:
    b = a * (1.0 + sign * gap)
    value = float(dd(*_arr(a, b, t)))
    exponent = min(a, b) * t
    expected = ref_dd(a, b, t)
    if expected < 1e-300:  # below the float64 normal range: only require an absolute match
        assert value <= 1e-290
        return
    np.testing.assert_allclose(value, expected, rtol=4 * EPS * (1.0 + exponent), atol=0.0)


@given(a=positive_rates, t=durations)
def test_dd_at_equal_rates_is_the_garrett_limit(a: float, t: float) -> None:
    value = float(dd(*_arr(a, a, t)))
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        limit = float(_d(t) * (-_d(a) * _d(t)).exp())
    if limit < 1e-300:
        assert value <= 1e-290
        return
    np.testing.assert_allclose(value, limit, rtol=4 * EPS * (1.0 + a * t), atol=0.0)


@given(a=positive_rates, b=positive_rates, t=durations)
def test_dd_is_symmetric_bit_for_bit(a: float, b: float, t: float) -> None:
    assert float(dd(*_arr(a, b, t))) == float(dd(*_arr(b, a, t)))


@given(a=rates, gap=relative_gaps, t=durations)
def test_e2_matches_decimal(a: float, gap: float, t: float) -> None:
    b = a * (1.0 + gap)
    value = float(e2(*_arr(a, b, t)))
    expected = ref_e2(a, b, t)
    np.testing.assert_allclose(value, expected, rtol=4e-15 * (1.0 + min(a, b) * t), atol=0.0)


@pytest.mark.parametrize("rate", [1e-6, 0.37, 1.0, 5.0, 2e3])
@pytest.mark.parametrize("ratio", [0.0, 1e-12, 0.5, 1.0])
def test_e2_is_continuous_across_the_series_threshold(rate: float, ratio: float) -> None:
    # The Taylor branch is used for max(a, b) * t < 1 and the closed form above it.
    t_switch = 1.0 / rate
    for t in (np.nextafter(t_switch, 0.0), t_switch, np.nextafter(t_switch, np.inf)):
        value = float(e2(*_arr(ratio * rate, rate, float(t))))
        np.testing.assert_allclose(value, ref_e2(ratio * rate, rate, float(t)), rtol=4e-15)


@given(a=positive_rates, b=positive_rates)
def test_e2_to_infinity(a: float, b: float) -> None:
    value = float(e2(*_arr(a, b, math.inf)))
    np.testing.assert_allclose(value, 1.0 / (a * b), rtol=2 * EPS)


def test_e1_to_infinity_and_at_zero() -> None:
    assert float(e1(*_arr(0.25, math.inf))) == 4.0
    assert float(e1(*_arr(0.0, 3.0))) == 3.0
    assert float(e1(*_arr(5.0, 0.0))) == 0.0
    # rate * t is subnormal: psi is 1 to double precision and the value is t itself.
    assert float(e1(*_arr(1e-300, 1e-10))) == 1e-10
    # tiny times: e1 = t (1 - rate t / 2 + ...) = t to double precision
    assert float(e1(*_arr(3.0, 1e-300))) == 1e-300
    assert float(dd(*_arr(3.0, 3.0000000000000004, 1e-300))) == 1e-300
    assert float(e2(*_arr(3.0, 7.0, 1e-150))) == 0.5e-300


def test_kernels_stay_exact_when_rate_times_t_overflows() -> None:
    with np.errstate(over="ignore"):
        assert float(e1(*_arr(1e300, 1e10))) == 1e-300
        np.testing.assert_allclose(float(dd(*_arr(1e-3, 1e300, 1e10))), 0.0, atol=1e-300)
        assert math.isfinite(float(e2(*_arr(1.0, 1e300, 1e10))))


@given(
    a=positive_rates,
    gap=relative_gaps,
    factor=st.sampled_from([1.0, 1.0 - 1e-14, 1.0 + 1e-9, 0.5, 2.0, 10.0]),
    t=durations,
)
def test_conv3_matches_decimal(a: float, gap: float, factor: float, t: float) -> None:
    b, c = a * (1.0 + gap), a * factor
    value = float(conv3(*_arr(a, b, c, t)))
    expected = ref_conv3(a, b, c, t)
    if expected < 1e-300:
        assert value <= 1e-290
        return
    np.testing.assert_allclose(value, expected, rtol=8 * EPS * (1.0 + min(a, b, c) * t), atol=0.0)


def test_conv3_is_symmetric() -> None:
    args = (0.3, 1.7, 0.30000000000001, 2.5)
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    values = {float(conv3(*_arr(*(args[i] for i in perm), args[3]))) for perm in perms}
    assert len(values) == 1
