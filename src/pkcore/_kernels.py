"""Numerically stable kernels for sums of exponentials.

Every response computed by pkcore is assembled from the functions below. They take
non-negative rate constants and non-negative times as NumPy arrays that broadcast
together, and they are evaluated without subtractive cancellation:

``e1(r, t)``
    ``integral_0^t exp(-r s) ds = t * psi(r t)`` with ``psi(x) = (1 - exp(-x)) / x``,
    ``psi(0) = 1``: the response to a unit-rate input.
``dd(a, b, t)``
    ``integral_0^t exp(-a (t - s)) exp(-b s) ds = (exp(-a t) - exp(-b t)) / (b - a)``:
    the divided difference of the exponential that appears in the Bateman function.
``e2(a, b, t)``
    ``integral_0^t dd(a, b, s) ds``.
``conv3(a, b, c, t)``
    the convolution of three exponentials (``dd`` with a third rate).

The textbook expressions lose every significant digit when two rates nearly coincide
(for example ``ka ~ k``). Here ``dd`` is evaluated as
``t * exp(-min(a, b) t) * psi(|a - b| t)``, which is exact in the limit ``a == b``
(Garrett 1994, doi:10.1007/BF02353538) and well conditioned near it, because
``|a - b|`` is computed exactly for nearby floating-point numbers and ``psi`` is a
smooth function evaluated with ``expm1``. Second-order divided differences use a
Taylor series for small arguments and a cancellation-free closed form otherwise, the
strategy of McCurdy, Ng & Parlett (1984), doi:10.1090/S0025-5718-1984-0758198-0.

Large arguments are safe: no expression contains ``exp`` of a positive argument, and
``t * psi(r t)`` is evaluated as ``-expm1(-r t) / r``, which stays exact when ``r t``
overflows to infinity. Callers evaluate these kernels under
``np.errstate(all="ignore")`` and check the final result for finiteness.
"""

import math

import numpy as np

from pkcore._types import FloatArray

# The second divided difference of exp(-z) at {0, x, y} is evaluated by its Taylor
# series when max(x, y) < 1. With 20 terms the truncation error is below 1e-19.
_SERIES_BOUND = 1.0
_SERIES_COEFFICIENTS = tuple((-1.0) ** n / math.factorial(n + 2) for n in range(20))
_TINY = float(np.finfo(np.float64).tiny)


def e1(rate: FloatArray, t: FloatArray) -> FloatArray:
    """Return ``integral_0^t exp(-rate s) ds = t * psi(rate t)`` for ``rate >= 0``, ``t >= 0``.

    Evaluated as ``-expm1(-rate t) / rate``: one rounding in ``rate t``, ``expm1`` and
    the division, and exact when ``rate t`` overflows (``t = inf`` gives ``1 / rate``).
    When ``rate t`` is zero or subnormal, ``psi(rate t)`` is 1 to double precision and
    the result is ``t``.
    """
    x = rate * t
    normal = x >= _TINY
    return np.where(normal, -np.expm1(-x) / np.where(normal, rate, 1.0), t)


def dd(a: FloatArray, b: FloatArray, t: FloatArray) -> FloatArray:
    """Return ``integral_0^t exp(-a (t - s)) exp(-b s) ds`` for ``a, b >= 0``, finite ``t >= 0``.

    Equal to ``(exp(-a t) - exp(-b t)) / (b - a)`` for ``a != b`` and to
    ``t exp(-a t)`` for ``a == b``; symmetric in ``a`` and ``b``. Evaluated as
    ``exp(-min(a, b) t) * e1(|a - b|, t) = t * exp(-min(a, b) t) * psi(|a - b| t)``,
    a product of positive factors (no cancellation).
    """
    return np.exp(-np.minimum(a, b) * t) * e1(np.abs(a - b), t)


def _second_divided_difference_series(x: FloatArray, y: FloatArray) -> FloatArray:
    """Return the second divided difference of ``exp(-z)`` at ``{0, x, y}``, ``0 <= x <= y < 1``.

    Uses ``sum_n (-1)^n h_n(x, y) / (n + 2)!``, where ``h_n(x, y) = sum_i x^i y^(n - i)``
    is the complete homogeneous symmetric polynomial. The series alternates, but the sum
    of the absolute values of its terms is at most about 4 times the result for
    ``y < 1``, so rounding errors stay at a few units in the last place.
    """
    h = np.ones_like(y)
    x_power = np.ones_like(x)
    total = _SERIES_COEFFICIENTS[0] * h
    for coefficient in _SERIES_COEFFICIENTS[1:]:
        x_power = x_power * x
        h = y * h + x_power
        total = total + coefficient * h
    return total


def e2(a: FloatArray, b: FloatArray, t: FloatArray) -> FloatArray:
    """Return ``integral_0^t dd(a, b, s) ds`` for ``a, b >= 0`` and ``0 <= t <= inf``.

    With ``lo = min(a, b)`` and ``hi = max(a, b)``: for ``hi t < 1`` the value is
    ``t^2`` times a Taylor series; otherwise it is
    ``(e1(lo, t) - exp(-lo t) e1(hi - lo, t)) / hi``, where in that region the
    subtracted term is at most about 0.64 of the first one, so the subtraction costs
    at most about 1.5 bits. ``t = inf`` requires ``a, b > 0`` and gives ``1 / (a b)``.
    """
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    finite = np.isfinite(t)
    t_fin = np.where(finite, t, 0.0)
    series = hi * t_fin < _SERIES_BOUND
    t_near = np.where(series, t_fin, 0.0)
    near = t_near * t_near * _second_divided_difference_series(lo * t_near, hi * t_near)
    far = (e1(lo, t_fin) - np.exp(-lo * t_fin) * e1(hi - lo, t_fin)) / np.where(series, 1.0, hi)
    tail = 1.0 / np.where(finite, 1.0, lo * hi)
    return np.where(finite, np.where(series, near, far), tail)


def conv3(a: FloatArray, b: FloatArray, c: FloatArray, t: FloatArray) -> FloatArray:
    """Return the convolution of ``exp(-a s)``, ``exp(-b s)`` and ``exp(-c s)`` at finite ``t``.

    Shifting all rates by their minimum ``lo`` turns the convolution into
    ``exp(-lo t) * e2(mid - lo, hi - lo, t)``; the order statistics are selected
    exactly (no arithmetic), so the shifted rates carry no rounding error beyond
    one subtraction.
    """
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    mid = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
    return np.exp(-lo * t) * e2(mid - lo, hi - lo, t)
