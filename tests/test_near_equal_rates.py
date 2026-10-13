"""Absorption rate equal or nearly equal to a disposition rate constant (ka ~ k).

The textbook solutions divide by ``ka - lambda`` and lose all accuracy as the two
rates approach each other; pkcore's closed forms must not. Expected values are the
textbook formulas evaluated in 60-digit decimal arithmetic (cancellation is then
harmless), Garrett's (1994) limit ``C = F D k t exp(-k t) / V`` for ``ka == k``, and
identities. Relative gaps of 1e-14, 1e-12 and 1e-6 are tested on both sides.

Tolerance ``rtol = 1e-13``: the only error amplification left is the conditioning of
``exp(-k t)`` (about ``k t`` units in the last place, ``k t <= 25`` here).
"""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

import pkcore as pk
from pk_strategies import administered, model_of
from pkcore import Dose, Route
from reference import (
    Parameters,
    decimal_oral_auc,
    decimal_oral_concentration,
    decimal_oral_peripheral_amount,
)

GAPS = [0.0, 1e-14, -1e-14, 1e-12, -1e-12, 1e-6, -1e-6]
TIMES = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 48.0, 96.0, 200.0])
DOSE = 100.0
RTOL = 1e-13

# k = cl / v = 0.125 exactly in binary floating point.
CL, V, F, TLAG = 1.0, 8.0, 0.6, 0.5
K = CL / V


def garrett_limit(t: float) -> float:
    """``F D k (t - tlag) exp(-k (t - tlag)) / V`` (Garrett 1994), in decimal arithmetic."""
    with localcontext() as ctx:
        ctx.prec = 60
        k, tau = Decimal(K), Decimal(t) - Decimal(TLAG)
        if tau <= 0:
            return 0.0
        return float(Decimal(F) * Decimal(DOSE) * k * tau * (-k * tau).exp() / Decimal(V))


def relative_change(values: np.ndarray, base: np.ndarray) -> np.ndarray:
    """``|values / base - 1|`` where ``base > 0`` and ``|values - base|`` elsewhere."""
    positive = base > 0.0
    return np.where(
        positive, np.abs(values / np.where(positive, base, 1.0) - 1.0), abs(values - base)
    )


def test_equal_rates_reproduce_the_garrett_limit() -> None:
    assert K == 0.125
    model = pk.OneCompartment(CL, V, ka=K, f=F, tlag=TLAG)
    c = pk.concentration(model, [Dose(0.0, DOSE, Route.ORAL)], TIMES)
    np.testing.assert_allclose(c, [garrett_limit(t) for t in TIMES], rtol=RTOL)


@pytest.mark.parametrize("gap", GAPS)
def test_one_compartment_near_equal_rates(gap: float) -> None:
    ka = K * (1.0 + gap)
    p = Parameters(cl=CL, v1=V, ka=ka, f=F, tlag=TLAG)
    model = model_of(p)
    history = [Dose(0.0, DOSE, Route.ORAL)]
    c = pk.concentration(model, history, TIMES)
    np.testing.assert_allclose(
        c, [decimal_oral_concentration(p, DOSE, t) for t in TIMES], rtol=RTOL
    )
    # Agreement with the limit formula, to within the true first-order change:
    # |d log C / d log ka| <= 1 + ka (t - tlag).
    limit = np.array([garrett_limit(t) for t in TIMES])
    bound = abs(gap) * (1.0 + K * (TIMES - TLAG)) * 1.01 + RTOL
    assert np.all(relative_change(c, limit) <= bound)
    auc = [float(pk.auc(model, history, 0.0, t)) for t in TIMES]
    np.testing.assert_allclose(auc, [decimal_oral_auc(p, DOSE, t) for t in TIMES], rtol=RTOL)
    np.testing.assert_allclose(float(pk.auc(model, history, 0.0, math.inf)), F * DOSE / CL)


def test_one_compartment_is_continuous_through_equality() -> None:
    """C(ka) is smooth: tiny changes of ka give proportionally tiny changes of C."""
    history = [Dose(0.0, DOSE, Route.ORAL)]
    at_equality = pk.concentration(pk.OneCompartment(CL, V, ka=K, f=F, tlag=TLAG), history, TIMES)
    for step in range(-40, 41):
        ka = K + step * math.ulp(K)
        c = pk.concentration(pk.OneCompartment(CL, V, ka=ka, f=F, tlag=TLAG), history, TIMES)
        gap = abs(ka / K - 1.0)
        bound = gap * (1.0 + K * (TIMES - TLAG)) * 1.01 + 1e-14
        assert np.all(relative_change(c, at_equality) <= bound)


def test_textbook_formula_in_float64_fails_where_the_engine_does_not() -> None:
    """Sanity check that the decimal comparison above is sensitive to the problem."""
    ka = K * (1.0 + 1e-12)
    p = Parameters(cl=CL, v1=V, ka=ka, f=F, tlag=0.0)
    t = np.array([1.0, 4.0, 16.0])
    naive = F * DOSE * ka / (V * (ka - K)) * (np.exp(-K * t) - np.exp(-ka * t))
    reference = np.array([decimal_oral_concentration(p, DOSE, x) for x in t])
    engine = pk.concentration(model_of(p), [Dose(0.0, DOSE, Route.ORAL)], t)
    assert np.max(np.abs(naive / reference - 1.0)) > 1e-6
    assert np.max(np.abs(engine / reference - 1.0)) < RTOL


TWO = Parameters(cl=3.0, v1=15.0, q=5.0, v2=40.0, f=0.8, tlag=0.25)


def disposition_rates(p: Parameters) -> list[float]:
    """Eigenvalues of the two-compartment rate matrix, from LAPACK (independent)."""
    k10, k12, k21 = p.cl / p.v1, p.q / p.v1, p.q / p.v2
    matrix = np.array([[-(k10 + k12), k21], [k12, -k21]])
    return sorted(float(-e) for e in np.linalg.eigvals(matrix).real)


@pytest.mark.parametrize("which", [0, 1], ids=["ka~lambda1", "ka~lambda2"])
@pytest.mark.parametrize("gap", GAPS)
def test_two_compartment_near_equal_rates(which: int, gap: float) -> None:
    rate = disposition_rates(TWO)[which]
    p = Parameters(TWO.cl, TWO.v1, TWO.q, TWO.v2, ka=rate * (1.0 + gap), f=TWO.f, tlag=TWO.tlag)
    model = model_of(p)
    history = [Dose(0.0, DOSE, Route.ORAL)]
    c = pk.concentration(model, history, TIMES)
    np.testing.assert_allclose(
        c, [decimal_oral_concentration(p, DOSE, t) for t in TIMES], rtol=1e-12
    )
    amounts = pk.amounts(model, history, TIMES)
    np.testing.assert_allclose(
        amounts.peripheral,
        [decimal_oral_peripheral_amount(p, DOSE, t) for t in TIMES],
        rtol=1e-12,
    )
    total = amounts.depot + amounts.central + amounts.peripheral + amounts.eliminated
    expected = [administered(p, history, t) for t in TIMES]
    np.testing.assert_allclose(total, expected, rtol=1e-12)
    np.testing.assert_allclose(
        float(pk.auc(model, history, -math.inf, math.inf)), p.f * DOSE / p.cl, rtol=1e-13
    )
