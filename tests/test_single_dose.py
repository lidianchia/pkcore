"""Single doses of every route: the engine against independent solutions.

Expected values come from numerical ODE integration (Radau), matrix-exponential
propagation and, for one compartment, the textbook closed forms in float64 with
well-separated rate constants. Tolerances are explained in ``pk_strategies``.
"""

import numpy as np
import pytest

import pkcore as pk
from pk_strategies import EXPM_ATOL, EXPM_RTOL, ODE_ATOL, ODE_RTOL, model_of
from pkcore import Dose, Route
from reference import Parameters, expm_solution, ode_solution

DOSE = 100.0
DURATION = 3.0
MODELS = {
    "one-compartment": Parameters(cl=2.0, v1=20.0, ka=1.3, f=0.7, tlag=0.25),
    "two-compartment": Parameters(cl=3.0, v1=15.0, q=5.0, v2=40.0, ka=0.9, f=0.8, tlag=0.5),
}
# Includes the dose time, the end of the lag times and the end of the infusion.
TIMES = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0, 96.0])


def single(route: Route) -> Dose:
    return Dose(0.0, DOSE, route, DURATION if route is Route.IV_INFUSION else None)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("route", list(Route))
def test_concentration_matches_ode_and_expm(name: str, route: Route) -> None:
    p = MODELS[name]
    history = [single(route)]
    model = model_of(p)
    scale = DOSE / p.v1
    c = pk.concentration(model, history, TIMES)
    ode = ode_solution(p, history, TIMES)
    exact = expm_solution(p, history, TIMES)
    np.testing.assert_allclose(c, ode.concentration, rtol=ODE_RTOL, atol=ODE_ATOL * scale)
    np.testing.assert_allclose(c, exact.concentration, rtol=EXPM_RTOL, atol=EXPM_ATOL * scale)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("route", list(Route))
def test_amounts_and_auc_match_expm(name: str, route: Route) -> None:
    p = MODELS[name]
    history = [single(route)]
    model = model_of(p)
    exact = expm_solution(p, history, TIMES)
    amounts = pk.amounts(model, history, TIMES)
    for state in ("depot", "central", "peripheral", "eliminated"):
        np.testing.assert_allclose(
            getattr(amounts, state),
            exact.right[state],
            rtol=EXPM_RTOL,
            atol=EXPM_ATOL * DOSE,
            err_msg=state,
        )
    auc = np.array([float(pk.auc(model, history, 0.0, t)) for t in TIMES])
    np.testing.assert_allclose(
        auc, exact.right["auc"], rtol=EXPM_RTOL, atol=EXPM_ATOL * DOSE / p.v1 * TIMES[-1]
    )


def test_one_compartment_textbook_formulas() -> None:
    cl, v, ka, f, tlag = 2.0, 20.0, 1.3, 0.7, 0.25
    k = cl / v
    model = pk.OneCompartment(cl, v, ka=ka, f=f, tlag=tlag)
    t = TIMES

    bolus = pk.concentration(model, [single(Route.IV_BOLUS)], t)
    np.testing.assert_allclose(bolus, DOSE / v * np.exp(-k * t), rtol=1e-14)

    rate = DOSE / DURATION
    infusion = pk.concentration(model, [single(Route.IV_INFUSION)], t)
    during = rate / cl * (1.0 - np.exp(-k * t))
    after = rate / cl * (1.0 - np.exp(-k * DURATION)) * np.exp(-k * (t - DURATION))
    np.testing.assert_allclose(infusion, np.where(t <= DURATION, during, after), rtol=1e-13)

    # Bateman function; ka and k differ by a factor 13, so float64 cancellation costs
    # at most about two digits here.
    oral = pk.concentration(model, [single(Route.ORAL)], t)
    tau = np.maximum(t - tlag, 0.0)
    bateman = f * DOSE * ka / (v * (ka - k)) * (np.exp(-k * tau) - np.exp(-ka * tau))
    np.testing.assert_allclose(oral, bateman, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name", MODELS)
def test_bolus_is_included_at_its_own_time(name: str) -> None:
    p = MODELS[name]
    model = model_of(p)
    history = [Dose(5.0, DOSE, Route.IV_BOLUS)]
    at_dose = pk.concentration(model, history, [5.0])
    assert at_dose[0] == pytest.approx(DOSE / p.v1, rel=1e-15)
    assert pk.trough(model, history, [5.0])[0] == 0.0
    assert pk.concentration(model, history, [np.nextafter(5.0, 0.0)])[0] == 0.0


@pytest.mark.parametrize("name", MODELS)
def test_oral_dose_starts_after_the_lag_time(name: str) -> None:
    p = MODELS[name]
    model = model_of(p)
    history = [Dose(2.0, DOSE, Route.ORAL)]
    before = np.array([0.0, 2.0, 2.0 + p.tlag / 2, 2.0 + p.tlag])
    assert np.all(pk.concentration(model, history, before) == 0.0)
    assert np.all(pk.amounts(model, history, before[:-1]).depot == 0.0)
    after = pk.amounts(model, history, [2.0 + p.tlag]).depot
    assert after[0] == pytest.approx(p.f * DOSE, rel=1e-15)
    assert pk.concentration(model, history, [2.0 + p.tlag + 1e-3])[0] > 0.0


def test_vectorised_draws_match_scalar_simulations() -> None:
    draws = pk.TwoCompartment(
        cl=[1.0, 3.0, 8.0], v1=[10.0, 15.0, 30.0], q=5.0, v2=[20.0, 40.0, 80.0], ka=0.9, f=0.8
    )
    history = [single(route) for route in Route]
    c = pk.concentration(draws, history, TIMES)
    assert c.shape == (3, TIMES.size)
    for i, (cl, v1, v2) in enumerate([(1.0, 10.0, 20.0), (3.0, 15.0, 40.0), (8.0, 30.0, 80.0)]):
        scalar = pk.TwoCompartment(cl, v1, 5.0, v2, ka=0.9, f=0.8)
        np.testing.assert_allclose(c[i], pk.concentration(scalar, history, TIMES), rtol=1e-15)
