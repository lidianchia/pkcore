"""No dosing events (or none that has started): every output is exactly zero."""

import numpy as np
import pytest

import pkcore as pk
from pkcore import Dose, Route

MODELS: list[pk.Model] = [
    pk.OneCompartment(cl=2.0, v=20.0, ka=1.0),
    pk.TwoCompartment(cl=3.0, v1=15.0, q=5.0, v2=40.0, ka=0.9),
    pk.TwoCompartment(cl=[3.0, 4.0, 5.0], v1=15.0, q=5.0, v2=40.0, ka=0.9),
]
TIMES = np.array([-10.0, 0.0, 1.0, 24.0, 1e4])


@pytest.mark.parametrize("model", MODELS)
def test_empty_history_gives_zeros_everywhere(model: pk.Model) -> None:
    draws = () if model.n_draws is None else (model.n_draws,)
    c = pk.concentration(model, [], TIMES)
    assert c.shape == (*draws, TIMES.size)
    assert np.all(c == 0.0)
    assert np.all(pk.trough(model, [], TIMES) == 0.0)
    amounts = pk.amounts(model, [], TIMES)
    for state in (amounts.depot, amounts.central, amounts.peripheral, amounts.eliminated):
        assert state.shape == c.shape
        assert np.all(state == 0.0)
    assert np.all(pk.auc(model, [], -np.inf, np.inf) == 0.0)
    for metric in (pk.cmax, pk.cmin):
        extremum = metric(model, [], 0.0, 24.0)
        assert extremum.value.shape == draws
        assert np.all(extremum.value == 0.0)
        assert np.all(extremum.time == 0.0)  # earliest time at which 0 is attained
    bands = pk.percentile_bands(model, [], TIMES)
    assert np.all(bands.values == 0.0)


def test_scalar_time_gives_scalar_shaped_output() -> None:
    assert pk.concentration(MODELS[0], [], 3.0).shape == ()
    assert pk.concentration(MODELS[2], [], 3.0).shape == (3,)


def test_doses_after_the_observations_contribute_nothing() -> None:
    later = [
        Dose(50.0, 100.0, Route.IV_BOLUS),
        Dose(50.0, 100.0, Route.IV_INFUSION, duration=2.0),
        Dose(49.0, 100.0, Route.ORAL),
    ]
    for model in MODELS:
        assert np.all(pk.concentration(model, later, TIMES[:4]) == 0.0)
        assert np.all(pk.auc(model, later, -np.inf, 24.0) == 0.0)


def test_zero_amounts_and_zero_bioavailability_contribute_nothing() -> None:
    model = pk.TwoCompartment(cl=3.0, v1=15.0, q=5.0, v2=40.0, ka=0.9, f=0.0)
    history = [
        Dose(0.0, 0.0, Route.IV_BOLUS),
        Dose(0.0, 0.0, Route.IV_INFUSION, duration=1.0),
        Dose(0.0, 500.0, Route.ORAL),  # f = 0: nothing is absorbed
    ]
    assert np.all(pk.concentration(model, history, TIMES) == 0.0)
    assert np.all(pk.amounts(model, history, TIMES).depot == 0.0)
