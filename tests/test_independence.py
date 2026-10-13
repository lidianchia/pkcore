"""Simulations of different drugs are independent.

pkcore simulates one model (one drug) per call. It keeps no state between calls and
has no function that combines the concentrations of different drugs, so simulating
drug A gives bit-identical results whether or not drug B was simulated before, after
or concurrently. These are consistency checks: they compare pkcore with itself.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import pkcore as pk
from pk_strategies import doses, model_of, observation_times, parameters
from pkcore import Dose, Route
from reference import Parameters

DRUG_A = pk.OneCompartment(cl=2.0, v=20.0, ka=1.3, f=0.7, tlag=0.25)
DOSES_A = [Dose(0.0, 100.0, Route.ORAL), Dose(12.0, 100.0, Route.ORAL)]
DRUG_B = pk.TwoCompartment(cl=3.0, v1=15.0, q=5.0, v2=40.0, ka=0.9)
DOSES_B = [Dose(1.0, 250.0, Route.IV_INFUSION, duration=0.5), Dose(9.0, 50.0, Route.IV_BOLUS)]
TIMES = np.linspace(0.0, 48.0, 97)


def everything(model: pk.Model, history: list[Dose]) -> list[np.ndarray]:
    """Every result pkcore computes for one drug."""
    amounts = pk.amounts(model, history, TIMES)
    peak = pk.cmax(model, history, 0.0, 48.0)
    low = pk.cmin(model, history, 12.0, 24.0)
    return [
        pk.concentration(model, history, TIMES),
        pk.trough(model, history, TIMES),
        amounts.depot,
        amounts.central,
        amounts.peripheral,
        amounts.eliminated,
        pk.auc(model, history, 0.0, 48.0),
        peak.value,
        peak.time,
        low.value,
        low.time,
    ]


def assert_bit_identical(first: list[np.ndarray], second: list[np.ndarray]) -> None:
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert a.tobytes() == b.tobytes()


def test_drug_a_does_not_depend_on_drug_b() -> None:
    alone_a = everything(DRUG_A, DOSES_A)
    alone_b = everything(DRUG_B, DOSES_B)
    # Interleave with B, failed calls and other models; then repeat A and B.
    for _ in range(3):
        everything(DRUG_B, DOSES_B)
        with pytest.raises(ValueError, match="finite"):
            pk.concentration(DRUG_A, [Dose(0.0, 1.0, Route.ORAL)], [np.nan])
        pk.concentration(pk.OneCompartment(cl=[1.0, 2.0], v=5.0), DOSES_B, TIMES)
    assert_bit_identical(everything(DRUG_A, DOSES_A), alone_a)
    assert_bit_identical(everything(DRUG_B, DOSES_B), alone_b)


def test_concurrent_simulations_match_sequential_ones() -> None:
    expected = {"A": everything(DRUG_A, DOSES_A), "B": everything(DRUG_B, DOSES_B)}
    jobs = ["A", "B"] * 8
    cases: dict[str, tuple[pk.Model, list[Dose]]] = {"A": (DRUG_A, DOSES_A), "B": (DRUG_B, DOSES_B)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda name: (name, everything(*cases[name])), jobs))
    for name, result in results:
        assert_bit_identical(result, expected[name])


def test_inputs_are_not_modified_and_outputs_are_fresh() -> None:
    times = TIMES.copy()
    history = list(DOSES_A)
    first = pk.concentration(DRUG_A, history, times)
    first[:] = -1.0  # mutating a result must not affect later calls
    second = pk.concentration(DRUG_A, history, times)
    assert np.all(second >= 0.0)
    np.testing.assert_array_equal(times, TIMES)
    assert history == DOSES_A
    assert np.all(DRUG_A.cl == 2.0)


@given(
    pa=parameters(),
    pb=parameters(),
    history_a=doses(),
    history_b=doses(),
    times=observation_times(),
    b_first=st.booleans(),
)
def test_random_drug_pairs_are_independent(
    *,
    pa: Parameters,
    pb: Parameters,
    history_a: list[Dose],
    history_b: list[Dose],
    times: list[float],
    b_first: bool,
) -> None:
    model_a, model_b = model_of(pa), model_of(pb)
    alone = pk.concentration(model_a, history_a, times)
    if b_first:
        pk.concentration(model_b, history_b, times)
    again = pk.concentration(model_a, history_a, times)
    pk.concentration(model_b, history_b, times)
    assert alone.tobytes() == again.tobytes()
    assert alone.tobytes() == pk.concentration(model_a, history_a, times).tobytes()


def test_draw_rows_are_independent() -> None:
    """Row i of a vectorised simulation equals the scalar simulation of draw i."""
    draws = pk.TwoCompartment(cl=[3.0, 30.0], v1=[15.0, 1.5], q=[5.0, 0.05], v2=40.0, ka=[0.9, 9.0])
    history = [*DOSES_A, *DOSES_B]
    together = pk.concentration(draws, history, TIMES)
    for i, (cl, v1, q, ka) in enumerate([(3.0, 15.0, 5.0, 0.9), (30.0, 1.5, 0.05, 9.0)]):
        alone = pk.concentration(pk.TwoCompartment(cl, v1, q, 40.0, ka=ka), history, TIMES)
        np.testing.assert_allclose(together[i], alone, rtol=1e-15, atol=0.0)
