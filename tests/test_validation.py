"""Invalid inputs raise clear exceptions; valid objects are immutable."""

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pkcore as pk
from pkcore import Dose, Route

BOLUS = [Dose(0.0, 100.0, Route.IV_BOLUS)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cl": 0.0, "v": 10.0}, "cl must be finite and > 0"),
        ({"cl": -1.0, "v": 10.0}, "cl must be finite and > 0"),
        ({"cl": math.nan, "v": 10.0}, "cl must be finite and > 0"),
        ({"cl": math.inf, "v": 10.0}, "cl must be finite and > 0"),
        ({"cl": 1.0, "v": 0.0}, "v must be finite and > 0"),
        ({"cl": 1.0, "v": -5.0}, "v must be finite and > 0"),
        ({"cl": 1.0, "v": 10.0, "ka": 0.0}, "ka must be finite and > 0"),
        ({"cl": 1.0, "v": 10.0, "ka": -0.5}, "ka must be finite and > 0"),
        ({"cl": 1.0, "v": 10.0, "ka": math.nan}, "ka must be finite and > 0"),
        ({"cl": 1.0, "v": 10.0, "f": -0.1}, r"f must be finite and >= 0"),
        ({"cl": 1.0, "v": 10.0, "f": 1.1}, r"f must be in \[0, 1\]"),
        ({"cl": 1.0, "v": 10.0, "f": math.nan}, "f must be finite"),
        ({"cl": 1.0, "v": 10.0, "tlag": -0.1}, "tlag must be finite and >= 0"),
        ({"cl": 1.0, "v": 10.0, "tlag": math.inf}, "tlag must be finite and >= 0"),
        ({"cl": [1.0, -2.0], "v": 10.0}, "cl must be finite and > 0; got -2.0 at index 1"),
        ({"cl": [1.0, 2.0], "v": [10.0, 20.0, 30.0]}, "must have the same length"),
        ({"cl": [[1.0]], "v": 10.0}, "1-D array of parameter draws"),
        ({"cl": [], "v": 10.0}, "cl must not be empty"),
    ],
)
def test_one_compartment_rejects_invalid_parameters(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pk.OneCompartment(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"q": 0.0}, "q must be finite and > 0"),
        ({"q": -1.0}, "q must be finite and > 0"),
        ({"v1": 0.0}, "v1 must be finite and > 0"),
        ({"v2": math.inf}, "v2 must be finite and > 0"),
        ({"v2": [10.0, math.nan]}, "v2 must be finite and > 0; got nan at index 1"),
        ({"cl": -3.0}, "cl must be finite and > 0"),
    ],
)
def test_two_compartment_rejects_invalid_parameters(kwargs: dict[str, Any], message: str) -> None:
    arguments: dict[str, Any] = {"cl": 3.0, "v1": 15.0, "q": 5.0, "v2": 40.0} | kwargs
    with pytest.raises(ValueError, match=message):
        pk.TwoCompartment(**arguments)


@pytest.mark.parametrize("value", [True, "1.0", 1 + 2j, None, [1.0, "a"]])
def test_parameters_must_be_real_numbers(value: Any) -> None:
    with pytest.raises(TypeError, match="cl must contain real numbers"):
        pk.OneCompartment(cl=value, v=10.0)


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((math.nan, 100.0, Route.IV_BOLUS), {}, "Dose.time must not be NaN"),
        ((math.inf, 100.0, Route.IV_BOLUS), {}, "Dose.time must be finite"),
        ((0.0, -1.0, Route.IV_BOLUS), {}, "Dose.amount must be >= 0"),
        ((0.0, math.inf, Route.ORAL), {}, "Dose.amount must be finite"),
        ((0.0, 100.0, "intramuscular"), {}, "route must be one of"),
        ((0.0, 100.0, Route.IV_INFUSION), {}, "an IV infusion needs a duration"),
        ((0.0, 100.0, Route.IV_INFUSION), {"duration": 0.0}, "Dose.duration must be > 0"),
        ((0.0, 100.0, Route.IV_INFUSION), {"duration": -2.0}, "Dose.duration must be > 0"),
        ((0.0, 100.0, Route.IV_INFUSION), {"duration": math.nan}, "Dose.duration must not"),
        ((0.0, 100.0, Route.IV_BOLUS), {"duration": 1.0}, "duration applies only to IV"),
        ((0.0, 100.0, Route.ORAL), {"duration": 1.0}, "duration applies only to IV"),
    ],
)
def test_dose_rejects_invalid_values(
    args: tuple[Any, ...], kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Dose(*args, **kwargs)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("0", 100.0, Route.ORAL), "Dose.time must be a real number"),
        ((0.0, True, Route.ORAL), "Dose.amount must be a real number"),
    ],
)
def test_dose_rejects_non_numbers(args: tuple[Any, ...], message: str) -> None:
    with pytest.raises(TypeError, match=message):
        Dose(*args)


def test_dose_accepts_route_strings_and_numpy_scalars() -> None:
    dose = Dose(np.float64(1.5), np.int64(100), "iv_infusion", duration=2)  # type: ignore[arg-type]
    assert dose.route is Route.IV_INFUSION
    assert dose == Dose(1.5, 100.0, Route.IV_INFUSION, 2.0)
    assert dose.rate == 50.0
    assert Dose(0.0, 1.0, Route.ORAL).rate is None


def test_oral_dose_requires_an_absorption_rate() -> None:
    model = pk.OneCompartment(cl=1.0, v=10.0)
    with pytest.raises(ValueError, match="oral doses need a model with an absorption rate"):
        pk.concentration(model, [Dose(0.0, 100.0, Route.ORAL)], [1.0])
    with pytest.raises(ValueError, match="no absorption rate constant"):
        pk.apparent_oral_terminal_rate(model)


@pytest.mark.parametrize(
    ("doses", "error"),
    [(Dose(0.0, 1.0, Route.IV_BOLUS), TypeError), ([(0.0, 1.0, "iv_bolus")], TypeError)],
)
def test_doses_must_be_an_iterable_of_dose_objects(doses: object, error: type[Exception]) -> None:
    model = pk.OneCompartment(cl=1.0, v=10.0)
    with pytest.raises(error, match="Dose"):
        pk.concentration(model, doses, [1.0])  # type: ignore[arg-type]


def test_models_must_be_supported_types() -> None:
    with pytest.raises(TypeError, match="expected OneCompartment or TwoCompartment"):
        pk.concentration("model", BOLUS, [1.0])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("times", "error", "message"),
    [
        ([0.0, math.nan], ValueError, "times must be finite; got nan at index 1"),
        ([math.inf], ValueError, "times must be finite"),
        ([[0.0, 1.0]], ValueError, "times must be a scalar or a 1-D array"),
        (["a"], TypeError, "times must contain real numbers"),
    ],
)
def test_times_must_be_finite_and_one_dimensional(
    times: Any, error: type[Exception], message: str
) -> None:
    model = pk.OneCompartment(cl=1.0, v=10.0)
    with pytest.raises(error, match=message):
        pk.concentration(model, BOLUS, times)
    with pytest.raises(error, match=message):
        pk.trough(model, BOLUS, times)
    with pytest.raises(error, match=message):
        pk.amounts(model, BOLUS, times)


@pytest.mark.parametrize(
    ("start", "end", "error", "message"),
    [
        (2.0, 1.0, ValueError, "need start <= end"),
        (math.inf, math.inf, ValueError, "need start <= end"),
        (-math.inf, -math.inf, ValueError, "need start <= end"),
        (math.nan, 1.0, ValueError, "start must not be NaN"),
        (0.0, "1", TypeError, "end must be a real number"),
    ],
)
def test_auc_rejects_invalid_intervals(
    start: object, end: object, error: type[Exception], message: str
) -> None:
    model = pk.OneCompartment(cl=1.0, v=10.0)
    with pytest.raises(error, match=message):
        pk.auc(model, BOLUS, start, end)  # type: ignore[arg-type]


@pytest.mark.parametrize("metric", [pk.cmax, pk.cmin])
def test_extremum_window_is_validated(metric: Callable[..., pk.Extremum]) -> None:
    model = pk.OneCompartment(cl=1.0, v=10.0)
    with pytest.raises(ValueError, match="start must be <= end"):
        metric(model, BOLUS, 5.0, 1.0)
    with pytest.raises(ValueError, match="end must be finite"):
        metric(model, BOLUS, 0.0, math.inf)
    with pytest.raises(TypeError, match="start must be a real number"):
        metric(model, BOLUS, "0", 1.0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: pk.OneCompartment(cl=1e300, v=1e-300), "cl / v must be a finite rate"),
        (lambda: pk.OneCompartment(cl=1e-300, v=1e300), "cl / v must be a finite rate"),
        (lambda: pk.TwoCompartment(1e300, 1e-10, 1.0, 1.0), "cl / v1 must be a finite rate"),
        (lambda: pk.TwoCompartment(1.0, 1e-300, 1e300, 1.0), "q / v1 must be a finite rate"),
        (lambda: pk.TwoCompartment(1.0, 1.0, 1e-300, 1e300), "q / v2 must be a finite rate"),
        (
            lambda: pk.TwoCompartment(1e-200, 1.0, 1e-200, 1e100),
            "slower disposition rate constant must be a finite rate",
        ),
    ],
)
def test_derived_rate_constants_must_be_representable(
    factory: Callable[[], pk.Model], message: str
) -> None:
    # pytest turns warnings into errors here, so this also checks that no numpy
    # RuntimeWarning escapes from the overflow or underflow.
    with pytest.raises(ValueError, match=message):
        factory()


def test_event_times_must_be_representable() -> None:
    model = pk.OneCompartment(cl=1.0, v=1.0, ka=1.0, tlag=1e308)
    with pytest.raises(ValueError, match="infusion end time"):
        pk.concentration(model, [Dose(1e308, 1.0, Route.IV_INFUSION, duration=1e308)], [0.0])
    with pytest.raises(ValueError, match="oral input time"):
        pk.concentration(model, [Dose(1e308, 1.0, Route.ORAL)], [0.0])


def test_negative_zero() -> None:
    with pytest.raises(ValueError, match=r"Dose\.amount must be >= 0"):
        Dose(0.0, -0.0, Route.ORAL)
    model = pk.OneCompartment(cl=1.0, v=1.0, ka=1.0, f=-0.0, tlag=-0.0)
    assert math.copysign(1.0, float(model.tlag)) == 1.0
    assert math.copysign(1.0, float(model.f)) == 1.0


def test_huge_integers_are_rejected_as_values() -> None:
    with pytest.raises(ValueError, match=r"Dose\.amount must be finite"):
        Dose(0.0, 10**400, Route.ORAL)


@pytest.mark.parametrize("percentiles", [[101.0], [-1.0], [], [[5.0, 95.0]], [math.nan]])
def test_percentiles_are_validated(percentiles: Any) -> None:
    model = pk.OneCompartment(cl=[1.0, 2.0], v=10.0)
    with pytest.raises(ValueError, match="percentiles"):
        pk.percentile_bands(model, BOLUS, [1.0], percentiles)


@pytest.mark.parametrize("percentiles", [[True, False], ["50"], [None]])
def test_percentiles_must_be_numbers(percentiles: Any) -> None:
    model = pk.OneCompartment(cl=[1.0, 2.0], v=10.0)
    with pytest.raises(TypeError, match="percentiles must be real numbers"):
        pk.percentile_bands(model, BOLUS, [1.0], percentiles)


def test_models_and_doses_are_immutable() -> None:
    model = pk.OneCompartment(cl=[1.0, 2.0], v=10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.cl = np.array([3.0, 4.0])  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        model.cl[0] = 5.0
    dose = Dose(0.0, 1.0, Route.IV_BOLUS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        dose.amount = 2.0  # type: ignore[misc]


def test_model_copies_its_inputs() -> None:
    draws = np.array([1.0, 2.0])
    model = pk.OneCompartment(cl=draws, v=10.0)
    draws[0] = 100.0
    assert model.cl[0] == 1.0
    assert draws.flags.writeable
