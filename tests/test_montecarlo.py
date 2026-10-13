"""Monte Carlo helper: all draws in one vectorised call, pointwise percentile bands.

Expected values: one scalar simulation per draw (a separate call each) and
``numpy.percentile`` over those curves. The draws are made by the test itself, as a
caller would; pkcore samples nothing.
"""

import numpy as np

import pkcore as pk
from pkcore import Dose, Route

HISTORY = [
    Dose(0.0, 100.0, Route.ORAL),
    Dose(12.0, 100.0, Route.ORAL),
    Dose(24.0, 150.0, Route.IV_INFUSION, duration=1.0),
]
TIMES = np.linspace(0.0, 48.0, 25)
LEVELS = (5.0, 25.0, 50.0, 75.0, 95.0)


def test_bands_are_percentiles_of_per_draw_simulations() -> None:
    rng = np.random.default_rng(20261012)
    n = 200
    cl = 3.0 * rng.lognormal(0.0, 0.3, n)
    v1 = 15.0 * rng.lognormal(0.0, 0.2, n)
    ka = 0.9 * rng.lognormal(0.0, 0.4, n)
    model = pk.TwoCompartment(cl=cl, v1=v1, q=5.0, v2=40.0, ka=ka, f=0.8, tlag=0.25)
    bands = pk.percentile_bands(model, HISTORY, TIMES, LEVELS)
    curves = np.array(
        [
            pk.concentration(
                pk.TwoCompartment(cl[i], v1[i], 5.0, 40.0, ka=ka[i], f=0.8, tlag=0.25),
                HISTORY,
                TIMES,
            )
            for i in range(n)
        ]
    )
    assert bands.n_draws == n
    assert bands.values.shape == (len(LEVELS), TIMES.size)
    np.testing.assert_array_equal(bands.percentiles, LEVELS)
    np.testing.assert_allclose(bands.values, np.percentile(curves, LEVELS, axis=0), rtol=1e-13)
    assert np.all(np.diff(bands.values, axis=0) >= 0.0)


def test_median_of_three_draws_is_the_middle_curve() -> None:
    model = pk.OneCompartment(cl=[1.0, 2.0, 4.0], v=20.0, ka=1.0)
    curves = pk.concentration(model, HISTORY, TIMES)
    bands = pk.percentile_bands(model, HISTORY, TIMES, [50.0])
    np.testing.assert_array_equal(bands.values[0], np.sort(curves, axis=0)[1])


def test_scalar_model_counts_as_one_draw() -> None:
    model = pk.OneCompartment(cl=2.0, v=20.0, ka=1.0)
    bands = pk.percentile_bands(model, HISTORY, TIMES)
    assert bands.n_draws == 1
    curve = pk.concentration(model, HISTORY, TIMES)
    for row in bands.values:
        np.testing.assert_array_equal(row, curve)


def test_scalar_time_gives_one_value_per_percentile() -> None:
    model = pk.OneCompartment(cl=[1.0, 2.0, 4.0], v=20.0, ka=1.0)
    bands = pk.percentile_bands(model, HISTORY, 6.0)
    assert bands.values.shape == (3,)
    assert not bands.percentiles.flags.writeable
