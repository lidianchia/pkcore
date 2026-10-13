"""Percentile bands over caller-supplied parameter draws.

pkcore does not sample parameters: the caller draws them (from whatever distribution
and with whatever random generator they choose) and passes them to a model as 1-D
arrays. All draws are then simulated in one vectorised call.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pkcore import _validation
from pkcore._types import FloatArray
from pkcore.doses import Dose
from pkcore.models import Model
from pkcore.simulation import concentration


@dataclass(frozen=True, eq=False)
class PercentileBands:
    """Pointwise percentiles of simulated concentrations.

    Attributes:
        percentiles: The requested percentiles, shape ``(p,)``.
        values: Concentrations (mg/L), shape ``(p,) + times.shape``; ``values[i]`` is
            the ``percentiles[i]``-th percentile across draws at each time.
        n_draws: Number of parameter draws the percentiles were computed from.
    """

    percentiles: FloatArray
    values: FloatArray
    n_draws: int


def percentile_bands(
    model: Model,
    doses: Iterable[Dose],
    times: npt.ArrayLike,
    percentiles: npt.ArrayLike = (5.0, 50.0, 95.0),
) -> PercentileBands:
    """Simulate every parameter draw of ``model`` and return pointwise percentile bands.

    Percentiles are computed independently at each time with
    ``numpy.percentile(..., method="linear")``, so a band is not the curve of any
    single draw. A model with scalar parameters counts as one draw.
    """
    levels = np.array(percentiles, dtype=np.float64)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("percentiles must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(levels) & (levels >= 0.0) & (levels <= 100.0)):
        raise ValueError(f"percentiles must lie in [0, 100], got {levels.tolist()}")
    t = _validation.times("times", times)
    curves = concentration(model, doses, t)
    draws = curves.reshape((model.n_draws or 1, *t.shape))
    values = np.percentile(draws, levels, axis=0, method="linear")
    levels.setflags(write=False)
    return PercentileBands(percentiles=levels, values=values, n_draws=int(draws.shape[0]))
