"""Concentration and amount time courses.

Output shapes: for a model whose parameters are all scalars the result has the shape
of ``times``; for a model with ``n`` parameter draws it has shape
``(n,) + times.shape``, one row per draw. The simulation of one model never involves
any other model: there is no shared state and nothing is ever summed across calls.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pkcore import _validation
from pkcore._engine import (
    Side,
    central_amount,
    central_integral,
    depot_amount,
    finish,
    peripheral_amount,
    prepare,
)
from pkcore._types import FloatArray
from pkcore.doses import Dose
from pkcore.models import Model


def concentration(model: Model, doses: Iterable[Dose], times: npt.ArrayLike) -> FloatArray:
    """Return the central-compartment concentration ``C(t)`` (mg/L) at ``times`` (h).

    ``C`` is right-continuous: an IV bolus given at ``t0`` is included in ``C(t0)``.
    Oral doses contribute from ``t0 + tlag``; infusions from ``t0``. Use
    :func:`pkcore.trough` for the pre-dose value (the left limit). With no doses the
    result is all zeros.

    The value is the superposition of closed-form unit responses (sums of
    exponentials evaluated without cancellation, see docs/conventions.md).
    """
    kin, table = prepare(model, doses)
    t = _validation.times("times", times)
    with np.errstate(all="ignore"):
        amount = central_amount(kin, table, t.reshape(1, 1, -1), Side.RIGHT)
        values = amount / kin.v1[:, :, 0]
    return finish(values, kin, t.shape, "concentration")


@dataclass(frozen=True, eq=False)
class Amounts:
    """Amounts (mg) at the requested times, each with the shape of a concentration result.

    Attributes:
        depot: Bioavailable oral amount not yet absorbed (doses whose lag has elapsed).
        central: Amount in the central compartment (``central / v1`` is ``C(t)``).
        peripheral: Amount in the peripheral compartment (zero for one compartment).
        eliminated: Cumulative amount eliminated, ``cl`` times the AUC since the doses.

    For every time ``t`` these four add up to the input received by ``t``: IV bolus
    amounts, the infused part of each infusion, and ``f`` times each oral amount whose
    lag time has elapsed (mass balance).
    """

    depot: FloatArray
    central: FloatArray
    peripheral: FloatArray
    eliminated: FloatArray


def amounts(model: Model, doses: Iterable[Dose], times: npt.ArrayLike) -> Amounts:
    """Return the amount in each compartment and the cumulative eliminated amount.

    Right-continuous like :func:`pkcore.concentration`. The eliminated amount is
    ``cl`` times the exact integral of ``C`` from before the first dose to ``t``.
    """
    kin, table = prepare(model, doses)
    t = _validation.times("times", times)
    t3 = t.reshape(1, 1, -1)
    with np.errstate(all="ignore"):
        depot = depot_amount(kin, table, t3)
        central = central_amount(kin, table, t3, Side.RIGHT)
        peripheral = peripheral_amount(kin, table, t3)
        integral = central_integral(kin, table, np.full((1, 1, 1), -np.inf), t3)
        eliminated = integral * (kin.cl / kin.v1)[:, :, 0]
    return Amounts(
        depot=finish(depot, kin, t.shape, "depot amount"),
        central=finish(central, kin, t.shape, "central amount"),
        peripheral=finish(peripheral, kin, t.shape, "peripheral amount"),
        eliminated=finish(eliminated, kin, t.shape, "eliminated amount"),
    )
