"""Reduction of a model to a sum of exponentials (internal)."""

from dataclasses import dataclass

import numpy as np

from pkcore._disposition import two_compartment_modes
from pkcore._types import FloatArray, IndexArray
from pkcore.models import Model, OneCompartment, TwoCompartment


@dataclass(frozen=True, eq=False)
class Kinetics:
    """A model in modal form, one row per parameter draw.

    The amount in the central compartment ``t`` hours after a unit IV bolus is
    ``sum_i fractions[i] * exp(-rates[i] * t)``, with ``rates`` in ascending order and
    ``sum_i fractions[i] == 1``. Every array has shape ``(rows, 1, 1)`` so that it
    broadcasts against a dose axis and a time axis; ``rows`` is 1 for scalar models.
    """

    shape: tuple[int, ...]
    rates: tuple[FloatArray, ...]
    fractions: tuple[FloatArray, ...]
    v1: FloatArray
    cl: FloatArray
    k12: FloatArray | None
    ka: FloatArray | None
    f: FloatArray
    tlag: FloatArray

    @property
    def rows(self) -> int:
        return int(self.v1.shape[0])

    def take(self, rows: IndexArray) -> "Kinetics":
        """Return the kinetics of the given rows (draw indices), in that order."""

        def pick(a: FloatArray) -> FloatArray:
            return a[rows]

        def maybe(a: FloatArray | None) -> FloatArray | None:
            return None if a is None else pick(a)

        return Kinetics(
            shape=(int(rows.shape[0]),),
            rates=tuple(pick(r) for r in self.rates),
            fractions=tuple(pick(c) for c in self.fractions),
            v1=pick(self.v1),
            cl=pick(self.cl),
            k12=maybe(self.k12),
            ka=maybe(self.ka),
            f=pick(self.f),
            tlag=pick(self.tlag),
        )


def kinetics(model: Model) -> Kinetics:
    """Return the modal form of ``model``."""
    if not isinstance(model, OneCompartment | TwoCompartment):
        raise TypeError(f"expected OneCompartment or TwoCompartment, got {type(model).__name__}")
    # The rate constants were validated on construction; no floating-point warning can
    # occur here, and none must escape to callers that turn warnings into errors.
    with np.errstate(all="ignore"):
        return _modal_form(model)


def _modal_form(model: Model) -> Kinetics:
    n = model.n_draws
    rows = 1 if n is None else n

    def column(a: FloatArray) -> FloatArray:
        return np.broadcast_to(a, (rows,)).reshape(rows, 1, 1)

    absorption = None if model.ka is None else column(model.ka)
    if isinstance(model, OneCompartment):
        cl, v = column(model.cl), column(model.v)
        return Kinetics(
            shape=() if n is None else (n,),
            rates=(cl / v,),
            fractions=(np.ones_like(cl),),
            v1=v,
            cl=cl,
            k12=None,
            ka=absorption,
            f=column(model.f),
            tlag=column(model.tlag),
        )
    cl, v1 = column(model.cl), column(model.v1)
    q, v2 = column(model.q), column(model.v2)
    slow, fast, slow_fraction, fast_fraction = two_compartment_modes(cl, v1, q, v2)
    return Kinetics(
        shape=() if n is None else (n,),
        rates=(slow, fast),
        fractions=(slow_fraction, fast_fraction),
        v1=v1,
        cl=cl,
        k12=q / v1,
        ka=absorption,
        f=column(model.f),
        tlag=column(model.tlag),
    )
