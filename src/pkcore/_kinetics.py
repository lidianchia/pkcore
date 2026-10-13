"""Reduction of a model to a sum of exponentials (internal)."""

from dataclasses import dataclass

import numpy as np

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


def two_compartment_modes(
    cl: FloatArray, v1: FloatArray, q: FloatArray, v2: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(lambda1, lambda2, c1, c2)`` for the two-compartment disposition.

    With ``k10 = cl / v1``, ``k12 = q / v1`` and ``k21 = q / v2`` the central amount
    after a unit bolus is ``c1 exp(-lambda1 t) + c2 exp(-lambda2 t)`` with
    ``lambda1 < lambda2`` the roots of
    ``lambda^2 - (k10 + k12 + k21) lambda + k10 k21 = 0`` and
    ``c1 = (k21 - lambda1) / (lambda2 - lambda1)``,
    ``c2 = (lambda2 - k21) / (lambda2 - lambda1)`` (the micro-constant form of the
    ADVAN-style solutions, Abuhelwa, Foster & Upton 2015,
    doi:10.1016/j.vascn.2015.03.004).

    The only subtraction is ``k10 - k21``, which is exact when the two are within a
    factor of two of each other (Sterbenz) and harmless otherwise; everything else is
    a sum or product of positive terms:

    * the discriminant is ``(k10 - k21)^2 + k12 (k12 + 2 k10 + 2 k21)``, strictly
      positive because ``k12 > 0``;
    * ``lambda2 = (s + r) / 2`` with ``r = sqrt(discriminant)`` and
      ``lambda1 = k10 k21 / lambda2`` (Vieta), instead of ``(s - r) / 2``;
    * ``k21 - lambda1`` and ``lambda2 - k21`` have product ``k12 k21`` and difference
      ``d = (k10 - k21) + k12``: the larger one is ``(r + |d|) / 2`` and the smaller
      one is ``k12 k21`` divided by the larger one.

    Both fractions lie in (0, 1) because ``lambda1 < k21 < lambda2``.
    """
    k10 = cl / v1
    k12 = q / v1
    k21 = q / v2
    total = k10 + k12 + k21
    gap = k10 - k21
    root = np.sqrt(gap * gap + k12 * (k12 + 2.0 * k10 + 2.0 * k21))
    fast = 0.5 * (total + root)
    slow = (k10 * k21) / fast
    difference = gap + k12
    larger = 0.5 * (root + np.abs(difference))
    smaller = (k12 * k21) / larger
    fast_gap = np.where(difference >= 0.0, larger, smaller)  # lambda2 - k21
    slow_gap = np.where(difference >= 0.0, smaller, larger)  # k21 - lambda1
    return slow, fast, slow_gap / root, fast_gap / root
