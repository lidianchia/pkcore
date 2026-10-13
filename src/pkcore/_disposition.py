"""Disposition rate constants of the two-compartment model (internal)."""

import numpy as np

from pkcore._types import FloatArray


def two_compartment_modes(
    cl: FloatArray, v1: FloatArray, q: FloatArray, v2: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(lambda1, lambda2, c1, c2)`` for the two-compartment disposition.

    With ``k10 = cl / v1``, ``k12 = q / v1`` and ``k21 = q / v2`` the central amount
    after a unit bolus is ``c1 exp(-lambda1 t) + c2 exp(-lambda2 t)`` with
    ``lambda1 < lambda2`` the roots of
    ``lambda^2 - (k10 + k12 + k21) lambda + k10 k21 = 0`` and
    ``c1 = (k21 - lambda1) / (lambda2 - lambda1)``,
    ``c2 = (lambda2 - k21) / (lambda2 - lambda1)``: the micro-constant form of the
    closed-form (ADVAN-style) solutions, see Abuhelwa, Foster & Upton 2015,
    doi:10.1016/j.vascn.2015.03.004.

    The cancellation-free evaluation below is pkcore's own reformulation. The only
    subtraction is ``k10 - k21``, which is exact when the two are within a factor of
    two of each other (Sterbenz) and harmless otherwise; everything else is a sum or
    product of positive terms:

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
