"""Dose records.

A dosing history is any iterable of :class:`Dose` objects: times may be irregular,
unsorted and negative, and infusions may overlap. A missed dose is simply left out.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

from pkcore import _validation


class Route(StrEnum):
    """Route of administration."""

    IV_BOLUS = "iv_bolus"
    """Instantaneous input into the central compartment."""

    IV_INFUSION = "iv_infusion"
    """Zero-order input into the central compartment at ``amount / duration`` per hour."""

    ORAL = "oral"
    """First-order absorption from a depot (model parameters ``ka``, ``f``, ``tlag``)."""


@dataclass(frozen=True, init=False)
class Dose:
    """One dose event.

    Attributes:
        time: Administration time (h), finite. For oral doses the bioavailable amount
            enters the depot at ``time + tlag``.
        amount: Amount administered (mg), finite and >= 0. For oral doses only the
            fraction ``f`` (bioavailability) of it is absorbed.
        route: A :class:`Route`; the constructor also accepts its string value, e.g.
            ``"oral"``.
        duration: Infusion duration (h), finite and > 0. Required for
            ``Route.IV_INFUSION`` and must be ``None`` for the other routes.
    """

    time: float
    amount: float
    route: Route
    duration: float | None

    def __init__(
        self, time: float, amount: float, route: Route | str, duration: float | None = None
    ) -> None:
        try:
            checked_route = Route(route)
        except ValueError:
            choices = ", ".join(repr(r.value) for r in Route)
            raise ValueError(f"route must be one of {choices}; got {route!r}") from None
        checked_time = _validation.real_number("Dose.time", time)
        checked_amount = _validation.real_number("Dose.amount", amount)
        if checked_amount < 0.0 or math.copysign(1.0, checked_amount) < 0.0:
            raise ValueError(f"Dose.amount must be >= 0, got {checked_amount!r}")
        checked_duration: float | None = None
        if checked_route is Route.IV_INFUSION:
            if duration is None:
                raise ValueError("an IV infusion needs a duration (h) > 0")
            checked_duration = _validation.real_number("Dose.duration", duration)
            if checked_duration <= 0.0:
                raise ValueError(f"Dose.duration must be > 0, got {checked_duration!r}")
        elif duration is not None:
            raise ValueError(
                f"duration applies only to IV infusions, not to route {checked_route.value!r}"
            )
        object.__setattr__(self, "time", checked_time)
        object.__setattr__(self, "amount", checked_amount)
        object.__setattr__(self, "route", checked_route)
        object.__setattr__(self, "duration", checked_duration)

    @property
    def rate(self) -> float | None:
        """Infusion rate ``amount / duration`` (mg/h), or ``None`` for other routes."""
        return None if self.duration is None else self.amount / self.duration
