"""Dose records.

A dosing history is any iterable of :class:`Dose` objects: times may be irregular,
unsorted and negative, and infusions may overlap. A missed dose is simply left out.
"""

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


@dataclass(frozen=True)
class Dose:
    """One dose event.

    Attributes:
        time: Administration time (h), finite. For oral doses the bioavailable amount
            enters the depot at ``time + tlag``.
        amount: Amount administered (mg), finite and >= 0. For oral doses only the
            fraction ``f`` (bioavailability) of it is absorbed.
        route: A :class:`Route` (or its string value, e.g. ``"oral"``).
        duration: Infusion duration (h), finite and > 0. Required for
            ``Route.IV_INFUSION`` and must be ``None`` for the other routes.
    """

    time: float
    amount: float
    route: Route
    duration: float | None = None

    def __post_init__(self) -> None:
        try:
            route = Route(self.route)
        except ValueError:
            choices = ", ".join(repr(r.value) for r in Route)
            raise ValueError(f"route must be one of {choices}; got {self.route!r}") from None
        time = _validation.real_number("Dose.time", self.time)
        amount = _validation.real_number("Dose.amount", self.amount)
        if amount < 0.0:
            raise ValueError(f"Dose.amount must be >= 0, got {amount!r}")
        duration: float | None = None
        if route is Route.IV_INFUSION:
            if self.duration is None:
                raise ValueError("an IV infusion needs a duration (h) > 0")
            duration = _validation.real_number("Dose.duration", self.duration)
            if duration <= 0.0:
                raise ValueError(f"Dose.duration must be > 0, got {duration!r}")
        elif self.duration is not None:
            raise ValueError(f"duration applies only to IV infusions, not to route {route.value!r}")
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "duration", duration)

    @property
    def rate(self) -> float | None:
        """Infusion rate ``amount / duration`` (mg/h), or ``None`` for other routes."""
        return None if self.duration is None else self.amount / self.duration
