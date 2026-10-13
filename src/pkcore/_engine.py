"""Closed-form superposition engine (internal).

The amount in each compartment is a sum over doses of closed-form unit responses
(built from :mod:`pkcore._kernels`) evaluated at ``t - t_dose``. Arrays are laid out
as ``(rows, doses, times)``: model parameters have shape ``(rows, 1, 1)``, dose data
``(1, doses, 1)`` and evaluation times ``(rows or 1, 1, times)``. Doses are processed
in chunks so that temporary arrays stay below a fixed element budget.

Nothing here keeps state between calls: every function depends only on its arguments.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from pkcore._kernels import conv3, dd, e1, e2
from pkcore._kinetics import Kinetics, kinetics
from pkcore._types import BoolArray, FloatArray
from pkcore.doses import Dose, Route
from pkcore.models import Model

_ELEMENT_BUDGET = 1 << 20


class Side(Enum):
    """Which one-sided value to return at an event time."""

    RIGHT = "right"
    """Right-continuous: an event at ``t`` is included in the value at ``t``."""

    LEFT = "left"
    """Left limit: only events strictly before ``t`` are included."""


@dataclass(frozen=True, eq=False)
class DoseTable:
    """Dose records split by route into flat arrays."""

    bolus_time: FloatArray
    bolus_amount: FloatArray
    infusion_time: FloatArray
    infusion_amount: FloatArray
    infusion_duration: FloatArray
    oral_time: FloatArray
    oral_amount: FloatArray


def prepare(model: Model, doses: Iterable[Dose]) -> tuple[Kinetics, DoseTable]:
    """Validate ``doses`` against ``model`` and return both in internal form."""
    kin = kinetics(model)
    if isinstance(doses, Dose):
        raise TypeError("doses must be an iterable of Dose objects, not a single Dose")
    items = list(doses)
    for item in items:
        if not isinstance(item, Dose):
            raise TypeError(f"doses must contain Dose objects, got {type(item).__name__}")

    def column(route: Route, attribute: str) -> FloatArray:
        return np.array([getattr(d, attribute) for d in items if d.route is route], dtype=float)

    table = DoseTable(
        bolus_time=column(Route.IV_BOLUS, "time"),
        bolus_amount=column(Route.IV_BOLUS, "amount"),
        infusion_time=column(Route.IV_INFUSION, "time"),
        infusion_amount=column(Route.IV_INFUSION, "amount"),
        infusion_duration=column(Route.IV_INFUSION, "duration"),
        oral_time=column(Route.ORAL, "time"),
        oral_amount=column(Route.ORAL, "amount"),
    )
    if table.oral_time.size and kin.ka is None:
        raise ValueError("oral doses need a model with an absorption rate constant (ka)")
    return kin, table


def finish(values: FloatArray, kin: Kinetics, shape: tuple[int, ...], what: str) -> FloatArray:
    """Reshape ``(rows, n)`` results to ``kin.shape + shape`` and check finiteness."""
    out = values.reshape(kin.shape + shape) if kin.shape else values[0].reshape(shape)
    if not np.isfinite(out).all():
        raise FloatingPointError(
            f"{what} is not finite (overflow); check the magnitudes of parameters, doses and times"
        )
    return out


def _accumulate(
    count: int, rows: int, columns: int, modes: int, term: Callable[[slice], FloatArray]
) -> FloatArray:
    """Return ``sum`` over dose chunks of ``term(chunk).sum(axis=1)``, shape (rows, columns)."""
    total = np.zeros((rows, columns))
    step = max(1, _ELEMENT_BUDGET // max(1, rows * columns * modes))
    for start in range(0, count, step):
        total += term(slice(start, start + step)).sum(axis=1)
    return total


def _modal(kin: Kinetics, response: Callable[[FloatArray], FloatArray]) -> FloatArray:
    """Return ``sum_i fractions[i] * response(rates[i])``."""
    total = kin.fractions[0] * response(kin.rates[0])
    for rate, fraction in zip(kin.rates[1:], kin.fractions[1:], strict=True):
        total = total + fraction * response(rate)
    return total


def _dims(kin: Kinetics, t: FloatArray) -> tuple[int, int, int]:
    return max(kin.rows, int(t.shape[0])), int(t.shape[2]), len(kin.rates) + 1


def _given(tau: FloatArray, side: Side) -> BoolArray:
    return tau >= 0.0 if side is Side.RIGHT else tau > 0.0


def _masked(mask: BoolArray, values: FloatArray) -> FloatArray:
    return np.where(mask, values, 0.0)


def _absorption(kin: Kinetics) -> FloatArray:
    if kin.ka is None:  # prepare() rejects oral doses for models without ka
        raise ValueError("oral doses need a model with an absorption rate constant (ka)")
    return kin.ka


def central_amount(kin: Kinetics, table: DoseTable, t: FloatArray, side: Side) -> FloatArray:
    """Amount in the central compartment at times ``t`` (shape ``(rows|1, 1, n)``)."""
    rows, columns, modes = _dims(kin, t)

    def bolus(s: slice) -> FloatArray:
        tau = t - table.bolus_time[None, s, None]
        given = _given(tau, side)
        elapsed = np.where(given, tau, 0.0)
        response = _modal(kin, lambda rate: np.exp(-rate * elapsed))
        return np.where(given, table.bolus_amount[None, s, None] * response, 0.0)

    def infusion(s: slice) -> FloatArray:
        duration = table.infusion_duration[None, s, None]
        tau = t - table.infusion_time[None, s, None]
        running = np.clip(tau, 0.0, duration)
        after = np.maximum(tau - duration, 0.0)
        rate_in = table.infusion_amount[None, s, None] / duration
        return rate_in * _modal(kin, lambda rate: np.exp(-rate * after) * e1(rate, running))

    def oral(s: slice) -> FloatArray:
        ka = _absorption(kin)
        elapsed = np.maximum(t - table.oral_time[None, s, None] - kin.tlag, 0.0)
        absorbed = kin.f * table.oral_amount[None, s, None] * ka
        return absorbed * _modal(kin, lambda rate: dd(rate, ka, elapsed))

    return (
        _accumulate(table.bolus_time.size, rows, columns, modes, bolus)
        + _accumulate(table.infusion_time.size, rows, columns, modes, infusion)
        + _accumulate(table.oral_time.size, rows, columns, modes, oral)
    )


def central_slope(kin: Kinetics, table: DoseTable, t: FloatArray, side: Side) -> FloatArray:
    """Time derivative of the central amount; one-sided (``side``) at event times."""
    rows, columns, modes = _dims(kin, t)

    def bolus(s: slice) -> FloatArray:
        tau = t - table.bolus_time[None, s, None]
        given = _given(tau, side)
        elapsed = np.where(given, tau, 0.0)
        response = _modal(kin, lambda rate: -rate * np.exp(-rate * elapsed))
        return np.where(given, table.bolus_amount[None, s, None] * response, 0.0)

    def infusion(s: slice) -> FloatArray:
        duration = table.infusion_duration[None, s, None]
        tau = t - table.infusion_time[None, s, None]
        if side is Side.RIGHT:
            rising = np.logical_and(tau >= 0.0, tau < duration)
            falling = tau >= duration
        else:
            rising = np.logical_and(tau > 0.0, tau <= duration)
            falling = tau > duration
        running = np.clip(tau, 0.0, duration)
        after = np.maximum(tau - duration, 0.0)
        rate_in = table.infusion_amount[None, s, None] / duration
        up = _modal(kin, lambda rate: np.exp(-rate * running))
        down = _modal(kin, lambda rate: -rate * e1(rate, duration) * np.exp(-rate * after))
        return rate_in * (_masked(rising, up) + _masked(falling, down))

    def oral(s: slice) -> FloatArray:
        ka = _absorption(kin)
        tau = t - table.oral_time[None, s, None] - kin.tlag
        given = _given(tau, side)
        elapsed = np.where(given, tau, 0.0)
        absorbed = kin.f * table.oral_amount[None, s, None] * ka
        response = _modal(kin, lambda rate: np.exp(-ka * elapsed) - rate * dd(rate, ka, elapsed))
        return np.where(given, absorbed * response, 0.0)

    return (
        _accumulate(table.bolus_time.size, rows, columns, modes, bolus)
        + _accumulate(table.infusion_time.size, rows, columns, modes, infusion)
        + _accumulate(table.oral_time.size, rows, columns, modes, oral)
    )


def peripheral_amount(kin: Kinetics, table: DoseTable, t: FloatArray) -> FloatArray:
    """Amount in the peripheral compartment (zero for one-compartment models)."""
    rows, columns, modes = _dims(kin, t)
    if kin.k12 is None:
        return np.zeros((rows, columns))
    k12 = kin.k12
    slow, fast = kin.rates

    def bolus(s: slice) -> FloatArray:
        elapsed = np.maximum(t - table.bolus_time[None, s, None], 0.0)
        return table.bolus_amount[None, s, None] * k12 * dd(slow, fast, elapsed)

    def infusion(s: slice) -> FloatArray:
        duration = table.infusion_duration[None, s, None]
        tau = t - table.infusion_time[None, s, None]
        running = np.clip(tau, 0.0, duration)
        after = np.maximum(tau - duration, 0.0)
        rate_in = table.infusion_amount[None, s, None] / duration
        response = dd(slow, fast, after) * e1(slow, running) + np.exp(-fast * after) * e2(
            slow, fast, running
        )
        return rate_in * k12 * response

    def oral(s: slice) -> FloatArray:
        ka = _absorption(kin)
        elapsed = np.maximum(t - table.oral_time[None, s, None] - kin.tlag, 0.0)
        absorbed = kin.f * table.oral_amount[None, s, None] * ka
        return absorbed * k12 * conv3(slow, fast, ka, elapsed)

    return (
        _accumulate(table.bolus_time.size, rows, columns, modes, bolus)
        + _accumulate(table.infusion_time.size, rows, columns, modes, infusion)
        + _accumulate(table.oral_time.size, rows, columns, modes, oral)
    )


def depot_amount(kin: Kinetics, table: DoseTable, t: FloatArray) -> FloatArray:
    """Bioavailable amount still in the depot (oral doses whose lag time has elapsed)."""
    rows, columns, modes = _dims(kin, t)

    def oral(s: slice) -> FloatArray:
        ka = _absorption(kin)
        tau = t - table.oral_time[None, s, None] - kin.tlag
        given = _given(tau, Side.RIGHT)
        elapsed = np.where(given, tau, 0.0)
        return np.where(
            given, kin.f * table.oral_amount[None, s, None] * np.exp(-ka * elapsed), 0.0
        )

    return _accumulate(table.oral_time.size, rows, columns, modes, oral)


def central_integral(kin: Kinetics, table: DoseTable, t1: FloatArray, t2: FloatArray) -> FloatArray:
    """Integral of the central amount over ``[t1, t2]``, exact up to rounding.

    ``t1 <= t2``; ``t1`` may be ``-inf`` and ``t2`` may be ``+inf``. Each dose's
    contribution is integrated analytically on its own time axis. For an interval
    ``[a, a + w]`` after the start of an input, the response is split into the part
    already in the system at ``a`` (which decays) and the input still to come, e.g.
    ``dd(r, ka, a + s) = exp(-r s) dd(r, ka, a) + exp(-ka a) dd(r, ka, s)``, so every
    term is non-negative and short intervals keep full relative accuracy.
    """
    rows = max(kin.rows, int(t1.shape[0]), int(t2.shape[0]))
    columns = max(int(t1.shape[2]), int(t2.shape[2]))
    modes = len(kin.rates) + 1

    def bolus(s: slice) -> FloatArray:
        start = np.maximum(t1 - table.bolus_time[None, s, None], 0.0)
        stop = np.maximum(t2 - table.bolus_time[None, s, None], 0.0)
        width = stop - start
        response = _modal(kin, lambda rate: np.exp(-rate * start) * e1(rate, width))
        return table.bolus_amount[None, s, None] * response

    def infusion(s: slice) -> FloatArray:
        duration = table.infusion_duration[None, s, None]
        tau1 = t1 - table.infusion_time[None, s, None]
        tau2 = t2 - table.infusion_time[None, s, None]
        run_start = np.clip(tau1, 0.0, duration)
        run_width = np.clip(tau2, 0.0, duration) - run_start
        after_start = np.maximum(tau1, duration)
        after_width = np.maximum(tau2, duration) - after_start
        rate_in = table.infusion_amount[None, s, None] / duration
        zero = np.zeros_like(duration)

        def response(rate: FloatArray) -> FloatArray:
            running = run_width * e1(rate, run_start) + np.exp(-rate * run_start) * e2(
                zero, rate, run_width
            )
            after = e1(rate, duration) * np.exp(-rate * (after_start - duration))
            return running + after * e1(rate, after_width)

        return rate_in * _modal(kin, response)

    def oral(s: slice) -> FloatArray:
        ka = _absorption(kin)
        start = np.maximum(t1 - table.oral_time[None, s, None] - kin.tlag, 0.0)
        width = np.maximum(t2 - table.oral_time[None, s, None] - kin.tlag, 0.0) - start
        absorbed = kin.f * table.oral_amount[None, s, None] * ka
        response = _modal(
            kin,
            lambda rate: (
                dd(rate, ka, start) * e1(rate, width) + np.exp(-ka * start) * e2(rate, ka, width)
            ),
        )
        return absorbed * response

    return (
        _accumulate(table.bolus_time.size, rows, columns, modes, bolus)
        + _accumulate(table.infusion_time.size, rows, columns, modes, infusion)
        + _accumulate(table.oral_time.size, rows, columns, modes, oral)
    )


def event_times(kin: Kinetics, table: DoseTable) -> FloatArray:
    """Times at which the curve may jump or have a kink, shape ``(rows, events)``.

    Bolus times, infusion starts and ends, and lag-shifted oral dose times.
    """
    fixed = np.concatenate(
        [table.bolus_time, table.infusion_time, table.infusion_time + table.infusion_duration]
    )
    oral = table.oral_time[None, :] + kin.tlag[:, :, 0]
    rows = kin.rows
    return np.concatenate(
        [np.broadcast_to(fixed, (rows, fixed.size)), np.broadcast_to(oral, (rows, oral.shape[1]))],
        axis=1,
    )
