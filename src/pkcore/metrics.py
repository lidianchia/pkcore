"""Summary metrics of simulated concentration curves.

Every metric has the draw shape of the model: a 0-d array for scalar parameters,
shape ``(n,)`` for ``n`` parameter draws. All metrics use the same closed forms as
:func:`pkcore.concentration`. AUC and troughs are analytic (exact up to rounding);
Cmax and Cmin are located by bracketing every critical point of ``C`` between events
and refining it with the analytic derivative (see :func:`cmax`).
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pkcore import _validation
from pkcore._engine import (
    DoseTable,
    Side,
    central_amount,
    central_bend,
    central_integral,
    central_slope,
    event_times,
    finish,
    prepare,
)
from pkcore._kinetics import Kinetics, kinetics
from pkcore._types import BoolArray, FloatArray, IndexArray
from pkcore.doses import Dose
from pkcore.models import Model

_MAX_REFINEMENTS = 400
_RESOLUTION = 8.0 * float(np.finfo(np.float64).eps)
_NOISE_FLOOR = 1e-280
_LN2 = math.log(2.0)


def trough(model: Model, doses: Iterable[Dose], times: npt.ArrayLike) -> FloatArray:
    """Return the left limit ``C(t-) = lim_{s -> t, s < t} C(s)`` (mg/L) at ``times``.

    This is the pre-dose concentration: an IV bolus given exactly at ``t`` is not
    included. Infusions and oral absorption are continuous, so for them (and at times
    without a bolus) the left limit equals :func:`pkcore.concentration`. The value is
    exact; it is not approximated by evaluating at ``t - epsilon``.
    """
    kin, table = prepare(model, doses)
    t = _validation.times("times", times)
    with np.errstate(all="ignore"):
        values = central_amount(kin, table, t.reshape(1, 1, -1), Side.LEFT) / kin.v1[:, :, 0]
    return finish(values, kin, t.shape, "trough concentration")


def auc(model: Model, doses: Iterable[Dose], start: float, end: float) -> FloatArray:
    """Return the exact area under ``C(t)`` over ``[start, end]`` (mg*h/L).

    ``start <= end``; ``start`` may be ``-inf`` (all doses) and ``end`` may be
    ``+inf`` (``AUC(start, inf)``, e.g. AUC0-inf for a single dose).

    Method: analytic integration of each dose's closed-form contribution, written as
    a sum of non-negative terms so that short intervals keep full relative accuracy
    (mass balance ``eliminated / cl`` would suffer cancellation there). Bolus jumps do
    not matter because the integrand is bounded.
    """
    kin, table = prepare(model, doses)
    t1 = _validation.extended_real_number("start", start)
    t2 = _validation.extended_real_number("end", end)
    if t1 == math.inf or t2 == -math.inf or t1 > t2:
        raise ValueError(f"need start <= end with start < inf and end > -inf; got [{t1}, {t2}]")
    with np.errstate(all="ignore"):
        integral = central_integral(kin, table, np.full((1, 1, 1), t1), np.full((1, 1, 1), t2))
        values = integral / kin.v1[:, :, 0]
    return finish(values, kin, (), "AUC")


@dataclass(frozen=True, eq=False)
class Extremum:
    """An extreme concentration and when it occurs.

    Attributes:
        value: The concentration (mg/L).
        time: The time (h). If the extreme value is a left limit (a minimum just before
            an IV bolus) this is the bolus time.
    """

    value: FloatArray
    time: FloatArray


def cmax(model: Model, doses: Iterable[Dose], start: float, end: float) -> Extremum:
    """Return the maximum of ``C(t)`` over the closed window ``[start, end]`` and its time.

    Cmax is ``max C(t)`` (attained, because ``C`` is right-continuous with upward
    jumps only); Tmax is the earliest time at which it is attained.

    Method (no sampling grid): the maximum lies at ``start``, ``end``, an event time
    (bolus, infusion start or end, lag-shifted oral dose) or a zero of ``C'`` between
    two events. There ``C'`` is a combination of at most three exponentials (rates
    ``lambda1``, ``lambda2``, ``ka``) and has at most two zeros, and the reduced
    derivative ``C'' + lambda1 C'`` changes sign at most once, between them (Rolle's
    theorem). Each interval between events is split where that sign change is
    (located by root finding); on each piece ``C'`` then has at most one zero, so the
    signs of ``C'`` at the two ends decide whether the piece contains a maximum, which
    is then located by root finding on ``C'`` (Illinois regula falsi interleaved with
    bisection, always bracketing). Derivative values below ``1e-280`` times the dose
    scale count as zero (near underflow their sign is unreliable).

    Precision: root finding stops when the bracket is at most 8 units in the last
    place of ``t`` wide. Tmax is then exact up to the band around the true Tmax in
    which rounding makes the sign of the computed ``C'`` unreliable, roughly
    ``eps * S / |C''(Tmax)|`` with ``S`` the sum of the magnitudes of the terms of
    ``C'`` (against closed forms, Tmax agreed to about 1e-15 relative).
    Cmax is evaluated at Tmax, and a Tmax error ``d`` changes ``C`` only by
    ``O(C'' d^2)``, so Cmax is accurate to rounding error. Two extrema closer together
    than that rounding band cannot be separated; Cmax is then still correct to
    rounding level.
    """
    return _extremum(model, doses, (start, end), maximum=True)


def cmin(model: Model, doses: Iterable[Dose], start: float, end: float) -> Extremum:
    """Return the infimum of ``C(t)`` over the closed window ``[start, end]`` and its time.

    Because ``C`` jumps up at IV bolus times, the infimum may be a left limit: a bolus
    at ``tb`` in ``(start, end]`` contributes ``C(tb-)`` (the trough before it), whereas
    a bolus exactly at ``start`` is part of ``C(start)``. For the pre-dose trough at a
    given time use :func:`trough`. Method and precision as for :func:`cmax`, with the
    derivative changing sign from - to +; ties resolve to the earliest time.
    """
    return _extremum(model, doses, (start, end), maximum=False)


def terminal_rate(model: Model) -> FloatArray:
    """Return ``lambda_z`` (1/h): the smallest disposition rate constant.

    For one compartment this is ``cl / v``; for two compartments it is the slower
    eigenvalue ``lambda1`` (beta). It governs the terminal decline after IV doses.
    """
    kin = kinetics(model)
    return finish(kin.rates[0][:, :, 0], kin, (), "terminal rate")


def terminal_half_life(model: Model) -> FloatArray:
    """Return the terminal (disposition) half-life ``ln 2 / lambda_z`` (h)."""
    return _LN2 / terminal_rate(model)


def apparent_oral_terminal_rate(model: Model) -> FloatArray:
    """Return the apparent terminal rate after oral doses, ``min(ka, lambda_z)`` (1/h).

    When absorption is slower than disposition (``ka < lambda_z``, "flip-flop"
    kinetics) the terminal phase of an oral curve reflects ``ka``, not elimination.
    Raises ``ValueError`` if the model has no ``ka``.
    """
    kin = kinetics(model)
    if kin.ka is None:
        raise ValueError("the model has no absorption rate constant (ka)")
    return finish(np.minimum(kin.ka, kin.rates[0])[:, :, 0], kin, (), "apparent terminal rate")


def apparent_oral_terminal_half_life(model: Model) -> FloatArray:
    """Return ``ln 2 / min(ka, lambda_z)`` (h); see :func:`apparent_oral_terminal_rate`."""
    return _LN2 / apparent_oral_terminal_rate(model)


def _extremum(
    model: Model, doses: Iterable[Dose], window: tuple[float, float], *, maximum: bool
) -> Extremum:
    kin, table = prepare(model, doses)
    ta = _validation.real_number("start", window[0])
    tb = _validation.real_number("end", window[1])
    if ta > tb:
        raise ValueError(f"start must be <= end; got [{ta}, {tb}]")
    rows = kin.rows
    what = "Cmax" if maximum else "Cmin"
    with np.errstate(all="ignore"):
        search = _Search.create(kin, table, maximum=maximum)
        grid, at_event = _grid(kin, table, ta, tb)
        event_row, event_column = np.nonzero(at_event)
        event_time = grid[event_row, event_column]
        all_rows = np.broadcast_to(np.arange(rows)[:, None], grid.shape)

        def one_sided(order: int) -> tuple[FloatArray, FloatArray]:
            right = search.at(all_rows.ravel(), grid.ravel(), Side.RIGHT, order).reshape(grid.shape)
            left = right.copy()
            left[event_row, event_column] = search.at(event_row, event_time, Side.LEFT, order)
            return right, left

        value = _evaluate(kin, table, grid, Side.RIGHT, 0)
        slope_right, slope_left = one_sided(1)
        bend_right, bend_left = one_sided(2)

        candidate_row = [all_rows.ravel()]
        candidate_value = [value.ravel()]
        candidate_time = [grid.ravel()]
        if not maximum:
            inside = event_time > ta
            candidate_row.append(event_row[inside])
            candidate_value.append(search.at(event_row[inside], event_time[inside], Side.LEFT, 0))
            candidate_time.append(event_time[inside])

        intervals = _Intervals(
            row=np.repeat(np.arange(rows), grid.shape[1] - 1),
            lo=grid[:, :-1].ravel(),
            hi=grid[:, 1:].ravel(),
            g_lo=slope_right[:, :-1].ravel(),
            g_hi=slope_left[:, 1:].ravel(),
        )
        bends = (bend_right[:, :-1].ravel(), bend_left[:, 1:].ravel())
        piece_row, lo, hi, g_lo, g_hi = _split(search, intervals, bends)
        # On each piece sign * C' has at most one zero: a change from + to <= 0 brackets a
        # local maximum of sign * C.
        turning = (g_lo > 0.0) & (g_hi <= 0.0) & (hi > lo)
        if turning.any():
            chosen = np.nonzero(turning)[0]
            critical_row = piece_row[chosen]

            def slope(x: FloatArray, index: IndexArray) -> FloatArray:
                return search.at(critical_row[index], x, Side.RIGHT, 1)

            critical = _refine(slope, (lo[chosen], hi[chosen]), (g_lo[chosen], g_hi[chosen]))
            candidate_row.append(critical_row)
            candidate_value.append(search.at(critical_row, critical, Side.RIGHT, 0))
            candidate_time.append(critical)

        values = np.concatenate(candidate_value)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{what} candidates are not finite (overflow)")
        best_value, best_time = _best(
            np.concatenate(candidate_row),
            values,
            np.concatenate(candidate_time),
            rows,
            maximum=maximum,
        )
    return Extremum(
        value=finish(best_value[:, None], kin, (), what),
        time=finish(best_time[:, None], kin, (), f"time of {what}"),
    )


@dataclass(frozen=True, eq=False)
class _Search:
    """Point evaluations for the extremum search.

    ``at(rows, times, side, order)`` returns ``C`` (order 0), ``sign * C'`` (order 1) or
    ``C'' + lambda1 C'`` (order 2) for row ``rows[i]`` at ``times[i]``. Derivative values
    whose magnitude is below ``_NOISE_FLOOR`` times the row's scale (dose amount over
    ``v1`` times the largest rate constant, to the power of the order) are set to
    zero: so close to underflow their sign is not reliable, and they correspond to a
    curve that is zero for all practical purposes.
    """

    kin: Kinetics
    table: DoseTable
    sign: float
    floors: tuple[FloatArray, FloatArray]

    @classmethod
    def create(cls, kin: Kinetics, table: DoseTable, *, maximum: bool) -> "_Search":
        amount = (
            table.bolus_amount.sum()
            + table.infusion_amount.sum()
            + kin.f[:, 0, 0] * table.oral_amount.sum()
        ) / kin.v1[:, 0, 0]
        fastest = kin.rates[-1][:, 0, 0]
        if kin.ka is not None and table.oral_amount.size:
            fastest = np.maximum(fastest, kin.ka[:, 0, 0])
        floors = (_NOISE_FLOOR * amount * fastest, _NOISE_FLOOR * amount * fastest * fastest)
        return cls(kin=kin, table=table, sign=1.0 if maximum else -1.0, floors=floors)

    def at(self, rows: IndexArray, times: FloatArray, side: Side, order: int) -> FloatArray:
        values = _pointwise(self.kin.take(rows), self.table, times, side, order)
        if order == 0:
            return values
        floor = self.floors[order - 1][rows]
        flushed = np.where(np.abs(values) < floor, 0.0, values)
        return self.sign * flushed if order == 1 else flushed


def _grid(kin: Kinetics, table: DoseTable, ta: float, tb: float) -> tuple[FloatArray, BoolArray]:
    """Sorted per-row points ``ta``, ``tb`` and every event time in ``(ta, tb]``.

    Other events are moved onto ``ta``, where duplicates are harmless; an event at
    ``tb`` matters through its left limit. Left limits differ from right values only
    at event times, so a point is marked if any entry with the same time is an event.
    """
    events = event_times(kin, table)
    events = np.where((events > ta) & (events <= tb), events, ta)
    ends = np.broadcast_to(np.array([ta, tb]), (kin.rows, 2))
    combined = np.concatenate([ends, events], axis=1)
    order = np.argsort(combined, axis=1, kind="stable")
    grid = np.take_along_axis(combined, order, axis=1)
    return grid, _shares_time_with_event(grid, order >= 2)


def _shares_time_with_event(grid: FloatArray, is_event: BoolArray) -> BoolArray:
    """Mark the entries of each sorted row whose time equals the time of an event entry."""
    rows, columns = grid.shape
    starts = np.ones(grid.shape, dtype=bool)
    starts[:, 1:] = grid[:, 1:] != grid[:, :-1]
    run = np.cumsum(starts, axis=1) - 1 + columns * np.arange(rows)[:, None]
    counts = np.bincount(run.ravel(), weights=is_event.ravel(), minlength=rows * columns)
    return np.asarray(counts[run] > 0.0, dtype=np.bool_)


class _Intervals(NamedTuple):
    """Flat per-interval data: row, ends and ``sign * C'`` just inside both ends."""

    row: IndexArray
    lo: FloatArray
    hi: FloatArray
    g_lo: FloatArray
    g_hi: FloatArray


def _split(
    search: _Search, intervals: _Intervals, bends: tuple[FloatArray, FloatArray]
) -> _Intervals:
    """Split every interval where the reduced derivative ``C'' + lambda1 C'`` changes sign.

    ``bends`` holds the reduced derivative just inside both ends of each interval. On
    the resulting pieces ``C'`` has at most one zero. A value that is zero (in practice
    below the noise floor, near underflow) counts as a change of sign, so the split
    then lands at the boundary of the region where the sign is known; extra splits are
    harmless.
    """
    row, lo, hi, g_lo, g_hi = intervals
    bend_lo, bend_hi = bends
    split = (bend_lo != 0.0) & (np.sign(bend_hi) != np.sign(bend_lo)) & (hi > lo)
    if not split.any():
        return intervals
    chosen = np.nonzero(split)[0]
    orientation = np.sign(bend_lo[chosen])
    split_row = row[chosen]

    def bend(x: FloatArray, index: IndexArray) -> FloatArray:
        return orientation[index] * search.at(split_row[index], x, Side.RIGHT, 2)

    middle = _refine(
        bend,
        (lo[chosen], hi[chosen]),
        (orientation * bend_lo[chosen], orientation * bend_hi[chosen]),
    )
    g_middle = search.at(split_row, middle, Side.RIGHT, 1)
    keep = np.nonzero(~split)[0]
    return _Intervals(
        row=np.concatenate([row[keep], split_row, split_row]),
        lo=np.concatenate([lo[keep], lo[chosen], middle]),
        hi=np.concatenate([hi[keep], middle, hi[chosen]]),
        g_lo=np.concatenate([g_lo[keep], g_lo[chosen], g_middle]),
        g_hi=np.concatenate([g_hi[keep], g_middle, g_hi[chosen]]),
    )


def _evaluate(
    kin: Kinetics, table: DoseTable, times: FloatArray, side: Side, order: int
) -> FloatArray:
    """``C``, ``C'`` or ``C'' + lambda1 C'`` (``order`` 0, 1, 2) at per-row times (rows, n)."""
    t = times[:, None, :]
    if order == 0:
        amount = central_amount(kin, table, t, side)
    elif order == 1:
        amount = central_slope(kin, table, t, side)
    else:
        amount = central_bend(kin, table, t, side)
    return amount / kin.v1[:, :, 0]


def _pointwise(
    kin: Kinetics, table: DoseTable, times: FloatArray, side: Side, order: int
) -> FloatArray:
    """Like :func:`_evaluate` at ``times[i]`` for row ``i`` of ``kin``."""
    return _evaluate(kin, table, times[:, None], side, order)[:, 0]


def _refine(
    g: Callable[[FloatArray, IndexArray], FloatArray],
    bracket: tuple[FloatArray, FloatArray],
    values: tuple[FloatArray, FloatArray],
) -> FloatArray:
    """Locate the change of ``g`` from ``> 0`` to ``<= 0`` inside each bracket ``[lo, hi]``.

    ``g(x, index)`` evaluates the function of the brackets ``index`` at ``x``;
    ``values`` holds it just inside both ends (``> 0`` at ``lo``, ``<= 0`` at ``hi``).
    Uses the Illinois variant of regula falsi with a bisection step every fourth
    iteration, so that each bracket at least halves every four steps, and stops
    refining a bracket once it is at most 8 units in the last place of its ends wide.
    """
    lo, hi = bracket[0].copy(), bracket[1].copy()
    g_lo, g_hi = values[0].copy(), values[1].copy()
    last_moved_lo = np.zeros(lo.shape, dtype=bool)
    last_moved_hi = np.zeros(lo.shape, dtype=bool)
    for iteration in range(_MAX_REFINEMENTS):
        width = hi - lo
        active = np.nonzero(width > _RESOLUTION * np.maximum(np.abs(lo), np.abs(hi)))[0]
        if active.size == 0:
            break
        a_lo, a_hi, a_width = lo[active], hi[active], width[active]
        middle = 0.5 * (a_lo + a_hi)
        if iteration % 4 == 3:
            x = middle
        else:
            x = a_lo + a_width * (g_lo[active] / (g_lo[active] - g_hi[active]))
            x = np.where((x > a_lo) & (x < a_hi), x, middle)
        gx = g(x, active)
        move_lo = gx > 0.0
        move_hi = ~move_lo
        # Illinois: when the same end moves twice in a row, halve the value kept at the
        # other end so that the next secant point falls on the other side of the change.
        g_hi[active] = np.where(move_lo & last_moved_lo[active], 0.5 * g_hi[active], g_hi[active])
        g_lo[active] = np.where(move_hi & last_moved_hi[active], 0.5 * g_lo[active], g_lo[active])
        lo[active] = np.where(move_lo, x, a_lo)
        hi[active] = np.where(move_hi, x, a_hi)
        g_lo[active] = np.where(move_lo, gx, g_lo[active])
        g_hi[active] = np.where(move_hi, gx, g_hi[active])
        last_moved_lo[active] = move_lo
        last_moved_hi[active] = move_hi
    return 0.5 * (lo + hi)


def _best(
    rows: IndexArray, values: FloatArray, times: FloatArray, count: int, *, maximum: bool
) -> tuple[FloatArray, FloatArray]:
    """Per row, the largest (or smallest) value and its earliest time."""
    order = np.lexsort((times, -values if maximum else values, rows))
    first = np.searchsorted(rows[order], np.arange(count))
    chosen = order[first]
    return values[chosen], times[chosen]
