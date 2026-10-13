"""Summary metrics of simulated concentration curves.

Every metric has the draw shape of the model: a 0-d array for scalar parameters,
shape ``(n,)`` for ``n`` parameter draws. All metrics use the same closed forms as
:func:`pkcore.concentration`. AUC and troughs are exact; Cmax and Cmin are located on
a dense grid and then refined with the analytic derivative (see :func:`cmax`).
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pkcore import _validation
from pkcore._engine import (
    DoseTable,
    Side,
    central_amount,
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
_RESOLUTION = 4.0 * float(np.finfo(np.float64).eps)
_WINDOW_TOLERANCE = 1e-12
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


def cmax(
    model: Model, doses: Iterable[Dose], start: float, end: float, *, grid_points: int = 1001
) -> Extremum:
    """Return the maximum of ``C(t)`` over the closed window ``[start, end]`` and its time.

    Cmax is ``max C(t)`` (attained, because ``C`` is right-continuous with upward
    jumps only); Tmax is the earliest time at which it is attained.

    Method: ``C`` is evaluated on a grid of ``grid_points`` equally spaced times plus
    every event time in the window (bolus times, infusion starts and ends,
    lag-shifted oral dose times), so that ``C`` is smooth between neighbouring grid
    points. Every grid interval in which the analytic derivative changes sign from
    + to - is refined by root finding on the derivative (Illinois regula falsi with
    interleaved bisection, always keeping a bracket); the candidate with the largest
    value wins.

    Precision: the refinement stops when the bracket is narrower than
    ``1e-12 * (end - start)`` (or a few units in the last place of ``t``); in tests
    Tmax agrees with the closed form to better than 1e-11 relative. Cmax is evaluated
    at Tmax, and a Tmax error ``d`` changes ``C`` only by ``O(d^2)``, so Cmax is
    accurate to rounding error (about 1e-14 relative). This holds provided that no
    grid interval contains two local extrema; increase ``grid_points`` for curves
    with features narrower than ``(end - start) / grid_points``.
    """
    return _extremum(model, doses, (start, end), grid_points, maximum=True)


def cmin(
    model: Model, doses: Iterable[Dose], start: float, end: float, *, grid_points: int = 1001
) -> Extremum:
    """Return the infimum of ``C(t)`` over the closed window ``[start, end]`` and its time.

    Because ``C`` jumps up at IV bolus times, the infimum may be a left limit: a bolus
    at ``tb`` in ``(start, end]`` contributes ``C(tb-)`` (the trough before it), whereas
    a bolus exactly at ``start`` is part of ``C(start)``. For the pre-dose trough at a
    given time use :func:`trough`. Method and precision as for :func:`cmax`, with the
    derivative changing sign from - to +; ties resolve to the earliest time.
    """
    return _extremum(model, doses, (start, end), grid_points, maximum=False)


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
    model: Model,
    doses: Iterable[Dose],
    window: tuple[float, float],
    grid_points: int,
    *,
    maximum: bool,
) -> Extremum:
    kin, table = prepare(model, doses)
    ta = _validation.real_number("start", window[0])
    tb = _validation.real_number("end", window[1])
    if ta > tb:
        raise ValueError(f"start must be <= end; got [{ta}, {tb}]")
    if isinstance(grid_points, bool) or not isinstance(grid_points, int):
        raise TypeError("grid_points must be an int")
    if grid_points < 2:
        raise ValueError("grid_points must be >= 2")
    rows = kin.rows
    sign = 1.0 if maximum else -1.0
    with np.errstate(all="ignore"):
        # Grid: equally spaced points plus every event time in (ta, tb] (other events
        # are moved onto `ta`, where duplicates are harmless; an event at `tb` matters
        # through its left limit).
        events = event_times(kin, table)
        events = np.where((events > ta) & (events <= tb), events, ta)
        base = np.broadcast_to(np.linspace(ta, tb, grid_points), (rows, grid_points))
        combined = np.concatenate([base, events], axis=1)
        order = np.argsort(combined, axis=1, kind="stable")
        grid = np.take_along_axis(combined, order, axis=1)
        # Left limits differ from right values only at event times, so a grid point
        # needs its own left limit if any entry with the same time is an event.
        at_event = _shares_time_with_event(grid, order >= grid_points)
        event_row, event_column = np.nonzero(at_event)
        event_time = grid[event_row, event_column]

        right_value = _evaluate(kin, table, grid, Side.RIGHT, slope=False)
        right_slope = _evaluate(kin, table, grid, Side.RIGHT, slope=True)
        left_slope = right_slope.copy()
        left_slope[event_row, event_column] = _pointwise(
            kin.take(event_row), table, event_time, Side.LEFT, slope=True
        )

        row_index = np.broadcast_to(np.arange(rows)[:, None], grid.shape)
        candidate_row = [row_index.ravel()]
        candidate_value = [right_value.ravel()]
        candidate_time = [grid.ravel()]
        if not maximum:
            inside = event_time > ta
            candidate_row.append(event_row[inside])
            candidate_value.append(
                _pointwise(
                    kin.take(event_row[inside]), table, event_time[inside], Side.LEFT, slope=False
                )
            )
            candidate_time.append(event_time[inside])

        # Grid intervals in which sign * dC/dt changes from + to - contain a local
        # maximum of sign * C; C is smooth inside every interval.
        turning = (sign * right_slope[:, :-1] > 0.0) & (sign * left_slope[:, 1:] < 0.0)
        flagged_row, flagged_column = np.nonzero(turning)
        if flagged_row.size:
            flagged = kin.take(flagged_row)

            def g(x: FloatArray) -> FloatArray:
                return sign * _pointwise(flagged, table, x, Side.RIGHT, slope=True)

            refined_time = _refine(
                g,
                (grid[flagged_row, flagged_column], grid[flagged_row, flagged_column + 1]),
                (
                    sign * right_slope[flagged_row, flagged_column],
                    sign * left_slope[flagged_row, flagged_column + 1],
                ),
                _WINDOW_TOLERANCE * (tb - ta),
            )
            candidate_row.append(flagged_row)
            candidate_value.append(
                _pointwise(flagged, table, refined_time, Side.RIGHT, slope=False)
            )
            candidate_time.append(refined_time)
        values = np.concatenate(candidate_value)
        what = "Cmax" if maximum else "Cmin"
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{what} candidates are not finite (overflow)")
        value, time = _best(
            np.concatenate(candidate_row),
            values,
            np.concatenate(candidate_time),
            rows,
            maximum=maximum,
        )
    return Extremum(
        value=finish(value[:, None], kin, (), what),
        time=finish(time[:, None], kin, (), f"time of {what}"),
    )


def _shares_time_with_event(grid: FloatArray, is_event: BoolArray) -> BoolArray:
    """Mark the entries of each sorted row whose time equals the time of an event entry."""
    rows, columns = grid.shape
    starts = np.ones(grid.shape, dtype=bool)
    starts[:, 1:] = grid[:, 1:] != grid[:, :-1]
    run = np.cumsum(starts, axis=1) - 1 + columns * np.arange(rows)[:, None]
    counts = np.bincount(run.ravel(), weights=is_event.ravel(), minlength=rows * columns)
    return np.asarray(counts[run] > 0.0, dtype=np.bool_)


def _evaluate(
    kin: Kinetics, table: DoseTable, times: FloatArray, side: Side, *, slope: bool
) -> FloatArray:
    """Concentration or its time derivative at per-row times of shape (rows, n)."""
    t = times[:, None, :]
    amount = central_slope(kin, table, t, side) if slope else central_amount(kin, table, t, side)
    return amount / kin.v1[:, :, 0]


def _pointwise(
    kin: Kinetics, table: DoseTable, times: FloatArray, side: Side, *, slope: bool
) -> FloatArray:
    """Concentration or its time derivative at ``times[i]`` for row ``i`` of ``kin``."""
    return _evaluate(kin, table, times[:, None], side, slope=slope)[:, 0]


def _refine(
    g: Callable[[FloatArray], FloatArray],
    bracket: tuple[FloatArray, FloatArray],
    values: tuple[FloatArray, FloatArray],
    tolerance: float,
) -> FloatArray:
    """Locate a sign change of ``g`` from + to - inside each bracket ``[lo, hi]``.

    ``values`` holds ``g`` just inside both ends (positive at ``lo``, negative at
    ``hi``). Uses the Illinois variant of regula falsi, with a bisection step every
    fourth iteration so that each bracket at least halves every four steps, and stops
    once every bracket is narrower than ``tolerance`` or a few units in the last place
    of its end points.
    """
    lo, hi = bracket
    g_lo, g_hi = values
    last_moved_lo = np.zeros(lo.shape, dtype=bool)
    last_moved_hi = np.zeros(lo.shape, dtype=bool)
    for iteration in range(_MAX_REFINEMENTS):
        width = hi - lo
        active = width > np.maximum(tolerance, _RESOLUTION * np.maximum(np.abs(lo), np.abs(hi)))
        if not active.any():
            break
        middle = 0.5 * (lo + hi)
        if iteration % 4 == 3:
            x = middle
        else:
            x = lo + width * (g_lo / (g_lo - g_hi))
            x = np.where((x > lo) & (x < hi), x, middle)
        gx = g(x)
        move_lo = active & (gx > 0.0)
        move_hi = active & (gx < 0.0)
        root = active & (gx == 0.0)
        lo = np.where(move_lo | root, x, lo)
        hi = np.where(move_hi | root, x, hi)
        # Illinois: when the same end moves twice in a row, halve the value kept at the
        # other end so that the next secant point falls on the other side of the root.
        g_hi = np.where(move_lo & last_moved_lo, 0.5 * g_hi, g_hi)
        g_lo = np.where(move_hi & last_moved_hi, 0.5 * g_lo, g_lo)
        g_lo = np.where(move_lo, gx, g_lo)
        g_hi = np.where(move_hi, gx, g_hi)
        last_moved_lo = np.where(active, move_lo, last_moved_lo)
        last_moved_hi = np.where(active, move_hi, last_moved_hi)
    return 0.5 * (lo + hi)


def _best(
    rows: IndexArray, values: FloatArray, times: FloatArray, count: int, *, maximum: bool
) -> tuple[FloatArray, FloatArray]:
    """Per row, the largest (or smallest) value and its earliest time."""
    order = np.lexsort((times, -values if maximum else values, rows))
    first = np.searchsorted(rows[order], np.arange(count))
    chosen = order[first]
    return values[chosen], times[chosen]
