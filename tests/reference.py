"""Independent reference implementations, used only by the tests.

Nothing here imports pkcore's numerical code: the references read dose records
(plain data) and the raw parameters, and solve the compartmental model in their own,
unrelated ways.

* :func:`ode_solution` integrates the ODE system with ``scipy.integrate.solve_ivp``
  (Radau or LSODA, analytic Jacobian, ``rtol = 1e-10``), restarting the integration
  at every event (bolus, oral input after the lag, infusion start and stop) and every
  observation time.
* :func:`expm_solution` propagates the state exactly with ``scipy.linalg.expm``; a
  constant infusion rate enters through the Van Loan augmented matrix
  ``[[A, b], [0, 0]]`` (Van Loan 1978, doi:10.1109/TAC.1978.1101743). SciPy's ``expm``
  loses accuracy for triangular matrices with nearly equal diagonal entries (for
  example ``ka`` close to a disposition eigenvalue), so this reference is used only
  for well-conditioned parameter sets (see :func:`well_conditioned`).
* :func:`decimal_oral_concentration` evaluates the textbook single-dose oral formulas
  in 60-digit decimal arithmetic, so their cancellation for ``ka ~ lambda`` is
  harmless; the ``ka == k`` limit is Garrett (1994), doi:10.1007/BF02353538.
* :func:`decimal_extremum` locates Cmax or Cmin of IV bolus and oral doses by
  bisection on the textbook derivative, also in 60-digit decimal arithmetic.

All references use the same state vector: depot, central, peripheral, eliminated
amounts and the AUC of the central concentration.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pkcore import Dose, Route

FloatArray = npt.NDArray[np.float64]

STATES = ("depot", "central", "peripheral", "eliminated", "auc")
_SHORT_SEGMENT = 1e-9  # h


@dataclass(frozen=True)
class Parameters:
    """Raw model parameters. ``q = 0`` gives a one-compartment model with volume ``v1``."""

    cl: float
    v1: float
    q: float = 0.0
    v2: float = 1.0
    ka: float = 0.0
    f: float = 1.0
    tlag: float = 0.0

    def matrix(self) -> FloatArray:
        """System matrix ``A`` of ``dy/dt = A y + b`` for the state vector ``STATES``."""
        k10 = self.cl / self.v1
        k12 = self.q / self.v1
        k21 = self.q / self.v2
        return np.array(
            [
                [-self.ka, 0.0, 0.0, 0.0, 0.0],
                [self.ka, -(k10 + k12), k21, 0.0, 0.0],
                [0.0, k12, -k21, 0.0, 0.0],
                [0.0, k10, 0.0, 0.0, 0.0],
                [0.0, 1.0 / self.v1, 0.0, 0.0, 0.0],
            ]
        )


@dataclass(frozen=True)
class Solution:
    """States at the observation times: right-continuous values and left limits."""

    right: dict[str, FloatArray]
    left: dict[str, FloatArray]
    v1: float

    @property
    def concentration(self) -> FloatArray:
        return self.right["central"] / self.v1

    @property
    def trough(self) -> FloatArray:
        return self.left["central"] / self.v1


def infusion_rate(doses: Sequence[Dose], t: float, *, left: bool = False) -> float:
    """Total infusion rate just after ``t`` (or just before it, with ``left=True``)."""
    total = 0.0
    for d in doses:
        if d.duration is None:
            continue
        end = d.time + d.duration
        if (d.time < t <= end) if left else (d.time <= t < end):
            total += d.amount / d.duration
    return total


def _jumps(p: Parameters, doses: Sequence[Dose], t: float) -> FloatArray:
    """Instantaneous state change at time ``t`` (boluses and oral inputs after the lag)."""
    jump = np.zeros(len(STATES))
    for d in doses:
        if d.route is Route.IV_BOLUS and d.time == t:
            jump[1] += d.amount
        elif d.route is Route.ORAL and d.time + p.tlag == t:
            jump[0] += p.f * d.amount
    return jump


def _breakpoints(p: Parameters, doses: Sequence[Dose], times: FloatArray) -> list[float]:
    points = {float(t) for t in times}
    for d in doses:
        if d.route is Route.ORAL:
            points.add(d.time + p.tlag)
        else:
            points.add(d.time)
        if d.route is Route.IV_INFUSION and d.duration is not None:
            points.add(d.time + d.duration)
    return sorted(points)


Step = Callable[[FloatArray, float, float, float], FloatArray]
"""Propagates a state from ``t0`` to ``t1`` with a constant infusion ``rate``."""


def _solve(p: Parameters, doses: Sequence[Dose], times: npt.ArrayLike, step: Step) -> Solution:
    obs = np.asarray(times, dtype=float)
    right = {name: np.zeros(obs.shape) for name in STATES}
    left = {name: np.zeros(obs.shape) for name in STATES}
    points = _breakpoints(p, doses, obs)
    y = np.zeros(len(STATES))
    for i, t in enumerate(points):
        hit = obs == t
        for k, name in enumerate(STATES):
            left[name][hit] = y[k]
        y = y + _jumps(p, doses, t)
        for k, name in enumerate(STATES):
            right[name][hit] = y[k]
        if i + 1 < len(points):
            y = step(y, t, points[i + 1], infusion_rate(doses, t))
    return Solution(right=right, left=left, v1=p.v1)


def _ode_step(p: Parameters, method: str, rtol: float, atol: float) -> Step:
    a = p.matrix()

    def jac(_t: float, _y: FloatArray) -> FloatArray:
        return a

    def step(y: FloatArray, t0: float, t1: float, rate: float) -> FloatArray:
        b = np.zeros(len(STATES))
        b[1] = rate

        def rhs(_t: float, state: FloatArray) -> FloatArray:
            return a @ state + b

        if t1 - t0 < _SHORT_SEGMENT:
            # Adaptive integrators can stall on absurdly short spans (e.g. [0, 1e-299]);
            # one Euler step is exact to O(h^2) there, far below the tolerances used.
            return y + (t1 - t0) * rhs(t0, y)
        result = solve_ivp(rhs, (t0, t1), y, method=method, jac=jac, rtol=rtol, atol=atol)
        if not result.success:
            raise RuntimeError(f"solve_ivp failed: {result.message}")
        return np.asarray(result.y[:, -1], dtype=float)

    return step


def _expm_step(p: Parameters) -> Step:
    a = p.matrix()
    n = len(STATES)

    def step(y: FloatArray, t0: float, t1: float, rate: float) -> FloatArray:
        augmented = np.zeros((n + 1, n + 1))
        augmented[:n, :n] = a
        augmented[1, n] = rate
        propagator = np.asarray(expm(augmented * (t1 - t0)), dtype=np.float64)
        return np.asarray(propagator[:n, :n] @ y + propagator[:n, n], dtype=np.float64)

    return step


def ode_solution(
    p: Parameters,
    doses: Sequence[Dose],
    times: npt.ArrayLike,
    *,
    method: str = "Radau",
    rtol: float = 1e-10,
) -> Solution:
    """Reference solution by numerical ODE integration, segmented at every event.

    ``method`` is ``"Radau"`` (implicit Runge-Kutta, the default) or ``"LSODA"``
    (ODEPACK, much faster); both use the analytic Jacobian and ``atol`` scaled to
    the largest dose.
    """
    scale = max([1.0] + [d.amount for d in doses])
    return _solve(p, doses, times, _ode_step(p, method, rtol=rtol, atol=1e-14 * scale))


def expm_solution(p: Parameters, doses: Sequence[Dose], times: npt.ArrayLike) -> Solution:
    """Reference solution by matrix-exponential propagation (well-conditioned cases only)."""
    return _solve(p, doses, times, _expm_step(p))


def well_conditioned(p: Parameters, *, separation: float = 0.05) -> bool:
    """True if ``ka`` (when used) is well separated from both disposition eigenvalues."""
    if p.ka == 0.0:
        return True
    eigenvalues = np.linalg.eigvals(p.matrix()[1:3, 1:3]) if p.q > 0 else [-p.cl / p.v1]
    return all(abs(p.ka + float(np.real(e))) > separation * p.ka for e in eigenvalues)


def decimal_oral_concentration(p: Parameters, dose: float, t: float) -> float:
    """Single oral dose at time 0: textbook closed form in 60-digit decimal arithmetic.

    One compartment: the Bateman function, or ``f D k t exp(-k t) / V`` when
    ``ka == k`` exactly. Two compartments: the standard tri-exponential solution with
    coefficients ``(k21 - lambda_i) / ((ka - lambda_i)(lambda_j - lambda_i))`` and
    ``(k21 - ka) / ((lambda1 - ka)(lambda2 - ka))``.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        cl, v1, q, v2 = (Decimal(x) for x in (p.cl, p.v1, p.q, p.v2))
        ka, f, amount = Decimal(p.ka), Decimal(p.f), Decimal(dose)
        tau = Decimal(t) - Decimal(p.tlag)
        if tau <= 0:
            return 0.0
        scale = f * amount * ka / v1
        if q == 0:
            k = cl / v1
            if ka == k:
                value = f * amount * k * tau * (-k * tau).exp() / v1
            else:
                value = scale * ((-k * tau).exp() - (-ka * tau).exp()) / (ka - k)
            return float(value)
        k10, k12, k21 = cl / v1, q / v1, q / v2
        total = k10 + k12 + k21
        root = (total * total - 4 * k10 * k21).sqrt()
        lam1, lam2 = (total - root) / 2, (total + root) / 2
        value = (
            (k21 - lam1) / ((ka - lam1) * (lam2 - lam1)) * (-lam1 * tau).exp()
            + (k21 - lam2) / ((ka - lam2) * (lam1 - lam2)) * (-lam2 * tau).exp()
            + (k21 - ka) / ((lam1 - ka) * (lam2 - ka)) * (-ka * tau).exp()
        )
        return float(scale * value)


def decimal_oral_peripheral_amount(p: Parameters, dose: float, t: float) -> float:
    """Single oral dose at time 0: peripheral amount of the two-compartment model.

    ``f D ka k12 sum_i exp(-r_i tau) / prod_{j != i} (r_j - r_i)`` over the three rates
    ``r = (lambda1, lambda2, ka)``, in 60-digit decimal arithmetic.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        cl, v1, q, v2 = (Decimal(x) for x in (p.cl, p.v1, p.q, p.v2))
        ka, f, amount = Decimal(p.ka), Decimal(p.f), Decimal(dose)
        tau = Decimal(t) - Decimal(p.tlag)
        if tau <= 0:
            return 0.0
        k10, k12, k21 = cl / v1, q / v1, q / v2
        total = k10 + k12 + k21
        root = (total * total - 4 * k10 * k21).sqrt()
        rates = ((total - root) / 2, (total + root) / 2, ka)
        value = Decimal(0)
        for i, r in enumerate(rates):
            denominator = Decimal(1)
            for j, other in enumerate(rates):
                if j != i:
                    denominator *= other - r
            value += (-r * tau).exp() / denominator
        return float(f * amount * ka * k12 * value)


def decimal_oral_auc(
    p: Parameters, dose: float, t: float, *, start: float | None = None, dose_time: float = 0.0
) -> float:
    """Single oral dose, one compartment: AUC over ``[start, t]`` (default from the dose).

    Integral of the Bateman function, or of Garrett's limit when ``ka == k``, as the
    difference of its antiderivative at the two ends, in 80-digit decimal arithmetic.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        cl, v = Decimal(p.cl), Decimal(p.v1)
        ka, f, amount = Decimal(p.ka), Decimal(p.f), Decimal(dose)
        k = cl / v

        def antiderivative(x: float) -> Decimal:
            tau = Decimal(x) - Decimal(dose_time) - Decimal(p.tlag)
            if tau <= 0:
                return Decimal(0)
            if ka == k:
                return f * amount / (v * k) * (1 - (-k * tau).exp() * (1 + k * tau))
            one_minus = ((1 - (-k * tau).exp()) / k) - ((1 - (-ka * tau).exp()) / ka)
            return f * amount * ka / (v * (ka - k)) * one_minus

        lower = Decimal(0) if start is None else antiderivative(start)
        return float(antiderivative(t) - lower)


def concentration_derivatives(
    p: Parameters, doses: Sequence[Dose], times: npt.ArrayLike
) -> dict[str, FloatArray]:
    """One-sided time derivatives of ``C`` from exact states and the ODE right-hand side.

    The states come from :func:`expm_solution` (right values and left limits). Between
    events ``dy/dt = A y + b`` and ``d2y/dt2 = A (A y + b)`` with the infusion rate in
    ``b`` taken just after (right) or just before (left) each time. Returns the first
    derivative (``"slope_right"``, ``"slope_left"``), ``C'' + lambda1 C'`` with
    ``lambda1`` the smallest disposition eigenvalue from LAPACK (``"bend_right"``,
    ``"bend_left"``), and, for tolerances, the sums of the magnitudes of the terms
    (``"slope_scale"``, ``"bend_scale"``).
    """
    obs = np.asarray(times, dtype=float)
    solution = expm_solution(p, doses, obs)
    a = p.matrix()
    if p.q > 0:
        disposition = a[1:3, 1:3]
        lam1 = float(np.min(-np.linalg.eigvals(disposition).real))
    else:
        lam1 = p.cl / p.v1
    out = {
        name: np.zeros(obs.shape)
        for name in (
            "slope_right",
            "slope_left",
            "bend_right",
            "bend_left",
            "slope_scale",
            "bend_scale",
        )
    }
    for i, t in enumerate(obs):
        for side, states, rate in (
            ("right", solution.right, infusion_rate(doses, float(t))),
            ("left", solution.left, infusion_rate(doses, float(t), left=True)),
        ):
            y = np.array([states[name][i] for name in STATES])
            b = np.zeros(len(STATES))
            b[1] = rate
            dy = a @ y + b
            d2y = a @ dy
            out[f"slope_{side}"][i] = dy[1] / p.v1
            out[f"bend_{side}"][i] = (d2y[1] + lam1 * dy[1]) / p.v1
            slope_terms = float(np.abs(a[1]) @ np.abs(y)) + rate
            bend_terms = float(np.abs(a[1]) @ np.abs(dy)) + lam1 * abs(float(dy[1]))
            out["slope_scale"][i] = max(out["slope_scale"][i], slope_terms / p.v1)
            out["bend_scale"][i] = max(out["bend_scale"][i], bend_terms / p.v1)
    return out


def concentration_slopes(
    p: Parameters, doses: Sequence[Dose], times: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One-sided ``dC/dt`` (right, left) and a scale; see :func:`concentration_derivatives`."""
    d = concentration_derivatives(p, doses, times)
    return d["slope_right"], d["slope_left"], d["slope_scale"]


def decimal_one_compartment_auc(p: Parameters, dose: Dose, start: float, end: float) -> float:
    """AUC over ``[start, end]`` of one dose in the one-compartment model, in decimal.

    The difference of the antiderivative of the textbook concentration (bolus, constant-
    rate infusion, or Bateman function) at the two ends, in 80-digit arithmetic.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        cl, v = Decimal(p.cl), Decimal(p.v1)
        k = cl / v
        t0, amount = Decimal(dose.time), Decimal(dose.amount)

        def exposure(x: float) -> Decimal:
            """Integral of C from minus infinity to x."""
            t = Decimal(x)
            if dose.route is Route.IV_BOLUS:
                tau = t - t0
                return Decimal(0) if tau <= 0 else amount / cl * (1 - (-k * tau).exp())
            if dose.route is Route.IV_INFUSION and dose.duration is not None:
                duration = Decimal(dose.duration)
                rate = amount / duration
                tau = t - t0
                if tau <= 0:
                    return Decimal(0)
                running = min(tau, duration)
                during = rate / cl * (running - (1 - (-k * running).exp()) / k)
                if tau <= duration:
                    return during
                at_end = rate / cl * (1 - (-k * duration).exp())
                return during + at_end / k * (1 - (-k * (tau - duration)).exp())
            ka, f = Decimal(p.ka), Decimal(p.f)
            tau = t - t0 - Decimal(p.tlag)
            if tau <= 0:
                return Decimal(0)
            one_minus = (1 - (-k * tau).exp()) / k - (1 - (-ka * tau).exp()) / ka
            return f * amount * ka / (v * (ka - k)) * one_minus

        return float(exposure(end) - exposure(start))


def decimal_bolus_oral(p: Parameters, doses: Sequence[Dose], t: Decimal, order: int) -> Decimal:
    """``C`` (order 0) or ``dC/dt`` (order 1) of a two-compartment model, in decimal.

    Textbook bi- and tri-exponential solutions for IV boluses and oral doses (no
    infusions), summed over the doses; 60-digit arithmetic. Use within a
    ``localcontext`` with at least that precision.
    """
    cl, v1, q, v2 = (Decimal(x) for x in (p.cl, p.v1, p.q, p.v2))
    ka, f, tlag = Decimal(p.ka), Decimal(p.f), Decimal(p.tlag)
    k10, k12, k21 = cl / v1, q / v1, q / v2
    total = k10 + k12 + k21
    root = (total * total - 4 * k10 * k21).sqrt()
    lam = ((total - root) / 2, (total + root) / 2)
    weight = ((k21 - lam[0]) / (lam[1] - lam[0]), (lam[1] - k21) / (lam[1] - lam[0]))

    def term(rate: Decimal, coefficient: Decimal, tau: Decimal) -> Decimal:
        return coefficient * (-rate) ** order * (-rate * tau).exp()

    value = Decimal(0)
    for d in doses:
        if d.route is Route.IV_BOLUS:
            tau = t - Decimal(d.time)
            if tau >= 0:
                value += (
                    Decimal(d.amount)
                    / v1
                    * sum(term(r, c, tau) for r, c in zip(lam, weight, strict=True))
                )
        elif d.route is Route.ORAL:
            tau = t - Decimal(d.time) - tlag
            if tau > 0:
                a = [(k21 - lam[i]) / ((ka - lam[i]) * (lam[1 - i] - lam[i])) for i in (0, 1)]
                b = (k21 - ka) / ((lam[0] - ka) * (lam[1] - ka))
                parts = term(lam[0], a[0], tau) + term(lam[1], a[1], tau) + term(ka, b, tau)
                value += f * Decimal(d.amount) * ka / v1 * parts
        else:
            raise ValueError("decimal_bolus_oral supports IV boluses and oral doses only")
    return value


def decimal_extremum(
    p: Parameters, doses: Sequence[Dose], start: float, end: float, *, maximum: bool
) -> tuple[float, float]:
    """(value, time) of the maximum or minimum of ``C`` over ``[start, end]``, in decimal.

    Candidates are both ends and every zero of ``dC/dt`` found by a sign scan on 400
    sub-intervals followed by 200 bisection steps (the window must contain no dose, and
    two zeros within one sub-interval would be missed).
    """
    with localcontext() as ctx:
        ctx.prec = 60
        a, b = Decimal(start), Decimal(end)
        points = [a + (b - a) * i / 400 for i in range(401)]
        slopes = [decimal_bolus_oral(p, doses, x, 1) for x in points]
        candidates = [a, b]
        for i in range(400):
            if (slopes[i] > 0) != (slopes[i + 1] > 0):
                lo, hi = points[i], points[i + 1]
                rising = slopes[i] > 0
                for _ in range(200):
                    mid = (lo + hi) / 2
                    if (decimal_bolus_oral(p, doses, mid, 1) > 0) == rising:
                        lo = mid
                    else:
                        hi = mid
                candidates.append((lo + hi) / 2)
        values = [(decimal_bolus_oral(p, doses, x, 0), x) for x in candidates]
        value, time = max(values) if maximum else min(values)
        return float(value), float(time)
