# Conventions and numerical methods

This document states exactly what pkcore v0.1 computes. The conventions, metrics and
numerical methods below are exercised by the test suite (see [Verification](#verification));
the units table and the steady-state guidance are definitions or advice and are not tested,
and of the notes on other engines only the rxode2 and mrgsolve behaviour is tested.

## Units

pkcore never converts units. All inputs and outputs use one canonical set:

| Quantity | Unit |
| --- | --- |
| amount (dose, compartment amounts) | mg |
| volume (`v`, `v1`, `v2`) | L |
| time (dose times, durations, `tlag`, observation times) | h |
| clearance (`cl`, `q`) | L/h |
| rate constant (`ka`, disposition rates) | 1/h |
| concentration | mg/L |
| AUC | mg·h/L |

Any consistent set works if the caller converts everything beforehand (for example µg, L and
h give µg/L); the engine itself only sees numbers.

## Models

`OneCompartment(cl, v, *, ka=None, f=1.0, tlag=0.0)`:
`dA/dt = -(cl/v) A + input(t)`, concentration `C = A / v`.

`TwoCompartment(cl, v1, q, v2, *, ka=None, f=1.0, tlag=0.0)`, with micro-constants
`k10 = cl/v1`, `k12 = q/v1`, `k21 = q/v2`:

```
dA1/dt = -(k10 + k12) A1 + k21 A2 + input(t)
dA2/dt =  k12 A1         - k21 A2
C = A1 / v1
```

Oral doses enter a depot that empties into the central compartment at the first-order rate
`ka`. Elimination is linear and from the central compartment only.

Parameters are validated on construction: all finite; `cl`, `v`, `v1`, `v2`, `q`, `ka` > 0;
`f` in [0, 1]; `tlag` >= 0 (`ValueError` otherwise, `TypeError` for values that are not real
numbers). The rate constants (`ka` and `cl/v`, or `ka`, `cl/v1`, `q/v1`, `q/v2` and both
disposition eigenvalues) must be normal floating-point numbers no larger than `1e12` per hour
(a half-life of about 2.5 ns; faster processes are instantaneous for any pharmacokinetic
purpose); otherwise construction fails with `ValueError` rather than overflowing or
underflowing later. Each parameter is a scalar or a 1-D
array of parameter draws; all arrays of one model must have the same length `n`. Results
then have one row per draw: shape `(n,) + times.shape` instead of `times.shape`. Models,
doses and parameter arrays are immutable.

## Doses

`Dose(time, amount, route, duration=None)` with `route` one of

- `Route.IV_BOLUS`: the whole amount enters the central compartment at `time`;
- `Route.IV_INFUSION`: zero-order input at rate `amount / duration` from `time` to
  `time + duration` (`duration` > 0 is required);
- `Route.ORAL`: `f * amount` enters the depot at `time + tlag`.

Validation (`ValueError`, or `TypeError` for values that are not real numbers): `time` finite,
`amount` finite and >= 0 (negative zero is rejected), `duration` finite and > 0 for infusions
and absent for the other routes; `time + duration` and `time + tlag` must be finite. A dosing
history is any iterable of doses: times may be irregular, unsorted or negative, infusions may
overlap, and several doses may share a time. **Missed doses are simply not passed in** (the
caller filters them).

## Conventions

- **Right-continuity.** `concentration(model, doses, t)` is right-continuous: an IV bolus at
  `t0` is included in `C(t0)`. An observation exactly at a dose time therefore includes that
  dose.
- **Pre-dose values.** `trough(model, doses, t)` returns the left limit `C(t-)`: everything
  given strictly before `t`. It differs from `concentration` only where an IV bolus is given
  exactly at `t`, because infusions and first-order absorption are continuous. It is computed
  exactly, not by evaluating at `t - epsilon`.
- **Lag time.** An oral dose contributes from `t0 + tlag` on (nothing during the lag).
- **Event times in floating point.** The time at which an oral dose enters the depot is the
  floating-point number `t0 + tlag`, and an infusion ends at `t_end = t0 + duration` (rounded);
  every comparison (is a time before or after the event?) uses exactly these numbers. For
  example with `t0 = 0.7` and `tlag = 0.1` the input happens at `0.7999999999999999`. An
  infusion runs at the rate `amount / (t_end - t0)`, so the full amount is infused and `C` is
  continuous at `t_end`. Compared with exact arithmetic, the rounding (at most half a unit
  in the last place of the event time) changes `C` by at most about
  `ulp(t_end) / (2 duration)` relative during an infusion and `lambda * ulp(t) / 2` after an
  event (`lambda` the fastest rate constant involved).
- **Bioavailability.** `f` scales the oral amount. In v0.1 neither `f` nor `tlag` is applied
  to IV boluses or infusions.
- **Steady state.** There are no steady-state (SS) dose records in v0.1. Simulate enough prior
  doses instead: after `n` doses every `tau`, the remaining distance to steady state is of
  the order of `exp(-r n tau)`, where `r` is the slowest rate in the system:
  `terminal_rate(model)` for IV doses and `apparent_oral_terminal_rate(model)`
  (`min(ka, lambda_z)`) for oral doses.
- **No dosing events.** An empty history gives exactly zero for every output.
- **One drug per call.** A model plus its doses describes one drug. pkcore keeps no state
  between calls and has no function that adds up concentrations of different drugs.

### Where other engines may differ

NONMEM-style engines (for example NONMEM, mrgsolve and rxode2) can apply bioavailability
and lag times to IV dose records too. For infusions, whether such an engine changes the
infusion duration or its rate when `F` != 1 depends on whether the record fixes the rate or
the duration. They also support steady-state records. Engines that process event records in
order (NONMEM does) decide by the order of the records whether an observation at a dose time
is pre- or post-dose. Check these settings before comparing pkcore with another engine:
pkcore always reports the post-dose value from `concentration` and the pre-dose value from
`trough`.

The cross-engine golden tests (`tests/golden`) pin down the behaviour of rxode2 5.1.7.1 and
mrgsolve 2.0.1 for the cases pkcore supports. At a time with both a dose and an observation,
rxode2 reports the post-dose value whatever the record order, and mrgsolve follows the record
order. Bioavailability and lag times defined on the depot only behave as in pkcore. Their
closed forms (rxode2 `linCmt()`, mrgsolve `$PKMODEL`) fail when `ka` equals a disposition
rate: mrgsolve refuses `ka == k` for one compartment, and when `ka` equals the slow rate of
two compartments rxode2's closed form is off by −16 % to +42 % and mrgsolve's by +17 % to
+127 %, while their ODE paths agree with pkcore.

## Metrics

All metrics have the draw shape of the model (`()` or `(n,)`).

- `auc(model, doses, start, end)`: the integral of `C` over `[start, end]`, analytic and exact
  up to rounding; `start` may be `-inf` and `end` may be `+inf`.
- `cmax(model, doses, start, end)`: the maximum of `C` over the closed window and the earliest
  time at which it is attained (attained because `C` only jumps upwards). A bolus at `end` is
  part of `C(end)`.
- `cmin(model, doses, start, end)`: the infimum of `C` over the closed window. Because of the
  upward jumps it can be a left limit: a bolus at `tb` in `(start, end]` contributes `C(tb-)`,
  reported with time `tb`; a bolus at `start` is part of `C(start)`.
- `trough(model, doses, t)`: the left limit `C(t-)`.
- `terminal_rate(model)`: `lambda_z`, the smallest disposition rate constant (`cl/v`, or the
  slower eigenvalue of the two-compartment system); `terminal_half_life = ln 2 / lambda_z`.
- `apparent_oral_terminal_rate(model)`: `min(ka, lambda_z)`. When `ka < lambda_z` ("flip-flop"
  kinetics) the terminal phase after oral doses reflects absorption, not elimination.
  `apparent_oral_terminal_half_life = ln 2 / min(ka, lambda_z)`.
- `percentile_bands(model, doses, times, percentiles)`: pointwise percentiles across the
  model's parameter draws (`numpy.percentile`, linear interpolation). pkcore draws nothing;
  the caller passes parameter draws as arrays.

**Cmax/Tmax and Cmin method and precision.** No sampling grid is used. An extremum lies at
`start`, `end`, an event time (bolus, infusion start or end, lag-shifted oral dose) or at a
zero of `C'` between two events. Between two events `C'` is a combination of at most three
exponentials (rates `lambda1`, `lambda2`, `ka`), so it has at most two zeros there, and the
reduced derivative `C'' + lambda1 C'` = `exp(-lambda1 t) (exp(lambda1 t) C')'` has at most one
sign change, which by Rolle's theorem lies between them. This argument needs at most two
disposition rate constants plus `ka`, as in every model of v0.1; `cmax` and `cmin` raise
`NotImplementedError` for kinetics with more modes (three compartments, transit compartments,
metabolites) rather than search them without a proof. Each interval between events is split
at that sign change, after which `C'` has at most one zero per piece, so the signs of `C'` at
the ends of a piece decide whether it holds a maximum (or minimum); that zero is then located
by root finding on `C'` (Illinois regula falsi interleaved with bisection, always
bracketing). An interval whose ends have opposite signs of `C'` holds exactly one zero and is
not split, but only if both signs are trusted: `|C'|` must exceed `4096 eps S` at both ends,
with `S` the sum of the magnitudes of the terms of `C'` (the rounding error of `C'` is a
modest multiple of `eps S`; it grows with the arguments of the exponentials, which stay below
about 745 until they underflow). Near a turning point the computed sign of `C'` is rounding
noise, and spuriously opposite signs could hide two zeros. Derivative values below `1e-280`
times the dose scale (total dose over `v1`, times the largest rate constant per order of the
derivative) are treated as zero, because so close to underflow their sign is unreliable; such
a zero counts as a change of sign.

Root finding stops when the bracket is at most `8 eps |t|` wide (8 to 16 units in the last
place of `t`). Tmax is then exact up to the band around the true Tmax in which rounding makes
the sign of the computed `C'` unreliable, roughly `eps * S / |C''(Tmax)|` (against closed
forms, Tmax agreed to about 1e-15 relative; the tests assert 1e-12). Cmax and Cmin are
evaluated at the located time; an error `d` in the time changes `C` only by `O(C'' d^2)`, so
they are accurate to rounding error. Remaining limitation: a turning point within that band
of a window end, an event time or another turning point is not separated from it (the
computed signs of `C'` cannot tell on which side it lies); Cmax or Cmin is then the value
there, which differs only at rounding level. Ties between equal computed values resolve to the
earliest time; peaks that are equal only up to rounding (for example at steady state) may be
reported at any of their times.

## Numerical methods

**Superposition of closed-form unit responses.** The model is reduced to its modal form: the
central amount after a unit IV bolus is `sum_i c_i exp(-lambda_i t)` (one term for one
compartment, two for two compartments). Every dose contributes a closed-form response at
`t - t_dose`, and contributions are summed in chunks of doses vectorised with NumPy over
time points and parameter draws. All responses are sums of non-negative terms, so the
summation itself has no cancellation. There is no ODE solver and no matrix exponential in the
production path. (SciPy's `expm` loses accuracy for triangular matrices with nearly equal
diagonal entries, i.e. exactly when `ka` approaches a disposition rate.)

**Two-compartment modes.** The closed form is the micro-constant form of the ADVAN-style
solutions (Abuhelwa, Foster & Upton 2015); its cancellation-free evaluation is pkcore's own.
The discriminant is computed in the form
`(k10 - k21)^2 + k12 (k12 + 2 k10 + 2 k21)` (strictly positive), `lambda2 = (s + r)/2` and
`lambda1 = k10 k21 / lambda2` (no subtraction), and the coefficients
`c1 = (k21 - lambda1)/r`, `c2 = (lambda2 - k21)/r` from whichever of the two gaps has no
cancellation, using `(k21 - lambda1)(lambda2 - k21) = k12 k21` and
`(lambda2 - k21) - (k21 - lambda1) = (k10 - k21) + k12`. The only subtraction, `k10 - k21`,
is exact when the two are within a factor of two (Sterbenz). Both coefficients lie in
(0, 1).

**Kernels.** With `psi(x) = (1 - exp(-x))/x` (`psi(0) = 1`), evaluated with `expm1`:

| Route | contribution of mode `i` (per unit amount, central compartment) |
| --- | --- |
| IV bolus | `c_i exp(-lambda_i t)` for `t >= 0` |
| infusion, rate `R` | `R c_i u psi(lambda_i u) exp(-lambda_i w)` with `u = min(t, T)`, `w = max(t - T, 0)` |
| oral | `f ka c_i DD(lambda_i, ka; t - tlag)` |

where `DD(a, b; t) = (exp(-a t) - exp(-b t))/(b - a)` is evaluated as
`t exp(-min(a, b) t) psi(|b - a| t)`. This form is exact for `a == b` (giving
`t exp(-a t)`, Garrett's limit `C = F D k t exp(-k t)/V` for one compartment; Garrett 1994)
and stays accurate as `a -> b`: `|b - a|` is computed exactly for nearby floating-point
numbers and `psi` is smooth, so no digits are lost when `ka` and `k` agree to 6, 12 or 14
significant digits, whereas the textbook form loses about that many digits. The products
`t psi(r t)` are evaluated as `-expm1(-r t)/r`, which stays exact when `r t` overflows. No
expression contains `exp` of a positive argument.

**AUC.** Each dose's contribution is integrated analytically on its own time axis. For an
interval `[a, a + w]` the response is split into what is already in the system at `a`
(which decays) and what is still to come, e.g.
`DD(r, ka; a + s) = exp(-r s) DD(r, ka; a) + exp(-ka a) DD(r, ka; s)`. Every term is
non-negative, and the remaining integrals are `integral_0^w exp(-r s) ds = w psi(r w)` and
`integral_0^w DD(a, b; s) ds`, a second-order divided difference of the exponential. The
latter is evaluated with a Taylor series when `max(a, b) w < 1` and with a closed form that
loses at most about 1.5 bits otherwise (the approach of McCurdy, Ng & Parlett 1984).
Interval widths are differences of the original bounds, for example
`max(t2, t0) - max(t1, t0)` for an input starting at `t0`, which is exact for nearby bounds
(Sterbenz), so short intervals far from the dose keep full relative accuracy. Mass balance
(`AUC = eliminated amount / cl`) was not used because `eliminated = input - remaining`
cancels catastrophically for short intervals; instead, mass balance is tested as an identity
between the amount formulas and the AUC integrals.

**Compartment amounts.** `amounts(model, doses, times)` returns the depot, central and
peripheral amounts and the cumulative eliminated amount (`cl` times the AUC since the doses).
The peripheral response involves the same kernels plus the convolution of three
exponentials, evaluated as `exp(-lo t)` times the second divided difference above.

**Floating point.** Kernels run under `numpy.errstate(all="ignore")` because some
intermediate products may harmlessly overflow to infinity; every public result is then
checked, and a non-finite result raises `FloatingPointError` instead of being returned.
Results below the normal floating-point range (about 1e-308) are subnormal and lose
relative precision.

## Verification

Three levels, all of them verification (not validation): V1, analytic identities and
literature closed forms; V2, independent numerical methods; V3, agreement with established
engines. Most tests compare the engine with methods that share no code with it:

- numerical integration of the ODEs with `scipy.integrate.solve_ivp` (Radau or LSODA,
  `rtol = 1e-10`, analytic Jacobian), restarted at every event: agreement within
  `rtol = 1e-7` plus `1e-9` of the dose-scaled concentration;
- exact propagation with `scipy.linalg.expm`, with a Van Loan augmented matrix for infusions
  (Van Loan 1978), only for well-conditioned parameters (`ka` at least 5 % away from the
  disposition rates): `rtol = 1e-10`; the one-sided first and reduced second derivatives
  used by the metrics are checked against the ODE right-hand side on these exact states;
- the textbook formulas evaluated in 60- to 120-digit decimal arithmetic (for `ka` near or
  equal to a disposition rate, short AUC intervals, and the kernels themselves): `rtol`
  between a few units in the last place and `1e-12`;
- identities with independent expected values: single-dose AUC0-inf = F·D/CL for every
  route, the IV-bolus accumulation ratio `1/(1 - exp(-k tau))`, steady-state AUC over one
  interval, and mass balance (input = depot + central + peripheral + eliminated);
- literature closed forms for Tmax and Cmax (single dose, steady state, `ka == k`), root
  finding on textbook and reference derivatives, and textbook limits for extreme parameters;
- cross-engine golden tests (V3, `tests/golden`): 26 scenarios simulated with rxode2
  (`linCmt()` and ODE) and mrgsolve (`$PKMODEL` and ODE) in a pinned container; pkcore agrees
  within `1e-8` relative with the two closed forms (observed about 1e-15 away from `ka ~ k`)
  and `1e-7` with the ODE solutions (observed better than 1e-10; both use LSODA, so they
  count as one method). A CI job regenerates the engine outputs and fails on drift.

Consistency checks compare pkcore results with each other: linearity in dose,
superposition, time-shift invariance (bit-identical on a dyadic time grid), AUC additivity,
drug independence (bit-identical results whatever else is simulated, also concurrently),
vectorised against per-draw simulations, and extrema against densely sampled concentrations.
They cannot detect errors that respect the identity being checked.

Property-based tests (Hypothesis) draw random parameters and random dosing histories; CI uses
a derandomized profile.

## References

- Garrett ER. The Bateman function revisited: a critical reevaluation of the quantitative
  expressions to characterize concentrations in the one compartment body model as a function
  of time with first-order invasion and first-order elimination. *J Pharmacokinet
  Biopharm* 1994;22(2):103-128. doi:10.1007/BF02353538
- McCurdy AC, Ng KC, Parlett BN. Accurate computation of divided differences of the
  exponential function. *Math Comp* 1984;43(168):501-528. doi:10.1090/S0025-5718-1984-0758198-0
- Abuhelwa AY, Foster DJR, Upton RN. ADVAN-style analytical solutions for common
  pharmacokinetic models. *J Pharmacol Toxicol Methods* 2015;73:42-48.
  doi:10.1016/j.vascn.2015.03.004
- Van Loan CF. Computing integrals involving the matrix exponential. *IEEE Trans Automat
  Control* 1978;23(3):395-404. doi:10.1109/TAC.1978.1101743
