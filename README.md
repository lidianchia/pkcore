# pkcore

A small, verifiable pharmacokinetic (PK) computation engine for Python.

> **Not a medical device.** pkcore is for research and education. It does not recommend
> doses, and its output must not be used to start, stop or change any medication.

**Status:** pre-alpha. Nothing is released and the API will change.

## Scope

pkcore computes drug-amount and concentration time courses from a model and a dosing history
that the caller supplies. It ships **no drug-specific models or parameters**: every parameter
comes from the caller, together with the caller's own source for it.

Version 0.1 provides:

- Linear one- and two-compartment models with IV bolus, zero-order IV infusion and
  first-order absorption with lag time and bioavailability; arbitrary, irregular dosing
  histories (overlapping infusions, unsorted or negative times; a missed dose is simply left
  out).
- Closed-form superposition (no ODE solver), numerically stable also when the absorption rate
  equals or nearly equals a disposition rate, vectorised over time points and parameter draws.
- Exact AUC over any interval (including AUC to infinity) and summary metrics on simulated
  curves: Cmax/Tmax, Cmin over an explicitly defined interval, the pre-dose trough, the
  terminal half-life and the apparent terminal rate after oral doses.
- Percentile bands over parameter draws that the caller supplies.

Units are canonical and never converted: amounts in mg, volumes in L, times in h, clearances
in L/h, rate constants in 1/h, concentrations in mg/L. Not in v0.1: three-compartment models,
steady-state (SS) dose records, unit conversion, nonlinear elimination.

The conventions (what happens exactly at a dose time, where F and the lag time apply, how
other engines may differ) and the numerical methods are documented in
[docs/conventions.md](docs/conventions.md).

## Example

The parameters below are fictional and purely illustrative.

```python
import numpy as np

import pkcore as pk

model = pk.TwoCompartment(cl=4.0, v1=30.0, q=6.0, v2=60.0, ka=1.2, f=0.8, tlag=0.5)
doses = [pk.Dose(time=12.0 * i, amount=100.0, route=pk.Route.ORAL) for i in range(6)]
doses.append(pk.Dose(time=80.0, amount=200.0, route=pk.Route.IV_INFUSION, duration=2.0))

times = np.linspace(0.0, 96.0, 193)  # h
concentration = pk.concentration(model, doses, times)  # mg/L, shape (193,)

auc = pk.auc(model, doses, 60.0, 72.0)  # mg*h/L, exact
peak = pk.cmax(model, doses, 60.0, 72.0)  # Cmax (peak.value) and Tmax (peak.time)
pre_dose = pk.trough(model, doses, 60.0)  # left limit just before the 60 h dose
print(f"AUC 60-72 h: {auc:.3f} mg*h/L; Cmax {peak.value:.3f} mg/L at {peak.time:.2f} h")

# Monte Carlo: the caller draws parameters; all draws are simulated in one call.
rng = np.random.default_rng(seed=1)
draws = pk.TwoCompartment(
    cl=4.0 * rng.lognormal(0.0, 0.3, size=500), v1=30.0, q=6.0, v2=60.0, ka=1.2, f=0.8, tlag=0.5
)
bands = pk.percentile_bands(draws, doses, times, percentiles=(5, 50, 95))  # shape (3, 193)
```

## Verification, not validation

The test suite **verifies** that the code reproduces the mathematics: analytic identities
(for example AUC0-inf = F·D/CL, mass balance, linearity, time-shift invariance), independent
numerical ODE integration and matrix-exponential solutions, and the textbook formulas evaluated
in high-precision decimal arithmetic. Documented cross-checks against established open-source
PK engines are planned. Passing these tests says nothing about whether any particular drug
model describes any particular person; that is model **validation**, which needs external
clinical data and is outside this library.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io): CI runs a fixed,
derandomized profile; set `HYPOTHESIS_PROFILE=thorough` locally for many more examples.

## License

MIT, see [LICENSE](LICENSE). Provided without warranty of any kind.
