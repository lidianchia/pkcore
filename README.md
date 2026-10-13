# pkcore

A small, verifiable pharmacokinetic (PK) computation engine for Python.

> **Not a medical device.** pkcore is for research and education. It does not recommend
> doses, and its output must not be used to start, stop or change any medication.

**Status:** pre-alpha. Nothing is released and the API will change.

## Scope

pkcore computes drug-amount and concentration time courses from a model and a dosing history
that the caller supplies. It ships **no drug-specific models or parameters**: every parameter
comes from the caller, together with the caller's own source for it.

Planned for the first version:

- Event-driven simulation of linear compartmental models: IV bolus, zero-order infusion,
  first-order absorption with lag time and bioavailability; irregular and missed doses.
- Exact AUC over any interval, and summary metrics on simulated curves (Cmax, Tmax,
  Cmin over an explicitly defined interval, terminal half-life where defined).
- Unit handling for mass, volume, time and molar concentrations.
- Monte Carlo prediction intervals from caller-supplied parameter distributions.

## Verification, not validation

The test suite **verifies** that the code reproduces the mathematics: analytic identities,
an independent numerical ODE integration, and documented cross-checks against established
open-source PK engines. Passing these tests says nothing about whether any particular drug
model describes any particular person; that is model **validation**, which needs external
clinical data and is outside this library.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

## License

MIT, see [LICENSE](LICENSE). Provided without warranty of any kind.
