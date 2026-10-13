# Cross-engine golden tests

pkcore is verified at three levels:

1. **V1**: analytic identities and literature closed forms;
2. **V2**: independent numerical methods in Python (ODE integration, matrix exponential,
   decimal arithmetic), see `docs/conventions.md`;
3. **V3** (this directory): agreement with established open-source PK engines.

All three are **verification**: they show that pkcore computes what the model equations
say. None of them is **validation**, which would need clinical data and is outside this
library.

## Layout

- `scenarios/*.json` defines one dosing scenario per file: model parameters, doses,
  observation times and, at a time with both a dose and an observation, whether dose
  or observation records come first. All parameter values are synthetic.
- `generate.R` simulates every scenario with four engine paths and writes
  `outputs/<scenario>.json`, plus `outputs/session.json` and `outputs/sessionInfo.txt`.
  The four paths are:
  - rxode2 `linCmt()`, a closed form;
  - rxode2 ODE (liblsoda, `atol = rtol = 1e-12`);
  - mrgsolve `$PKMODEL`, a closed form;
  - mrgsolve ODE (DLSODA from ODEPACK, `atol = rtol = 1e-12`).

  The two ODE paths use the same numerical method (LSODA) in two packages, so they are
  not two independent methods; the closed forms are independent implementations.
- `install.R` installs the pinned engine versions.
- `compare.R` compares a regenerated output directory with the committed one.
- `../test_golden.py` compares pkcore with the committed outputs on every test run.

## Pinned environment

- Container: `rocker/r-ver:4.6.1@sha256:4e6e6696ff54a86625c7f80046478e67014a2184e942b85eeff43db7b992d93e`
  (R 4.6.1 on Ubuntu 24.04).
- Packages: rxode2 5.1.7.1, mrgsolve 2.0.1 and jsonlite 2.0.0. They are installed from the
  Posit Package Manager snapshot of 2026-10-09
  (`https://p3m.dev/cran/__linux__/noble/2026-10-09`); `install.R` stops if any
  installed version differs from its pin.

The `golden (R engines)` CI job regenerates the outputs in this environment. It fails if
any value differs from the committed one by more than `1e-9` relative (plus `1e-12` of the
largest value), or if an engine starts or stops failing. Outputs declared as known engine
issues are excluded from this numeric check, because they are rounding noise. The
regenerated files are uploaded as the `golden-outputs` artifact.

To regenerate locally with Docker:

```bash
docker run --rm -v "$PWD":/work -w /work \
  -e PKCORE_CRAN_SNAPSHOT=https://p3m.dev/cran/__linux__/noble/2026-10-09 \
  rocker/r-ver:4.6.1@sha256:4e6e6696ff54a86625c7f80046478e67014a2184e942b85eeff43db7b992d93e \
  bash -c "Rscript tests/golden/install.R && Rscript tests/golden/generate.R"
```

## Conventions and differences between the engines

- **Bioavailability and lag time.** Both are defined only for the depot in the engine
  models, so they apply to oral doses only, as in pkcore. Engine-specific semantics of
  F and lag times on IV records are not compared, because pkcore v0.1 does not support
  them.
- **Same-time order.** At a time with both a dose and an observation, rxode2 always
  reports the post-dose value. mrgsolve follows the record order. The test therefore
  compares rxode2 with `pkcore.concentration`, and mrgsolve with `pkcore.trough` in the
  `*_observation_before_dose` scenarios.
- **`ka` equal to `k` (one compartment).** mrgsolve `$PKMODEL` refuses this case ("k10 is
  too close to ka"); the scenario declares that error.
- **`ka` equal to the slow disposition rate (two compartments).** Neither closed form
  handles this case. Relative to the exact solution, rxode2 `linCmt()` is off by −16 % to
  +42 % and mrgsolve `$PKMODEL` by +17 % to +127 % over the scenario's times (worst at
  4 h). Both ODE paths agree with pkcore to about 1e-10. pkcore's values for nearly equal
  rates are also verified against decimal arithmetic in `tests/test_near_equal_rates.py`.
  The scenario records these as `known_engine_issues`, and the test checks that the
  deviation is still present, so the documentation stays true.
- **`ka` near `k`.** The closed forms lose digits to cancellation: about 1e-9 relative for a
  relative gap of 1e-6.

Tolerances in `test_golden.py`: `1e-8` relative against the closed-form engines (observed
agreement is about 1e-15 away from `ka ~ k`) and `1e-7` against the ODE engines
(observed better than 1e-10), each plus an absolute term scaled by dose / central volume.
