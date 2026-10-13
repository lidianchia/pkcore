"""Cross-engine golden tests: pkcore against rxode2 and mrgsolve (verification level V3).

``tests/golden/scenarios`` defines dosing scenarios as data; ``tests/golden/generate.R``
simulates them with four engine paths (rxode2 ``linCmt()`` and ODE, mrgsolve
``$PKMODEL`` and ODE) and the outputs are committed under ``tests/golden/outputs``
together with the R session information. A CI job regenerates them in a pinned container
and fails on any drift; this module compares pkcore with the committed outputs.

Tolerances (``|pkcore - engine| <= rtol |engine| + atol * dose / v1``):

* closed-form engines: ``rtol = 1e-8``, ``atol = 1e-12``. Observed agreement is about
  1e-15, except near ``ka ~ k`` where the engines' textbook forms lose digits (about 1e-9
  for a relative gap of 1e-6; pkcore's own values there are verified against decimal
  arithmetic in ``test_near_equal_rates``).
* ODE engines (``atol = rtol = 1e-12`` in the engines): ``rtol = 1e-7``, ``atol = 1e-9``.
  Observed agreement is better than 1e-10.

Semantics: both engines put bioavailability and lag time on the depot only, as pkcore
does. At a time with both an observation and a dose, rxode2 always reports the
post-dose value (compared with :func:`pkcore.concentration`), while mrgsolve follows the
record order (pre-dose for ``observation_first``, compared with :func:`pkcore.trough`).
Engine failures and known engine inaccuracies are declared in the scenario files
(``expected_errors``, ``known_engine_issues``) and are checked, not silently skipped.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pkcore as pk
from pkcore import Dose

GOLDEN = Path(__file__).parent / "golden"
SCENARIOS = sorted((GOLDEN / "scenarios").glob("*.json"))
ENGINES = ("rxode2_lincmt", "rxode2_ode", "mrgsolve_pkmodel", "mrgsolve_ode")
TOLERANCES = {
    "rxode2_lincmt": (1e-8, 1e-12),
    "mrgsolve_pkmodel": (1e-8, 1e-12),
    "rxode2_ode": (1e-7, 1e-9),
    "mrgsolve_ode": (1e-7, 1e-9),
}
FOLLOWS_RECORD_ORDER = {"mrgsolve_pkmodel", "mrgsolve_ode"}


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def model_of(spec: dict[str, Any]) -> tuple[pk.Model, float]:
    """The pkcore model of a scenario and its central volume."""
    absorption = {"ka": spec["ka"], "f": spec["f"], "tlag": spec["tlag"]}
    if spec["compartments"] == 1:
        return pk.OneCompartment(spec["cl"], spec["v"], **absorption), spec["v"]
    model = pk.TwoCompartment(spec["cl"], spec["v1"], spec["q"], spec["v2"], **absorption)
    return model, spec["v1"]


def test_every_scenario_has_outputs_from_every_engine() -> None:
    assert len(SCENARIOS) >= 20
    for path in SCENARIOS:
        output = load(GOLDEN / "outputs" / path.name)
        assert output["scenario"] == path.stem
        assert sorted(output["engines"]) == sorted(ENGINES)


def test_outputs_record_the_engine_versions() -> None:
    session = load(GOLDEN / "outputs" / "session.json")
    assert session["r_version"].startswith("R version 4.6.1")
    assert session["packages"]["rxode2"] == "5.1.7.1"
    assert session["packages"]["mrgsolve"] == "2.0.1"


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_pkcore_matches_the_engines(path: Path) -> None:
    scenario = load(path)
    output = load(GOLDEN / "outputs" / path.name)
    model, v1 = model_of(scenario["model"])
    doses = [Dose(d["time"], d["amount"], d["route"], d.get("duration")) for d in scenario["doses"]]
    times = np.array(scenario["observations"]["times"], dtype=float)
    np.testing.assert_array_equal(np.array(output["times"], dtype=float), times)
    post_dose = pk.concentration(model, doses, times)
    pre_dose = pk.trough(model, doses, times)
    observation_first = scenario["observations"]["order"] == "observation_first"
    scale = max(d["amount"] for d in scenario["doses"]) / v1
    expected_errors: dict[str, str] = scenario.get("expected_errors", {})
    known_issues: dict[str, str] = scenario.get("known_engine_issues", {})

    for engine in ENGINES:
        result = output["engines"][engine]
        if engine in expected_errors:
            assert expected_errors[engine] in result.get("error", ""), engine
            continue
        assert "error" not in result, f"{engine}: {result.get('error')}"
        values = np.array(result["cp"], dtype=float)
        mine = pre_dose if observation_first and engine in FOLLOWS_RECORD_ORDER else post_dose
        rtol, atol = TOLERANCES[engine]
        if engine in known_issues:
            # Documented engine inaccuracy: make sure the issue is still there (so the
            # documentation stays true) instead of comparing.
            assert not np.allclose(mine, values, rtol=rtol, atol=atol * scale), engine
            continue
        np.testing.assert_allclose(mine, values, rtol=rtol, atol=atol * scale, err_msg=engine)
