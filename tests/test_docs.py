"""The README example runs, and the public API is what the docs describe."""

import re
from pathlib import Path

import numpy as np

import pkcore

ROOT = Path(__file__).resolve().parent.parent


def test_readme_example_runs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert len(blocks) == 1
    namespace: dict[str, object] = {}
    exec(compile(blocks[0], "README.md", "exec"), namespace)  # noqa: S102
    bands = namespace["bands"]
    assert isinstance(bands, pkcore.PercentileBands)
    assert bands.values.shape == (3, 193)
    assert np.all(np.isfinite(bands.values))


def test_public_api() -> None:
    expected = {
        "Amounts",
        "Dose",
        "Extremum",
        "Model",
        "OneCompartment",
        "PercentileBands",
        "Route",
        "TwoCompartment",
        "__version__",
        "amounts",
        "apparent_oral_terminal_half_life",
        "apparent_oral_terminal_rate",
        "auc",
        "cmax",
        "cmin",
        "concentration",
        "percentile_bands",
        "terminal_half_life",
        "terminal_rate",
        "trough",
    }
    assert set(pkcore.__all__) == expected
    for name in expected:
        assert hasattr(pkcore, name)
