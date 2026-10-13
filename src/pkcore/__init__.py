"""pkcore: a small, verifiable pharmacokinetic computation engine.

Units are canonical and never converted: amounts in mg, volumes in L, times in h,
clearances in L/h, rate constants in 1/h and concentrations in mg/L. See
docs/conventions.md for the conventions and the numerical methods.
"""

from importlib.metadata import version

from pkcore.doses import Dose, Route
from pkcore.metrics import (
    Extremum,
    apparent_oral_terminal_half_life,
    apparent_oral_terminal_rate,
    auc,
    cmax,
    cmin,
    terminal_half_life,
    terminal_rate,
    trough,
)
from pkcore.models import Model, OneCompartment, TwoCompartment
from pkcore.montecarlo import PercentileBands, percentile_bands
from pkcore.simulation import Amounts, amounts, concentration

__version__ = version("pkcore")

__all__ = [
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
]
