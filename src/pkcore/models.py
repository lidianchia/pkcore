"""Linear compartmental models.

Units (pkcore never converts units; see docs/conventions.md): clearances in L/h,
volumes in L, rate constants in 1/h, times in h. With doses in mg, concentrations are
in mg/L.

Every parameter is either a real scalar or a 1-D array of parameter draws. All
array-valued parameters of one model must have the same length ``n``; simulations of
such a model return one row per draw.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from pkcore import _validation
from pkcore._disposition import two_compartment_modes
from pkcore._types import FloatArray


@dataclass(frozen=True, eq=False, init=False)
class OneCompartment:
    """One-compartment model with linear elimination and optional first-order absorption.

    Disposition: ``dA/dt = -(cl / v) A`` for the amount ``A`` in the body.

    Attributes:
        cl: Clearance (L/h), finite and > 0.
        v: Volume of distribution (L), finite and > 0.
        ka: First-order absorption rate constant (1/h) for oral doses, finite and > 0,
            or ``None`` if the model is only used with IV doses.
        f: Bioavailability of oral doses, in [0, 1]. Not applied to IV doses.
        tlag: Absorption lag time (h) of oral doses, >= 0. Not applied to IV doses.
    """

    cl: FloatArray
    v: FloatArray
    ka: FloatArray | None
    f: FloatArray
    tlag: FloatArray

    def __init__(
        self,
        cl: npt.ArrayLike,
        v: npt.ArrayLike,
        *,
        ka: npt.ArrayLike | None = None,
        f: npt.ArrayLike = 1.0,
        tlag: npt.ArrayLike = 0.0,
    ) -> None:
        values = {
            "cl": _positive("cl", cl),
            "v": _positive("v", v),
            "ka": None if ka is None else _positive("ka", ka),
            "f": _validation.fraction("f", f),
            "tlag": _validation.parameter("tlag", tlag, minimum=0.0, strict=False),
        }
        _initialise(self, values)
        with np.errstate(all="ignore"):
            _check_rate("cl / v", self.cl / self.v)

    @property
    def n_draws(self) -> int | None:
        """Number of parameter draws, or ``None`` if every parameter is a scalar."""
        return _draw_count(self.cl, self.v, self.ka, self.f, self.tlag)


@dataclass(frozen=True, eq=False, init=False)
class TwoCompartment:
    """Two-compartment model with linear elimination and optional first-order absorption.

    Disposition, with ``A1`` the central and ``A2`` the peripheral amount::

        dA1/dt = -(cl + q) / v1 * A1 + q / v2 * A2
        dA2/dt =  q / v1 * A1        - q / v2 * A2

    Doses enter the central compartment (IV) or a depot that empties into it at the
    rate ``ka`` (oral). Concentrations refer to the central compartment.

    Attributes:
        cl: Clearance from the central compartment (L/h), finite and > 0.
        v1: Central volume (L), finite and > 0.
        q: Inter-compartmental clearance (L/h), finite and > 0.
        v2: Peripheral volume (L), finite and > 0.
        ka: First-order absorption rate constant (1/h), finite and > 0, or ``None``.
        f: Bioavailability of oral doses, in [0, 1]. Not applied to IV doses.
        tlag: Absorption lag time (h) of oral doses, >= 0. Not applied to IV doses.
    """

    cl: FloatArray
    v1: FloatArray
    q: FloatArray
    v2: FloatArray
    ka: FloatArray | None
    f: FloatArray
    tlag: FloatArray

    def __init__(  # noqa: PLR0913 (one argument per model parameter)
        self,
        cl: npt.ArrayLike,
        v1: npt.ArrayLike,
        q: npt.ArrayLike,
        v2: npt.ArrayLike,
        *,
        ka: npt.ArrayLike | None = None,
        f: npt.ArrayLike = 1.0,
        tlag: npt.ArrayLike = 0.0,
    ) -> None:
        values = {
            "cl": _positive("cl", cl),
            "v1": _positive("v1", v1),
            "q": _positive("q", q),
            "v2": _positive("v2", v2),
            "ka": None if ka is None else _positive("ka", ka),
            "f": _validation.fraction("f", f),
            "tlag": _validation.parameter("tlag", tlag, minimum=0.0, strict=False),
        }
        _initialise(self, values)
        with np.errstate(all="ignore"):
            _check_rate("cl / v1", self.cl / self.v1)
            _check_rate("q / v1", self.q / self.v1)
            _check_rate("q / v2", self.q / self.v2)
            slow, fast, _, _ = two_compartment_modes(self.cl, self.v1, self.q, self.v2)
            _check_rate("the slower disposition rate constant", slow)
            _check_rate("the faster disposition rate constant", fast)

    @property
    def n_draws(self) -> int | None:
        """Number of parameter draws, or ``None`` if every parameter is a scalar."""
        return _draw_count(self.cl, self.v1, self.q, self.v2, self.ka, self.f, self.tlag)


Model: TypeAlias = OneCompartment | TwoCompartment
"""Any model accepted by the simulation functions."""


def _positive(name: str, value: npt.ArrayLike) -> FloatArray:
    return _validation.parameter(name, value, minimum=0.0, strict=True)


_SMALLEST_RATE = float(np.finfo(np.float64).tiny)


def _check_rate(name: str, rate: FloatArray) -> None:
    """Derived rate constants must be finite normal floating-point numbers."""
    bad = ~(np.isfinite(rate) & (rate >= _SMALLEST_RATE))
    if bad.any():
        value = float(np.asarray(rate).reshape(-1)[np.flatnonzero(bad)[0]])
        raise ValueError(
            f"{name} must be a finite rate constant >= {_SMALLEST_RATE:.3g} per hour, got "
            f"{value!r}: the parameter values are too large or too small to be combined "
            "in floating point"
        )


def _initialise(model: object, values: dict[str, FloatArray | None]) -> None:
    lengths = {name: v.shape[0] for name, v in values.items() if v is not None and v.ndim == 1}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{name}: {n}" for name, n in lengths.items())
        raise ValueError(f"array-valued parameters must have the same length ({detail})")
    for name, value in values.items():
        object.__setattr__(model, name, value)


def _draw_count(*values: FloatArray | None) -> int | None:
    for value in values:
        if value is not None and value.ndim == 1:
            return int(value.shape[0])
    return None
