"""Input validation shared by models, doses and metrics."""

import math
import numbers

import numpy as np
import numpy.typing as npt

from pkcore._types import FloatArray


def real_number(name: str, value: object) -> float:
    """Return ``value`` as a finite float, or raise ``TypeError``/``ValueError``."""
    result = extended_real_number(name, value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def extended_real_number(name: str, value: object) -> float:
    """Return ``value`` as a float that may be infinite but not NaN."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} must not be NaN")
    return result


def parameter(name: str, value: npt.ArrayLike, *, minimum: float, strict: bool) -> FloatArray:
    """Return a read-only float64 copy of a model parameter after validating it.

    A parameter is a real scalar or a non-empty 1-D array of draws. Every element
    must be finite and ``> minimum`` (``strict``) or ``>= minimum``.
    """
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers, got dtype {raw.dtype}")
    if raw.ndim > 1:
        raise ValueError(
            f"{name} must be a scalar or a 1-D array of parameter draws, got shape {raw.shape}"
        )
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    array = raw.astype(np.float64, copy=True)
    finite = np.isfinite(array)
    above = array > minimum if strict else array >= minimum
    requirement = f"finite and {'>' if strict else '>='} {minimum:g}"
    _require(name, array, finite & above, requirement)
    array.setflags(write=False)
    return array


def fraction(name: str, value: npt.ArrayLike) -> FloatArray:
    """Like :func:`parameter`, for values in the closed interval [0, 1]."""
    array = parameter(name, value, minimum=0.0, strict=False)
    _require(name, array, array <= 1.0, "in [0, 1]")
    return array


def _require(
    name: str, array: FloatArray, ok: npt.NDArray[np.bool_] | np.bool_, requirement: str
) -> None:
    valid = np.asarray(ok, dtype=bool).reshape(-1)
    if valid.all():
        return
    bad = np.flatnonzero(~valid)
    first = float(array.reshape(-1)[bad[0]])
    where = "" if array.ndim == 0 else f" at index {int(bad[0])}"
    more = f" ({bad.size} invalid values)" if bad.size > 1 else ""
    raise ValueError(f"{name} must be {requirement}; got {first!r}{where}{more}")


def times(name: str, value: npt.ArrayLike) -> FloatArray:
    """Return finite time points as a float64 array with at most one dimension."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers, got dtype {raw.dtype}")
    if raw.ndim > 1:
        raise ValueError(f"{name} must be a scalar or a 1-D array, got shape {raw.shape}")
    array = raw.astype(np.float64, copy=True)
    _require(name, array, np.isfinite(array), "finite")
    return array
