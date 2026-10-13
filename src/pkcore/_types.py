"""Shared type aliases."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

BoolArray: TypeAlias = npt.NDArray[np.bool_]
FloatArray: TypeAlias = npt.NDArray[np.float64]
IndexArray: TypeAlias = npt.NDArray[np.intp]
