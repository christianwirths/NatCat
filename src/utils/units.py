"""
Unit conversion utilities.

"""

from typing import Final 
import numpy as np 
from numpy.typing import NDArray


def nm2km(
        nm: float | NDArray[np.float_]
) -> float | NDArray[np.float_]:
    """
    Convert Nautical Miles to Kilometers.
    """
    return nm * 1.852

def kt2kmh(
        kt: float | NDArray[np.float_]
) -> float | NDArray[np.float_]:
    """
    Convert Knots to Kilometers per Hour.
    """
    return kt * 1.852