import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numbers import Number
from typing import Union, cast

Numeric = Union[Number, Quantity]

def round_sigfigs(val: Numeric, sigfigs: int) -> Numeric:
    """Round a float to specified number of significant figures."""

    if sigfigs < 1:
        raise ValueError("Cannot round to less than one significant figure.")
    
    # Separate value and unit for Quantity objects
    if isinstance(val, u.Quantity):
        unit = val.unit
        val = cast(Number, val.value)
    else:
        unit = 1
        val = cast(Number, val)

    rounder = -int(np.floor(np.log10(abs(val)))) + (sigfigs - 1)

    return round(val, rounder) * unit
