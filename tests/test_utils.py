import astropy.constants as c
import astropy.units as u
import pytest

from astroutils.utils import round_sigfigs


@pytest.mark.parametrize(
    "val, sigfigs, expected",
    [
        (4.214, 2, 4.2),
        (4.214e20, 2, 4.2e20),
        (1.333 * u.W, 2, 1.3 * u.W),
        (1 * c.R_sun, 4, 6.957e8 * u.m),
    ],
)
def test_round_sigfigs(val, sigfigs, expected):
    rounded = round_sigfigs(val, sigfigs)

    assert rounded == expected


@pytest.mark.parametrize(
    "val, sigfigs",
    [
        (4.214, 0),
        (4.214, -1),
    ],
)
def test_round_sigfigs_raises_valuerror_with_zero_sigfigs(val, sigfigs):
    with pytest.raises(ValueError):
        round_sigfigs(val, sigfigs)
