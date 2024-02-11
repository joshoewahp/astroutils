import logging
from typing import Optional

import astropy.units as u
import pandas as pd
from astropy.coordinates import Distance, SkyCoord
from astropy.time import Time
from astroquery.simbad import Simbad

logger = logging.getLogger(__name__)

Simbad.add_votable_fields(
    "otype",
    "ra(d)",
    "dec(d)",
    "parallax",
    "pmdec",
    "pmra",
    "distance",
    "sptype",
    "distance_result",
)

QueryResult = tuple[pd.DataFrame, SkyCoord]


def query_simbad(
    position: SkyCoord,
    radius: u.Quantity,
    obstime: Time,
) -> Optional[QueryResult]:
    """Query SIMBAD database with proper motion corrections, returning result DataFrame and corrected positions."""

    simbad = Simbad.query_region(position, radius=radius)

    # Catch SIMBAD failure either from None return of query or no stellar type matches in region
    try:
        simbad = simbad.to_pandas()
        assert len(simbad) > 0

    except (ValueError, AssertionError):
        logger.debug(f"No objects within {radius}.")
        return

    # Treat non-existent proper motion parameters as extremely distant objects
    simbad["PMRA"].fillna(0, inplace=True)
    simbad["PMDEC"].fillna(0, inplace=True)
    simbad["PLX_VALUE"].fillna(0.01, inplace=True)

    pmra = simbad["PMRA"].values * u.mas / u.yr
    pmdec = simbad["PMDEC"].values * u.mas / u.yr

    dist = Distance(parallax=simbad["PLX_VALUE"].values * u.mas)

    j2000pos = SkyCoord(
        ra=simbad["RA_d"].values * u.deg,
        dec=simbad["DEC_d"].values * u.deg,
        frame="icrs",
        distance=dist,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        obstime="J2000",
    )
    newpos = j2000pos.apply_space_motion(obstime)

    simbad_cols = {
        "MAIN_ID": "Object",
        "OTYPE": "Type",
        "SP_TYPE": "Spectral Type",
        "DISTANCE_RESULT": "Separation (arcsec)",
    }

    simbad = simbad.rename(columns=simbad_cols)
    simbad = simbad[simbad_cols.values()].copy()

    simbad["PM Corrected Separation (arcsec)"] = newpos.separation(position).arcsec
    simbad = simbad.sort_values("PM Corrected Separation (arcsec)")

    return simbad, newpos
