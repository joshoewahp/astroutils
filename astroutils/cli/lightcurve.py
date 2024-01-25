import click
import os
import logging
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from astroutils.io import get_surveys
from astroutils.source import measure_flux
from matplotlib.dates import DateFormatter

from astroutils.logger import setupLogger

logger = logging.getLogger(__name__)
logging.getLogger('forced_phot').setLevel(logging.CRITICAL)

@click.command
@click.option('-v', '--verbose', is_flag=True, default=False)
@click.argument('RA', type=str)
@click.argument('Dec', type=str)
def main(verbose, ra, dec):

    setupLogger(verbose)

    if not os.path.exists('fluxes.csv'):

        surveys = get_surveys()
        surveys = surveys[
            (surveys.survey == 'racs-low') | 
            (surveys.survey.str.contains('vast') & 
            surveys.image_path_v_T.str.contains('low'))
        ]

        print(surveys)

        c = SkyCoord(ra=ra, dec=dec, unit='deg')

        fluxes = measure_flux(
            c,
            epochs=surveys,
            size=15*u.arcsec,
            fluxtype='peak',
            stokes='i',
            tiletype='TILES',
        )

        fluxes.to_csv('fluxes.csv', index=False)

    fluxes = pd.read_csv('fluxes.csv')

    # Remove limits that are near the field edge
    fluxlim = fluxes[~fluxes.is_limit].flux.min()
    # fluxes = fluxes[
    #     ~((fluxes.flux > fluxlim) & (fluxes.is_limit))
    # ]

    print(fluxes[['epoch', 'field', 'obsdate', 'flux', 'flux_err', 'is_limit', 'dist_field_centre']])

    fig, ax = plt.subplots()

    limits = fluxes[fluxes.is_limit]
    fluxes = fluxes[~fluxes.is_limit]

    ax.scatter(
        pd.to_datetime(fluxes.obsdate),
        fluxes.flux,
        marker='.',
        color='k',
    )
    ax.errorbar(
        pd.to_datetime(limits.obsdate),
        limits.flux,
        yerr=0.1,
        uplims=True,
        marker='.',
        ls='none',
        color='gray',
    )
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    ax.set_ylabel('Flux Density (mJy)')

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.18,
        top=0.98,
    )

    plt.show()


if __name__ == '__main__':
    main()
