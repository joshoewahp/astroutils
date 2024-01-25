import click
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from astroutils.io import find_fields
from forced_phot import ForcedPhot

from astroutils.logger import setupLogger

@click.command()
@click.option('-s', '--survey', default=None, type=str)
@click.option('-v', '--verbose', is_flag=True, help="Enable verbose logging mode")
@click.argument('RA', type=str)
@click.argument('Dec', type=str)
def main(ra, dec, survey, verbose):

    setupLogger(verbose)

    unit = u.hourangle if ':' in ra or 'h' in ra else u.deg

    position = SkyCoord(ra=ra, dec=dec, unit=(unit, u.deg))

    if survey:
        fields = find_fields(position, survey)
    else:
        raise NotImplementedError("Only survey argument currently implemented.")

    print(fields)

    # image, background, and noise maps from ASKAPSoft
    image = 'image.i.SB9668.cont.VAST_0341-50A.linmos.taylor.0.restored.fits'
    background = 'meanMap.image.i.SB9668.cont.VAST_0341-50A.linmos.taylor.0.restored.fits'
    noise = 'noiseMap.image.i.SB9668.cont.VAST_0341-50A.linmos.taylor.0.restored.fits'

    # make the Forced Photometry object
    FP = ForcedPhot(image, background, noise)
    

if __name__ == '__main__':
    main()
