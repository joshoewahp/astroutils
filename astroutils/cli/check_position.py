import click
import logging
from astroutils.logger import setupLogger
from astroutils.source import get_all_catalogues

logger = logging.getLogger(__name__)
setupLogger(verbose=True)

@click.command()
@click.argument('ra')
@click.argument('dec')
def main(ra, dec):

    stokesi = get_all_catalogues(stokes='i')
    stokesv = get_all_catalogues(stokes='v')

    print(stokesi)
    print(stokesv)

if __name__ == '__main__':
    main()
