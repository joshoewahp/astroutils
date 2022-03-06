import configparser
import glob
import logging
import os
import sys
import re
import time
import warnings
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from pathlib import Path

warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

PACKAGE_ROOT = Path(__file__).parent
SURVEYS_PATH = PACKAGE_ROOT / "surveys.json"

logger = logging.getLogger(__name__)


class FITSException(Exception):
    pass


def get_config():
    """Load config.ini to expose required absolute paths."""

    config_path = PACKAGE_ROOT / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path.resolve())

    return config

# Make config available in this module
config = get_config()
aux_path = Path(config['DATA']['aux_path'])


def get_surveys():
    """Get the survey metadata, resolving image and selavy paths based upon system."""

    # Import survey data from JSON file, remove any already saved
    surveys = pd.read_json(SURVEYS_PATH)
    system = os.uname()[1]
    for col in ["selavy_path_i", "image_path_i", "selavy_path_v", "image_path_v"]:
        if system == "ada.physics.usyd.edu.au":
            surveys[col] = surveys.ada_root + surveys[col]
        elif system == 'vast-data':
            surveys[col] = surveys.nimbus_root + surveys[col]
        else:
            raise NotImplementedError(f"Data paths unknown for hostname {system}")
    surveys.drop(columns=['ada_root', 'nimbus_root'], inplace=True)

    return surveys


def get_survey(epoch):
    """Get a single survey epoch from surveys.json."""

    surveys = get_surveys()
    if epoch not in surveys.survey.values:
        raise NotImplementedError(f"{epoch} not a valid survey in surveys.json")

    survey = surveys[surveys.survey == epoch].iloc[0]
        
    return survey


def get_image(epoch: pd.Series, field: str, stokes: str, load: bool=False):
    """Get image header and data for a given field, epoch, and Stokes parameter."""

    image_file = list(Path(epoch[f'image_path_{stokes}']).glob(f'*{field}*.fits'))[0]
    with fits.open(image_file) as hdul:
        data = hdul[0].data if load else None
        header = hdul[0].header

    return data, header


def find_fields(position, epoch):
    """Return DataFrame of epoch fields containing position."""

    try:
        image_df = pd.read_csv(f'{aux_path}/fields/{epoch}_fields.csv')
    except FileNotFoundError:
        raise FITSException(f"Missing field metadata csv for {epoch}.")
    
    beam_centre = SkyCoord(ra=image_df['cr_ra_pix'], dec=image_df['cr_dec_pix'], unit=u.deg)
    image_df['dist_field_centre'] = beam_centre.separation(position).deg
    
    pbeamsize = 1 * u.degree if epoch == 'vlass' else 5 * u.degree
    fields = image_df[image_df.dist_field_centre < pbeamsize].reset_index(drop=True)

    return fields


def make_mwats_field_csv():
    image_path = config['DATA'][f'mwats_path']

    vals = []
    for field in glob.glob(image_path + '*I.fits'):
        try:
            with fits.open(field) as hdul:
                header = hdul[0].header
                w = WCS(header, naxis=2)
                size_x = header['NAXIS1']
                size_y = header['NAXIS2']

                central_coords = [[size_x / 2., size_y / 2.]]
                centre = w.wcs_pix2world(np.array(central_coords, np.float_), 1)

                epoch = header['DATE-OBS']
                cr_ra = header['CRVAL1']
                cr_dec = header['CRVAL2']
                name = 'mwats_{}_{}_{}'.format(epoch, cr_ra, cr_dec)

                vals.append({'field': name,
                             'cr_ra': cr_ra,
                             'cr_dec': cr_dec,
                             'cr_ra_pix': centre[0][0],
                             'cr_dec_pix': centre[0][1]})
        except Exception as e:
            print(e)
            raise
    df = pd.DataFrame(vals)
    df = df.dropna()

    print(df)
    df.to_csv(f'{aux_path}/fields/mwats_fields.csv', index=False)

    return


def build_field_csv(epoch):
    """Generate metadata csv for dataset fields."""

    survey = get_survey(epoch)
    paths = glob.glob(survey.image_path_i + '*.fits')

    pattern = re.compile(r'\S*(\d{4}[-+]\d{2})\S*')
    sbidpattern = re.compile(r'\S*(SB\d{4,5})\S*')

    vals = []
    for field in paths:

        name = pattern.sub(r'\1', field)
        if name == field:
            name = sbidpattern.sub(r'\1', field)

        sbid = sbidpattern.sub(r'\1', field)
        if sbid == field:
            sbid = 'SBXXX'

        with fits.open(field) as hdul:
            header = hdul[0].header
            w = WCS(header, naxis=2)
            size_x = header["NAXIS1"]
            size_y = header["NAXIS2"]

            central_coords = [[size_x / 2., size_y / 2.]]
            centre = w.wcs_pix2world(np.array(central_coords, np.float_), 1)


        params = {
            'field': name,
            'sbid': sbid,
            'cr_ra_pix': centre[0][0],
            'cr_dec_pix': centre[0][1],
            'bmaj': header['BMAJ'] * 3600,
            'bmin': header['BMIN'] * 3600,
        }

        vals.append(params)


    df = pd.DataFrame(vals)
    
    return df.dropna()


def make_vlass_fields(base_dir):

    pattern = re.compile(r'\S+(J\d{6}[-+]\d{6})\S+')
    fields = list(Path(base_dir).rglob("*subim.fits"))
    names = [f.parts[-1] for f in fields]
    df = pd.DataFrame({'filename': names,
                       'coord': [pattern.sub(r'\1', name) for name in names],
                       'epoch': [f.parts[4] for f in fields],
                       'tile': [f.parts[5] for f in fields],
                       'image': [f.parts[6] for f in fields]})

    vals = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)

        with fits.open(f'{base_dir}{row.epoch}/{row.tile}/{row.image}/{row.filename}') as hdul:
            header = hdul[0].header
            w = WCS(header, naxis=2)
            size_x = header["NAXIS1"]
            size_y = header["NAXIS2"]

            central_coords = [[size_x / 2., size_y / 2.]]
            centre = w.wcs_pix2world(np.array(central_coords, np.float_), 1)

            vals.append({'image': row.image,
                         'cr_ra_pix': centre[0][0],
                         'cr_dec_pix': centre[0][1],
                         'date': header['DATE-OBS']})

    df = df.merge(pd.DataFrame(vals), on='image')

    print(df)
    df.to_csv(f'{aux_path}/fields/vlass_fields.csv', index=False)


def make_raw_cat(epoch, pol):
    selavy_path = config['DATA'][f'selavy{pol}_path_{epoch}']

    components = [selavy_path + c for c in os.listdir(selavy_path) if
                  'components.xml' in c]

    pattern = re.compile(r'\S*(\d{4}[+-]\d{2})\S*')
    sbidpattern = re.compile(r'\S*(SB\d{4,5})\S*')

    csvs = []
    for csv in components:
        sign = -1 if csv.split('/')[-1][0] == 'n' else 1
        sbid = 'SBXXXX' if '-mm' in epoch else sbidpattern.sub(r'\1', csv)

        df = pd.read_fwf(csv, skiprows=[1, ])
        df.insert(0, 'sbid', sbid)
        df.insert(0, 'field', pattern.sub(r'\1', csv))
        df['sign'] = sign
        csvs.append(df)
    df = pd.concat(csvs, ignore_index=True, sort=False)
    df.to_csv(f'{aux_path}/raw_selavy/{epoch}_raw_selavy_cat.csv', index=False)

    return


def combine_field_csvs():

    dfs = []
    for epoch in ['vastp1-mm', 'vastp2-mm', 'vastp3x-mm', 'vastp4x-mm', 'vastp5x-mm',
                  'vastp6x-mm', 'vastp7x-mm', 'vastp8-mm', 'vastp9-mm', 'vastp10x-mm', 'vastp11x-mm']:
        df = pd.read_csv(f'{aux_path}/fields/{epoch}_fields.csv')
        df.insert(0, 'epoch', epoch)
        dfs.append(df)

    bigdf = pd.concat(dfs)
    bigdf.to_csv(f'{aux_path}/fields/vastp_fields.csv', index=False)

    return


def table2df(table):
    """Clean conversion of Astropy table to DataFrame. """
    df = table.to_pandas()
    return df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)


def timeit(func):
    """Decorator to time func"""

    def _timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        print('{} took {:2.2f} ms'.format(func.__name__, (te - ts) * 1000))
        return result

    return _timed
