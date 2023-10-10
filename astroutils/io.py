import configparser
import glob
import logging
import os
import re
import warnings
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from pathlib import Path
from typing import Union

warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

PACKAGE_ROOT = Path(__file__).parent
SURVEYS_PATH = PACKAGE_ROOT / "surveys.json"

Strset = Union[str, None]

logger = logging.getLogger(__name__)


class FITSException(Exception):
    pass

class DataError(Exception):
    pass

def get_config() -> configparser.ConfigParser:
    """Load config.ini to expose required absolute paths."""

    config_path = PACKAGE_ROOT / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path.resolve())

    return config

# Make config available in this module
config = get_config()
aux_path = Path(config['DATA']['aux_path'])


def get_surveys() -> pd.DataFrame:
    """Get the survey metadata, resolving image and selavy paths based upon system."""

    # Import survey data from JSON file, remove any already saved
    surveys = pd.read_json(SURVEYS_PATH)

    # Read system name to resolve base path
    system = os.uname()[1]
    if system == "ada.physics.usyd.edu.au":
        surveys['data_path'] = surveys.ada_root
    elif system == 'vast-data':
        surveys['data_path'] = surveys.nimbus_root
    else:
        raise NotImplementedError(f"Data paths unknown for hostname {system}")

    # Populate all relevant data paths
    def update(row):
        row['image_path_i_C'] = (row.data_path + 'COMBINED/STOKESI_IMAGES/') if row.combined_I else ''
        row['selavy_path_i_C'] = (row.data_path + 'COMBINED/STOKESI_SELAVY/') if row.combined_I else ''
        row['image_path_v_C'] = (row.data_path + 'COMBINED/STOKESV_IMAGES/') if row.combined_V else ''
        row['selavy_path_v_C'] = (row.data_path + 'COMBINED/STOKESV_SELAVY/') if row.combined_V else ''
        row['image_path_i_T'] = (row.data_path + 'TILES/STOKESI_IMAGES/') if row.tiles_I else ''
        row['selavy_path_i_T'] = (row.data_path + 'TILES/STOKESI_SELAVY/') if row.tiles_I else ''
        row['image_path_v_T'] = (row.data_path + 'TILES/STOKESV_IMAGES/') if row.tiles_V else ''
        row['selavy_path_v_T'] = (row.data_path + 'TILES/STOKESV_SELAVY/') if row.tiles_V else ''

        return row

    surveys = surveys.apply(update, axis=1)

    surveys.drop(columns=['ada_root', 'nimbus_root'], inplace=True)

    return surveys


def get_survey(epoch: str, is_name: bool=False) -> pd.Series:
    """Get a single survey epoch from surveys.json."""

    surveys = get_surveys()

    column = 'name' if is_name else 'survey'
    
    if epoch not in surveys[column].values:
        raise NotImplementedError(f"{epoch} not a valid survey in surveys.json")

    survey = surveys[surveys[column] == epoch].iloc[0]
        
    return survey

def get_image_data_header(image_path: Path, load_data: bool=True):
    """Open FITS image and fetch header / data in units of mJy/beam."""

    with fits.open(image_path) as hdul:
        header = hdul[0].header
        data = hdul[0].data if load_data else None

    if data is not None:
        unit = u.Jy if header.get('BUNIT') == 'Jy/beam' else u.mJy
        data = (data*unit).to(u.mJy).value
        header['BUNIT'] = 'mJy/beam'

    return data, header


def get_image_from_survey_params(epoch: pd.Series, field: str, stokes: str, tiletype: str, load: bool=True):
    """Get image header and data for a given field, epoch, and Stokes parameter."""

    image_path = list(Path(epoch[f'image_path_{stokes}_{tiletype[0]}']).glob(f'*{field}*.fits'))[0]
    data, header = get_image_data_header(image_path, load_data=load)

    return data, header


def find_fields(position: SkyCoord, epoch: str, tiletype: Strset, radius: Angle=5*u.deg) -> pd.DataFrame:
    """Return DataFrame of epoch fields containing position."""

    tilestr = f'_{tiletype.lower()}' if tiletype else ''
    path = f'{aux_path}/fields/{epoch}{tilestr}_fields.csv'
    try:
        image_df = pd.read_csv(path)
    except FileNotFoundError:
        raise FITSException(f"Missing field metadata csv for {epoch} at {path}.")
    
    beam_centre = SkyCoord(ra=image_df['cr_ra_pix'], dec=image_df['cr_dec_pix'], unit=u.deg)
    image_df['dist_field_centre'] = beam_centre.separation(position).deg
    
    fields = image_df[image_df.dist_field_centre < radius].reset_index(drop=True)

    return fields


def parse_image_filenames_in_dir(path: Path, stokes: str) -> list[Path]:
    """Return all valid image paths contained within directory path."""

    patterns = (
        f'image.{stokes.lower()}*fits',
        f'*{stokes.upper()}.fits',
        f'*{stokes.upper()}.conv.fits',
    )

    image_paths = [p for p in path.iterdir() if any(p.match(pattern) for pattern in patterns)]

    return image_paths


def build_field_csv(epoch: str, tiletype: str='TILES') -> pd.DataFrame:
    """Generate metadata csv for fields from epoch."""

    survey = get_survey(epoch)

    i_paths = parse_image_filenames_in_dir(Path(survey[f'image_path_i_{tiletype[0]}']), stokes='i')
    v_paths = parse_image_filenames_in_dir(Path(survey[f'image_path_v_{tiletype[0]}']), stokes='v')

    pattern = re.compile(r'\S*(\d{4}[-+]\d{2})\S*')
    sbidpattern = re.compile(r'\S*(SB\d{4,5})\S*')

    vals = []
    for i_path in i_paths:

        str_path = str(i_path)
        logger.debug(f"Reading image at {str_path}")

        # Try to parse field coordinate pattern first, then default to SBID pattern
        field = pattern.sub(r'\1', str_path)
        if field == str_path:
            field = sbidpattern.sub(r'\1', str_path)

        # If neither field or sbid is parseable, throw an error
        if field == str_path:
            raise FITSException(f"Could not parse field name or SBID from image at {str_path}")

        # Try to parse SBID, then default to placeholder
        sbid = sbidpattern.sub(r'\1', str_path)
        if sbid == str_path:
            sbid = 'SBXXX'

        _, header = get_image_data_header(i_path, load_data=False)

        # Locate coordinates of image pixel centre
        w = WCS(header, naxis=2)
        date_obs = header.get("DATE-OBS", "")
        size_x = header["NAXIS1"]
        size_y = header["NAXIS2"]
        central_coords = [[size_x / 2., size_y / 2.]]
        centre = w.wcs_pix2world(np.array(central_coords, np.float_), 1)

        if len(v_paths) == 0:
            v_path = ''
        else:
            v_path = [p for p in v_paths if field in str(p) and sbid in str(p)][0]

        params = {
            'field': field,
            'sbid': sbid,
            'date_obs': date_obs, 
            'i_path': str_path,
            'v_path': v_path,
            'cr_ra_pix': centre[0][0],
            'cr_dec_pix': centre[0][1],
            'bmaj': header['BMAJ'] * 3600,
            'bmin': header['BMIN'] * 3600,
        }

        vals.append(params)

    df = pd.DataFrame(vals)
    
    return df.dropna()


def build_vlass_field_csv(epoch: str) -> pd.DataFrame:

    survey = get_survey(epoch)
    base_dir = Path(survey.data_path)
    
    pattern = re.compile(r'\S+(J\d{6}[-+]\d{6})\S+')
    fields = [Path(f) for f in glob.glob(f"{survey.data_path}/**/*.subim.fits", recursive=True)]
    
    names = [f.parts[-1] for f in fields]

    df = pd.DataFrame({
        'image': names,
        'imagedir': [f.parts[-2] for f in fields],
        'tile': [f.parts[-3] for f in fields],
        'coord': [pattern.sub(r'\1', name) for name in names],
    })

    vals = []
    for _, row in df.iterrows():

        path = f'{base_dir}/{row.tile}/{row.imagedir}/{row.image}'
        with fits.open(path) as hdul:
            header = hdul[0].header
            w = WCS(header, naxis=2)
            size_x = header["NAXIS1"]
            size_y = header["NAXIS2"]

            central_coords = [[size_x / 2., size_y / 2.]]
            centre = w.wcs_pix2world(np.array(central_coords, np.float_), 1)

            vals.append({
                'image': row.image,
                'i_path': path,
                'cr_ra_pix': centre[0][0],
                'cr_dec_pix': centre[0][1],
                'date': header['DATE-OBS']
            })

    df = df.merge(pd.DataFrame(vals), on='image')

    return df
