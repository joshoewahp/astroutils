import re
import logging
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from pathlib import Path
from typing import Union, cast

from astroutils.io import find_fields, get_image_data_header, get_survey, get_image_from_survey_params

logger = logging.getLogger(__name__)


class SelavyCatalogue:

    def __init__(self, selavypath: Union[str, Path, list[str], list[Path]], correct_negative=True):

        if isinstance(selavypath, str):
            selavypath = [Path(selavypath)]
        elif isinstance(selavypath, Path):
            selavypath = [selavypath]
        elif isinstance(selavypath, list) and isinstance(selavypath[0], str):
            selavypath = [Path(p) for p in selavypath]

        self.selavypath = cast(list[Path], selavypath)
        self.components = pd.concat([self._load(p) for p in self.selavypath])
        if correct_negative:
            self._correct_negative_fluxes()

    def _load(self, selavypath: Path) -> pd.DataFrame:
        """Import selavy catalogue from multiple source formats to pandas DataFrame."""

        if selavypath.suffix in ['.xml', '.vot']:
            components = Table.read(
                selavypath, format="votable", use_names_over_ids=True
            ).to_pandas()
        elif selavypath.suffix == '.parq':
            components = pd.read_parquet(selavypath)
        else:
            components = pd.read_fwf(selavypath, skiprows=[1]).drop(columns=['#'])

        pattern = re.compile(r'\S*(\d{4}[+-]\d{2})\S*')
        sbidpattern = re.compile(r'\S*SB(\d{4,5})\S*')

        field = pattern.sub(r'\1', str(selavypath))
        sbid = sbidpattern.sub(r'\1', str(selavypath))

        # Replace fields without a name by SBID (e.g. GW fields)
        if field == str(selavypath):
            field = 'SB' + sbid

        components['sign'] = -1 if (selavypath.name[0] == 'n' or 'nimage' in selavypath.name) else 1
        components['field'] = field
        components['sbid'] = sbid

        return components

    @classmethod
    def from_params(cls, epoch: str, stokes: str, tiletype: str, fields: Union[str, list[str]]='', is_name: bool=False):

        if isinstance(fields, str):
            fields = [fields]
        
        survey = get_survey(epoch, is_name)

        selavypath = Path(survey[f'selavy_path_{stokes}_{tiletype[0]}'])
        selavy_files = []

        for field in fields:

            # First try locating catalogues in xml format, then try txt format
            files = list(selavypath.glob(f'*[._]{field}*components.xml'))
            if len(files) == 0:
                files = list(selavypath.glob(f'*[._]{field}*components.txt'))

            selavy_files.extend(files)
            
        if len(selavy_files) == 0:
            raise FileNotFoundError(f"Could not locate selavy files at {selavypath} in either xml or txt format.")

        return cls(selavy_files)

    @classmethod
    def from_aegean(cls, aegeanpath: Union[str, Path]):
        """Load Aegean source cat and convert to selavy format."""

        cat = cls(aegeanpath, correct_negative=False)

        columns = {
            'island': 'island_id',
            'source': 'component_id',
            'local_rms': 'rms_image',
            'rms_background': 'rms_image',
            'ra': 'ra_deg_cont',
            'dec': 'dec_deg_cont',
            'err_ra': 'ra_err',
            'err_dec': 'dec_err',
            'peak_flux': 'flux_peak',
            'err_peak_flux': 'flux_peak_err',
            'raw_peak_flux': 'flux_peak',
            'err_raw_peak_flux': 'flux_peak_err',
            'int_flux': 'flux_int',
            'err_int_flux': 'flux_int_err',
            'raw_total_flux': 'flux_int',
            'err_raw_total_flux': 'flux_int_err',
            'a': 'maj_axis',
            'bmaj': 'maj_axis',
            'err_a': 'maj_axis_err',
            'b': 'min_axis',
            'bmin': 'min_axis',
            'err_b': 'min_axis_err',
            'pa': 'pos_ang',
            'err_pa': 'pos_ang_err',
        }

        cat.components.rename(columns=columns, inplace=True)

        cat.components.maj_axis *= 3600
        cat.components.min_axis *= 3600

        return cat

    def cone_search(self, position: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
        """Find components within radius of position."""

        components = self.components.copy()

        selavy_coords = SkyCoord(ra=components.ra_deg_cont, dec=components.dec_deg_cont, unit=u.deg)
        components['d2d'] = position.separation(selavy_coords).arcsec

        return components[components.d2d < radius.to(u.arcsec)]

    def _correct_negative_fluxes(self):
        """Correct recent selavy update that writes fluxes as negative values for flagNegative runs."""

        for col in ['flux_peak', 'flux_peak_err', 'flux_int', 'flux_int_err']:
            self.components[col] = self.components[col].abs()
    
    def nearest_component(self, position: SkyCoord, radius: u.Quantity) -> Union[pd.Series, None]:
        """Return closest SelavyComponent within radius of position."""

        components = self.cone_search(position, radius)

        if components.empty:
            return None
        
        return components.sort_values('d2d').iloc[0]


def condon_flux_error(
        component: pd.Series,
        bmaj: u.Quantity,
        bmin: u.Quantity,
        flux_scale_error: float=0.,
        fluxtype: str='int'
) -> float:
    """Flux density errors of selavy component according to Condon (1997) / Condon (1998)."""

    if not isinstance(bmaj, u.Quantity) or not isinstance(bmin, u.Quantity):
        raise TypeError(f"bmaj and bmin must be of type Quantity")

    bmaj = bmaj.to(u.arcsec).value
    bmin = bmin.to(u.arcsec).value

    flux = component[f'flux_{fluxtype}']
    rms = component['rms_image']
    snr = flux / rms

    theta_M = component['maj_axis']
    theta_m = component['min_axis']
    alpha_M = 1.5
    alpha_m = 1.5

    # Potential future parameters
    # clean_bias = 0
    clean_bias_error = 0

    # Condon (1997) Equation 41
    rho_sq = ((theta_M * theta_m / (4 * bmaj * bmin)) *
              ((1 + (bmaj/theta_M)**2)) ** alpha_M *
              ((1 + (bmin/theta_m)**2)) ** alpha_m *
              snr ** 2)

    # Optionally correct for local noise gradient and clean bias
    # Unclear how selavy's peak / integrated fluxes are calculated
    # w.r.t. clean_bias or noise gradients, so ignoring for now.
    # flux += -noise**2 / flux + clean_bias

    if fluxtype == 'peak':
        flux *= np.sqrt(theta_M * theta_m / (bmaj * bmin))

    # Condon (1997) Equation 21 w/ clean and calibration errors
    flux_error = np.sqrt(
        (flux_scale_error * flux) ** 2 +
        clean_bias_error ** 2 +
        (2 * flux ** 2 / rho_sq)
    )

    return flux_error


def fractional_pol_error(
        sources: pd.DataFrame,
        flux_col: str='flux',
        fp_col: str='fp',
        corr_errors: bool=False,
) -> pd.DataFrame:
    """Propagate uncertainty from Stokes I and V flux density to fractional polarisation.

    NOTE: Accounting for error correlation requires more thought.
          Need to look at how individual contributions to flux density errors 
          (e.g. local rms gradient, clean bias, fitting bias, snr) 
          each correlate between Stokes I and V.
    """
        
    if corr_errors:
        corr_xy = sources.corr().loc[f'{flux_col}_i', f'{flux_col}_v']
        logger.info(f"Using corr(I, V) = {corr_xy:.4f} for f_p error propagation.")
    else:
        corr_xy = 0

    I_rel_err = sources[f'{flux_col}_err_i'] / sources[f'{flux_col}_i']
    V_rel_err = sources[f'{flux_col}_err_v'] / sources[f'{flux_col}_v']

    pol_error = np.sqrt(
        (I_rel_err ** 2 + V_rel_err ** 2) * sources[fp_col] ** 2 -
        2 * sources[fp_col]**2 * I_rel_err * V_rel_err * corr_xy
    )

    return pol_error


def measure_flux(
        position: SkyCoord,
        epochs: pd.DataFrame,
        size: u.Quantity,
        fluxtype: str,
        stokes: str,
        tiletype: str,
        name: str='source'
) -> pd.DataFrame:
    """Measure flux or 3-sigma nondetection limit in multiple epochs at position."""

    flux_list = []
    for _, epoch in epochs.iterrows():

        # Get fields containing the source
        fields = find_fields(position, epoch.survey)

        if len(fields) == 0:
            continue

        if len(fields) > 1:
            logger.warning(f"{epoch['name']} has more than one field containing {name}")

        for _, field in fields.iterrows():

            fieldname = field.field
            _, header = get_image_from_survey_params(epoch, fieldname, stokes, tiletype, load=False)
        
            selavy = SelavyCatalogue.from_params(epoch['name'], fieldname, stokes)
            component = selavy.nearest_component(position, radius=size)

            if component is None:
                # Measure 3-sigma limit from RMS map
                image_file = list(Path(epoch[f'image_path_{stokes}_{tiletype[0]}']).glob(f'*{fieldname}*.fits'))[0]
                rms_image = Path(
                    str(image_file)
                    .replace('IMAGES', 'RMSMAPS')
                    .replace(f'.{stokes.upper()}.fits', f'.{stokes.upper()}_rms.fits')
                )

                flux = measure_limit(position, rms_image, 30*u.arcsec)
                flux_err = np.nan
                limit = True
            else:
                # Handle potential negative Stokes V fluxes
                flux = component.flux_int * component.sign
                
                # Calculate Condon flux error if component was found
                bmaj, bmin = header['BMAJ']*u.deg, header['BMIN']*u.deg
                flux_err = condon_flux_error(component, bmaj, bmin, fluxtype=fluxtype)
                limit = False

            data = {
                'source': name,
                'epoch': epoch['name'],
                'field': field.field,
                'flux': flux,
                'flux_err': flux_err,
                'limit': limit,
                'dist_centre': field.dist_field_centre,
            }
            flux_list.append(data)

    flux_table = pd.DataFrame(flux_list)

    return flux_table 


def measure_limit(position: SkyCoord, image_path: Path, size: u.Quantity) -> pd.Series:
    """Calculate 3-sigma rms limit from at position in image."""

    data, header = get_image_data_header(image_path)
    wcs = WCS(header, naxis=2)
        
    try:
        cutout = Cutout2D(data[0, 0, :, :], position, size, wcs=wcs)
    except IndexError:
        cutout = Cutout2D(data, position, size, wcs=wcs)

    if 'rms' in image_path.name:
        return 3 * np.median(cutout.data)
    else:
        return 3 * np.sqrt(np.mean(np.square(cutout.data)))

