import logging
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from pathlib import Path

from astroutils.io import find_fields, get_survey, get_image

logger = logging.getLogger(__name__)


class SelavyCatalogue:

    def __init__(self, selavypath):

        if isinstance(selavypath, str):
            selavypath = [Path(selavypath)]
        elif isinstance(selavypath, Path):
            selavypath = [selavypath]
        elif isinstance(selavypath, list) and isinstance(selavypath[0], str):
            selavypath = [Path(p) for p in selavypath]
            
        if not (isinstance(selavypath, list) and all([isinstance(p, Path) for p in selavypath])):
            raise TypeError(f"{selavypath} must be either a Path or list of Paths.")

        self.selavypath = selavypath
        self.components = pd.concat([self._load(p) for p in self.selavypath])

    def _load(self, selavypath: Path) -> pd.DataFrame:
        """Import selavy catalogue from multiple source formats to pandas DataFrame."""

        if selavypath.suffix in ['.xml', '.vot']:
            components = Table.read(
                selavypath, format="votable", use_names_over_ids=True
            ).to_pandas()
        elif selavypath.suffix == '.csv':
            # CSVs from CASDA have all lowercase column names
            components = pd.read_csv(selavypath).rename(
                columns={"spectral_index_from_tt": "spectral_index_from_TT"}
            )
            # Remove unused columns for consistency
            components.drop(columns=['id', 'catalogue_id', 'first_sbid', 'other_sbids',
                                     'project_id', 'quality_level', 'released_date'],
                        inplace=True)
        else:
            components = pd.read_fwf(selavypath, skiprows=[1]).drop(columns=['#'])

        components['sign'] = -1 if (selavypath.name[0] == 'n' or 'nimage' in selavypath.name) else 1

        return components

    @classmethod
    def from_params(cls, epoch: str, field: str, stokes: str):
        epoch = get_survey(epoch)

        selavypath = Path(epoch[f'selavy_path_{stokes}'])
        # First try locating catalogues in xml format, then try txt format
        selavy_files = list(selavypath.glob(f'*{field}*components.xml'))
        if len(selavy_files) == 0:
            selavy_files = list(selavypath.glob(f'*{field}*components.txt'))

        if len(selavy_files) == 0:
            raise FileNotFoundError(f"Could not locate selavy files at {selavypath} in either xml or txt format.")

        return cls(selavy_files)

    @classmethod
    def from_aegean(cls, aegeanpath):
        """Load Aegean source components and convert to selavy format."""

        columns = {
            'island': 'island_id',
            'source': 'component_id',
            'local_rms': 'rms_image',
            'ra': 'ra_deg_cont',
            'dec': 'dec_deg_cont',
            'err_ra': 'ra_deg_cont_err',
            'err_dec': 'dec_deg_cont_err',
            'peak_flux': 'flux_peak',
            'err_peak_flux': 'flux_peak_err',
            'int_flux': 'flux_int',
            'err_int_flux': 'flux_int_err',
            'a': 'maj_axis',
            'err_a': 'maj_axis_err',
            'b': 'min_axis',
            'err_b': 'min_axis_err',
            'pa': 'pos_ang',
            'err_pa': 'pos_ang_err',
        }
        pass

    def cone_search(self, position: SkyCoord, radius):
        """Return DataFrame of components within radius of position, sorted by match distance."""

        components = self.components.copy()
        selavy_coords = SkyCoord(ra=components.ra_deg_cont, dec=components.dec_deg_cont, unit=u.deg)
        components['d2d'] = position.separation(selavy_coords).arcsec

        return components[components.d2d < radius.to(u.arcsec)]

    def nearest_component(self, position: SkyCoord, radius):
        """Return closest SelavyComponent within radius of position."""

        components = self.cone_search(position, radius)

        if components.d2d.min() > radius.to(u.arcsec).value:
            return

        return components.sort_values('d2d').iloc[0]


def condon_flux_error(component, bmaj, bmin, flux_scale_error=0, fluxtype='int'):
    """Flux density errors of selavy component according to Condon (1997) / Condon (1998)."""

    flux = component[f'flux_{fluxtype}']
    rms = component['rms_image']
    snr = flux / rms
    bmaj *= 3600
    bmin *= 3600

    theta_M = component['maj_axis']
    theta_m = component['min_axis']
    alpha_M = 1.5
    alpha_m = 1.5

    # Potential future parameters
    clean_bias = 0
    clean_bias_error = 0

    # Condon (1997) Equation 40
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


def fractional_pol_error(sources, flux_col='flux', fp_col='fp', corr_errors=False):
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

    return np.sqrt((I_rel_err ** 2 + V_rel_err ** 2) * sources[fp_col] ** 2 -
                2 * sources[fp_col]**2 * I_rel_err * V_rel_err * corr_xy)


def measure_flux(position, epochs, radius, fluxtype, stokes, name='source'):
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
            _, header = get_image(epoch, fieldname, stokes, load=False)
            selavy = SelavyCatalogue(epoch, fieldname, stokes)
            component = selavy.nearest_component(position, radius=radius*u.arcsec)

            if component is None:
                # Measure 3-sigma limit from RMS map
                image_file = list(Path(epoch[f'image_path_{stokes}']).glob(f'*{fieldname}*.fits'))[0]
                rms_image = Path(str(image_file)
                                .replace('IMAGES', 'RMSMAPS')
                                .replace(f'.{stokes.upper()}.fits', f'.{stokes.upper()}_rms.fits'))

                flux = measure_limit(position, rms_image, 30*u.arcsec)
                flux_err = np.nan
                limit = True
            else:
                # Handle potential negative Stokes V fluxes
                flux = component.flux_int * component.sign
                
                # Calculate Condon flux error if component was found
                bmaj, bmin = header['BMAJ'], header['BMIN']
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


def measure_limit(position: SkyCoord, image_path: Path, radius) -> pd.Series:
    """Calculate 3-sigma rms limit from at position in image."""

    with fits.open(image_path) as hdul:
        header, data = hdul[0].header, hdul[0].data * 1e3
        
    wcs = WCS(header, naxis=2)
        
    try:
        cutout = Cutout2D(data[0, 0, :, :], position, radius, wcs=wcs)
    except IndexError:
        cutout = Cutout2D(data, position, radius, wcs=wcs)

    if 'rms' in image_path.name:
        return 3 * np.median(cutout.data)
    else:
        return 3 * np.sqrt(np.mean(np.square(cutout.data)))

