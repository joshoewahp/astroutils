import logging
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from pathlib import Path

from astroutils.io import find_fields, load_image_selavy

logger = logging.getLogger(__name__)


def get_selavy_component(position: SkyCoord, selavy: pd.DataFrame, radius):
    """Find closest selavy component within radius of position."""
    selavy_coords = SkyCoord(ra=selavy.ra_deg_cont, dec=selavy.dec_deg_cont, unit=u.deg)
    selavy['d2d'] = position.separation(selavy_coords).arcsec

    if selavy.d2d.min() > radius.to(u.arcsec).value:
        return

    return selavy.sort_values('d2d').iloc[0]


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
            _, header, selavy = load_image_selavy(epoch, fieldname, stokes)
            component = get_selavy_component(position, selavy, radius=radius*u.arcsec)

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
                sign = component.sign if stokes == 'v' else 1
                flux = component.flux_int * sign
                
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

