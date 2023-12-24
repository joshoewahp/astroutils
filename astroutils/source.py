import logging
import re
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Union, cast

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.table import Table
from astropy.wcs import WCS
from forced_phot import ForcedPhot

from astroutils.io import (
    find_fields,
    get_image_data_header,
    get_image_from_survey_params,
    get_survey,
)

logger = logging.getLogger(__name__)

Pathset = Union[str, Path, list[str], list[Path]]
Strset = Union[str, list[str]]


class SelavyCatalogue:
    def __init__(self, selavypath: Pathset, correct_negative: bool = True):
        if isinstance(selavypath, str):
            selavypath = [Path(selavypath)]
        elif isinstance(selavypath, Path):
            selavypath = [selavypath]
        elif isinstance(selavypath, list) and isinstance(selavypath[0], str):
            selavypath = [Path(p) for p in selavypath]

        self.selavypath = cast(list[Path], selavypath)

        components = [self._load(p) for p in self.selavypath]
        components = [c for c in components if c is not None]

        if len(components) == 0:
            raise ValueError("No components found.")

        self.components = pd.concat(components).reset_index(drop=True)

        if correct_negative:
            self._correct_negative_fluxes()

    def _load(self, selavypath: Path) -> pd.DataFrame:
        """Import selavy catalogue from multiple source formats to pandas DataFrame."""

        # Read in variety of data formats
        if selavypath.suffix in [".xml", ".vot"]:
            components = Table.read(
                selavypath,
                format="votable",
                use_names_over_ids=True,
            ).to_pandas()
        elif selavypath.suffix == ".csv":
            components = pd.read_csv(selavypath)
        elif selavypath.suffix == ".parq":
            components = pd.read_parquet(selavypath)
        else:
            components = pd.read_fwf(selavypath, skiprows=[1])

        # Skip loading of empty component files
        if components.empty:
            return

        # Fix extra column in FWF files
        if "#" in components.columns:
            components.drop(columns=["#"], inplace=True)

        pattern = re.compile(r"\S*(\d{4}[+-]\d{2})\S*")
        sbidpattern = re.compile(r"\S*SB(\d{4,5})\S*")

        field = pattern.sub(r"\1", str(selavypath))
        sbid = sbidpattern.sub(r"\1", str(selavypath))

        # Replace fields without a name by SBID (e.g. GW fields)
        if field == str(selavypath):
            field = "SB" + sbid

        components["sign"] = (
            -1 if (selavypath.name[0] == "n" or "nimage" in selavypath.name) else 1
        )

        components["field"] = field
        components["sbid"] = sbid

        return components

    @classmethod
    def from_params(
        cls,
        epoch: str,
        stokes: str,
        tiletype: str,
        fields: Strset = "",
        sbids: Strset = "",
        is_name: bool = False,
    ):
        if isinstance(fields, str):
            fields = [fields]

        if sbids == "":
            sbids = ["" for field in fields]
        elif isinstance(sbids, str):
            sbids = [sbids]

        survey = get_survey(epoch, is_name)

        selavypath = Path(survey[f"selavy_path_{stokes}_{tiletype[0]}"])
        selavy_files = []

        for field, sbid in zip(fields, sbids):
            # First try locating catalogues in xml format
            files = list(selavypath.glob(f"*[._]{field}[AB.]*{sbid}.*components.xml"))

            # then try txt format
            if len(files) == 0:
                files = list(
                    selavypath.glob(f"*[._]{field}[AB.]*{sbid}.*components.txt")
                )

            # then try early RACS pattern with SBID first
            if len(files) == 0:
                files = list(
                    selavypath.glob(f"*[._]{sbid}*{field}[AB.].*components.xml")
                )

            # and finally early RACS text catalogues
            if len(files) == 0:
                files = list(
                    selavypath.glob(f"*[._]{sbid}*{field}[AB.].*components.txt")
                )

            selavy_files.extend(files)

        if len(selavy_files) == 0:
            raise FileNotFoundError(
                f"Could not locate selavy files at {selavypath} in either xml or txt format."
            )

        return cls(selavy_files)

    @classmethod
    def from_aegean(cls, aegeanpath: Pathset):
        """Load Aegean source cat and convert to selavy format."""

        cat = cls(aegeanpath, correct_negative=False)

        columns = {
            "island": "island_id",
            "source": "component_id",
            "local_rms": "rms_image",
            "rms_background": "rms_image",
            "ra": "ra_deg_cont",
            "dec": "dec_deg_cont",
            "err_ra": "ra_err",
            "err_dec": "dec_err",
            "peak_flux": "flux_peak",
            "err_peak_flux": "flux_peak_err",
            "raw_peak_flux": "flux_peak",
            "err_raw_peak_flux": "flux_peak_err",
            "int_flux": "flux_int",
            "err_int_flux": "flux_int_err",
            "raw_total_flux": "flux_int",
            "err_raw_total_flux": "flux_int_err",
            "a": "maj_axis",
            "bmaj": "maj_axis",
            "err_a": "maj_axis_err",
            "b": "min_axis",
            "bmin": "min_axis",
            "err_b": "min_axis_err",
            "pa": "pos_ang",
            "err_pa": "pos_ang_err",
        }

        cat.components.rename(columns=columns, inplace=True)

        cat.components.maj_axis *= 3600
        cat.components.min_axis *= 3600

        return cat

    def cone_search(self, position: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
        """Find components within radius of position."""

        components = self.components.copy()

        selavy_coords = SkyCoord(
            ra=components.ra_deg_cont, dec=components.dec_deg_cont, unit=u.deg
        )
        components["d2d"] = position.separation(selavy_coords).arcsec

        return components[components.d2d < radius.to(u.arcsec)]

    def _correct_negative_fluxes(self):
        """Correct recent selavy update that writes fluxes as negative values for flagNegative runs."""

        for col in ["flux_peak", "flux_peak_err", "flux_int", "flux_int_err"]:
            self.components[col] = self.components[col].abs()

    def nearest_component(
        self, position: SkyCoord, radius: u.Quantity
    ) -> Union[pd.Series, None]:
        """Return closest SelavyComponent within radius of position."""

        components = self.cone_search(position, radius)

        if components.empty:
            return None

        return components.sort_values("d2d").iloc[0]


def condon_flux_error(
    component: pd.Series,
    bmaj: u.Quantity,
    bmin: u.Quantity,
    flux_scale_error: float = 0.0,
    fluxtype: str = "int",
) -> float:
    """Flux density errors of selavy component according to Condon (1997) / Condon (1998)."""

    if not isinstance(bmaj, u.Quantity) or not isinstance(bmin, u.Quantity):
        raise TypeError("bmaj and bmin must be of type Quantity")

    bmaj = bmaj.to(u.arcsec).value
    bmin = bmin.to(u.arcsec).value

    flux = component[f"flux_{fluxtype}"]
    rms = component["rms_image"]
    snr = flux / rms

    theta_M = component["maj_axis"]
    theta_m = component["min_axis"]
    alpha_M = 1.5
    alpha_m = 1.5

    # Potential future parameters
    # clean_bias = 0
    clean_bias_error = 0

    # Condon (1997) Equation 41
    rho_sq = (
        (theta_M * theta_m / (4 * bmaj * bmin))
        * ((1 + (bmaj / theta_M) ** 2)) ** alpha_M
        * ((1 + (bmin / theta_m) ** 2)) ** alpha_m
        * snr**2
    )

    # Optionally correct for local noise gradient and clean bias
    # Unclear how selavy's peak / integrated fluxes are calculated
    # w.r.t. clean_bias or noise gradients, so ignoring for now.
    # flux += -noise**2 / flux + clean_bias

    if fluxtype == "peak":
        flux *= np.sqrt(theta_M * theta_m / (bmaj * bmin))

    # Condon (1997) Equation 21 w/ clean and calibration errors
    flux_error = np.sqrt(
        (flux_scale_error * flux) ** 2
        + clean_bias_error**2
        + (2 * flux**2 / rho_sq)
    )

    return flux_error


def fractional_pol_error(
    sources: pd.DataFrame,
    flux_col: str = "flux",
    fp_col: str = "fp",
    corr_errors: bool = False,
) -> pd.DataFrame:
    """Propagate uncertainty from Stokes I and V flux density to fractional polarisation.

    NOTE: Accounting for error correlation requires more thought.
          Need to look at how individual contributions to flux density errors
          (e.g. local rms gradient, clean bias, fitting bias, snr)
          each correlate between Stokes I and V.
    """

    if corr_errors:
        corr_xy = (
            sources[[f"{flux_col}_i", f"{flux_col}_v"]]
            .corr()
            .loc[f"{flux_col}_i", f"{flux_col}_v"]
        )
        logger.info(f"Using corr(I, V) = {corr_xy:.4f} for f_p error propagation.")
    else:
        corr_xy = 0

    I_rel_err = sources[f"{flux_col}_err_i"] / sources[f"{flux_col}_i"]
    V_rel_err = sources[f"{flux_col}_err_v"] / sources[f"{flux_col}_v"]

    pol_error = np.sqrt(
        (I_rel_err**2 + V_rel_err**2) * sources[fp_col] ** 2
        - 2 * sources[fp_col] ** 2 * I_rel_err * V_rel_err * corr_xy
    )

    return pol_error


def force_measure_flux(
    position: SkyCoord,
    image: Path,
    background: Path,
    noise: Path,
    size: u.Quantity,
) -> pd.Series:
    # Perform forced extraction
    try:
        FP = ForcedPhot(str(image), str(background), str(noise), verbose=True)
        flux, flux_err, chisq, DOF, cluster_id = FP.measure(
            position,
            cluster_threshold=0,
        )

        is_limit = False
        flux *= 1000
        flux_err *= 1000

        # Ensure forced fit is above 3-sigma significance
        rms_limit = measure_limit(position, noise, 30 * u.arcsec)
        rms = rms_limit / 3

        if abs(flux) < rms_limit or np.isnan(flux):
            flux = rms_limit
            flux_err = np.nan
            is_limit = True

    except (FileNotFoundError, AttributeError, IndexError):
        logger.debug("Missing background/noise maps, measuring RMS from image.")

        # If rms/background files missing, just use average survey RMS
        try:
            flux = measure_limit(position, image, size)
            rms = flux / 3
            flux_err = np.nan
            is_limit = True
        except NoOverlapError:
            logger.debug(
                f"Image does not contain position <{position.ra:.2f},{position.dec:.2f}>"
            )
            return None, None

    component = pd.Series(
        {
            "flux_peak": flux,
            "flux_int": flux,
            "flux_peak_err": flux_err,
            "flux_int_err": flux_err,
            "rms_image": rms,
        }
    )

    return component, is_limit


def measure_epoch_flux(
    position: SkyCoord,
    epoch: pd.Series,
    size: u.Quantity,
    fluxtype: str,
    stokes: str,
    tiletype: str,
    name: str = "source",
):
    # Get fields containing the source
    ttype = "COMBINED" if epoch.survey == "racs-low" and stokes == "v" else tiletype

    fields = find_fields(position, epoch.survey, tiletype=ttype)

    if len(fields) == 0:
        return

    if len(fields) > 1:
        logger.debug(f"{epoch['name']} has more than one field containing {name}")

    components = []
    for _, field in fields.iterrows():
        is_limit = False
        is_forced = False

        fieldname = field.field
        _, header = get_image_from_survey_params(
            epoch, fieldname, stokes, ttype, load=False
        )

        # First look for flux from selavy
        try:
            selavy = SelavyCatalogue.from_params(
                epoch.survey,
                stokes=stokes,
                tiletype=ttype,
                fields=fieldname,
            )
            component = selavy.nearest_component(position, radius=size)
        except (FileNotFoundError, ValueError):
            component = None

        if component is None:
            logger.debug(
                f"{epoch['name']} {fieldname} - no selavy component, performing forced fit"
            )

            image_file = list(
                Path(epoch[f"image_path_{stokes}_{ttype[0]}"]).glob(
                    f"*{fieldname}*.fits"
                )
            )[0]
            noise_filepattern = str(image_file).replace("IMAGES", "RMSMAPS")
            bkg_file = Path(noise_filepattern.replace("image.", "meanMap.image."))
            rms_file = Path(noise_filepattern.replace("image.", "noiseMap.image."))

            # Then force a measurement in the image and measure 3-sigma limit from RMS map
            component, is_limit = force_measure_flux(
                position,
                image_file,
                bkg_file,
                rms_file,
                size,
            )
            is_forced = True

            # Neither selavy or forced fit possible. Move to next field.
            if component is None:
                continue

        if not is_forced:
            # Handle potential negative Stokes V fluxes
            flux = component[f"flux_{fluxtype}"] * component.sign

            # Calculate Condon flux error if component was found
            bmaj, bmin = header["BMAJ"] * u.deg, header["BMIN"] * u.deg
            flux_err = condon_flux_error(component, bmaj, bmin, fluxtype=fluxtype)
            rms = component.rms_image
        else:
            flux = component[f"flux_{fluxtype}"]
            flux_err = component[f"flux_{fluxtype}_err"]
            rms = component.rms_image

        try:
            data = {
                "source": name,
                "epoch": epoch["name"],
                "obsdate": header.get("DATE-OBS", ""),
                "field": field.field,
                "flux": flux,
                "flux_err": flux_err,
                "rms_image": rms,
                "snr": abs(flux / rms),
                "is_limit": is_limit,
                "is_forced": is_forced,
                "dist_field_centre": field.dist_field_centre,
            }
        except KeyError:
            print(name, stokes, epoch["name"])
            print(header.keys)
            raise

        components.append(data)

    return pd.DataFrame(components)


def measure_flux(
    position: SkyCoord,
    epochs: pd.DataFrame,
    size: u.Quantity,
    fluxtype: str,
    stokes: str,
    tiletype: str,
    name: str = "source",
) -> pd.DataFrame:
    """Measure flux or 3-sigma nondetection limit in multiple epochs at position."""

    with ProcessPoolExecutor(max_workers=10) as executor:
        processes = [
            executor.submit(
                measure_epoch_flux,
                position,
                epoch,
                size,
                fluxtype,
                stokes,
                tiletype,
                name,
            )
            for _, epoch in epochs.iterrows()
        ]
    flux_list = [future.result() for future in processes if future.result() is not None]

    return pd.concat(flux_list)


def measure_limit(position: SkyCoord, image_path: Path, size: u.Quantity) -> pd.Series:
    """Calculate 3-sigma rms limit from at position in image."""

    try:
        data, header = get_image_data_header(image_path)
    except FileNotFoundError:
        return

    wcs = WCS(header, naxis=2)

    try:
        cutout = Cutout2D(data[0, 0, :, :], position, size, wcs=wcs)
    except IndexError:
        cutout = Cutout2D(data, position, size, wcs=wcs)

    if "noise" in image_path.name:
        return 3 * np.nanmedian(cutout.data)
    else:
        return 3 * np.sqrt(np.nanmean(np.square(cutout.data)))


def measure_polarised_source(
    position: SkyCoord,
    survey: str,
    field: str,
    tiletype: str = "TILES",
) -> pd.Series:
    # Get selavy fluxes
    cat_i = SelavyCatalogue.from_params(
        survey,
        stokes="i",
        tiletype=tiletype,
        fields=[field],
    )
    cat_v = SelavyCatalogue.from_params(
        survey,
        stokes="v",
        tiletype=tiletype,
        fields=[field],
    )
    comp_i = cat_i.cone_search(position, radius=0.1 * u.arcmin).iloc[0]
    comp_v = cat_v.cone_search(position, radius=0.1 * u.arcmin).iloc[0]

    # Get PSF size parameters
    epoch = get_survey(survey)
    _, header = get_image_from_survey_params(epoch, field, "i", tiletype, load=False)
    bmaj, bmin = header["bmaj"] * u.deg, header["bmin"] * u.deg

    # Calculate errors
    i_peak_err = condon_flux_error(comp_i, bmaj, bmin, fluxtype="peak")
    i_int_err = condon_flux_error(comp_i, bmaj, bmin, fluxtype="int")
    v_peak_err = condon_flux_error(comp_v, bmaj, bmin, fluxtype="peak")
    v_int_err = condon_flux_error(comp_v, bmaj, bmin, fluxtype="int")

    # Calculate fractional polarisation and error
    source = pd.Series(
        {
            "flux_peak_i": comp_i.flux_peak,
            "flux_peak_err_i": i_peak_err,
            "flux_int_i": comp_i.flux_int,
            "flux_int_err_i": i_int_err,
            "flux_peak_v": comp_v.flux_peak,
            "flux_peak_err_v": v_peak_err,
            "flux_int_v": comp_v.flux_int,
            "flux_int_err_v": v_int_err,
        }
    )
    source["fp_peak"] = source.flux_peak_v / source.flux_peak_i
    source["fp_peak_err"] = fractional_pol_error(
        source, flux_col="flux_peak", fp_col="fp_peak"
    )
    source["fp_int"] = source.flux_int_v / source.flux_int_i
    source["fp_int_err"] = fractional_pol_error(
        source, flux_col="flux_int", fp_col="fp_int"
    )

    return source


def get_stokes_matches(
    survey: str,
    tiletype: str = "TILES",
) -> pd.DataFrame:
    """Crossmatch all Stokes I and V selavy components in survey epoch.

    These are useful for characterising leakage, so we search for all matches within 20 arcsec
    in order to easily identify isolated sources that sample polarisation leakage across the field.
    """

    fields = pd.read_csv(
        f"/import/ada1/jpri6587/aux_data/fields/{survey}_{tiletype.lower()}_fields.csv"
    )

    components = []
    for _, field in fields.iterrows():
        try:
            sel_i = SelavyCatalogue.from_params(
                epoch=survey,
                stokes="i",
                tiletype=tiletype,
                fields=field.field,
            ).components
            sel_v = SelavyCatalogue.from_params(
                epoch=survey,
                stokes="v",
                tiletype=tiletype,
                fields=field.field,
            ).components
        except ValueError:
            logger.warning(f"Selavy file empty for {survey} {field.field}")
            continue
        except FileNotFoundError:
            logger.warning(f"No selavy files found for {survey} {field.field}")
            continue

        if len(sel_i) < 1000:
            logger.info(f"Small selavy file at {survey} {field.field}")

        cV = SkyCoord(ra=sel_v.ra_deg_cont, dec=sel_v.dec_deg_cont, unit="deg")
        cI = SkyCoord(ra=sel_i.ra_deg_cont, dec=sel_i.dec_deg_cont, unit="deg")

        idxI, idxV, d2d, _ = cV.search_around_sky(cI, 20 * u.arcsec)

        df = pd.DataFrame({"idx_I": idxI, "idx_V": idxV, "iv_dist": d2d.arcsec})

        cols = [
            "ra_deg_cont",
            "dec_deg_cont",
            "maj_axis",
            "min_axis",
            "component_id",
            "has_siblings",
            "flux_peak",
            "flux_int",
            "rms_image",
            "maj_axis",
            "min_axis",
            "field",
            "sbid",
            "sign",
        ]
        selv = sel_v[cols]
        seli = sel_i[cols]

        df = df.merge(
            seli.rename(columns={col: "I_" + col for col in cols}),
            left_on="idx_I",
            right_index=True,
        )

        df = df.merge(
            selv.rename(columns={col: "V_" + col for col in cols}),
            left_on="idx_V",
            right_index=True,
        )

        df["epoch"] = survey
        df = df.merge(
            fields,
            left_on="I_field",
            right_on="field",
        )
        df["cp"] = df.V_flux_peak / df.I_flux_peak
        df["nmatches"] = df.groupby("idx_V").transform("size")

        field_coord = SkyCoord(ra=df.cr_ra_pix, dec=df.cr_dec_pix, unit="deg")
        c = SkyCoord(ra=df.I_ra_deg_cont, dec=df.I_dec_deg_cont, unit="deg")
        ra_offset, dec_offset = field_coord.spherical_offsets_to(c)
        centre_dist = field_coord.separation(c)

        df["ra_offset"] = ra_offset.deg
        df["dec_offset"] = dec_offset.deg
        df["field_centre_dist"] = centre_dist.deg

        logger.debug(f"{survey} - {field.field}")
        logger.debug(f"{len(cI)} I components, {len(cV)} V components")
        logger.debug(f"{len(df)} matches")

        components.append(df)

    return pd.concat(components)
