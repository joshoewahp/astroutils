from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.nddata import NoOverlapError

from astroutils.source import (
    SelavyCatalogue,
    condon_flux_error,
    force_measure_flux,
    fractional_pol_error,
    measure_epoch_flux,
    measure_flux,
    measure_limit,
)

mock_survey = pd.Series(
    {
        "selavy_path_i_C": "tests/data",
        "selavy_path_v_C": "tests/data",
    }
)


@pytest.fixture()
def survey(mocker):
    mocker.patch("astroutils.source.get_survey", return_value=mock_survey)

    return


def validate_selavy_catalogue(cat, n_expected_rows):
    """Check that all necessary columns are named correctly in components DataFrame."""

    # Check basic object construction
    assert isinstance(cat.selavypath, list)
    assert isinstance(cat.selavypath[0], Path)
    assert isinstance(cat.components, pd.DataFrame)
    assert len(cat.components) == n_expected_rows

    # Check that required columns are correctly named
    required_cols = [
        "ra_deg_cont",
        "ra_err",
        "dec_deg_cont",
        "dec_err",
        "maj_axis",
        "min_axis",
        "pos_ang",
        "flux_int",
        "flux_int_err",
        "flux_peak",
        "flux_peak_err",
        "rms_image",
        "sign",
    ]
    for col in required_cols:
        assert col in cat.components.columns

    # Check that maj_axis / min_axis are in units of arcsec (should be > 1 as simple test)
    assert cat.components.iloc[0].maj_axis > 1
    assert cat.components.iloc[0].min_axis > 1

    return


txt_path = "tests/data/RACS_0000-12A.EPOCH00.I.selavy.components.txt"
xml_path = "tests/data/VAST_0012+00A.EPOCH12.I.selavy.components.xml"
stokesv_path = "tests/data/nimage.v.VAST_0021+00.SB29580.components.xml"
empty_path = "tests/data/empty_selavyfile.txt"


@pytest.mark.parametrize(
    "path, num_components",
    [
        (txt_path, 15),
        (xml_path, 23),
        (Path(xml_path), 23),
        ([txt_path, xml_path], 38),
    ],
)
def test_selavy_path_parsing(path, num_components):
    cat = SelavyCatalogue(path)

    validate_selavy_catalogue(cat, num_components)


def test_selavy_empty_components():
    with pytest.raises(ValueError):
        SelavyCatalogue(empty_path)


@pytest.mark.parametrize(
    "surveyname, fields, sbids, stokes, tiletype, num_components",
    [
        ("racs-low", "0000-12", "", "i", "COMBINED", 15),
        ("racs-low", ["0000-12", "1200-74"], "", "i", "COMBINED", 28),
        ("vastp12", "0012+00", "", "i", "COMBINED", 23),
        ("vastp23", "0021+00", "SB29580", "i", "COMBINED", 7),
    ],
)
def test_selavy_from_params(
    surveyname,
    fields,
    sbids,
    stokes,
    tiletype,
    num_components,
    survey,
):
    cat = SelavyCatalogue.from_params(
        surveyname,
        stokes=stokes,
        fields=fields,
        sbids=sbids,
        tiletype=tiletype,
    )

    validate_selavy_catalogue(cat, num_components)


def test_selavy_from_params_raises_error_if_no_path():
    with pytest.raises(FileNotFoundError):
        SelavyCatalogue.from_params(
            epoch="vastp1", fields="0012+10", stokes="i", tiletype="COMBINED"
        )


def test_selavy_from_aegean():
    cat = SelavyCatalogue.from_aegean("tests/data/mwats_test.parq")

    validate_selavy_catalogue(cat, 10)


def test_selavy_cone_search():
    cat = SelavyCatalogue(xml_path)
    position = SkyCoord(ra=3.292805, dec=0.856316, unit="deg")
    components = cat.cone_search(position, radius=0.5 * u.arcmin)

    assert len(components) == 2
    assert components.d2d.min() == 0
    assert components.d2d.max().round(3) == 27.959


def test_selavy_nearest_component_when_in_radius():
    cat = SelavyCatalogue(txt_path)

    position_ra0 = SkyCoord(ra=0, dec=-11, unit=u.deg)
    component = cat.nearest_component(position_ra0, radius=1 * u.deg)

    assert len(component) == 42


def test_selavy_nearest_component_when_not_in_radius():
    cat = SelavyCatalogue(txt_path)
    position_ra0 = SkyCoord(ra=0, dec=-11, unit=u.deg)
    component = cat.nearest_component(position_ra0, radius=15 * u.arcsec)

    assert component is None


@pytest.mark.parametrize(
    "column", ["flux_peak", "flux_peak_err", "flux_int", "flux_int_err"]
)
def test_selavy_negative_fluxes_corrected(column):
    cat = SelavyCatalogue(stokesv_path)

    assert np.all(cat.components[column] > 0)


@pytest.fixture
def component():
    comp = pd.Series(
        {
            "flux_peak": 1.41,
            "flux_int": 1.38,
            "rms_image": 0.23,
            "maj_axis": 16.3,
            "min_axis": 12.1,
        }
    )
    return comp


@pytest.mark.parametrize(
    "bmaj, bmin, fluxtype, expected_error",
    [
        (15 * u.arcsec, 12 * u.arcsec, "peak", 0.246),
        (15 * u.arcsec, 12 * u.arcsec, "int", 0.235),
        (15 / 3600 * u.deg, 12 / 3600 * u.deg, "peak", 0.246),
    ],
)
def test_condon_error_calculation(bmaj, bmin, fluxtype, expected_error, component):
    error = condon_flux_error(
        component,
        bmaj=bmaj,
        bmin=bmin,
        flux_scale_error=0,
        fluxtype=fluxtype,
    )

    assert error.round(3) == expected_error


def test_condon_error_bmaj_not_quantity_raises_typeerror(component):
    with pytest.raises(TypeError):
        condon_flux_error(
            component,
            bmaj=15,
            bmin=12 * u.arcsec,
            flux_scale_error=0,
            fluxtype="peak",
        )


@pytest.fixture
def fluxes():
    fluxes = pd.DataFrame(
        {
            "name": ["test"] * 3,
            "ra": [0.0, 0.015, 0.01],
            "dec": [0.0, 0.015, 0.01],
            "epoch": ["racs-low", "vastp1", "vastp2"],
            "field": ["0012+00"] * 3,
            "flux_i": [1.2, 1.1, 2.3],
            "flux_err_i": [0.22, 0.24, 0.21],
            "flux_v": [1.1, 0.64, 0.55],
            "flux_err_v": [0.22, 0.24, 0.21],
        }
    )
    fluxes["fp"] = fluxes.flux_v.abs() / fluxes.flux_i

    return fluxes


@pytest.mark.parametrize(
    "corr_errors, expected_errors",
    [(False, [0.248704, 0.252423, 0.093879]), (True, [0.311009, 0.308328, 0.105212])],
)
def test_fractional_pol_error_calculation(corr_errors, expected_errors, fluxes):
    fp_errors = fractional_pol_error(fluxes, corr_errors=corr_errors)

    for error, expected_error in zip(fp_errors, expected_errors):
        assert round(error, 6) == expected_error


@pytest.fixture
def epochs():
    epochs = pd.DataFrame(
        {
            "survey": ["vastp2", "vastp8"],
            "name": ["VAST P1-2", "VAST P1-8"],
            "image_path_i_T": ["TILES/STOKESI_IMAGES/", "TILES/STOKESI_IMAGES/"],
            "image_path_v_T": ["TILES/STOKESV_IMAGES/", "TILES/STOKESV_IMAGES/"],
            "image_path_i_C": ["COMBINED/STOKESI_IMAGES/", "COMBINED/STOKESI_IMAGES/"],
            "image_path_v_C": ["COMBINED/STOKESV_IMAGES/", "COMBINED/STOKESV_IMAGES/"],
        }
    )
    return epochs


mocked_fields = pd.DataFrame(
    {
        "field": ["1237+00", "1237-06"],
        "sbid": ["", ""],
        "image_path_i_T": [
            "tests/data/test_image_mJy.fits",
            "tests/data/test_image_mJy.fits",
        ],
        "image_path_v_T": [
            "tests/data/test_image_mJy.fits",
            "tests/data/test_image_mJy.fits",
        ],
        "cr_ra_pix": [189.30660013892276, 189.3064293311123],
        "cr_dec_pix": [0.0030764999999284, -6.298899722106099],
        "bmaj": [12.966916344916608, 12.813313530441672],
        "bmin": [12.688634591018316, 11.58917164487208],
        "dist_field_centre": [2.0110998437527563, 4.322889739848082],
    }
)


@pytest.mark.parametrize(
    "selavy_path, expected_flux",
    [
        ("tests/data/VAST_1237+00A.EPOCH02.I.selavy.components.xml", 1.405),
        ("tests/data/VAST_1237+00A.EPOCH08.I.selavy.components.xml", 16.261),
    ],
)
def test_measure_flux_value(selavy_path, expected_flux, epochs, mocker):
    position = SkyCoord(ra=189.0104, dec=-1.9861, unit="deg")
    size = 0.1 * u.deg
    fluxtype = "int"
    stokes = "i"
    tiletype = "TILES"

    mocked_glob_results = ["tests/data/test_image_mJy.fits"]
    sel = SelavyCatalogue(selavy_path)

    mocker.patch("astroutils.source.Path.glob", return_value=mocked_glob_results)
    mocker.patch("astroutils.source.find_fields", return_value=mocked_fields)
    mocker.patch("astroutils.source.SelavyCatalogue.from_params", return_value=sel)

    # Patch the future results from multiprocessing
    flux = measure_epoch_flux(
        position,
        epochs.iloc[0],
        size,
        fluxtype,
        stokes,
        tiletype,
        name="source",
    )
    mocker.patch("astroutils.source.Future.result", return_value=flux)

    fluxes = measure_flux(
        position,
        epochs,
        size=size,
        fluxtype=fluxtype,
        stokes=stokes,
        tiletype=tiletype,
    )

    assert fluxes.flux.iloc[0].round(3) == expected_flux


@pytest.mark.parametrize(
    "image_path, expected_flux",
    [
        (Path("tests/data/test_image_Jy.fits"), 1.1002),
        (Path("tests/data/test_image_mJy.fits"), 1.1002),
        (Path("tests/data/test_image_rms.fits"), 1.0427),
    ],
)
def test_measure_limit_values(image_path, expected_flux):
    position = SkyCoord(ra=189.0104, dec=-1.9861, unit="deg")
    limit = measure_limit(position=position, image_path=image_path, size=0.1 * u.deg)

    assert limit.round(4) == expected_flux


@pytest.mark.parametrize(
    "position, image, background, noise, expected_flux, is_limit, overlap",
    [
        (
            SkyCoord(ra=189.0104, dec=-1.9861, unit="deg"),
            Path("tests/data/test_image_Jy.fits"),
            Path("tests/data/noimage.fits"),
            Path("tests/data/noimage.fits"),
            1.1002,
            True,
            True,
        ),
        (
            SkyCoord(ra=189.0104, dec=-10.9861, unit="deg"),
            Path("tests/data/test_image_Jy.fits"),
            Path("tests/data/noimage.fits"),
            Path("tests/data/noimage.fits"),
            1.1002,
            True,
            False,
        ),
    ],
)
def test_force_measure_flux(
    position,
    image,
    background,
    noise,
    expected_flux,
    is_limit,
    overlap,
):
    if not overlap:
        with pytest.raises(NoOverlapError):
            component, limit_flag = force_measure_flux(
                position=position,
                image=image,
                background=background,
                noise=noise,
                size=0.1 * u.deg,
            )
        return

    component, limit_flag = force_measure_flux(
        position=position,
        image=image,
        background=background,
        noise=noise,
        size=0.1 * u.deg,
    )

    assert is_limit == limit_flag
    assert component.flux_peak.round(4) == expected_flux
