import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from astroutils.io import (
    SURVEYS_PATH,
    FITSException,
    build_field_csv,
    build_vlass_field_csv,
    find_fields,
    get_config,
    get_image_data_header,
    get_image_from_survey_params,
    get_survey,
    get_surveys,
    parse_image_filenames_in_dir,
)


def test_get_config_has_paths():
    config = get_config()
    assert list(config["DATA"].keys()) == ["aux_path", "mwats_path", "vlass_path"]


@pytest.mark.parametrize(
    "system, valid_path_roots",
    [
        (
            "ada.physics.usyd.edu.au",
            ["/import/ada1", "/import/ada2", "/import/ruby1/", ""],
        ),
        ("localhost", ["/import/ada1", "/import/ada2", "/import/ruby1/", ""]),
        ("vast-data", ["/data/pilot", "/home/joshoewahp/", ""]),
    ],
)
def test_surveys_on_valid_systems(system, valid_path_roots, mocker):
    mocker.patch("astroutils.io.os.uname", return_value=["", system])

    surveys = get_surveys()
    surveys = surveys[surveys.local].dropna(subset=["selavy_path_i_T"])

    assert "root" not in surveys.columns
    assert "ada_root" not in surveys.columns
    assert "nimbus_root" not in surveys.columns

    assert all(
        [
            any(p in c.selavy_path_i_T for p in valid_path_roots)
            for _, c in surveys.iterrows()
        ]
    )


def test_valid_survey_using_survey_codes(mocker):
    mocker.patch("astroutils.io.os.uname", return_value=["", "ada.physics.usyd.edu.au"])

    survey = get_survey("racs-low")

    # Should be a single survey (Series)
    assert isinstance(survey, pd.Series)


def test_valid_survey_using_names(mocker):
    mocker.patch("astroutils.io.os.uname", return_value=["", "ada.physics.usyd.edu.au"])

    survey = get_survey("RACS Low", is_name=True)

    # Should be a single survey (Series)
    assert isinstance(survey, pd.Series)


def test_invalid_survey_name_raises_not_implemented_error(mocker):
    mocker.patch("astroutils.io.os.uname", return_value=["", "ada.physics.usyd.edu.au"])

    with pytest.raises(NotImplementedError):
        get_survey("missing-survey")


@pytest.mark.parametrize(
    "image_path",
    [Path("tests/data/test_image_mJy.fits"), Path("tests/data/test_image_Jy.fits")],
)
def test_data_units_when_input_in_mJy(image_path):
    """Test that images in either mJy or Jy units get converted to mJy."""

    data, header = get_image_data_header(image_path, load_data=True)

    # In units of mJy this image should have no values further than 1e-6 from 0
    assert (np.abs(data) > 1e-6).all()
    assert header["BUNIT"] == "mJy/beam"


def test_load_is_false():
    """Test that using load=False only loads the header into memory."""

    image_path = Path("tests/data/test_image_mJy.fits")

    data, header = get_image_data_header(image_path, load_data=False)

    # Unloaded data should be assigned to None and occupy 16 bytes
    assert data is None
    assert sys.getsizeof(data) == 16

    # Check header still loads correctly
    assert isinstance(header, fits.header.Header)


def test_image_path_resolves():
    epoch = pd.Series({"image_path_i_T": "tests/data/", "field": "mJy"})
    _, header = get_image_from_survey_params(
        epoch, field="mJy", stokes="i", tiletype="TILES", load=True
    )

    # Check the correct image is loaded by looking at centre coordinates and MJD
    assert header["CRVAL1"] == 189.3062529167
    assert header["CRVAL2"] == 0.003423722222222
    assert header["MJD-OBS"] == 58865.844162986
    assert isinstance(header, fits.header.Header)


def test_askap_fields_load(mocker):
    expected = pd.DataFrame(
        {
            "field": ["TEST00+00"],
            "cr_ra_pix": [0],
            "cr_dec_pix": [0],
        }
    )
    mocker.patch("astroutils.io.pd.read_csv", return_value=expected.copy())

    position = SkyCoord(ra=4, dec=0, unit="deg")

    fields = find_fields(position, "vastp1", tiletype="TILES")

    expected["dist_field_centre"] = 4.0

    assert (fields == expected).all().all()


def test_small_primary_beam_size_survey(mocker):
    expected = pd.DataFrame(
        {
            "field": ["TEST00+00"],
            "cr_ra_pix": [0],
            "cr_dec_pix": [0],
        }
    )
    mocker.patch("astroutils.io.pd.read_csv", return_value=expected.copy())

    position = SkyCoord(ra=4, dec=0, unit="deg")

    # VLASS has a primary beam of ~1 degree, and so should return empty here
    fields = find_fields(position, "vlass1", tiletype=None, radius=1 * u.deg)

    assert fields.empty


def test_invalid_survey_name_raises_error():
    with pytest.raises(FITSException):
        position = SkyCoord(ra=4, dec=0, unit="deg")
        find_fields(position, "vastp0", tiletype="COMBINED")


@pytest.mark.parametrize(
    "surveyname, testdata_dir, num_fields, field_list, sbid_list",
    [
        (
            "vastp3x",
            "multi_field_with_fieldname_no_sbid",
            2,
            ["0012+00", "0012-00"],
            ["SBXXX", "SBXXX"],
        ),
        ("gw1", "single_field_with_sbid_no_fieldname", 1, ["SB9602"], ["SB9602"]),
    ],
)
def test_multi_field_survey_with_fieldname_no_sbid(
    surveyname, testdata_dir, num_fields, field_list, sbid_list, mocker
):
    survey = pd.Series(
        {
            "survey": surveyname,
            "image_path_i_T": f"tests/data/{testdata_dir}/STOKESI_IMAGES/",
            "image_path_v_T": f"tests/data/{testdata_dir}/STOKESV_IMAGES/",
        }
    )
    mocker.patch("astroutils.io.get_survey", return_value=survey)

    fields = build_field_csv(surveyname)

    assert len(fields) == num_fields
    assert sorted(list(fields.field)) == sorted(field_list)
    assert sorted(list(fields.sbid)) == sorted(sbid_list)


def test_field_with_no_fieldname_or_sbid_raises_error(mocker):
    survey = pd.Series(
        {
            "survey": "gw1",
            "image_path_i_T": "tests/data/no_fieldname_or_sbid/STOKESI_IMAGES/",
            "image_path_v_T": "tests/data/no_fieldname_or_sbid/STOKESV_IMAGES/",
        }
    )
    mocker.patch("astroutils.io.get_survey", return_value=survey)

    with pytest.raises(FITSException):
        build_field_csv("gw1", tiletype="TILES")


image_dot_stokes_path_i = Path(
    "tests/data/multi_field_with_fieldname_no_sbid/STOKESI_IMAGES/"
)
image_dot_stokes_path_v = Path(
    "tests/data/multi_field_with_fieldname_no_sbid/STOKESV_IMAGES/"
)
stokes_dot_fits_path_i = Path(
    "tests/data/single_field_with_sbid_no_fieldname/STOKESI_IMAGES/"
)
stokes_dot_fits_path_v = Path(
    "tests/data/single_field_with_sbid_no_fieldname/STOKESV_IMAGES/"
)


@pytest.mark.parametrize("i_stokes", [("I"), ("i")])
@pytest.mark.parametrize("v_stokes", [("V"), ("v")])
@pytest.mark.parametrize(
    "stokesi_path, stokesv_path, num_paths_i, num_paths_v",
    [
        (image_dot_stokes_path_i, image_dot_stokes_path_v, 2, 2),
        (stokes_dot_fits_path_i, stokes_dot_fits_path_v, 1, 1),
    ],
)
def test_image_filename_parsing(
    i_stokes, v_stokes, stokesi_path, stokesv_path, num_paths_i, num_paths_v
):
    """This pattern should follow a pattern like image.i.*.fits for Stokes I."""

    i_paths = parse_image_filenames_in_dir(stokesi_path, stokes=i_stokes)
    v_paths = parse_image_filenames_in_dir(stokesv_path, stokes=v_stokes)

    assert len(i_paths) == num_paths_i
    assert len(v_paths) == num_paths_v


vlass = pd.Series(
    {
        "data_path": "tests/data/vlass/VLASS1.1/",
    }
)


def test_parsing_of_vlass_subdirectories(mocker):
    mocker.patch("astroutils.io.get_survey", return_value=vlass)
    fields = build_vlass_field_csv("vlass1")

    assert len(fields) == 1
