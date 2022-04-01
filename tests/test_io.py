import pytest
import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from pathlib import Path

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
    assert list(config['DATA'].keys()) == ['aux_path', 'mwats_path', 'vlass_path']


@pytest.mark.parametrize(
    "system, valid_path_roots",
    [
        ("ada.physics.usyd.edu.au", ["/import/ada1", "/import/ada2"]),
        ("vast-data", ["/data/pilot", "/home/joshoewahp/"]),
    ]
)
def test_surveys_on_valid_systems(system, valid_path_roots, mocker):

    mocker.patch('astroutils.io.os.uname', return_value=['', system])

    surveys = get_surveys()
    surveys = surveys[surveys.local].dropna(subset=['selavy_path_i'])

    assert 'ada_root' not in surveys.columns
    assert 'nimbus_root' not in surveys.columns

    assert all([
        any(p in c.selavy_path_i for p in valid_path_roots)
        for _, c in surveys.iterrows()
    ])


def test_invalid_system_raises_error(mocker):
    mocker.patch('astroutils.io.os.uname', return_value=['', 'localhost'])

    with pytest.raises(NotImplementedError):
        get_surveys()


surveys = pd.read_json(SURVEYS_PATH).survey
@pytest.mark.parametrize("survey", list(surveys))
def test_valid_survey_names(survey, mocker):
    mocker.patch('astroutils.io.os.uname', return_value=['', 'ada.physics.usyd.edu.au']) 

    survey = get_survey(survey)

    # Should be a single survey (Series)
    assert isinstance(survey, pd.Series)

    # Check each survey has 24 parameters
    assert len(survey) == 24


def test_invalid_survey_name_raises_not_implemented_error(mocker):
    mocker.patch('astroutils.io.os.uname', return_value=['', 'ada.physics.usyd.edu.au']) 

    with pytest.raises(NotImplementedError):
        get_survey('missing-survey')


@pytest.mark.parametrize(
    "image_path",
    [
        Path('tests/data/test_image_mJy.fits'),
        Path('tests/data/test_image_Jy.fits')
    ]
)
def test_data_units_when_input_in_mJy(image_path):
    """Test that images in either mJy or Jy units get converted to mJy."""

    data, header = get_image_data_header(image_path, load_data=True)

    # In units of mJy this image should have no values further than 1e-6 from 0
    assert (np.abs(data) > 1e-6).all()
    assert header['BUNIT'] == 'mJy/beam'


def test_load_is_false():
    """Test that using load=False only loads the header into memory."""

    image_path = Path('tests/data/test_image_mJy.fits')

    data, header = get_image_data_header(image_path, load_data=False)

    # Unloaded data should be assigned to None and occupy 16 bytes
    assert data is None
    assert sys.getsizeof(data) == 16

    # Check header still loads correctly
    assert isinstance(header, fits.header.Header)


def test_image_path_resolves():

    epoch = pd.Series({
        'image_path_i': 'tests/data/',
        'field': 'mJy'
    })
    _, header = get_image_from_survey_params(epoch, field='mJy', stokes='i', load=True)

    # Check the correct image is loaded by looking at centre coordinates and MJD
    assert header['CRVAL1'] == 189.3062529167
    assert header['CRVAL2'] == 0.003423722222222
    assert header['MJD-OBS'] == 58865.844162986
    assert isinstance(header, fits.header.Header)


def test_askap_fields_load(mocker):
    expected = pd.DataFrame({
        'field': ['TEST00+00'],
        'cr_ra_pix': [0],
        'cr_dec_pix': [0],
    })
    mocker.patch('astroutils.io.pd.read_csv', return_value = expected.copy())

    position = SkyCoord(ra=4, dec=0, unit='deg')

    fields = find_fields(position, 'vastp1')

    expected['dist_field_centre'] = 4.

    assert (fields == expected).all().all()


def test_small_primary_beam_size_survey(mocker):
    expected = pd.DataFrame({
        'field': ['TEST00+00'],
        'cr_ra_pix': [0],
        'cr_dec_pix': [0],
    })
    mocker.patch('astroutils.io.pd.read_csv', return_value = expected.copy())

    position = SkyCoord(ra=4, dec=0, unit='deg')

    # VLASS has a primary beam of ~1 degree, and so should return empty here
    fields = find_fields(position, 'vlass')

    assert fields.empty


def test_invalid_survey_name_raises_error():

    with pytest.raises(FITSException):
        position = SkyCoord(ra=4, dec=0, unit='deg')
        find_fields(position, 'vastp0')

        
@pytest.mark.parametrize(
    "surveyname, testdata_dir, num_fields, field_list, sbid_list",
    [
        ('vastp3x', 'multi_field_with_fieldname_no_sbid', 2, ['0012+00', '0012-00'], ['SBXXX', 'SBXXX']),
        ('gw1', 'single_field_with_sbid_no_fieldname', 1, ['SB9602'], ['SB9602']),
    ]
)
def test_multi_field_survey_with_fieldname_no_sbid(surveyname, testdata_dir, num_fields, field_list, sbid_list, mocker):
    survey = pd.Series({
        'survey': surveyname,
        'image_path_i': f'tests/data/{testdata_dir}/STOKESI_IMAGES/',
        'image_path_v': f'tests/data/{testdata_dir}/STOKESV_IMAGES/',
    })
    mocker.patch('astroutils.io.get_survey', return_value=survey)

    fields = build_field_csv(surveyname)

    assert len(fields) == num_fields
    assert sorted(list(fields.field)) == sorted(field_list)
    assert sorted(list(fields.sbid)) == sorted(sbid_list)


def test_field_with_no_fieldname_or_sbid_raises_error(mocker):
    survey = pd.Series({
        'survey': 'gw1',
        'image_path_i': 'tests/data/no_fieldname_or_sbid/STOKESI_IMAGES/',
        'image_path_v': 'tests/data/no_fieldname_or_sbid/STOKESV_IMAGES/',
    })
    mocker.patch('astroutils.io.get_survey', return_value=survey)

    with pytest.raises(FITSException):
        build_field_csv('gw1')


image_dot_stokes_path_i = Path('tests/data/multi_field_with_fieldname_no_sbid/STOKESI_IMAGES/')
image_dot_stokes_path_v = Path('tests/data/multi_field_with_fieldname_no_sbid/STOKESV_IMAGES/')
stokes_dot_fits_path_i = Path('tests/data/single_field_with_sbid_no_fieldname/STOKESI_IMAGES/')
stokes_dot_fits_path_v = Path('tests/data/single_field_with_sbid_no_fieldname/STOKESV_IMAGES/')

@pytest.mark.parametrize("i_stokes", [("I"), ("i")])
@pytest.mark.parametrize("v_stokes", [("V"), ("v")])
@pytest.mark.parametrize(
    "stokesi_path, stokesv_path, num_paths_i, num_paths_v",
    [
        (image_dot_stokes_path_i, image_dot_stokes_path_v, 2, 2),
        (stokes_dot_fits_path_i, stokes_dot_fits_path_v, 1, 1),
    ]
)
def test_image_filename_parsing(i_stokes, v_stokes, stokesi_path, stokesv_path, num_paths_i, num_paths_v):
    """This pattern should follow a pattern like image.i.*.fits for Stokes I."""

    i_paths = parse_image_filenames_in_dir(stokesi_path, stokes=i_stokes)
    v_paths = parse_image_filenames_in_dir(stokesv_path, stokes=v_stokes)

    assert len(i_paths) == num_paths_i
    assert len(v_paths) == num_paths_v

def test_parsing_of_vlass1_subdirectories():

    path = Path('tests/data/vlass/VLASS1.1/')
    fields = build_vlass_field_csv(path)

    assert len(fields) == 1
