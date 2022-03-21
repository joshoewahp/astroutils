import sys
from astropy.coordinates.sky_coordinate import SkyCoord
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

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


mock_survey = pd.Series({
    'selavy_path_i': 'tests/data',
    'selavy_path_v': 'tests/data',
})


class GetConfigTest(TestCase):

    def setUp(self):
        self.config = get_config()

    def test_config_contains_paths(self):

        self.assertEqual(
            list(self.config['DATA'].keys()),
            ['aux_path', 'mwats_path', 'vlass_path']
        )

class GetSurveysTest(TestCase):

    def check_extra_columns_dropped(self, surveys):
        self.assertFalse('ada_root' in surveys.columns)
        self.assertFalse('nimbus_root' in surveys.columns)
    
    @patch('astroutils.io.os.uname', return_value=['', 'ada.physics.usyd.edu.au'])
    def test_surveys_on_ada(self, *args):
        surveys = get_surveys()
        ada_surveys = surveys[surveys.local].dropna(subset=['selavy_path_i'])
        
        # Survey paths are all located at /import/ada1 or /import/ada2
        # This may change in future
        self.assertTrue(
            all(['/import/ada1' in c.selavy_path_i or
                 '/import/ada2' in c.selavy_path_i
                 for _, c in ada_surveys.iterrows()])
        )

        self.check_extra_columns_dropped(surveys)

    @patch('astroutils.io.os.uname', return_value=['', 'vast-data'])
    def test_surveys_on_nimbus(self, *args):
        surveys = get_surveys()
        nimbus_surveys = surveys[surveys.local].dropna(subset=['selavy_path_i'])
        
        # Survey paths are all located at /data/pilot or /home/joshoewahp
        # This may change in future
        self.assertTrue(
            all(['/data/pilot/' in c.selavy_path_i or
                 '/home/joshoewahp/' in c.selavy_path_i
                 for _, c in nimbus_surveys.iterrows()])
        )

        self.check_extra_columns_dropped(surveys)

    @patch('astroutils.io.os.uname', return_value=['', 'localhost'])
    def test_invalid_system_raises_error(self, *args):

        self.assertRaises(NotImplementedError, get_surveys)


@patch('astroutils.io.os.uname', return_value=['', 'ada.physics.usyd.edu.au'])
class GetSurveyTest(TestCase):

    def test_valid_survey_names(self, *args):

        surveys = pd.read_json(SURVEYS_PATH).survey

        for survey in surveys:
            survey = get_survey(survey)

            # Should be a single survey (Series)
            self.assertIsInstance(survey, pd.Series)

            # Check each survey has 24 parameters
            self.assertEqual(len(survey), 24)

    def test_invalid_survey_name_raises_not_implemented_error(self, *args):

        self.assertRaises(NotImplementedError, get_survey, 'missing-survey')


class GetImageHeaderDataTest(TestCase):

    def test_data_units_when_input_in_mJy(self):
        """Test that images in either mJy or Jy units get converted to mJy."""

        image_paths = [
            Path('tests/data/test_image_mJy.fits'),
            Path('tests/data/test_image_Jy.fits')
        ]

        for image_path in image_paths:
            data, header = get_image_data_header(image_path, load_data=True)

            # In units of mJy this image should have no values further than 1e-6 from 0
            self.assertTrue((np.abs(data) > 1e-6).all())
            self.assertTrue(header['BUNIT'] == 'mJy/beam')

    def test_load_is_false(self):
        """Test that using load=False only loads the header into memory."""


        image_path = Path('tests/data/test_image_mJy.fits')

        data, header = get_image_data_header(image_path, load_data=False)

        # Unloaded data should be assigned to None and occupy 16 bytes
        self.assertIsNone(data)
        self.assertEqual(sys.getsizeof(data), 16)

        # Check header still loads correctly
        self.assertIsInstance(header, fits.header.Header)


class GetImageTest(TestCase):

    def test_image_path_resolves(self):

        epoch = pd.Series({
            'image_path_i': 'tests/data/',
            'field': 'mJy'
        })
        _, header = get_image_from_survey_params(epoch, field='mJy', stokes='i', load=True)
 
        # Check the correct image is loaded by looking at centre coordinates and MJD
        self.assertEqual(header['CRVAL1'], 189.3062529167)
        self.assertEqual(header['CRVAL2'], 0.003423722222222)
        self.assertEqual(header['MJD-OBS'], 58865.844162986)
        self.assertIsInstance(header, fits.header.Header)


class FindFieldsTest(TestCase):

    def setUp(self):
        self.position = SkyCoord(ra=4, dec=0, unit='deg')

    @patch('astroutils.io.pd.read_csv')
    def test_askap_fields_load(self, mocked_read_csv):

        mocked_read_csv.return_value = pd.DataFrame({
            'field': ['TEST00+00'],
            'cr_ra_pix': [0],
            'cr_dec_pix': [0],
        })

        fields = find_fields(self.position, 'vastp1')

        expected = mocked_read_csv.return_value.copy()
        expected['dist_field_centre'] = 4.

        self.assertTrue((fields == expected).all().all())


    @patch('astroutils.io.pd.read_csv')
    def test_small_primary_beam_size_survey(self, mocked_read_csv):
        
        mocked_read_csv.return_value = pd.DataFrame({
            'field': ['TEST00+00'],
            'cr_ra_pix': [0],
            'cr_dec_pix': [0],
        })

        # VLASS has a primary beam of ~1 degree, and so should return empty here
        fields = find_fields(self.position, 'vlass')

        self.assertTrue(fields.empty)

    def test_invalid_survey_name_raises_error(self):

        self.assertRaises(FITSException, find_fields, self.position, 'vastp0')


class BuildFieldCsvTest(TestCase):

    @patch('astroutils.io.get_survey')
    def test_multi_field_survey_with_fieldname_no_sbid(self, mocked_get_survey):

        mocked_get_survey.return_value = pd.Series({
            'survey': 'vastp3x',
            'image_path_i': 'tests/data/multi_field_with_fieldname_no_sbid/STOKESI_IMAGES/',
            'image_path_v': 'tests/data/multi_field_with_fieldname_no_sbid/STOKESV_IMAGES/',
        })
        
        fields = build_field_csv('vastp3x')

        self.assertEqual(len(fields), 2)
        self.assertEqual(list(fields.field), ['0012+00', '0012-00'])
        self.assertEqual(list(fields.sbid), ['SBXXX', 'SBXXX'])

    @patch('astroutils.io.get_survey')
    def test_single_field_survey_with_sbid_no_fieldname(self, mocked_get_survey):

        mocked_get_survey.return_value = pd.Series({
            'survey': 'gw1',
            'image_path_i': 'tests/data/single_field_with_sbid_no_fieldname/STOKESI_IMAGES/',
            'image_path_v': 'tests/data/single_field_with_sbid_no_fieldname/STOKESV_IMAGES/',
        })
        
        fields = build_field_csv('gw1')

        self.assertEqual(len(fields), 1)
        self.assertEqual(list(fields.field), ['SB9602'])
        self.assertEqual(list(fields.sbid), ['SB9602'])

    @patch('astroutils.io.get_survey')
    def test_field_with_no_fieldname_or_sbid_raises_error(self, mocked_get_survey):

        mocked_get_survey.return_value = pd.Series({
            'survey': 'gw1',
            'image_path_i': 'tests/data/no_fieldname_or_sbid/STOKESI_IMAGES/',
            'image_path_v': 'tests/data/no_fieldname_or_sbid/STOKESV_IMAGES/',
        })
        
        self.assertRaises(FITSException, build_field_csv, 'gw1')


class ParseImageFilenamesInDirTest(TestCase):

    def setUp(self):
        self.image_dot_stokes_path_i = Path('tests/data/multi_field_with_fieldname_no_sbid/STOKESI_IMAGES/')
        self.image_dot_stokes_path_v = Path('tests/data/multi_field_with_fieldname_no_sbid/STOKESV_IMAGES/')
        self.stokes_dot_fits_path_i = Path('tests/data/single_field_with_sbid_no_fieldname/STOKESI_IMAGES/')
        self.stokes_dot_fits_path_v = Path('tests/data/single_field_with_sbid_no_fieldname/STOKESV_IMAGES/')

    def test_image_dot_stokes_pattern(self):
        """This pattern should follow a pattern like image.i.*.fits for Stokes I."""

        i_paths = parse_image_filenames_in_dir(self.image_dot_stokes_path_i, stokes='i')
        v_paths = parse_image_filenames_in_dir(self.image_dot_stokes_path_v, stokes='v')

        self.assertEqual(len(i_paths), 2)
        self.assertEqual(len(v_paths), 2)

    def test_stokes_dot_fitspattern(self):
        """This pattern should follow a pattern like *I.fits for Stokes I."""

        i_paths = parse_image_filenames_in_dir(self.stokes_dot_fits_path_i, stokes='I')
        v_paths = parse_image_filenames_in_dir(self.stokes_dot_fits_path_v, stokes='V')

        self.assertEqual(len(i_paths), 1)
        self.assertEqual(len(v_paths), 1)

    def test_stokes_capitalisation_irrelevant(self):
        """Capitalisation of stokes parameter should be handled within the function."""

        i_paths1 = parse_image_filenames_in_dir(self.image_dot_stokes_path_i, stokes='I')
        i_paths2 = parse_image_filenames_in_dir(self.stokes_dot_fits_path_i, stokes='i')
        v_paths1 = parse_image_filenames_in_dir(self.image_dot_stokes_path_v, stokes='V')
        v_paths2 = parse_image_filenames_in_dir(self.stokes_dot_fits_path_v, stokes='v')

        self.assertEqual(len(i_paths1), 2)
        self.assertEqual(len(v_paths1), 2)
        self.assertEqual(len(i_paths2), 1)
        self.assertEqual(len(v_paths2), 1)


class BuildVlassFieldsTest(TestCase):

    def test_parsing_of_vlass1_subdirectories(self):

        path = Path('tests/data/vlass/VLASS1.1/')
        fields = build_vlass_field_csv(path)

        self.assertEqual(len(fields), 1)
