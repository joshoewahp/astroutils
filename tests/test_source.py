import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from astroutils.source import (
    SelavyCatalogue,
    condon_flux_error,
    fractional_pol_error,
    measure_flux,
    measure_limit,
)


mock_survey = pd.Series({
    'selavy_path_i': 'tests/data',
    'selavy_path_v': 'tests/data',
})

@patch('astroutils.source.get_survey', return_value=mock_survey)
class SelavyCatalogueTest(TestCase):

    required_cols = [
        'ra_deg_cont',
        'ra_err',
        'dec_deg_cont',
        'dec_err',
        'maj_axis',
        'min_axis',
        'pos_ang',
        'flux_int',
        'flux_int_err',
        'flux_peak',
        'flux_peak_err',
        'rms_image',
        'sign'
    ]

    def setUp(self):
        self.racspath_ra0 = 'tests/data/RACS_0000-12A.EPOCH00.I.selavy.components.txt'
        self.vastpath = 'tests/data/VAST_0012+00A.EPOCH12.I.selavy.components.xml'
        self.mwatspath = 'tests/data/mwats_test.parq'
        self.position_ra0 = SkyCoord(ra=0, dec=-11, unit=u.deg)
        self.position_near_pole = SkyCoord(ra=0, dec=-85, unit=u.deg)

    def validate_selavy_catalogue(self, cat, n_expected_rows):
        """Check that all necessary columns are named correctly in components DataFrame."""

        # Check basic object construction
        self.assertIsInstance(cat.selavypath, list)
        self.assertIsInstance(cat.selavypath[0], Path)
        self.assertIsInstance(cat.components, pd.DataFrame)
        self.assertEqual(len(cat.components), n_expected_rows)

        # Check that required columns are correctly named
        for col in self.required_cols:
            self.assertIn(col, cat.components.columns)

        # Check that maj_axis / min_axis are in units of arcsec (should be > 1 as simple test)
        self.assertGreater(cat.components.iloc[0].maj_axis, 1)
        self.assertGreater(cat.components.iloc[0].min_axis, 1)

    def test_selavy_single_txt(self, *args):
        cat = SelavyCatalogue(self.racspath_ra0)
        
        self.validate_selavy_catalogue(cat, 15)

    def test_selavy_single_xml_path(self, *args):
        cat = SelavyCatalogue(self.vastpath)

        self.validate_selavy_catalogue(cat, 23)

    def test_selavy_single_xml_pathlib_path(self, *args):
        path = Path(self.vastpath)
        cat = SelavyCatalogue(path)

        self.validate_selavy_catalogue(cat, 23)

    def test_selavy_list_of_mixed_type_paths(self, *args):
        cat = SelavyCatalogue([self.racspath_ra0, self.vastpath])

        self.validate_selavy_catalogue(cat, 38)

    def test_from_params_txt(self, *args):
        cat = SelavyCatalogue.from_params('racs-low', field='0000-12', stokes='i')

        self.validate_selavy_catalogue(cat, 15)

    def test_from_params_xml(self, *args):
        cat = SelavyCatalogue.from_params('vastp1', field='0012+00', stokes='i')

        self.validate_selavy_catalogue(cat, 23)

    def test_from_params_raises_error_if_no_path(self, *args):

        self.assertRaises(
            FileNotFoundError,
            SelavyCatalogue.from_params, 
            epoch='vastp1',
            field='0012+10',
            stokes='i'
        )
        
    def test_from_aegean(self, *args):
        cat = SelavyCatalogue.from_aegean(self.mwatspath)

        self.validate_selavy_catalogue(cat, 10)

    def test_cone_search(self, *args):

        cat = SelavyCatalogue(self.vastpath)
        position = SkyCoord(ra=3.292805, dec=0.856316, unit='deg')
        components = cat.cone_search(position, radius=0.5*u.arcmin)

        self.assertEqual(len(components), 2)
        self.assertEqual(components.d2d.min(), 0)
        self.assertEqual(components.d2d.max().round(3), 27.959)

    def test_nearest_component_when_in_radius(self, *args):

        cat = SelavyCatalogue(self.racspath_ra0)
        component = cat.nearest_component(self.position_ra0, radius=1*u.deg)
        self.assertTrue(len(component), 1)

    def test_nearest_component_when_not_in_radius(self, *args):

        cat = SelavyCatalogue(self.racspath_ra0)
        component = cat.nearest_component(self.position_ra0, radius=15*u.arcsec)
        self.assertEqual(component, None)


class CondonFluxErrorTest(TestCase):

    def setUp(self):
        self.component = pd.Series({
            'flux_peak': 1.41,
            'flux_int': 1.38,
            'rms_image': 0.23,
            'maj_axis': 16.3,
            'min_axis': 12.1,
        })
        self.bmin = 12*u.arcsec
        self.bmaj = 15*u.arcsec

    def test_integrated_flux_calculation(self):

        error = condon_flux_error(self.component, bmaj=self.bmaj, bmin=self.bmin, flux_scale_error=0, fluxtype='int')

        self.assertEqual(error.round(3), 0.235)

    def test_peak_flux_calculation(self):

        error = condon_flux_error(self.component, bmaj=self.bmaj, bmin=self.bmin, flux_scale_error=0, fluxtype='peak')
        self.assertEqual(error.round(3), 0.246)

    def test_bmaj_bmin_in_degrees(self):
        bmaj = 15/3600 * u.deg
        bmin = 12/3600 * u.deg
        error = condon_flux_error(self.component, bmaj=bmaj, bmin=bmin, flux_scale_error=0, fluxtype='peak')

        self.assertEqual(error.round(3), 0.246)

    def test_bmaj_not_quantity_raises_typeerror(self):

        self.assertRaises(
            TypeError,
            condon_flux_error,
            self.component,
            bmaj=15,
            bmin=12*u.arcsec,
            flux_scale_error=0,
            fluxtype='peak',
        )


class FractionalPolErrorTest(TestCase):

    def setUp(self):

        self.fluxes = pd.DataFrame({
            'name': ['test'] *3,
            'ra': [0.0, 0.015, 0.01],
            'dec': [0.0, 0.015, 0.01],
            'epoch': ['racs-low', 'vastp1', 'vastp2'],
            'field': ['0012+00'] * 3,
            'flux_i': [1.2, 1.1, 2.3],
            'flux_err_i': [0.22, 0.24, 0.21],
            'flux_v': [1.1, 0.64, 0.55],
            'flux_err_v': [0.22, 0.24, 0.21],
        })
        self.fluxes['fp'] = self.fluxes.flux_v.abs() / self.fluxes.flux_i

    def test_calculation(self):
        fp_errors = fractional_pol_error(self.fluxes, corr_errors=False)

        expected_errors = [0.248704, 0.252423, 0.093879]
        for error, expected_error in zip(fp_errors, expected_errors):
            self.assertEqual(round(error, 6), expected_error)

    def test_calculation_with_corr(self):
        fp_errors = fractional_pol_error(self.fluxes, corr_errors=True)

        expected_errors = [0.311009, 0.308328, 0.105212]
        for error, expected_error in zip(fp_errors, expected_errors):
            self.assertEqual(round(error, 6), expected_error)

            
mocked_fields = pd.DataFrame({
    'field': ['1237+00', '1237-06'],
    'sbid': ['SBXXX', 'SBXXX'],
    'i_path': ['tests/data/test_image_mJy.fits',
               'tests/data/test_image_mJy.fits'],
    'v_path': ['tests/data/test_image_mJy.fits',
               'tests/data/test_image_mJy.fits'],
    'cr_ra_pix': [189.30660013892276, 189.3064293311123],
    'cr_dec_pix': [0.0030764999999284, -6.298899722106099],
    'bmaj': [12.966916344916608, 12.813313530441672],
    'bmin': [12.688634591018316, 11.58917164487208],
    'dist_field_centre': [2.0110998437527563, 4.322889739848082],
})

@patch('astroutils.source.Path.glob', return_value=['tests/data/test_image_mJy.fits'])
class MeasureFluxTest(TestCase):

    def setUp(self):
        self.position = SkyCoord(ra=189.0104, dec=-1.9861, unit='deg')
        self.epochs = pd.DataFrame({
            "survey": ["vastp2", "vastp8"],
            "name": ["VAST P1-2", "VAST P1-8"],
            "image_path_i": ["COMBINED/STOKESI_IMAGES/", "COMBINED/STOKESI_IMAGES/"],
            "image_path_v": ["COMBINED/STOKESV_IMAGES/", "COMBINED/STOKESV_IMAGES/"],
        })

    @patch('astroutils.source.find_fields', return_value=mocked_fields)
    @patch('astroutils.source.SelavyCatalogue.from_params', return_value=SelavyCatalogue('tests/data/VAST_1237+00A.EPOCH08.I.selavy.components.xml'))
    def test_position_with_component_in_range(self, *args):

        fluxes = measure_flux(
            self.position,
            self.epochs,
            size=0.1*u.deg,
            fluxtype='int',
            stokes='i'
        )

        self.assertEqual(fluxes.flux.iloc[0].round(3), 16.261)
        
    @patch('astroutils.source.find_fields', return_value=mocked_fields)
    @patch('astroutils.source.SelavyCatalogue.from_params', return_value=SelavyCatalogue('tests/data/VAST_1237+00A.EPOCH02.I.selavy.components.xml'))
    def test_position_with_no_component_in_range(self, *args):

        fluxes = measure_flux(
            self.position,
            self.epochs,
            size=0.1*u.deg,
            fluxtype='int',
            stokes='i'
        )

        self.assertEqual(fluxes.flux.iloc[0].round(3), 1.405)

    @patch('astroutils.source.find_fields', return_value=pd.DataFrame())
    def test_coordinates_with_no_fields(self, *args):

        fluxes = measure_flux(
            self.position,
            self.epochs,
            size=0.1*u.deg,
            fluxtype='int',
            stokes='i'
        )

        self.assertTrue(fluxes.empty)


class MeasureLimitTest(TestCase):

    def setUp(self):
        self.position = SkyCoord(ra=189.0104, dec=-1.9861, unit='deg')
    
    def test_image_with_Jy_units(self):

        image_path_Jy = Path('tests/data/test_image_Jy.fits')
        limit = measure_limit(
            position=self.position,
            image_path=image_path_Jy,
            size=0.1*u.deg
        )

        self.assertEqual(limit.round(4), 1.1002)

    def test_image_with_mJy_units(self):

        image_path_mJy = Path('tests/data/test_image_mJy.fits')
        limit = measure_limit(
            position=self.position,
            image_path=image_path_mJy,
            size=0.1*u.deg
        )

        self.assertEqual(limit.round(4), 1.1002)

    def test_rms_image_uses_median(self):

        image_path_rms = Path('tests/data/test_image_rms.fits')
        limit = measure_limit(
            position=self.position,
            image_path=image_path_rms,
            size=0.1*u.deg
        )

        self.assertEqual(limit.round(4), 1.0408)
