import os
import pandas as pd
from click.testing import CliRunner
from unittest import TestCase
from unittest.mock import patch

from astroutils.cli.build_fields import main

test_field_df = pd.DataFrame({'field': ['test']})

class BuildFieldsScriptTest(TestCase):

    def setUp(self):
        os.system('mkdir -p tests/fields/')

    def tearDown(self):
        os.system('rm -r tests/fields')

    @patch('astroutils.cli.build_fields.build_field_csv', return_value=test_field_df)
    @patch('astroutils.cli.build_fields.aux_path', 'tests/')
    def test_script_works(self, *args):
        runner = CliRunner()
        _ = runner.invoke(main, 'vastp3x'.split(), input='1')

        self.assertTrue(os.path.exists('tests/fields/vastp3x_fields.csv'))

