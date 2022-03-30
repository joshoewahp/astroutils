import os
import pandas as pd
from click.testing import CliRunner

from astroutils.cli.build_fields import main

test_field_df = pd.DataFrame({'field': ['test']})

def test_script_works(mocker):

    mocker.patch('astroutils.cli.build_fields.build_field_csv', return_value=test_field_df)
    mocker.patch('astroutils.cli.build_fields.aux_path', 'tests/')

    os.system('mkdir -p tests/fields/')

    runner = CliRunner()
    _ = runner.invoke(main, 'vastp3x'.split(), input='1')

    assert os.path.exists('tests/fields/vastp3x_fields.csv')

    os.system('rm -r tests/fields')

