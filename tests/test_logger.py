import os
import logging
from astroutils.logger import setupLogger

logger = logging.getLogger(__name__)

def test_filename():

    setupLogger(verbose=True, filename='tests/main.log')
    logger.debug("Test log message.")

    with open('tests/main.log') as f:
        message_parts = f.readlines()[0].split(' - ')

        assert message_parts[1] == 'tests.test_logger'

    os.system('rm -r tests/main.log')
