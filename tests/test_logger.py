import os
import logging
from unittest.case import TestCase
from astroutils.logger import setupLogger

logger = logging.getLogger(__name__)

class SetupLoggerTest(TestCase):

    def test_verbose(self):
        
        setupLogger(verbose=True)

        with self.assertLogs(level='DEBUG') as captured:
            logger.debug("Test debug log message.")
            logger.info("Test info log message.")

        self.assertEqual(len(captured.records), 2)
        
    def test_filename(self):

        setupLogger(verbose=True, filename='tests/main.log')
        logger.debug("Test log message.")

        with open('tests/main.log') as f:
            message_parts = f.readlines()[0].split(' - ')

        self.assertEqual(message_parts[1], 'tests.test_logger')

        os.system('rm -r tests/main.log')
