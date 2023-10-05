"""
Module to run all unit tests for NeuralArticleFinder project.

This module provides functionalities to run all tests and log their results.
"""
import unittest
import logging

logging.basicConfig(
    filename='unittest.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class LoggedTestSuite(unittest.TestSuite):
    """Test suite class that logs the result of each test case."""

    def run(self, result, debug=False):
        for index, test in enumerate(self):
            if result.shouldStop:
                break

            if not debug:
                test(result)
            else:
                test.debug()

            if result.wasSuccessful():
                logging.info('Test case %d (%s) passed.', index + 1, test)
            else:
                logging.info('Test case %d (%s) failed.', index + 1, test)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = LoggedTestSuite(loader.discover('.'))
    unittest.TextTestRunner().run(suite)
