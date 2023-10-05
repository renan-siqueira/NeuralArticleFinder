import unittest
import logging

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(filename='log/unittest.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LoggedTestSuite(unittest.TestSuite):

    def run(self, result, debug=False):
        for index, test in enumerate(self):
            if result.shouldStop:
                break

            if not debug:
                test(result)
            else:
                test.debug()

            if result.wasSuccessful():
                logging.info(f'Test case {index + 1} ({test}) passed.')
            else:
                logging.info(f'Test case {index + 1} ({test}) failed.')


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = LoggedTestSuite(loader.discover('.'))
    unittest.TextTestRunner().run(suite)
