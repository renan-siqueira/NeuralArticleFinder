import unittest
from src.preprocessing.preprocess import preprocess


class TestPreprocessFunction(unittest.TestCase):

    def test_preprocess(self):
        input_text = "This is a sample text with some stopwords and other relevant words."
        expected_output = ['sample', 'text', 'stopwords', 'relevant', 'word']

        output = preprocess(input_text)
        self.assertEqual(output, expected_output)


if __name__ == '__main__':
    unittest.main()
