import os
import unittest

from src.training.train_model import load_processed_data, Word2Vec, Callback


class TestTrainingFunctions(unittest.TestCase):

    def test_load_processed_data(self):
        with open('temp_test_file.txt', 'w', encoding='utf-8') as file:
            file.write("hello world\n")
            file.write("testing load function\n")

        expected_output = [['hello', 'world'], ['testing', 'load', 'function']]
        output = load_processed_data('temp_test_file.txt')

        # Remove temp file
        os.remove('temp_test_file.txt')

        self.assertEqual(output, expected_output)

    def test_word2vec_model_saving(self):
        docs = [['hello', 'world'], ['testing', 'load', 'function']]

        model_save_path = 'temp_test_model.model'
        model = Word2Vec(
            sentences=docs,
            vector_size=10,
            window=2,
            min_count=1,
            workers=1,
            epochs=2,
            callbacks=[Callback()]
        )

        model.save(model_save_path)

        # Check if the model was saved correctly
        self.assertTrue(os.path.exists(model_save_path))

        # Remove the saved model after testing
        os.remove(model_save_path)


if __name__ == '__main__':
    unittest.main()
