"""
Training module for the NeuralArticleFinder project.

This module provides functionalities for training a Word2Vec model on preprocessed data.
"""
import os

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


# A simple callback to print the progress of training
class Callback(CallbackAny2Vec):
    """Callback class to print training progress after each epoch."""
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, _model):
        """Prints training progress after the end of each epoch.
        
        Args:
            _model: Instance of the Word2Vec model being trained.
        """
        print(f"Epoch: {self.epoch + 1} completed.")
        self.epoch += 1


def load_processed_data(filepath):
    """Loads and returns preprocessed data from a given filepath.

    Args:
        filepath (str): Path to the preprocessed data file.

    Returns:
        list: List of documents, where each document is a list of tokens.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        documents = [line.strip().split() for line in file.readlines()]
    return documents


if __name__ == '__main__':
    data_path = os.path.join('data', 'processed', 'processed_dataset.txt')
    model_save_path = os.path.join('models', 'word2vec.model')

    # Loading data
    docs = load_processed_data(data_path)

    # Training the model
    model = Word2Vec(
        sentences=docs,
        vector_size=100,
        window=5,
        min_count=5,
        workers=8,
        epochs=10,
        callbacks=[Callback()]
    )
    model.save(model_save_path)

    print("Model successfully trained and saved!")
