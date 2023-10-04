import os

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


# A simple callback to print the progress of training
class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print(f"Epoch: {self.epoch + 1} completed.")
        self.epoch += 1


def load_processed_data(filepath):
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
