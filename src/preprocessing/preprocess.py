"""
Module for preprocessing textual data.

This module provides utilities to preprocess textual data,
including tokenization, stopword removal, and lemmatization.
"""
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def preprocess(corpus):
    """
    Preprocess a given text corpus.

    Tokenizes the corpus, converts to lowercase, removes punctuation and stopwords,
    and then lemmatizes the tokens.

    Args:
        corpus (str): The text to be preprocessed.

    Returns:
        list: A list of preprocessed tokens.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(corpus)

    # Convert to lower and remove punctuation
    tokens = [token.lower() for token in tokens if token.isalnum()]

    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


if __name__ == '__main__':
    input_path = os.path.join('data', 'raw', 'articles.txt')
    output_path = os.path.join('data', 'processed', 'processed_dataset.txt')

    with open(input_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()

    processed_texts = [" ".join(preprocess(text)) for text in texts]

    with open(output_path, 'w', encoding='utf-8') as file:
        for text in processed_texts:
            file.write(f'{text}\n')
