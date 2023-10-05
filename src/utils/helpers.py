"""
Utility functions for the NeuralArticleFinder project.

This module provides helper functions used across the project.
"""
from sklearn.datasets import fetch_20newsgroups


def create_newsgroupt_dataset():
    """
    Fetch the 20newsgroups dataset and save it to a text file.

    This function fetches the 20newsgroups dataset using the scikit-learn library,
    removes headers, footers, and quotes from each entry, and saves the processed dataset
    to 'data/raw/articles.txt'.
    """
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    with open("data/raw/articles.txt", 'w', encoding='utf-8') as file:
        for item in newsgroups.data:
            file.write(f"{item}\n")


if __name__ == '__main__':
    create_newsgroupt_dataset()
