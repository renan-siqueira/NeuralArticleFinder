"""
Recommendation module for the NeuralArticleFinder project.

This module provides functionalities for vectorizing documents and finding the most
similar documents to a given target document.
"""
import os

from gensim.models import Word2Vec
import numpy as np


def vectorize_document(document_tokens, w2v_model):
    """
    Vectorize a document using Word2Vec model.

    Args:
        document_tokens (list): List of tokens in a document.
        w2v_model: Trained Word2Vec model.

    Returns:
        numpy.ndarray: Vector representation of the document.
    """
    vectors = [w2v_model.wv[word] for word in document_tokens if word in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(w2v_model.vector_size)


def most_similar_docs(target_tokens, w2v_model, all_documents):
    """
    Find and rank the most similar documents to the target document.

    Args:
        target_tokens (list): List of tokens in the target document.
        w2v_model: Trained Word2Vec model.
        all_documents (list): List of all documents to search from.

    Returns:
        list: List of tuples containing the document and its similarity score.
    """
    target_vector = vectorize_document(target_tokens, w2v_model)
    document_vectors = [vectorize_document(doc, w2v_model) for doc in all_documents]

    if np.all(target_vector == 0):
        print("The target document has no representation in the trained model.")
        return []

    norms = np.linalg.norm(document_vectors, axis=1) * np.linalg.norm(target_vector)
    valid_indices = np.where(norms != 0)[0]

    similarities = np.zeros(len(document_vectors))

    similarities[valid_indices] = np.dot(
        np.array(document_vectors)[valid_indices],
        target_vector
    ) / norms[valid_indices]

    sorted_indexes = np.argsort(similarities)[::-1]
    return [(all_documents[i], similarities[i]) for i in sorted_indexes]


if __name__ == '__main__':
    model_path = os.path.join('models', 'word2vec.model')
    data_path = os.path.join('data', 'processed', 'processed_dataset.txt')

    model = Word2Vec.load(model_path)

    with open(data_path, 'r', encoding='utf-8') as file:
        documents = [line.strip().split() for line in file.readlines()]

    target_doc = input('Enter the content of the article of interest or provide keywords: ').split()

    top_matches = most_similar_docs(target_doc, model, documents)[:15]

    if not top_matches:
        print("No matching articles found.")
    else:
        print('\nRecommended Articles:\n')
        for idx, (doc, sim) in enumerate(top_matches, 1):
            print(f"{idx}. Similarity: {sim:.4f}")
            print(' '.join(doc[:100]) + "...\n")
