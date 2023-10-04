import os

from gensim.models import Word2Vec
import numpy as np


def vectorize_document(doc, model):
    vectors = [model.wv[word] for word in doc if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def most_similar_docs(target_doc, model, documents):
    target_vector = vectorize_document(target_doc, model)
    document_vectors = [vectorize_document(doc, model) for doc in documents]

    if np.all(target_vector == 0):
        print("The target document has no representation in the trained model.")
        return []

    norms = np.linalg.norm(document_vectors, axis=1) * np.linalg.norm(target_vector)
    valid_indices = np.where(norms != 0)[0]

    similarities = np.zeros(len(document_vectors))
    similarities[valid_indices] = np.dot(np.array(document_vectors)[valid_indices], target_vector) / norms[valid_indices]

    sorted_indexes = np.argsort(similarities)[::-1]
    return [(documents[i], similarities[i]) for i in sorted_indexes]


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
