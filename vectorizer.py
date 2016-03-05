import json
import numpy as np
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from batcher import batch
from keras.utils import np_utils
import config


def bag_of_words_vectorizer(datafile, k_features):
    """
    Computes sparse term-document matrix of datafile documents, selects k best features by chi2 test.
    Yields batches of BATCH_SIZE of dense tdm vectors and vector of labels, transformed for keras nn.
    """
    data = []
    labels = []

    for jsoned_entity in open("data.json", errors="ignore").readlines():
        entity = json.loads(jsoned_entity)
        if entity["lang"] == "en":
            data.append(entity["text"])
            labels.append(entity["label"])

    vectorizer = TfidfVectorizer(stop_words=get_stop_words("english"))
    data = vectorizer.fit_transform(data)
    data = SelectKBest(chi2, k=k_features).fit_transform(data, labels)

    for vector_label_batch in batch(zip(data, labels), config.BATCH_SIZE):
        vectors = []
        labels = []
        for vec_label in vector_label_batch:
            vectors.append(vec_label[0].toarray())
            labels.append(vec_label[1])

        X = np.vstack(vectors)
        Y = np_utils.to_categorical(labels, 2)
        yield X, Y



