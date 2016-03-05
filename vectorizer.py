import json
import numpy as np
from text_density import text_density
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from batcher import batch
import config


def bag_of_words_vectorizer(datafile, k_features):
    """
    Computes sparse term-document matrix of datafile documents, selects k best features by chi2 test.
    Yields batches of BATCH_SIZE of dense tdm vectors.
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

    for vector_batch in batch(data, config.BATCH_SIZE):
        yield np.vstack([i.toarray() for i in vector_batch])

