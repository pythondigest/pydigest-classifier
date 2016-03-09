from sklearn.pipeline import Pipeline
import json
from sklearn import cross_validation
from vectorizer import GeomFeatureExtractor
from random import shuffle
from classifier import ChainedClassifier


if __name__ == '__main__':

    input_data = []
    input_labels = []

    raw_docs = json.loads(open("data_final.json").read())
    shuffle(raw_docs)
    for i in raw_docs:
        input_data.append(i)
        input_labels.append(i["label"])

    text_clf = Pipeline([('vect', GeomFeatureExtractor()),
                         ('clf', ChainedClassifier())
                         ])

    scores = cross_validation.cross_val_score(text_clf, input_data, input_labels, cv=6, scoring="precision", n_jobs=6)
    print(scores)