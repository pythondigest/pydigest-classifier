from sklearn.pipeline import Pipeline
import json
from sklearn.externals import joblib

from vectorizer import GeomFeatureExtractor
from classifier import ChainedClassifier

input_data = []
input_labels = []

raw_docs = json.loads(open("data_final.json").read())
for i in raw_docs:
    input_data.append(i)
    input_labels.append(i["label"])

text_clf = Pipeline([('vect', GeomFeatureExtractor()),
                     ('clf', ChainedClassifier())
                    ])

text_clf.fit(input_data, input_labels)
joblib.dump(text_clf, "classifier_dump.pkl", compress=1)
