import sys
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from vectorizer import GeomFeatureExtractor
from classifier import ChainedClassifier


"""
Trains and dumps the classifier.
Accepts 2 args: first one is input path, second one is output path
Classifier is trained on all files with .json extension from input path
Classifier is saved to the file specified by output path
"""


if __name__ == "__main__":
    input_path = sys.argv[1]
    out_path = sys.argv[2]
    input_data = []
    input_labels = []

    for file in os.listdir(input_path):
        if file.endswith(".json"):
            raw_docs = json.loads(open(os.path.join(input_path, file), errors="ignore").read())["links"]
            for i in raw_docs:
                input_data.append(i)
                input_labels.append(i["data"]["label"])

    text_clf = Pipeline([('vect', GeomFeatureExtractor()),
                        ('clf', ChainedClassifier())
                        ])

    text_clf.fit(input_data, input_labels)
    joblib.dump(text_clf, out_path, compress=1)
