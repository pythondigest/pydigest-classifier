"""
Trains and dumps the classifier.
Accepts 2 args: first one is input path, second one is output path
Classifier is trained on all files with .json extension from input path
Classifier is saved to the file specified by output path
"""

import json
import os
import sys

import joblib
from sklearn.pipeline import Pipeline

from src.model.classifier import ChainedClassifier
from src.model.vectorizer import GeomFeatureExtractor

if __name__ == "__main__":
    input_path = sys.argv[1]  # folder
    out_path = sys.argv[2]  # filename
    input_data = []
    input_labels = []

    # read raw data
    for file in os.listdir(input_path):
        if file.endswith(".json"):
            raw_docs = json.loads(open(os.path.join(input_path, file), errors="ignore").read())
            for i in raw_docs["links"]:
                if i["data"]["article"]:
                    input_data.append(i)
                    input_labels.append(i["data"]["label"])

    # prepare model
    model_pipeline = Pipeline(
        [
            ("vect", GeomFeatureExtractor()),
            ("clf", ChainedClassifier()),
        ],
        verbose=True,
    )
    # train model
    model_pipeline.fit(input_data, input_labels)

    # save results
    joblib.dump(model_pipeline, out_path, compress=1)
