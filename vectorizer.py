from sklearn.base import BaseEstimator
from math import log


class GeomFeatureExtractor(BaseEstimator):
    """
    Computes various geometric properties of articles, returns them in pair with title text.
    Title text is used later to compute probability of article being good by semantic classification.
    This probability is used as additional feature for gradient boosting.
    """
    def __init__(self):
        pass

    def fit(self, raw_documents):
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        doc_vecs = []
        for document in raw_documents:
            total_len = log(document["total_len"])
            taglen = document["tag_len"]
            textlen = document["textlen"]
            titlelen = len(str(document["title"]))
            descr = str(document["descr"])
            if descr == "" or descr is None: descr = "1"
            descrlen = len(descr)
            if document["lang"] == "en":
                lang = 1
            else:
                lang = 0
            density = textlen / total_len

            doc_vec = {
                "geom_features": [total_len, taglen, textlen, titlelen, descrlen, lang, density],
                "title_text": document["title"]
            }
            doc_vecs.append(doc_vec)
        return doc_vecs

    def transform(self, raw_documents):
        return self.fit_transform(raw_documents)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
