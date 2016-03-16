from sklearn.base import BaseEstimator
from math import log
from text_metrics import get_totallen, get_text_density, get_taglen, get_textlen


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
        for raw in raw_documents:
            document = raw["data"]

            title = document["title"]
            lang = document["language"]
            article = document["article"]
            descr = document["description"]
            type = document["type"]

            if descr is None or descr is "":
                descr = "1"
            if title is None or title is "":
                title = "1"
            if article is None or article == "":
                article = "1"

            if lang == "en":
                lang = 1
            else:
                lang = 0

            if type == "article":
                type = 1
            else:
                type = 0

            total_len = log(get_totallen(article))
            textlen = get_textlen(article)
            titlelen = len(title)
            taglen = get_taglen(article)
            density = get_text_density(article)
            descrlen = len(descr)

            doc_vec = {
                "geom_features": [total_len, taglen, textlen, titlelen, descrlen, lang, density, type],
                "title_text": title
            }
            doc_vecs.append(doc_vec)
        return doc_vecs

    def transform(self, raw_documents):
        return self.fit_transform(raw_documents)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
