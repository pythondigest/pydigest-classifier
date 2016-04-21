from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.base import TransformerMixin
from langid.langid import LanguageIdentifier, model
import nltk
import string

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier.set_languages(["en", "ru"])
english_stemmer = nltk.stem.LancasterStemmer()
russian_stemmer = nltk.stem.snowball.RussianStemmer()


def stem_tokens(tokens):
    """
    Returns a list of stems from a list of tokens, works for russian and english words.
    """
    stems = []
    for token in tokens:
        lang = identifier.classify(token)[0]

        if lang == "ru":
            stem = russian_stemmer.stem(token)
        else:
            stem = english_stemmer.stem(token)
        stems.append(stem)

    return stems


def tokenize(text):
    """
    Returns a bag of words from a given text string.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens)
    return stems


class ChainedClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom classifier, chains semantic analysis with geometric.
    SVM trained on tfidf of titles predicts probability of article being good,
        then this probability is appended to geometric features.
    Finally, GradientBoosting makes a prediction, based on geometric features and probability output from bag-of-words model.
    ATTENTION: experimental "buzzword_score" feature is implemented in this version, performance is degraded
    """
    def __init__(self, learning_rate=0.01, max_depth=6, min_samples_leaf=20, max_features=None):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.gradboost = GradientBoostingClassifier(n_estimators=3000, learning_rate=self.learning_rate, max_depth=self.max_depth,
                                                    min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)

        self.title_semantic = Pipeline([('vect', TfidfVectorizer(tokenizer=tokenize, stop_words=(get_stop_words("english") + get_stop_words("russian")))),
                                        ('clf', SVC(probability=True))
                                        ])

        self.buzzwords = []

    def fit_buzzword_list(self, X, y):
        """
        Creates a list of most valuable features in titles.
        This list is ised to compute buzzword_score
        """
        vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=(get_stop_words("english") + get_stop_words("russian")))
        selector = SelectKBest(chi2, k=5000)
        title_texts = [i["title_text"] for i in X]
        tdm = vectorizer.fit_transform(title_texts)
        selector.fit_transform(tdm, y)

        for word in np.array(vectorizer.get_feature_names())[selector.get_support()]:
            for title, label in zip(title_texts, y):
                if label is True and word in title:
                    self.buzzwords.append(word)
                    break

    def buzzword_score(self, article_text):
        if len(article_text) == 0:
            return 0
        score = sum(article_text.count(i) for i in self.buzzwords)
        return score

    def fit(self, X, y):

        self.fit_buzzword_list(X, y)

        title_texts = [i["title_text"] for i in X]
        self.title_semantic.fit(title_texts, y)
        title_probs = self.title_semantic.predict_proba(title_texts)

        article_texts = [i["article_text"] for i in X]
        buzzword_score = [self.buzzword_score(i) for i in article_texts]

        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(title_probs[i][1])
            geom_features[i].append(buzzword_score[i])
        self.gradboost.fit(geom_features, y)

        return self

    def predict(self, X):
        title_texts = [i["title_text"] for i in X]
        title_probs = self.title_semantic.predict_proba(title_texts)

        article_texts = [i["article_text"] for i in X]
        buzzword_score = [self.buzzword_score(i) for i in article_texts]


        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(title_probs[i][1])
            geom_features[i].append(buzzword_score[i])
        return self.gradboost.predict(geom_features)

    def predict_proba(self, X):
        title_texts = [i["title_text"] for i in X]
        title_probs = self.title_semantic.predict_proba(title_texts)

        article_texts = [i["article_text"] for i in X]
        buzzword_score = [self.buzzword_score(i) for i in article_texts]

        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(title_probs[i][1])
            geom_features[i].append(buzzword_score[i])
        return self.gradboost.predict_proba(geom_features)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
