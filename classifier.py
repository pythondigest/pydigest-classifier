from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier


class ChainedClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom classifier, chains semantic analysis with geometric.
    SVM trained on tfidf of titles predicts probability of article being good,
        then this probability is appended to geometric features.
    Finally, GradientBoosting makes a prediction, based on geometric features and probability output from bag-of-words model.
    """
    def __init__(self):
        self.gradboost = GradientBoostingClassifier(learning_rate=0.01, max_depth=15, n_estimators=30, subsample=0.257)
        self.knn = Pipeline([('vect', TfidfVectorizer()),
                         ('clf', SVC(probability=True))
                         ])

        self.tfidf_vec = TfidfVectorizer()

    def fit(self, X, y):
        title_texts = [i["title_text"] for i in X]
        self.knn.fit(title_texts, y)
        probs = self.knn.predict_proba(title_texts)

        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(probs[i][1])

        self.gradboost.fit(geom_features, y)

        return self

    def predict(self, X):
        title_texts = [i["title_text"] for i in X]
        probs = self.knn.predict_proba(title_texts)
        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(probs[i][1])
        return self.gradboost.predict(geom_features)

    def predict_proba(self, X):
        title_texts = [i["title_text"] for i in X]
        probs = self.knn.predict_proba(title_texts)
        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(probs[i][1])
        return self.gradboost.predict_proba(geom_features)


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
