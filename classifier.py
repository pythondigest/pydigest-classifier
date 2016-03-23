from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2

class ChainedClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom classifier, chains semantic analysis with geometric.
    SVM trained on tfidf of titles predicts probability of article being good,
        then this probability is appended to geometric features.
    Finally, GradientBoosting makes a prediction, based on geometric features and probability output from bag-of-words model.
    """
    def __init__(self, learning_rate=0.01, max_depth=6, min_samples_leaf=20, max_features=None):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.gradboost = GradientBoostingClassifier(n_estimators=3000, learning_rate=self.learning_rate, max_depth=self.max_depth,
                                                    min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        self.title_semantic = Pipeline([('vect', TfidfVectorizer()),
                                        ('clf', SVC(probability=True))
                                        ])


    def fit(self, X, y):
        title_texts = [i["title_text"] for i in X]
        self.title_semantic.fit(title_texts, y)
        title_probs = self.title_semantic.predict_proba(title_texts)


        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(title_probs[i][1])
        print(self.gradboost.learning_rate)
        self.gradboost.fit(geom_features, y)

        return self

    def predict(self, X):
        title_texts = [i["title_text"] for i in X]
        title_probs = self.title_semantic.predict_proba(title_texts)


        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(title_probs[i][1])

        return self.gradboost.predict(geom_features)

    def predict_proba(self, X):
        title_texts = [i["title_text"] for i in X]
        title_probs = self.title_semantic.predict_proba(title_texts)


        geom_features = [i["geom_features"] for i in X]
        for i in range(0, len(geom_features)):
            geom_features[i].append(title_probs[i][1])

        return self.gradboost.predict_proba(geom_features)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
