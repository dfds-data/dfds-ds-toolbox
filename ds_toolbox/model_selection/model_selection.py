from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ClfSwitcher(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimator=RandomForestClassifier(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


class RegSwitcher(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator=RandomForestRegressor(),
    ):
        """
        A Custom BaseEstimator that can switch between regressors.
        :param estimator: sklearn object - The regressor
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
