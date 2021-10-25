from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ClfSwitcher(BaseEstimator, ClassifierMixin):
    """A Custom BaseEstimator that can switch between classifiers.

    Attributes:
        estimator: The classifier
    """

    def __init__(
        self,
        estimator: BaseEstimator = RandomForestClassifier(),
    ):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


class RegSwitcher(BaseEstimator, RegressorMixin):
    """A Custom BaseEstimator that can switch between classifiers.

    Attributes:
        estimator: The regressor
    """

    def __init__(
        self,
        estimator: BaseEstimator = RandomForestRegressor(),
    ):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
