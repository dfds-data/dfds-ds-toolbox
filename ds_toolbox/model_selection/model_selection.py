from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ClfSwitcher(BaseEstimator, ClassifierMixin):
    """A Custom BaseEstimator that can switch between classifiers.

    Attributes:
        estimator: The classifier
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
    ):
        if estimator is None:
            estimator = RandomForestClassifier()
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> ClfSwitcher:
        """Fit `estimator` with training data.

        Args:
            X: Training data
            y: Target values
        """
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        """Run `predict` on `estimator`.

        Args:
            X: Data used for prediction

        Returns:
            Predictions
        """
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """Run `predict_proba` on `estimator`.

        Args:
            X: Data used for prediction

        Returns:
            Probabilities
        """
        return self.estimator.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Run `score` on `estimator`."""
        return self.estimator.score(X, y)


class RegSwitcher(BaseEstimator, RegressorMixin):
    """A Custom BaseEstimator that can switch between classifiers.

    Attributes:
        estimator: The regressor
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
    ):
        if estimator is None:
            estimator = RandomForestRegressor()
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> RegSwitcher:
        """Fit `estimator` with training data.

        Args:
            X: Training data
            y: Target values
        """
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        """Run `predict` on `estimator`.

        Args:
            X: Data used for prediction

        Returns:
            Predictions
        """
        return self.estimator.predict(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Run `score` on `estimator`."""
        return self.estimator.score(X, y)
