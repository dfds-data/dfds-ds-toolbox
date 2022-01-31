import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV

from dfds_ds_toolbox.model_selection.model_selection import ClfSwitcher, RegSwitcher


def get_data(model_type):
    # Load dataset
    if model_type == "clf":
        return datasets.load_iris()
    elif model_type == "reg":
        return datasets.load_boston()


def test_ClfSwitcher():
    # Get datasets
    data = get_data(model_type="clf")

    # Define estimator
    model = ClfSwitcher(RandomForestClassifier())
    model.fit(data.data, data.target)

    # Test that score function is working
    assert model.score(data.data, data.target) > 0


def test_ClfSwitcher_gridsearch():
    # Get datasets
    data = get_data(model_type="clf")

    # Define estimators and parameters to be tested
    parameters = [
        {
            "estimator": [RandomForestClassifier()],
            "estimator__n_estimators": [150, 200],
            "estimator__max_depth": [2, 3],
        },
        {"estimator": [SGDClassifier()], "estimator__alpha": (1e-2, 1e-3, 1e-1)},
    ]

    # Perform grid search
    gs = GridSearchCV(ClfSwitcher(), parameters, cv=3, n_jobs=3, verbose=10)
    gs.fit(data.data, data.target)

    # Get cv results
    cv_res = pd.DataFrame(gs.cv_results_)

    # Check that mean score is bigger than 0
    assert cv_res["mean_test_score"].sum() > 0, "The mean test score should be bigger than 0"


def test_RegSwitcher():
    # Get datasets
    data = get_data(model_type="reg")

    # Define estimator
    model = RegSwitcher(RandomForestRegressor())
    model.fit(data.data, data.target)

    # Test that score function is working
    assert model.score(data.data, data.target) > 0


def test_RegSwitcher_gridsearch():
    # Get datasets
    data = get_data(model_type="reg")

    # Define estimators and parameters to be tested
    parameters = [
        {
            "estimator": [RandomForestRegressor()],
            "estimator__n_estimators": [150, 200],
            "estimator__max_depth": [2, 3],
        },
        {"estimator": [LinearRegression()], "estimator__fit_intercept": [True, False]},
    ]

    # Perform grid search
    gs = GridSearchCV(
        RegSwitcher(), parameters, cv=3, n_jobs=3, verbose=10, scoring="neg_mean_squared_error"
    )
    gs.fit(data.data, data.target)

    # Get cv results
    cv_res = pd.DataFrame(gs.cv_results_)

    # Check that mean score is bigger than 0
    assert cv_res["mean_test_score"].sum() < 0, "The mean test score should be bigger than 0"
