import pytest
from matplotlib.figure import Figure
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="function")
def classification_dataset():
    # Make a synthetic dataset
    return datasets.make_classification(random_state=0)


def test_prediction_histogram(classification_dataset):
    from ds_toolbox.analysis.plotting import plot_classification_proba_histogram

    X, y = classification_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression()
    # Train a simple model
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]  # Only take predictions for class 1
    # Plot this example dataset
    fig = plot_classification_proba_histogram(y_true=y_test, y_pred=preds)
    # Check that the figure is a matplotlib Figure
    assert isinstance(
        fig, Figure
    ), "plot_classification_proba_histogram did not return a matplotlib Figure"


def test_lift_curve(classification_dataset):
    from ds_toolbox.analysis.plotting import plot_lift_curve

    X, y = classification_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Train a simple model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]  # Select probabilities for class 1

    fig = plot_lift_curve(
        y_true=y_test,
        y_pred=y_pred,
        n_bins=20,
    )

    # Check that the figure is a matplotlib Figure
    assert isinstance(fig, Figure), "plot_lift_curve did not return a matplotlib Figure"
