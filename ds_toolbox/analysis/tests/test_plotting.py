from matplotlib.figure import Figure
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def test_prediction_histogram():
    from ds_toolbox.analysis.plotting import plot_classification_proba_histogram

    # Make a synthetic dataset
    data = datasets.make_classification()
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = LogisticRegression()
    # Train a simple model
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]  # Only take predictions for class 1
    # Plot this example dataset
    fig = plot_classification_proba_histogram(y_true=y_test, y_pred=preds)
    # Check that the figure is a matplotlib Figure
    assert isinstance(fig, Figure)
