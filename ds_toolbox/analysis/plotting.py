import pandas as pd
import seaborn as sns


def plot_prediction_histogram(y_true, y_pred):
    """
    Plot histogram of predictions for binary classifiers.
    :param y_true: 1D array of binary target values, 0 or 1
    :param y_pred: 1D array of predicted target values, probability of class 1.
    :return: matplotlib.Figure.
    """
    df = pd.DataFrame()
    df["Actual class"] = y_true
    df["Probability of class 1"] = y_pred
    fig = sns.histplot(
        df,
        x="Probability of class 1",
        hue="Actual class",
        multiple="dodge",
        kde=True,
        binrange=(0, 1),
    )
    return fig
