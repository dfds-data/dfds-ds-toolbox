from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve

from dfds_ds_toolbox.analysis.plotting_utils import (
    _get_equally_grouped_data,
    _get_trend_changes,
    _get_trend_correlation,
    _univariate_plotter,
)


def plot_classification_proba_histogram(
    y_true: Sequence[int], y_pred: Sequence[float], ax: Axes = None
) -> Figure:
    """Plot histogram of predictions for binary classifiers.

    Args:
        y_true: 1D array of binary target values, 0 or 1.
        y_pred: 1D array of predicted target values, probability of class 1.
        ax: Optional pre-existing axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    bins = np.linspace(0, 1, 11)
    df = pd.DataFrame()
    df["Actual class"] = y_true
    df["Probability of class 1"] = y_pred
    df_actual_1 = df[df["Actual class"] == 1]
    df_actual_0 = df[df["Actual class"] == 0]
    ax.hist(
        x=df_actual_0["Probability of class 1"], bins=bins, label="Actual class 0", histtype="step"
    )
    ax.hist(
        x=df_actual_1["Probability of class 1"], bins=bins, label="Actual class 1", histtype="step"
    )
    ax.set_xlabel("Probability of class 1")
    ax.set_ylabel("Counts")
    ax.legend()
    return fig


def plot_univariate_dependencies(
    data: pd.DataFrame,
    target_col: str,
    features_list: List[str] = None,
    bins: int = 10,
    data_test: pd.DataFrame = None,
):
    """Creates univariate dependence plots for features in the dataset

    Args:
        data: dataframe containing features and target columns
        target_col: target column name
        features_list: by default creates plots for all features. If list passed, creates plots of only those features.
        bins: number of bins to be created from continuous feature
        data_test: test data which has to be compared with input data for correlation

    Returns:
         Draws univariate plots for all columns in data
    """
    if features_list is None:
        features_list = list(data.columns)
        features_list.remove(target_col)

    for cols in features_list:
        if cols != target_col and data[cols].dtype == "O":
            print(cols + " is categorical. Categorical features not supported yet.")
        elif cols != target_col and data[cols].dtype != "O":
            _univariate_plotter(
                feature=cols, data=data, target_col=target_col, bins=bins, data_test=data_test
            )


def get_trend_stats(
    data: pd.DataFrame,
    target_col: str,
    features_list: List[str] = None,
    bins: int = 10,
    data_test: pd.DataFrame = None,
) -> pd.DataFrame:
    """Calculates trend changes and correlation between train/test for list of features.

    Args:
        data: dataframe containing features and target columns
        target_col: target column name
        features_list: by default creates plots for all features. If list passed, creates plots of only those features.
        bins: number of bins to be created from continuous feature
        data_test: test data which has to be compared with input data for correlation

    Returns:
        dataframe with trend changes and trend correlation (if test data passed)
    """

    if features_list is None:
        features_list = list(data.columns)
        features_list.remove(target_col)

    stats_all = []
    has_test = type(data_test) == pd.core.frame.DataFrame
    ignored = []
    for feature in features_list:
        if data[feature].dtype == "O" or feature == target_col:
            ignored.append(feature)
        else:
            cuts, grouped = _get_equally_grouped_data(
                input_data=data, feature=feature, target_col=target_col, bins=bins
            )
            trend_changes = _get_trend_changes(
                grouped_data=grouped, feature=feature, target_col=target_col
            )
            if has_test:
                grouped_test = _get_equally_grouped_data(
                    input_data=data_test.reset_index(drop=True),  # type: ignore[union-attr]
                    feature=feature,
                    target_col=target_col,
                    bins=bins,
                    cuts=cuts,
                )
                trend_corr = _get_trend_correlation(grouped, grouped_test, feature, target_col)
                trend_changes_test = _get_trend_changes(
                    grouped_data=grouped_test, feature=feature, target_col=target_col
                )
                stats = [feature, trend_changes, trend_changes_test, trend_corr]
            else:
                stats = [feature, trend_changes]
            stats_all.append(stats)
    stats_all_df = pd.DataFrame(stats_all)
    stats_all_df.columns = (
        ["Feature", "Trend_changes"]
        if has_test is False
        else ["Feature", "Trend_changes", "Trend_changes_test", "Trend_correlation"]
    )
    if len(ignored) > 0:
        print(
            "Categorical features "
            + str(ignored)
            + " ignored. Categorical features not supported yet."
        )

    print("Returning stats for all numeric features")
    return stats_all_df


def plot_regression_predicted_vs_actual(
    y_true: Sequence[float], y_pred: Sequence[float], alpha: float = 0.2, ax: Axes = None
) -> Figure:
    """Scatter plot of the predicted vs true targets for regression problems.

    Args:
        y_true: array with observed values
        y_pred: array with predicted values
        alpha: transparency of the dots on the scatter plot
        ax: Optional pre-existing axis to plot on


    Returns:
        Figure
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val])
    ax.scatter(y_true, y_pred, alpha=alpha)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return fig


def plot_roc_curve(
    y_true: Sequence[int], y_pred: Sequence[float], label: str = "Train", ax: Axes = None
) -> Figure:
    """plot roc curve for train and test

    Args:
        y_true: array with observed classes
        y_pred: array with predicted probabilities
        label: extra text to add, e.g. "Train" or "Test"
        ax: Optional pre-existing axis to plot on

    Returns:
        Figure
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    fpr_train, tpr_train, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    ax.plot(fpr_train, tpr_train, label=f"{label} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")
    return fig


def plot_lift_curve(
    y_true: Sequence[int], y_pred: Sequence[float], n_bins: int = 10, ax: Axes = None
) -> Figure:
    """Plot lift curve, i.e. how much better than baserate is the model at different thresholds.

    Lift of 1 corresponds to predicting the baserate for the whole sample.

    Args:
        y_true: array with observed values, either 0 or 1.
        y_pred: array with predicted probabilities, float between 0 and 1.
        n_bins: number of bins to use
        ax: Optional pre-existing axis to plot on


    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # Ensure numpy arrays. Save to new variable to avoid redefining a type. Mypy doesn't like that.
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    # Sort true and pred by predicted probability, in descending order
    sorted_idx = np.argsort(y_pred_array)[::-1]
    y_true_array, y_pred_array = y_true_array[sorted_idx], y_pred_array[sorted_idx]
    # Compute running mean
    y_true_running_mean = y_true_array.cumsum() / np.arange(1, len(y_true_array) + 1)
    # Lift is the running rate of positive events over the baserate.
    lift = y_true_running_mean / y_true_array.mean()
    bins = np.linspace(0, 1, n_bins)
    binned_lift = np.quantile(lift, bins)[::-1]  # reverse to get descending order
    # Plot
    ax.plot(bins, binned_lift, marker="o")
    ax.set_xlabel("Fraction of sample")
    ax.set_ylabel("Cumulative lift")
    # Baseline
    ax.plot([0, 1], [1, 1], color="black", linestyle="--", label="Baseline")
    return fig


def plot_gain_chart(
    y_true: Sequence[int], y_pred: Sequence[float], n_bins: int = 10, ax: Axes = None
) -> Figure:
    """The cumulative gains chart shows the percentage of the overall number of cases in a given
     category "gained" by targeting a percentage of the total number of cases.

    Args:
        y_true: array with observed values, either 0 or 1.
        y_pred: array with predicted probabilities, float between 0 and 1.
        n_bins: number of bins to use
        ax: Optional pre-existing axis to plot on

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # Ensure numpy arrays. Save to new variable to avoid redefining a type. Mypy doesn't like that.
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    # Sort true and pred by predicted probability, in descending order
    sorted_idx = np.argsort(y_pred_array)[::-1]
    y_true_array, y_pred_array = y_true_array[sorted_idx], y_pred_array[sorted_idx]
    # Compute cumulative positive events
    y_true_running_sum = y_true_array.cumsum()
    # Gain is the running sum of positive events over total number of positive events
    gain = y_true_running_sum / y_true_array.sum()
    # Make sure initial gain is 0. When we have no samples, we have no gain.
    gain = np.concatenate(([0], gain))
    bins = np.linspace(0, 1, n_bins)
    binned_gain = np.quantile(gain, bins)
    # Plot
    ax.plot(bins, binned_gain, marker="o")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Baseline")
    ax.set_xlabel("Fraction of sample")
    ax.set_ylabel("Gain")
    ax.set_title("Gain chart")
    return fig
