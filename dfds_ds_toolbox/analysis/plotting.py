from typing import List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve

from dfds_ds_toolbox.analysis.plotting_utils import (
    _get_equally_grouped_data,
    _get_trend_changes,
    _get_trend_correlation,
    _univariate_plotter,
)


def plot_classification_proba_histogram(y_true: Sequence[int], y_pred: Sequence[float]) -> Figure:
    """Plot histogram of predictions for binary classifiers.

    Args:
        y_true: 1D array of binary target values, 0 or 1.
        y_pred: 1D array of predicted target values, probability of class 1.
    """
    bins = np.linspace(0, 1, 11)
    df = pd.DataFrame()
    df["Actual class"] = y_true
    df["Probability of class 1"] = y_pred
    df_actual_1 = df[df["Actual class"] == 1]
    df_actual_0 = df[df["Actual class"] == 0]
    plt.hist(
        x=df_actual_0["Probability of class 1"], bins=bins, label="Actual class 0", histtype="step"
    )
    plt.hist(
        x=df_actual_1["Probability of class 1"], bins=bins, label="Actual class 1", histtype="step"
    )
    plt.xlabel("Probability of class 1")
    plt.ylabel("Counts")
    plt.legend()
    return plt.gcf()


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
    y_true: np.array, y_pred: np.array, title: str = "", extra_text: str = "", band_pct: float = 0.1
) -> Figure:
    """Scatter plot of the predicted vs true targets

    Args:
        y_true: array with observed values
        y_pred: array with predicted values
        title: Title of plot
        extra_text: Legend to add
        band_pct: width of band (in %, between 0 and 1)

    Returns:
        Figure
    """

    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["figure.figsize"] = (10, 7)
    matplotlib.rcParams["font.size"] = 12
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min(), y_true.max() * 1.1])
    ax.set_ylim([y_true.min(), y_true.max() * 1.1])
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda y, _: format(int(y), ",")))
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, _: format(int(y), ",")))
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0)
    ax.legend([extra], [extra_text], loc="upper left")
    ax.set_title(title)
    # create a confidence band of +/- 10% error
    y_lower = [i - band_pct * i for i in sorted(y_true)]
    y_upper = [i + band_pct * i for i in sorted(y_true)]
    # plot our confidence band
    ax.fill_between(sorted(y_true), y_lower, y_upper, alpha=0.2, color="tab:orange")
    plt.tight_layout()
    plt.show()
    return fig


def plot_roc_curve(data_train: pd.DataFrame, data_test: pd.DataFrame, label: str) -> Figure:
    """plot roc curve for train and test

    Args:
        data_train: dataframe containing features and target columns
        data_test: dataframe containing features and target columns
        label: extra test to add
    Returns:
        Figure
    """
    fpr_train, tpr_train, _ = roc_curve(data_train.target, data_train.predProba)
    roc_auc_train = roc_auc_score(data_train.target, data_train.predProba)

    f, ax1 = plt.subplots(1)
    lw = 2
    ax1.plot(
        fpr_train, tpr_train, "red", lw=lw, label="Train (AUC = {0:.2f})".format(roc_auc_train)
    )
    ax1.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    if len(data_test) > 0:
        fpr_test, tpr_test, _ = roc_curve(data_test.target, data_test.predProba)
        roc_auc_test = roc_auc_score(data_test.target, data_test.predProba)
        ax1.plot(
            fpr_test,
            tpr_test,
            color="blue",
            lw=lw,
            label="Test (AUC = {0:.2f})".format(roc_auc_test),
        )
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Receiver operating characteristic: " + label)
    ax1.legend(loc="lower right")
    ax1.set_aspect("equal", "datalim")
    return f


def plot_lift_curve(y_true: Sequence[int], y_pred: Sequence[float], n_bins: int = 10) -> Figure:
    """Plot lift curve, i.e. how much better than baserate is the model at different thresholds.

    Lift of 1 corresponds to predicting the baserate for the whole sample.

    Args:
        y_true: array with observed values, either 0 or 1.
        y_pred: array with predicted probabilities, float between 0 and 1.
        n_bins: number of bins to use

    Returns:
        matplotlib Figure
    """
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
    plt.plot(bins, binned_lift, marker="o")
    plt.xlabel("Fraction of sample")
    plt.ylabel("Cumulative lift")
    # Baseline
    plt.plot([0, 1], [1, 1], color="black", linestyle="--", label="Baseline")
    return plt.gcf()


def plot_gain_chart(y_true: Sequence[int], y_pred: Sequence[float], n_bins: int = 10) -> Figure:
    """The cumulative gains chart shows the percentage of the overall number of cases in a given
     category "gained" by targeting a percentage of the total number of cases.

    Args:
        y_true: array with observed values, either 0 or 1.
        y_pred: array with predicted probabilities, float between 0 and 1.
        n_bins: number of bins to use

    Returns:
        matplotlib Figure
    """
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
    plt.plot(bins, binned_gain, marker="o")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Baseline")
    plt.xlabel("Fraction of sample")
    plt.ylabel("Gain")
    plt.suptitle("Gain chart")
    return plt.gcf()
