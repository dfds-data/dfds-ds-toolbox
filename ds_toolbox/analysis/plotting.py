from typing import List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve

from ds_toolbox.analysis.plotting_utils import (
    _get_equally_grouped_data,
    _get_gains_curve,
    _get_lift_curve,
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
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: format(int(y), ",")))
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: format(int(y), ",")))
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


def plot_roc_curve(dataTrain: pd.DataFrame, dataTest: pd.DataFrame, label: str) -> Figure:
    """plot roc curve for train and test

    Args:
        dataTrain: dataframe containing features and target columns
        dataTest: dataframe containing features and target columns
        label: extra test to add
    Returns:
        Figure
    """
    fprTrain, tprTrain, _ = roc_curve(dataTrain.target, dataTrain.predProba)
    roc_aucTrain = roc_auc_score(dataTrain.target, dataTrain.predProba)

    f, ax1 = plt.subplots(1)
    lw = 2
    ax1.plot(fprTrain, tprTrain, "red", lw=lw, label="Train (AUC = {0:.2f})".format(roc_aucTrain))
    ax1.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    if len(dataTest) > 0:
        fprTest, tprTest, _ = roc_curve(dataTest.target, dataTest.predProba)
        roc_aucTest = roc_auc_score(dataTest.target, dataTest.predProba)
        ax1.plot(
            fprTest, tprTest, color="blue", lw=lw, label="Test (AUC = {0:.2f})".format(roc_aucTest)
        )
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Receiver operating characteristic: " + label)
    ax1.legend(loc="lower right")
    ax1.set_aspect("equal", "datalim")
    return f


def plot_lift_curve(
    dataTrain: pd.DataFrame,
    dataTest: pd.DataFrame,
    noBins: int = 50,
    label: str = "",
    eventBaseRate: float = None,
) -> Figure:
    """Plot cumulative precision and lift evolution in 2 axis

    Args:
        dataTrain: dataframe containing features and target columns
        dataTest: dataframe containing features and target columns
        noBins: number of bins to be created from continuous feature
        eventBaseRate: ??
    Returns:
        figure
    """
    trainLift = _get_lift_curve(dataTrain, noBins)
    if eventBaseRate is None:
        eventRateBase = dataTrain.target.sum() / len(dataTrain)

    f, ax1 = plt.subplots(1)

    ax1.plot(trainLift["Quantile"], trainLift["EventRate"], "r", label="Train")

    if len(dataTest) > 0:
        testLift = _get_lift_curve(dataTest, noBins)
        ax1.plot(testLift["Quantile"], testLift["EventRate"], "b", label="Test")

    y1Min, y1Max = ax1.get_ylim()
    ax2 = ax1.twinx()
    ax2.set_ylim(y1Min / eventRateBase, y1Max / eventRateBase)

    ax1.legend()
    ax1.set_title("Cumulative Precision and Lift: " + label)
    ax1.set_ylabel("Cumulative Precision")
    ax1.set_xlabel("Quantile")
    ax2.set_ylabel("Cumulative Lift")

    ax1.text(
        0.05,
        0.05,
        "Base Event Rate: {0:.2f}".format(eventRateBase),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax1.transAxes,
    )
    return f


def plot_gain_chart(
    dataTrain: pd.DataFrame, dataTest: pd.DataFrame = None, noBins: int = 50
) -> Figure:
    """calculate the percentage of ones within the best n featues and compare this with the average number of ones

    Args:
        dataTrain:
        dataTest:
        noBins:

    Returns:

    """
    trainLift = _get_gains_curve(dataTrain, noBins)

    f, ax1 = plt.subplots(1)

    ax1.plot(trainLift["Quantile"], trainLift["PctEvents"], "red", label="Train")
    if dataTest is not None and len(dataTest) > 0:
        testLift = _get_gains_curve(dataTest, noBins)
        ax1.plot(testLift["Quantile"], testLift["PctEvents"], "blue", label="Test")
    ax1.plot(trainLift["Quantile"], trainLift["Quantile"], "--", color="black", label="Random")

    y1Min, y1Max = ax1.get_ylim()

    ax1.legend()
    ax1.set_title("Gains Chart: ")
    ax1.set_ylabel("% of Events")
    ax1.set_xlabel("% of Customers")

    return f
