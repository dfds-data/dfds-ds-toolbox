from typing import List, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve


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


def plot_single_univariate_dependency(
    input_data: pd.DataFrame, feature: str, target_col: str, trend_correlation: float = None
) -> Figure:
    """Draws univariate dependence plots for a feature

    Args:
        input_data: grouped data contained bins of feature and target mean.
        feature: feature column name.
        target_col: target column name.
        trend_correlation: correlation between train and test trends of feature wrt target

    Returns:
        Figure trend plots for feature
    """
    trend_changes = _get_trend_changes(
        grouped_data=input_data, feature=feature, target_col=target_col
    )
    f = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(input_data[target_col + "_mean"], marker="o")
    ax1.set_xticks(np.arange(len(input_data)))
    ax1.set_xticklabels((input_data[feature]).astype("str"))
    plt.xticks(rotation=45)
    ax1.set_xlabel("Bins of " + feature)
    ax1.set_ylabel("Average of " + target_col)
    comment = "Trend changed " + str(trend_changes) + " times"
    if trend_correlation == 0:
        comment = comment + "\n" + "Correlation with train trend: NA"
    elif trend_correlation is not None:
        comment = (
            comment
            + "\n"
            + "Correlation with train trend: "
            + str(int(trend_correlation * 100))
            + "%"
        )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    ax1.text(
        0.05,
        0.95,
        comment,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
        transform=ax1.transAxes,
    )
    plt.title("Average of " + target_col + " wrt " + feature)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(np.arange(len(input_data)), input_data["Samples_in_bin"], alpha=0.5)
    ax2.set_xticks(np.arange(len(input_data)))
    ax2.set_xticklabels((input_data[feature]).astype("str"))
    plt.xticks(rotation=45)
    ax2.set_xlabel("Bins of " + feature)
    ax2.set_ylabel("Bin-wise sample size")
    plt.title("Samples in bins of " + feature)
    plt.tight_layout()
    plt.show()
    return f


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


def _get_equally_grouped_data(
    input_data: pd.DataFrame,
    feature: str,
    target_col: str,
    bins: int,
    cuts: Union[int, List[int]] = None,
):
    """Bins continuous features into equal sample size buckets and returns the target mean in each bucket.
    Separates out nulls into another bucket.

    Helper function for other plotting functions.

    Args:
        input_data: dataframe containing features and target column.
        feature: feature column name
        target_col: target column name
        bins: Number bins required
        cuts: if buckets of certain specific cuts are required. Used on test data to use cuts from train.

    Returns:
        If cuts are passed only grouped data is returned, else cuts and grouped data is returned
    """
    has_null = pd.isnull(input_data[feature]).sum() > 0
    if has_null == 1:
        data_null = input_data[pd.isnull(input_data[feature])]
        input_data = input_data[~pd.isnull(input_data[feature])]
        input_data.reset_index(inplace=True, drop=True)

    is_train = 0
    if cuts is None:
        is_train = 1
        prev_cut = min(input_data[feature]) - 1
        cuts = [prev_cut]
        reduced_cuts = 0
        for i in range(1, bins + 1):
            next_cut = np.percentile(input_data[feature], i * 100 / bins)
            # float numbers shold be compared with some threshold!
            if next_cut > prev_cut + 0.000001:
                cuts.append(next_cut)
            else:
                reduced_cuts = reduced_cuts + 1
            prev_cut = next_cut

        cut_series = pd.cut(input_data[feature], cuts)
    else:
        cut_series = pd.cut(input_data[feature], cuts)

    grouped = input_data.groupby([cut_series], as_index=True).agg(
        {target_col: [np.size, np.mean], feature: [np.mean]}
    )
    grouped.columns = ["_".join(cols).strip() for cols in grouped.columns.values]
    grouped[grouped.index.name] = grouped.index
    grouped.reset_index(inplace=True, drop=True)
    grouped = grouped[[feature] + list(grouped.columns[0:3])]
    grouped = grouped.rename(index=str, columns={target_col + "_size": "Samples_in_bin"})
    grouped = grouped.reset_index(drop=True)
    corrected_bin_name = (
        "["
        + str(np.round(min(input_data[feature]), 3))
        + ", "
        + str(grouped.loc[0, feature]).split(",")[1]
    )
    grouped[feature] = grouped[feature].astype("category")
    grouped[feature] = grouped[feature].cat.add_categories(corrected_bin_name)
    grouped.loc[0, feature] = corrected_bin_name

    if has_null == 1:
        grouped_null = grouped.loc[0:0, :].copy()
        grouped_null[feature] = grouped_null[feature].astype("category")
        grouped_null[feature] = grouped_null[feature].cat.add_categories("Nulls")
        grouped_null.loc[0, feature] = "Nulls"
        grouped_null.loc[0, "Samples_in_bin"] = len(data_null)
        grouped_null.loc[0, target_col + "_mean"] = data_null[target_col].mean()
        grouped_null.loc[0, feature + "_mean"] = np.nan
        grouped[feature] = grouped[feature].astype("str")
        grouped = pd.concat([grouped_null, grouped], axis=0)
        grouped.reset_index(inplace=True, drop=True)

    grouped[feature] = grouped[feature].astype("str").astype("category")
    if is_train == 1:
        return (cuts, grouped)
    else:
        return grouped


def _get_trend_changes(
    grouped_data: pd.DataFrame, feature: str, target_col: str, threshold: float = 0.03
) -> int:
    """Calculates number of times the trend of feature wrt target changed direction.

    Helper function for other plotting functions.

    Args:
        grouped_data: grouped dataset
        feature: feature column name
        target_col: target column
        threshold: minimum % difference required to count as trend change (between 0 and 1)
    Returns:
         number of trend changes for the feature
    """
    grouped_data = grouped_data.loc[grouped_data[feature] != "Nulls", :].reset_index(drop=True)
    target_diffs = grouped_data[target_col + "_mean"].diff()
    target_diffs = target_diffs[~np.isnan(target_diffs)].reset_index(drop=True)
    max_diff = grouped_data[target_col + "_mean"].max() - grouped_data[target_col + "_mean"].min()
    target_diffs_mod = target_diffs.fillna(0).abs()
    low_change = target_diffs_mod < threshold * max_diff
    target_diffs_norm = target_diffs.divide(target_diffs_mod)
    target_diffs_norm[low_change] = 0
    target_diffs_norm = target_diffs_norm[target_diffs_norm != 0]
    target_diffs_lvl2 = target_diffs_norm.diff()
    changes = target_diffs_lvl2.fillna(0).abs() / 2
    tot_trend_changes = int(changes.sum()) if ~np.isnan(changes.sum()) else 0
    return tot_trend_changes


def _get_trend_correlation(
    grouped: pd.DataFrame, grouped_test: pd.DataFrame, feature: str, target_col: str
) -> float:
    """Calculates correlation between train and test trend of feature wrt target.

    Helper function for other plotting functions.

    Args:
        grouped: train grouped data
        grouped_test: test grouped data
        feature: feature column name
        target_col: target column name

    Returns:
        trend correlation between train and test
    """
    grouped = grouped[grouped[feature] != "Nulls"].reset_index(drop=True)
    grouped_test = grouped_test[grouped_test[feature] != "Nulls"].reset_index(drop=True)

    if grouped_test.loc[0, feature] != grouped.loc[0, feature]:
        grouped_test[feature] = grouped_test[feature].cat.add_categories(grouped.loc[0, feature])
        grouped_test.loc[0, feature] = grouped.loc[0, feature]
    grouped_test_train = grouped.merge(
        grouped_test[[feature, target_col + "_mean"]],
        on=feature,
        how="left",
        suffixes=("", "_test"),
    )
    nan_rows = pd.isnull(grouped_test_train[target_col + "_mean"]) | pd.isnull(
        grouped_test_train[target_col + "_mean_test"]
    )
    grouped_test_train = grouped_test_train.loc[~nan_rows, :]
    if len(grouped_test_train) > 1:
        trend_correlation = np.corrcoef(
            grouped_test_train[target_col + "_mean"], grouped_test_train[target_col + "_mean_test"]
        )[0, 1]
    else:
        trend_correlation = 0
        print("Only one bin created for " + feature + ". Correlation can't be calculated")

    return trend_correlation


def _univariate_plotter(
    feature: str,
    data: pd.DataFrame,
    target_col: str,
    bins: int = 10,
    data_test: pd.DataFrame = None,
):
    """Calls the draw plot function and editing around the plots

    Helper function for `get_univariate_plots`.

    Args:
        feature: feature column name
        data: dataframe containing features and target columns
        target_col: target column name
        bins: number of bins to be created from continuous feature
        data_test: test data which has to be compared with input data for correlation

    Returns:
        grouped data if only train passed, else (grouped train data, grouped test data)
    """
    print(" {:^100} ".format("Plots for " + feature))
    if data[feature].dtype == "O":
        print("Categorical feature not supported")
    else:
        cuts, grouped = _get_equally_grouped_data(
            input_data=data, feature=feature, target_col=target_col, bins=bins
        )
        has_test = type(data_test) == pd.core.frame.DataFrame
        if has_test:
            grouped_test = _get_equally_grouped_data(
                input_data=data_test.reset_index(drop=True),  # type: ignore[union-attr]
                feature=feature,
                target_col=target_col,
                bins=bins,
                cuts=cuts,
            )
            trend_corr = _get_trend_correlation(grouped, grouped_test, feature, target_col)
            print(" {:^100} ".format("Train data plots"))

            plot_single_univariate_dependency(
                input_data=grouped, feature=feature, target_col=target_col
            )
            print(" {:^100} ".format("Test data plots"))

            plot_single_univariate_dependency(
                input_data=grouped_test,
                feature=feature,
                target_col=target_col,
                trend_correlation=trend_corr,
            )
        else:
            plot_single_univariate_dependency(
                input_data=grouped, feature=feature, target_col=target_col
            )
        print("\n")
        if has_test:
            return (grouped, grouped_test)
        else:
            return grouped


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


def _get_lift_curve(dataSet: pd.DataFrame, noBins: int) -> pd.DataFrame:
    datasetSorted = dataSet.sort_values("predProba", ascending=False)
    datasetSize = len(datasetSorted)
    datasetStep = round(datasetSize / noBins)

    frameOut = pd.DataFrame([], columns=["Quantile", "EventRate"])
    for i in range(datasetStep, datasetSize, datasetStep):
        frameOut = frameOut.append(
            pd.DataFrame(
                [[i / datasetSize, (datasetSorted.target.iloc[:i]).sum() / i]],
                columns=["Quantile", "EventRate"],
            )
        )
    return frameOut


def _get_gains_curve(dataSet: pd.DataFrame, noBins: int) -> pd.DataFrame:
    datasetSorted = dataSet.sort_values("predProba", ascending=False)
    datasetSize = len(datasetSorted)
    total_events = datasetSorted.target.sum()
    datasetStep = round(datasetSize / noBins)

    frameOut = pd.DataFrame([], columns=["Quantile", "EventRate"])
    for i in range(datasetStep, datasetSize, datasetStep):
        frameOut = frameOut.append(
            pd.DataFrame(
                [
                    [
                        i / datasetSize,
                        (datasetSorted.target.iloc[:i]).sum() / i,
                        (datasetSorted.target.iloc[:i]).sum() / total_events,
                    ]
                ],
                columns=["Quantile", "EventRate", "PctEvents"],
            )
        )
    return frameOut


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
