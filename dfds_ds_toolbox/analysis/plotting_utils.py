"""
Helper functions for dfds_ds_toolbox.analysis.plotting.
"""
from typing import List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


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

            _draw_single_univariate_dependency(
                input_data=grouped, feature=feature, target_col=target_col
            )
            print(" {:^100} ".format("Test data plots"))

            _draw_single_univariate_dependency(
                input_data=grouped_test,
                feature=feature,
                target_col=target_col,
                trend_correlation=trend_corr,
            )
        else:
            _draw_single_univariate_dependency(
                input_data=grouped, feature=feature, target_col=target_col
            )
        print("\n")
        if has_test:
            return (grouped, grouped_test)
        else:
            return grouped


def _draw_single_univariate_dependency(
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
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
    return fig
