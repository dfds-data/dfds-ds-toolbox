from __future__ import annotations

import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import deprecated


@deprecated("Use sklearn.feature_selection.SequentialFeatureSelector instead.")
def stepwise_selection(
    X: pd.DataFrame,
    y: np.array,
    initial_list: List[str] = None,
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
    verbose: bool = False,
) -> List[str]:
    """Perform a forward-backward feature selection

    Based on p-value from statsmodels.api.OLS

    Args:
        X: DataFrame with candidate features
        y: list-like with the target
        initial_list: list of features to start with (column names of X)
        threshold_in: include a feature if its p-value < threshold_in
        threshold_out: exclude a feature if its p-value > threshold_out
        verbose: whether to print the sequence of inclusions and exclusions

    Returns:
        list of selected features

    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    step_df = pd.DataFrame(columns=["action", "Feature", "p_val"])
    if initial_list is None:
        included = []
    else:
        included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            #             print(new_column)
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            action = "Add"
            feature = best_feature
            p_val = best_pval
            if verbose:
                print("Add  {:30} with p-value {:.6}".format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            action = "Drop"
            feature = worst_feature
            p_val = worst_pval
            if verbose:
                print("Drop {:30} with p-value {:.6}".format(worst_feature, worst_pval))
        if changed:
            vector_data = [action, feature, p_val]
            df_adj = pd.DataFrame([vector_data], columns=list(step_df.columns))
            step_df = step_df.append(df_adj, ignore_index=True)
        if not changed:
            break
    return included


@deprecated("Use sklearn.feature_selection.SelectFromModel instead.")
def rf_prim_columns(
    X: pd.DataFrame, y: pd.Series, n_trees: int = 10, top_cols: int = 10
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Returns dictionary counting the number of times each column appears among the `top_cols` most significant columns
    in each of the ten Random Forests applied to the data set
    Also returns a list of the MAE values for further validation of the ability to predict the data set

    Args:
        X:
        y:
        n_trees:
        top_cols:

    Returns:
        missing
    """
    col_names = list(X.columns)
    rand_list = np.random.randint(
        low=0,
        high=1000,
        size=n_trees,  # number of trees
    )
    sss = ShuffleSplit(n_splits=n_trees, test_size=0.3, random_state=0)
    sss.get_n_splits(X)
    fitted_models = []
    score_list = []
    idx = 1
    y_pred = y.copy()

    for idx, (train_idx, valid_idx) in enumerate(sss.split(X)):
        rf = RandomForestRegressor(random_state=rand_list[idx])
        rf = rf.fit(X.iloc[train_idx, :].values, y.iloc[train_idx])
        pred_tgt = rf.predict(X.iloc[valid_idx, :])
        fitted_models.append(rf)
        score_list.append(mean_absolute_error(y.iloc[valid_idx], pred_tgt))
        y_pred.iloc[valid_idx] = rf.predict(X.iloc[valid_idx, :])
        idx += 1
    cols_df = pd.DataFrame(columns=["Feature", "RF_Count", "RF_importance"])
    for mdl in fitted_models:
        top_features = pd.DataFrame(
            mdl.feature_importances_, index=col_names, columns=["RF_importance"]
        ).sort_values("RF_importance", ascending=False)

        for c in list(top_features.head(top_cols).index):
            if c in list(cols_df["Feature"]):
                cols_df.loc[cols_df["Feature"] == c, "RF_Count"] += 1
                cols_df.loc[cols_df["Feature"] == c, "RF_importance"] += top_features.loc[
                    c, "RF_importance"
                ]
            else:
                cols_df = cols_df.append(
                    {
                        "Feature": c,
                        "RF_Count": 1,
                        "RF_importance": top_features.loc[c, "RF_importance"],
                    },
                    ignore_index=True,
                )
    cols_df = cols_df.sort_values("RF_Count", ascending=False)
    return (cols_df, score_list)


@deprecated("Use sklearn.feature_selection module instead.")
class RegFeatureSelector:
    """Selects useful features.

    Several strategies are possible (filter and wrapper methods).
    Works for regression problems only.

    Attributes:
        strategy:
            default = "l1"
            The strategy to select features.
            Available strategies = ["variance", "l1", "rf_feature_importance", 'rf_top_features', "stepwise"]
        threshold:
            defaut = 0.3
            The percentage of variable to discard according the strategy.
            Must be between 0. and 1.
    """

    def __init__(self, strategy: str = "l1", threshold: float = 0.3):
        self.strategy = strategy
        self.threshold = threshold
        self.__fitOK = False
        self.__to_discard: List[Any] = []
        self._available_strategies = [
            "variance",
            "l1",
            "rf_feature_importance",
            "rf_top_features",
            "stepwise",
        ]

    def _get_params(self):
        return {"strategy": self.strategy, "threshold": self.threshold}

    def _set_params(self, **params):
        self.__fitOK = False

        for k, v in params.items():
            if k not in self._get_params():
                warnings.warn(
                    "Invalid parameter a for feature selector"
                    "Reg_feature_selector. Parameter IGNORED. Check "
                    "the list of available parameters with "
                    "`feature_selector.get_params().keys()`"
                )
            else:
                setattr(self, k, v)

    def fit(self, df_train: pd.DataFrame, y_train: pd.Series) -> RegFeatureSelector:
        """Fits Reg_feature_selector.

        Args:
            df_train:
                The train dataset with numerical features and no NA.
                With shape = (n_train, n_features).
            y_train:
                The target for regression task.
                With shape = (n_train, ).

        Returns:
            self
        """

        # sanity checks
        if not isinstance(df_train, pd.DataFrame):
            raise ValueError("df_train must be a DataFrame")

        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train must be a Series")

        if self.strategy == "variance":
            coef = df_train.std()
            abstract_threshold = np.percentile(coef, 100.0 * self.threshold)
            self.__to_discard = coef[coef <= abstract_threshold].index
            self.__fitOK = True

        elif self.strategy == "l1":
            model = Lasso(alpha=100.0, random_state=0)  # to be tuned
            model.fit(df_train, y_train)
            coef = np.abs(model.coef_)
            abstract_threshold = np.percentile(coef, 100.0 * self.threshold)
            self.__to_discard = df_train.columns[coef <= abstract_threshold]
            self.__fitOK = True

        elif self.strategy == "rf_feature_importance":
            model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=0)  # to be tuned
            model.fit(df_train, y_train)
            coef = model.feature_importances_
            abstract_threshold = np.percentile(coef, 100.0 * self.threshold)
            self.__to_discard = df_train.columns[coef <= abstract_threshold]
            self.__fitOK = True

        elif self.strategy == "rf_top_features":
            (cols_df, score_list) = rf_prim_columns(df_train, y_train)
            selected_cols = list(
                cols_df[cols_df.RF_Count == 10]["Feature"]
            )  # to be tuned, taken only columns that appear in top 10 in the 10 trees
            self.__to_discard = [col for col in df_train.columns if col not in selected_cols]
            self.__fitOK = True

        elif self.strategy == "stepwise":
            selected_cols = stepwise_selection(df_train, y_train)
            self.__to_discard = [col for col in df_train.columns if col not in selected_cols]
            self.__fitOK = True
        else:
            raise ValueError(
                "Strategy invalid. Please choose between ",
                self._available_strategies
                # "'variance', 'l1','stepwise' or 'rf_feature_importance' or 'rf_top_features'"
            )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the dataset

        Args:
            df:
                pandas dataframe of shape = (n, n_features).
                The dataset with numerical features and no NA

        Returns:
            pandas dataframe of shape = (n_train, n_features*(1-threshold))
            The train dataset with relevant features
        """

        if self.__fitOK:

            # sanity checks
            if type(df) != pd.DataFrame:
                raise ValueError("df must be a DataFrame")

            return df.drop(self.__to_discard, axis=1)
        else:
            raise ValueError("call fit or fit_transform function before")

    def fit_transform(self, df_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
        """Fits Reg_feature_selector and transforms the dataset

        Args:
            df_train:
                The train dataset with numerical features and no NA.
                With shape = (n_train, n_features).
            y_train:
                The target for regression task.
                With shape = (n_train, ).

        Returns:
            pandas dataframe of shape = (n_train, n_features*(1-threshold))
            The train dataset with relevant features
        """

        self.fit(df_train, y_train)

        return self.transform(df_train)
