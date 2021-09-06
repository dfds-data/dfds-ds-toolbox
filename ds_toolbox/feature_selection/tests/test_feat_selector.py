import numpy as np
import pandas as pd

from ds_toolbox.feature_selection.feat_selector import RegFeatureSelector


# import data
def get_data():
    from sklearn.datasets import load_boston

    boston_dataset = load_boston()

    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston["target"] = boston_dataset.target  # MEDV
    df = boston.copy()

    target_col = "target"
    # make list of numeric and string columns
    numeric_cols = []  # could still have ordinal data
    string_cols = []  # could have ordinal or nominal data

    for col in df.columns:
        if col != target_col:
            if (
                df.dtypes[col] == np.int64
                or df.dtypes[col] == np.int32
                or df.dtypes[col] == np.float64
            ):
                numeric_cols.append(col)  # True integer or float columns

            if df.dtypes[col] == object:  # Nominal and ordinal columns
                string_cols.append(col)

    # print("\n> Number of numerical features", len(numeric_cols), numeric_cols)
    # print("\n> Number of string features", len(string_cols), string_cols)

    df = df[numeric_cols + [target_col]].dropna().copy()
    df = df.reset_index(drop=True)

    # print("\n> Split in train/test")
    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(df, test_size=0.33, random_state=0)

    # break into X and y dataframes
    X_train = df_train.reindex(
        columns=[x for x in df_train[numeric_cols].columns.values if x != target_col]
    ).reset_index(
        drop=True
    )  # separate out X
    y_train = df_train.reindex(columns=[target_col])  # separate out y
    y_train = y_train.values.reshape(-1, 1)
    y_train = np.ravel(y_train)  # flatten the y array

    # break into X and y dataframes
    X_test = df_test.reindex(
        columns=[x for x in df_test[numeric_cols].columns.values if x != target_col]
    ).reset_index(
        drop=True
    )  # separate out X
    y_test = df_test.reindex(columns=[target_col])  # separate out y
    y_test = y_test.values.reshape(-1, 1)
    y_test = np.ravel(y_test)  # flatten the y array

    # print("\n X,y review (TRAIN):", X_train.shape, y_train.shape)
    # print("\n X,y review (TEST):", X_test.shape, y_test.shape)
    return (X_train, y_train, X_test, y_test)


def test_feat_selector():
    X_train, y_train, X_test, y_test = get_data()
    fs = RegFeatureSelector()
    available_strategies = fs._available_strategies
    for strategy in available_strategies:
        # print("\nStrategy=", strategy)
        fs = RegFeatureSelector(strategy=strategy)
        X_adj = fs.fit_transform(X_train, pd.Series(y_train))
        selected_cols = list(X_adj.columns)
        assert len(selected_cols) > 0 & len(selected_cols) <= X_train.shape[0]
        # print("selected_cols=", len(selected_cols), sorted(selected_cols))
