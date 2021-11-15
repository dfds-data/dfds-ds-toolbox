"""
Regression results (Pred vs Real)
================================

Given a trained model, it showcase the performance, along a error band
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, train_test_split

from ds_toolbox.analysis.plotting import plot_regression_predicted_vs_actual

# Create a dataset to fit and predict

X, y = load_diabetes(return_X_y=True, as_frame=True)
numeric_cols = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
data_train = X_train.copy()
data_train["target"] = y_train
data_test = X_test.copy()
data_test["target"] = y_test

est = RandomForestRegressor()

# CV predict
y_pred = cross_val_predict(est, X_train[numeric_cols], y_train, n_jobs=-1, verbose=0)

mae = (np.abs(y_train - y_pred)).mean(axis=0)
mae_text = (r"$MAE={:.2f}$").format(mae)

plot_regression_predicted_vs_actual(y_train, y_pred, title="RF Model", extra_text=mae_text)
