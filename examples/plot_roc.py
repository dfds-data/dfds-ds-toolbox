"""
Plot ROC (Receiver operating characteristic)
============================================

Given a trained model, it showcase the Area under the curve of both train and test data.
"""
import pandas as pd
from sklearn import datasets, model_selection, svm

from dfds_ds_toolbox.analysis.plotting import plot_roc_curve

X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
clf = svm.SVC(random_state=0, probability=True)
clf.fit(X_train, y_train)

data_train = pd.DataFrame(X_train.copy())
data_train["target"] = list(y_train)
data_train["predProba"] = list(clf.predict_proba(X_train)[:, 1])
print("data_train", data_train.shape)

data_test = pd.DataFrame(X_test.copy())
data_test["predProba"] = list(clf.predict_proba(X_test)[:, 1])
data_test["target"] = list(y_test)
print("data_test", data_test.shape)


fig = plot_roc_curve(y_true=data_train["target"], y_pred=data_train["predProba"], label="Train")
ax = fig.get_axes()[0]
fig = plot_roc_curve(y_true=data_test["target"], y_pred=data_test["predProba"], label="Test", ax=ax)
