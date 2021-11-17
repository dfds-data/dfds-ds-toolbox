"""
Plot ROC (Receiver operating characteristic)
============================================

Given a trained model, it showcase the Area under the curve of both train and test data. Remember to include a predProba field
"""
import pandas as pd
from sklearn import datasets, model_selection, svm

from ds_toolbox.analysis.plotting import plot_roc_curve

X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
clf = svm.SVC(random_state=0, probability=True)
clf.fit(X_train, y_train)

dataTrain = pd.DataFrame(X_train.copy())
dataTrain["target"] = list(y_train)
dataTrain["predProba"] = list(clf.predict_proba(X_train)[:, 1])
print("dataTrain", dataTrain.shape)

dataTest = pd.DataFrame(X_test.copy())
dataTest["predProba"] = list(clf.predict_proba(X_test)[:, 1])
dataTest["target"] = list(y_test)
print("dataTest", dataTest.shape)


fig = plot_roc_curve(dataTrain=dataTrain, dataTest=dataTest, label="Example")
