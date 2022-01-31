"""
Plot Lift curve
===============

Given a trained model, it showcase the accumulative lift curve of test data.
"""

from sklearn import datasets, model_selection, svm

from dfds_ds_toolbox.analysis.plotting import plot_lift_curve

X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
clf = svm.SVC(random_state=0, probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:, 1]  # Select probabilities for class 1

f = plot_lift_curve(
    y_true=y_test,
    y_pred=y_pred,
    n_bins=20,
)
