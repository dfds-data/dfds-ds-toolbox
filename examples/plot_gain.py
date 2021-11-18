"""
Plot Gain chart
===============

Given a trained model, show the fraction of events "gained" by targetting a percentaige of the total sample.

In this example, if we select the top 20% of the total sample, we expect to see a gain of about 50%, ie we will have captured 50% of the positive cases.
"""

from sklearn import datasets, model_selection, svm

from ds_toolbox.analysis.plotting import plot_gain_chart

X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
clf = svm.SVC(random_state=0, probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:, 1]  # Select probabilities for class 1

f = plot_gain_chart(
    y_true=y_test,
    y_pred=y_pred,
    n_bins=11,
)
