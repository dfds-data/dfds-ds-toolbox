"""
Histogram of predicted probabilities
====================================

When doing a classification, we want to see how much overlap there is in
the predicted probabilities.
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dfds_ds_toolbox.analysis.plotting import plot_classification_proba_histogram

# Create a dataset to classify
X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_redundant=2,
    n_informative=3,
    random_state=1,
    n_clusters_per_class=1,
)
# Train a model
model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model.fit(X_train, y_train)
# Get predictions
predictions = model.predict_proba(X_test)
proba_class_1 = predictions[:, 1]

# Compare predictions to ground truth
plot_classification_proba_histogram(y_true=y_test, y_pred=proba_class_1)
