#18
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create model
model = LogisticRegression(max_iter=200)

# -------------------------------
# 1. BOOTSTRAPPING
# -------------------------------

accuracy_boot = []
f1_boot = []

np.random.seed(42)

for i in range(10):

    # Generate random sample with replacement
    indices = np.random.choice(len(X), len(X), replace=True) #(range of values,number of elements to select,replace=True)

    X_sample = X[indices]
    y_sample = y[indices]

    # Train model
    model.fit(X_sample, y_sample)

    # Evaluate on original dataset
    y_pred = model.predict(X)

    accuracy_boot.append(accuracy_score(y, y_pred))
    f1_boot.append(f1_score(y, y_pred, average='weighted'))

print("BOOTSTRAPPING")
print("Mean Accuracy:", np.mean(accuracy_boot))
print("Mean F1 Score:", np.mean(f1_boot))


# -------------------------------
# 2. K-FOLD CROSS VALIDATION
# -------------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_cv = []
f1_cv = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    accuracy_cv.append(accuracy_score(y_test, y_pred))
    f1_cv.append(f1_score(y_test, y_pred, average='weighted'))

print("\nK-FOLD CROSS VALIDATION")
print("Mean Accuracy:", np.mean(accuracy_cv))
print("Mean F1 Score:", np.mean(f1_cv))


# -------------------------------
# 3. COMPARISON
# -------------------------------

print("\nCOMPARISON")
print("Bootstrapping Accuracy:", np.mean(accuracy_boot))
print("K-Fold Accuracy:", np.mean(accuracy_cv))

print("Bootstrapping F1:", np.mean(f1_boot))
print("K-Fold F1:", np.mean(f1_cv))

