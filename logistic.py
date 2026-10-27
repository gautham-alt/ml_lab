import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.pipeline import make_pipeline

# 1. Load dataset
data = fetch_openml(data_id=37, as_frame=True)

df = data.frame

X = df.drop('class', axis=1)
y = df['class'].astype(int)

# Replace zero values with median
cols = ['plas', 'pres', 'skin', 'insu', 'mass']

X[cols] = X[cols].replace(0, float('nan'))
X = X.fillna(X.median())

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Without scaling
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

# 4. With scaling
model2 = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

# 5. Evaluation
print("Without Scaling")
print("Accuracy:", accuracy_score(y_test, pred1))
print("Precision:", precision_score(y_test, pred1))
print("Recall:", recall_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1))

print("\nWith Scaling")
print("Accuracy:", accuracy_score(y_test, pred2))
print("Precision:", precision_score(y_test, pred2))
print("Recall:", recall_score(y_test, pred2))
print("F1 Score:", f1_score(y_test, pred2))
