#19
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Titanic dataset
data = fetch_openml('titanic', version=1, as_frame=True)
df = data.frame

# Select features and target
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch',
         'fare', 'embarked', 'survived']]

# Remove missing target values
df = df.dropna(subset=['survived'])

X = df.drop('survived', axis=1)
y = df['survived'].astype(int)

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

# Boosting (AdaBoost)
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

boosting.fit(X_train, y_train)
y_pred_boost = boosting.predict(X_test)

# Evaluation function
def evaluate(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

print("\nBAGGING RESULTS")
evaluate(y_test, y_pred_bag)

print("\nBOOSTING RESULTS")
evaluate(y_test, y_pred_boost)

