import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

# 1. Load dataset
data = fetch_openml('adult', version=2, as_frame=True)

df = data.frame

# 2. Preprocess
df = df.replace('?', pd.NA).dropna()

X = df.drop('class', axis=1)
y = df['class'].str.strip().str.replace('.', '', regex=False)

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='category').columns

# 3. Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Logistic Regression
lr = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=2000))
])

lr.fit(X_train, y_train)

pred1 = lr.predict(X_test)

# 6. Decision Tree
dt = Pipeline([
    ('preprocess', preprocessor),
    ('model', DecisionTreeClassifier(
        criterion='entropy',
        max_depth=8,
        random_state=42
    ))
])

dt.fit(X_train, y_train)

pred2 = dt.predict(X_test)

# 7. Evaluation function
def evaluate(name, pred):

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(
        y_test, pred, pos_label='>50K'
    ))
    print("Recall:", recall_score(
        y_test, pred, pos_label='>50K'
    ))
    print("F1 Score:", f1_score(
        y_test, pred, pos_label='>50K'
    ))

evaluate("Logistic Regression", pred1)
evaluate("Decision Tree", pred2)
