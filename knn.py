import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

X = data.data
y = data.target.astype(int)

# 2. Preprocess
X = X / 255.0

# Use a smaller subset to reduce execution time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=2000,
    random_state=42, stratify=y
)

# 3. Try different K values
for k in [1, 3, 5, 7]:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nK =", k)
    print("Accuracy:", accuracy_score(y_test, y_pred))
