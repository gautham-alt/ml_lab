import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_iris()

X = data.data[:, :2]
y = (data.target == 0).astype(int)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Linear SVM
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1)
)

model.fit(X_train, y_train)

# 4. Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# 5. Create meshgrid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# 6. Predict decision function
Z = model.decision_function(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# 7. Visualization
plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train, cmap='coolwarm', edgecolors='k'
)

# Decision boundary and margins
plt.contour(
    xx, yy, Z,
    levels=[-1, 0, 1],
    colors=['blue', 'black', 'blue'],
    linestyles=['--', '-', '--']
)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Linear SVM Decision Boundary')
plt.show()
