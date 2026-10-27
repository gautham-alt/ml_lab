import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df[['AveRooms']]
y = df['MedHouseVal']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gradient Descent
gd = make_pipeline(
    StandardScaler(),
    SGDRegressor(max_iter=10000, learning_rate='constant',
                 eta0=0.001, random_state=42)
)

gd.fit(X_train, y_train)
y_pred_gd = gd.predict(X_test)

# Normal Equation
ne = LinearRegression()
ne.fit(X_train, y_train)
y_pred_ne = ne.predict(X_test)

# Evaluation
print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2:", r2_score(y_test, y_pred_gd))

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, y_pred_ne))
print("R2:", r2_score(y_test, y_pred_ne))

# Visualization
plt.scatter(X_test, y_test)

idx = np.argsort(X_test.values[:, 0])

plt.plot(X_test.values[idx], y_pred_ne[idx], color='red')
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.show()
