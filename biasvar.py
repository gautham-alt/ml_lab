#20
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Load Boston Housing dataset
data = fetch_openml(name="boston", version=1, as_frame=True)

X = data.data
y = pd.to_numeric(data.target)

# 2. Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Store errors
train_errors = []
val_errors = []
degrees = range(1, 6)

# 4. Polynomial Regression for different degrees
for degree in degrees:

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Calculate MSE
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)

    train_errors.append(train_mse)
    val_errors.append(val_mse)

    print("Degree:", degree)
    print("Training MSE:", train_mse)
    print("Validation MSE:", val_mse)
    print()

# 5. Plot errors
plt.plot(degrees, train_errors, marker='o',
         label='Training Error')

plt.plot(degrees, val_errors, marker='o',
         label='Validation Error')

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff")
plt.xticks(list(degrees))
plt.legend()
plt.grid()
plt.show()

