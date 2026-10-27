import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_openml(name="autoMpg", version=1, as_frame=True)
df = data.frame.dropna()

X = df[['displacement']].astype(float)
y = df['mpg'].astype(float)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different polynomial degrees
for degree in range(1, 5):

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    print("\nDegree:", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    # Plot
    x_line = pd.DataFrame({
        'displacement': np.linspace(
            X.displacement.min(), X.displacement.max(), 200
        )
    })

    y_line = model.predict(poly.transform(x_line))

    plt.scatter(X, y)
    plt.plot(x_line, y_line, color='red')
    plt.title(f"Polynomial Degree {degree}")
    plt.xlabel("Displacement")
    plt.ylabel("MPG")
    plt.show()
