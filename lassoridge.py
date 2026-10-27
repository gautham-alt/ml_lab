from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear = make_pipeline(StandardScaler(), LinearRegression())
linear.fit(X_train, y_train)
pred = linear.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# Ridge
ridge = make_pipeline(StandardScaler(), Ridge())

params = {'ridge__alpha': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(ridge, params, cv=5)
grid.fit(X_train, y_train)

pred = grid.predict(X_test)

print("\nRidge")
print("Best Parameters:", grid.best_params_)
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# Lasso
lasso = make_pipeline(StandardScaler(), Lasso(max_iter=10000))

params = {'lasso__alpha': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(lasso, params, cv=5)
grid.fit(X_train, y_train)

pred = grid.predict(X_test)

print("\nLasso")
print("Best Parameters:", grid.best_params_)
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
