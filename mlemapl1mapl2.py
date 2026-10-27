from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLE
mle = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty=None, max_iter=10000)
)

# MAP L1
l1 = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l1', solver='liblinear',
                       C=1.0, max_iter=10000)
)

# MAP L2
l2 = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l2', max_iter=10000)
)

models = {
    'MLE': mle,
    'MAP L1': l1,
    'MAP L2': l2
}

# Train and evaluate
for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Log Loss:", log_loss(y_test, prob))
    print("Coefficients:", model[-1].coef_)
