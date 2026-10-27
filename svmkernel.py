from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

X = data.data
y = data.target.astype(int)

# 2. Preprocess
X = X / 255.0

# Use smaller subset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=5000, test_size=1000,
    random_state=42, stratify=y
)

# 3. Different SVM kernels
kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, degree=3, C=1)
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nKernel:", kernel)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
