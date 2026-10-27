from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import time

# 1. Load dataset
data = fetch_openml('mnist_784', version=1, as_frame=False)

X = data.data
y = data.target.astype(int)

# 2. Preprocess
X = X / 255.0

# Use smaller subset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=2000,
    random_state=42, stratify=y
)

# 3. Different activation functions
activations = ['logistic', 'relu', 'tanh']

for activation in activations:

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation=activation,
        max_iter=100,
        random_state=42,
        early_stopping=True
    )

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    y_pred = model.predict(X_test)

    print("\nActivation:", activation)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Training Time:", end - start)
    print("Iterations:", model.n_iter_)
