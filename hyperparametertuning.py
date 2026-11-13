#15
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Fashion MNIST dataset
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

X = data.data
y = data.target.astype(int)

# Use subset to reduce training time
X = X[:10000]
y = y[:10000]

# 2. Preprocessing (normalize pixel values)
X = X / 255.0

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Hyperparameter tuning
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = [50, 100]

best_accuracy = 0
best_model = None

for lr in learning_rates:
    for batch in batch_sizes:
        for ep in epochs:

            model = MLPClassifier(
                hidden_layer_sizes=(100,),
                learning_rate_init=lr,
                batch_size=batch,
                max_iter=ep,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print("Learning rate:", lr)
            print("Batch size:", batch)
            print("Epochs:", ep)
            print("Accuracy:", acc)
            print("--------------------")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model

# 5. Best model evaluation
print("Best Accuracy:", best_accuracy)

y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))

# 6. Plot loss curve
plt.plot(best_model.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Loss Curve")
plt.show()

