import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    sep=';'
)

# 2. Features and target
X = df.drop('quality', axis=1)
y = df['quality']

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# 4. Different MLP architectures
architectures = [
    (50,),
    (100,),
    (50, 30),
    (100, 50),
    (100, 50, 25)
]

for layers in architectures:

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=layers,
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nHidden Layers:", layers)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
