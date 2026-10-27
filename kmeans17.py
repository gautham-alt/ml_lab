import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load dataset
data = load_digits()

X = data.data

# 2. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Different K values
for k in [5, 8, 10, 12, 15]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    labels = model.labels_

    print("\nK =", k)
    print("Inertia:", model.inertia_)

    score = silhouette_score(
        X_scaled, labels,
        sample_size=1000,
        random_state=42
    )

    print("Silhouette Score:", score)

# 4. Elbow Method
inertia = []

for k in range(2, 16):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.plot(range(2, 16), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
