#16
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print(df.head())

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Number of clusters
k = 5

# ---------------- K-MEANS ----------------

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

kmeans_labels = kmeans.fit_predict(X_scaled)

# Evaluation
print("\nK-Means Results")
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, kmeans_labels))

# ---------------- AGGLOMERATIVE ----------------

agg = AgglomerativeClustering(n_clusters=k, linkage='ward')

agg_labels = agg.fit_predict(X_scaled)

# Evaluation
print("\nAgglomerative Results")
print("Silhouette Score:", silhouette_score(X_scaled, agg_labels))

# Calculate within-cluster SSE for Agglomerative
agg_sse = 0

for i in range(k):
    cluster_points = X_scaled[agg_labels == i]
    centroid = cluster_points.mean(axis=0)
    agg_sse += np.sum((cluster_points - centroid) ** 2)

print("Inertia (SSE):", agg_sse)

# ---------------- VISUALIZATION ----------------

plt.figure(figsize=(12, 5))

# K-Means plot
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=kmeans_labels, cmap='rainbow')
plt.title("K-Means Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

# Agglomerative plot
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=agg_labels, cmap='rainbow')
plt.title("Agglomerative Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()

