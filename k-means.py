import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate sample clustered data (3 groups like your image)
np.random.seed(42)

cluster1 = np.random.randn(150, 2) + np.array([-5, 3])
cluster2 = np.random.randn(150, 2) + np.array([0, 9])
cluster3 = np.random.randn(150, 2) + np.array([5, -4])

X = np.vstack((cluster1, cluster2, cluster3))

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Plot clusters
plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1],
            c='red', marker='^', s=200)

plt.title("K-Means Clustering Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()