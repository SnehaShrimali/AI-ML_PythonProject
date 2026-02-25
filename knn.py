import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Sample dataset (2D points)
X = np.array([
    [1, 2], [2, 3], [3, 3],
    [6, 5], [7, 7], [8, 6]
])

# Class labels
y = np.array([0, 0, 0, 1, 1, 1])

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1])
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1])

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Classification Example (k=3)")
plt.show()