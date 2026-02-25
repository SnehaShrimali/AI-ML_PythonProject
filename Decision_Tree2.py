import numpy as np

class DecisionTree:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y):
        # Choose random feature
        self.feature = np.random.randint(X.shape[1])
        self.threshold = np.mean(X[:, self.feature])

        left = y[X[:, self.feature] <= self.threshold]
        right = y[X[:, self.feature] > self.threshold]

        self.left_class = np.bincount(left).argmax()
        self.right_class = np.bincount(right).argmax()

    def predict(self, X):
        return np.where(
            X[:, self.feature] <= self.threshold,
            self.left_class,
            self.right_class
        )

# Sample data
X = np.array([
    [5.1, 3.5],
    [4.9, 3.0],
    [6.2, 3.4],
    [5.9, 3.0]
])
y = np.array([0, 0, 1, 1])

# Train
dt = DecisionTree()
dt.fit(X, y)

# Predict
predictions = dt.predict(X)
print("Predictions:", predictions)