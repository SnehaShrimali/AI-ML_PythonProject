import numpy as np

# Simple Decision Stump
class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.feature = np.random.randint(n_features)
        self.threshold = np.mean(X[:, self.feature])

        left = y[X[:, self.feature] <= self.threshold]
        right = y[X[:, self.feature] > self.threshold]

        self.left_class = np.bincount(left).argmax()
        self.right_class = np.bincount(right).argmax()

    def predict(self, X):
        predictions = np.where(
            X[:, self.feature] <= self.threshold,
            self.left_class,
            self.right_class
        )
        return predictions


# Random Forest
class RandomForest:
    def __init__(self, n_trees=10):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionStump()
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=tree_preds
        )


# Sample Data
X = np.array([
    [5.1, 3.5],
    [4.9, 3.0],
    [6.2, 3.4],
    [5.9, 3.0]
])
y = np.array([0, 0, 1, 1])

# Train Model
rf = RandomForest(n_trees=5)
rf.fit(X, y)

# Predict
predictions = rf.predict(X)
print("Predictions:", predictions)