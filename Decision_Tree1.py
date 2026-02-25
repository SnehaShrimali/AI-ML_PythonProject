from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Dataset
# =========================
X = np.array([
    [22, 20],
    [25, 25],
    [35, 40],
    [45, 60],
    [52, 80],
    [23, 22]
])

y = np.array([0, 0, 1, 1, 1, 0])

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# Model
# =========================
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Prediction & Accuracy
# =========================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# Visualization 🌳
# =========================
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=["Age", "Income"],
    class_names=["Not Buy", "Buy"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Visualization")
plt.show()

# =========================
# Predict New Data (Optional)
# =========================
new_person = [[30, 35]]  # Age=30, Income=35
result = model.predict(new_person)

print("Prediction:", "Buy" if result[0] == 1 else "Not Buy")