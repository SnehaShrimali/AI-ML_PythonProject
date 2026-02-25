from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
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
# Random Forest Model 🌲🌲
# =========================
rf_model = RandomForestClassifier(
    n_estimators=5,       # number of trees
    criterion="gini",
    max_depth=3,
    random_state=42
)

rf_model.fit(X_train, y_train)

# =========================
# Prediction & Accuracy
# =========================
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# Feature Importance 📊
# =========================
importances = rf_model.feature_importances_
features = ["Age", "Income"]

plt.figure(figsize=(6, 4))
plt.bar(features, importances)
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.show()

# =========================
# Visualize ONE Tree from the Forest 🌳
# =========================
plt.figure(figsize=(12, 8))
plot_tree(
    rf_model.estimators_[0],   # first tree
    feature_names=features,
    class_names=["Not Buy", "Buy"],
    filled=True,
    rounded=True
)
plt.title("One Decision Tree from Random Forest")
plt.show()

# =========================
# Predict New Data
# =========================
new_person = [[30, 35]]  # Age=30, Income=35
result = rf_model.predict(new_person)

print("Prediction:", "Buy" if result[0] == 1 else "Not Buy")