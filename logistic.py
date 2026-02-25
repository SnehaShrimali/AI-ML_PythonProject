from sklearn.linear_model import LogisticRegression
import numpy as np

# Data
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1])

# Model
model = LogisticRegression()
model.fit(X, y)

# Prediction
hours = np.array([[3.5]])
prediction = model.predict(hours)
probability = model.predict_proba(hours)

print("Pass/Fail:", prediction)
print("Probability:", probability)