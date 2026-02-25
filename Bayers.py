from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = [
    "free win offer",
    "free offer",
    "win money now",
    "meeting tomorrow",
    "project discussion"
]

labels = [1, 1, 1, 0, 0]  # 1 = Spam, 0 = Not Spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

test_email = vectorizer.transform(["free win"])
prediction = model.predict(test_email)

print("Spam" if prediction[0] == 1 else "Not Spam")