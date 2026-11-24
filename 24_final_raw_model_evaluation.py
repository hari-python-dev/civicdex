import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Load datasets
train = pd.read_csv("data/processed/raw_train.csv")
val = pd.read_csv("data/processed/raw_validation.csv")
test = pd.read_csv("data/processed/raw_test.csv")

# Combine training and validation
train_data = pd.concat([train, val], ignore_index=True)

X_train = train_data["raw_text"].fillna("").astype(str)
y_train = train_data["category"]

X_test = test["raw_text"].fillna("").astype(str)
y_test = test["category"]

# Word + Character TF-IDF
vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=3000
    )),
    ("char", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=5000
    ))
])

# Transform
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train final model
model = LogisticRegression(
    C=2,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_vec, y_train)

# Test evaluation
predictions = model.predict(X_test_vec)

print("===== FINAL RAW MODEL TEST =====")
print("Training records:", len(X_train))
print("Test records:", len(X_test))

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("Macro F1:", f1_score(y_test, predictions, average="macro"))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, predictions, zero_division=0))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/civicdex_raw_model.pkl")
joblib.dump(vectorizer, "models/civicdex_raw_vectorizer.pkl")

print("\nFinal raw model saved successfully!")