import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
train_df = pd.read_csv("data/processed/final_train.csv")
val_df = pd.read_csv("data/processed/final_validation.csv")

# Combine training and validation
data = pd.concat([train_df, val_df], ignore_index=True)

# Remove Welfare category
data = data[data["category"] != "welfare"]

X = data["normalized_text"].astype(str)
y = data["category"]

# Best configuration: Higher C
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X_vectorized = tfidf.fit_transform(X)

model = LogisticRegression(
    C=2,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_vectorized, y)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "models/civicdex_model.pkl")
joblib.dump(tfidf, "models/civicdex_tfidf.pkl")

print("===== FINAL MODEL TRAINING COMPLETED =====")
print("Training records:", len(data))
print("Categories:", model.classes_)
print("Model saved successfully!")
print("Vectorizer saved successfully!")