import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Load final model and vectorizer
model = joblib.load("models/civicdex_model.pkl")
tfidf = joblib.load("models/civicdex_tfidf.pkl")

# Load test dataset
test_df = pd.read_csv("data/processed/final_test.csv")

# Remove Welfare
test_df = test_df[test_df["category"] != "welfare"]

X_test = test_df["normalized_text"].astype(str)
y_test = test_df["category"]

# Transform test data
X_test_vectorized = tfidf.transform(X_test)

# Predictions
y_pred = model.predict(X_test_vectorized)

# Evaluation
print("===== FINAL 9 CATEGORY MODEL EVALUATION =====")

print("Test Records:", len(test_df))

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# Save predictions
test_df = test_df.copy()
test_df["predicted_category"] = y_pred

test_df.to_csv(
    "data/processed/final_test_predictions.csv",
    index=False
)

print("\nTest predictions saved successfully!")