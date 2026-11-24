import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load final datasets
train_df = pd.read_csv("data/processed/final_train.csv")
val_df = pd.read_csv("data/processed/final_validation.csv")

# Input and target
X_train = train_df["normalized_text"]
y_train = train_df["category"]

X_val = val_df["normalized_text"]
y_val = val_df["category"]

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# Predictions
train_predictions = model.predict(X_train_tfidf)
val_predictions = model.predict(X_val_tfidf)

# Accuracy
print("\n===== FINAL MODEL RESULTS =====")

print(
    "Training Accuracy:",
    accuracy_score(y_train, train_predictions)
)

print(
    "Validation Accuracy:",
    accuracy_score(y_val, val_predictions)
)

# Classification report
print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_val,
        val_predictions,
        zero_division=0
    )
)

# Confusion matrix
print("\n===== CONFUSION MATRIX =====")

print(confusion_matrix(
    y_val,
    val_predictions,
    labels=model.classes_
))

print("\n===== CLASS LABELS =====")
print(model.classes_)

# Show wrong validation predictions

error_df = pd.DataFrame({
    "complaint": X_val,
    "actual": y_val,
    "predicted": val_predictions
})

wrong_predictions = error_df[
    error_df["actual"] != error_df["predicted"]
]

print("\n===== WRONG PREDICTIONS =====")

print(wrong_predictions.to_string(index=False))

print("\nTotal wrong predictions:", len(wrong_predictions))

# Save wrong predictions for manual review

wrong_predictions.to_csv(
    "data/processed/validation_errors.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nValidation errors saved successfully!")

# ==========================================
# 15. FINAL TEST DATASET EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load untouched test dataset
test_df = pd.read_csv("data/processed/final_test.csv")

X_test = test_df["normalized_text"]
y_test = test_df["category"]

# Transform test text using already-trained TF-IDF
X_test_tfidf = tfidf.transform(X_test)

# Predict using already-trained model
test_predictions = model.predict(X_test_tfidf)

# Test accuracy
test_accuracy = accuracy_score(y_test, test_predictions)

print("\n===== FINAL TEST RESULTS =====")
print("Test Records:", len(test_df))
print("Test Accuracy:", test_accuracy)

print("\n===== TEST CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

print("\n===== TEST CONFUSION MATRIX =====")
print(confusion_matrix(
    y_test,
    test_predictions,
    labels=model.classes_
))

print("\n===== TEST CLASS LABELS =====")
print(model.classes_)