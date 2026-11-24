
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Load datasets
train_df = pd.read_csv("data/processed/final_train.csv")
val_df = pd.read_csv("data/processed/final_validation.csv")

# Remove welfare from ML datasets
train_df = train_df[train_df["category"] != "welfare"].copy()
val_df = val_df[val_df["category"] != "welfare"].copy()

print("===== DATASET INFORMATION =====")
print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Training categories:", train_df["category"].nunique())

# Prepare data
X_train = train_df["normalized_text"]
y_train = train_df["category"]

X_val = val_df["normalized_text"]
y_val = val_df["category"]

# Experiment configurations
experiments = [
    {"name": "Baseline", "C": 1.0, "ngram": (1, 2), "min_df": 1},
    {"name": "Unigram", "C": 1.0, "ngram": (1, 1), "min_df": 1},
    {"name": "Bigram", "C": 1.0, "ngram": (2, 2), "min_df": 1},
    {"name": "Lower C", "C": 0.1, "ngram": (1, 2), "min_df": 1},
    {"name": "Higher C", "C": 2.0, "ngram": (1, 2), "min_df": 1},
    {"name": "Min DF 2", "C": 1.0, "ngram": (1, 2), "min_df": 2}
]

results = []

best_f1 = -1
best_model_name = None
best_predictions = None

for exp in experiments:

    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=exp["ngram"],
        min_df=exp["min_df"]
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=exp["C"],
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    train_predictions = model.predict(X_train_tfidf)
    val_predictions = model.predict(X_val_tfidf)

    train_acc = accuracy_score(y_train, train_predictions)
    val_acc = accuracy_score(y_val, val_predictions)

    val_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Experiment": exp["name"],
        "Training Accuracy": round(train_acc, 4),
        "Validation Accuracy": round(val_acc, 4),
        "Validation Macro F1": round(val_f1, 4)
    })

    print("Completed:", exp["name"])

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_name = exp["name"]
        best_predictions = val_predictions

# Results comparison
results_df = pd.DataFrame(results)

print("\n===== 9 CATEGORY MODEL COMPARISON =====")
print(results_df.to_string(index=False))

print("\n===== BEST MODEL =====")
print("Best Configuration:", best_model_name)
print("Best Macro F1:", round(best_f1, 4))

print("\n===== BEST MODEL CLASSIFICATION REPORT =====")
print(classification_report(
    y_val,
    best_predictions,
    zero_division=0
))

# Save comparison
results_df.to_csv(
    "data/processed/model_comparison_9_categories.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n9 category model comparison saved successfully!")
