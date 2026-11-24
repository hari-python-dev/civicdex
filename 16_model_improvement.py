
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load training and validation datasets only
train_df = pd.read_csv("data/processed/final_train.csv")
val_df = pd.read_csv("data/processed/final_validation.csv")

X_train = train_df["normalized_text"]
y_train = train_df["category"]

X_val = val_df["normalized_text"]
y_val = val_df["category"]

# Different model configurations to compare
experiments = [
    {"name": "Baseline", "ngram": (1, 2), "C": 1.0, "min_df": 1},
    {"name": "Unigram", "ngram": (1, 1), "C": 1.0, "min_df": 1},
    {"name": "Bigram", "ngram": (2, 2), "C": 1.0, "min_df": 1},
    {"name": "Lower C", "ngram": (1, 2), "C": 0.1, "min_df": 1},
    {"name": "Higher C", "ngram": (1, 2), "C": 2.0, "min_df": 1},
    {"name": "Min DF 2", "ngram": (1, 2), "C": 1.0, "min_df": 2},
]

results = []
best_model = None
best_vectorizer = None
best_f1 = -1

for exp in experiments:

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=exp["ngram"],
        min_df=exp["min_df"]
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=exp["C"],
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    train_predictions = model.predict(X_train_tfidf)
    val_predictions = model.predict(X_val_tfidf)

    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Experiment": exp["name"],
        "Training Accuracy": round(train_accuracy, 4),
        "Validation Accuracy": round(val_accuracy, 4),
        "Validation Macro F1": round(val_f1, 4)
    })

    print("\nCompleted:", exp["name"])

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model = model
        best_vectorizer = vectorizer
        best_predictions = val_predictions

# Display comparison
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

print("\n===== BEST MODEL VALIDATION REPORT =====")
print("Best Macro F1:", round(best_f1, 4))

print(classification_report(
    y_val,
    best_predictions,
    zero_division=0
))

# Save experiment results
results_df.to_csv(
    "data/processed/model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nModel comparison saved successfully!")
