import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")
test_df = pd.read_csv("data/processed/test.csv")

print("===== DATASET LOADED =====")

print("Training Records:", len(train_df))
print("Validation Records:", len(val_df))
print("Testing Records:", len(test_df))

print("\n===== AVAILABLE COLUMNS =====")
print(train_df.columns.tolist())

# Select input and output columns

X_train = train_df["normalized_text"]
y_train = train_df["category"]

X_val = val_df["normalized_text"]
y_val = val_df["category"]

X_test = test_df["normalized_text"]
y_test = test_df["category"]


# Convert text into numerical features

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_val_tfidf = tfidf.transform(X_val)

X_test_tfidf = tfidf.transform(X_test)


print("\n===== TF-IDF VECTORIZATION =====")

print("Training Shape:", X_train_tfidf.shape)
print("Validation Shape:", X_val_tfidf.shape)
print("Testing Shape:", X_test_tfidf.shape)

print("\nTF-IDF Vocabulary Size:", len(tfidf.vocabulary_))

# Train Logistic Regression Model

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\n===== MODEL TRAINING COMPLETED =====")

print("Algorithm: Logistic Regression")
print("Training Accuracy:",
      model.score(X_train_tfidf, y_train))

print("Validation Accuracy:",
      model.score(X_val_tfidf, y_val))

# Validation Prediction

y_val_pred = model.predict(X_val_tfidf)

print("\n===== VALIDATION CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_val,
        y_val_pred,
        zero_division=0
    )
)

print("\n===== CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_val,
        y_val_pred,
        labels=model.classes_
    )
)

print("\n===== CATEGORY LABEL ORDER =====")
print(model.classes_)

# Validation Error Analysis

error_df = val_df.copy()

error_df["actual_category"] = y_val.values
error_df["predicted_category"] = y_val_pred

errors = error_df[
    error_df["actual_category"] != error_df["predicted_category"]
]

print("\n===== WRONG PREDICTIONS =====")

print(
    errors[
        ["normalized_text", "actual_category", "predicted_category"]
    ].to_string(index=False)
)

print("\nTotal Wrong Predictions:", len(errors))

# Category-wise Dataset Analysis

print("\n===== TRAINING CATEGORY COUNTS =====")

print(
    train_df["category"].value_counts().sort_index()
)

print("\n===== VALIDATION CATEGORY COUNTS =====")

print(
    val_df["category"].value_counts().sort_index()
)

print("\n===== TEST CATEGORY COUNTS =====")

print(
    test_df["category"].value_counts().sort_index()
)

# Inspect Roads and Welfare Training Data

for category_name in ["roads", "welfare"]:

    print(f"\n===== {category_name.upper()} COMPLAINTS =====")

    category_data = train_df[
        train_df["category"] == category_name
    ]

    for index, row in category_data.iterrows():

        print(
            f"{row['normalized_text']} | "
            f"Department: {row['department']}"
        )