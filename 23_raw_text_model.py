import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Load datasets
train_df = pd.read_csv("data/processed/raw_train.csv")
val_df = pd.read_csv("data/processed/raw_validation.csv")

# Original complaint text
X_train = train_df["raw_text"].fillna("").astype(str)
y_train = train_df["category"]

X_val = val_df["raw_text"].fillna("").astype(str)
y_val = val_df["category"]


# Word + Character TF-IDF
vectorizer = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=3000
        )
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=5000
        )
    )
])

# Transform text
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Train classifier
model = LogisticRegression(
    C=2,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_vec, y_train)

# Validation prediction
y_pred = model.predict(X_val_vec)

print("===== RAW TEXT MODEL RESULTS =====")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

print("\nAccuracy:", accuracy_score(y_val, y_pred))

print(
    "Macro F1:",
    f1_score(y_val, y_pred, average="macro")
)

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_val,
        y_pred,
        zero_division=0
    )
)


# Three language tests
test_complaints = [
    "சாலையில் அதிகமான பள்ளங்கள் உள்ளன",
    "There are many potholes on the road",
    "road la romba pothole iruku"
]

print("\n===== LANGUAGE TEST =====")

for complaint in test_complaints:

    complaint_vec = vectorizer.transform([complaint])

    prediction = model.predict(complaint_vec)[0]

    probabilities = model.predict_proba(complaint_vec)[0]

    confidence = max(probabilities) * 100

    print("\nComplaint:", complaint)
    print("Prediction:", prediction)
    print("Confidence:", round(confidence, 2), "%")