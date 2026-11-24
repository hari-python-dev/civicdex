import pandas as pd

from sklearn.model_selection import train_test_split

# Load final classifier dataset
df = pd.read_csv(
    "data/processed/civicdex_classifier_ready.csv"
)

# Input and target
X = df["normalized_text"]
y = df["category"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Create DataFrames
train_df = pd.DataFrame({
    "normalized_text": X_train,
    "category": y_train
})

val_df = pd.DataFrame({
    "normalized_text": X_val,
    "category": y_val
})

test_df = pd.DataFrame({
    "normalized_text": X_test,
    "category": y_test
})

# Save new splits
train_df.to_csv(
    "data/processed/final_train.csv",
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    "data/processed/final_validation.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    "data/processed/final_test.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== FINAL DATA SPLIT =====")

print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Testing records:", len(test_df))

print("\n===== TRAIN CATEGORY DISTRIBUTION =====")
print(train_df["category"].value_counts())

print("\nFinal train, validation and test files saved!")