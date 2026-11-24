import pandas as pd
import os

# Load original classifier dataset
df = pd.read_csv(
    "data/processed/civicdex_classifier_ready.csv"
)

# Remove Welfare
df = df[df["category"] != "welfare"].copy()

# Check required columns
required = [
    "raw_text",
    "category",
    "split",
    "language_type"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Preserve original text
df["raw_text"] = df["raw_text"].fillna("").astype(str)

# Create output folder
os.makedirs("data/processed", exist_ok=True)

# Create separate datasets
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]

# Save datasets
train.to_csv(
    "data/processed/raw_train.csv",
    index=False
)

val.to_csv(
    "data/processed/raw_validation.csv",
    index=False
)

test.to_csv(
    "data/processed/raw_test.csv",
    index=False
)

print("===== RAW TEXT SPLIT CREATED =====")

print("Training records:", len(train))
print("Validation records:", len(val))
print("Test records:", len(test))

print("\nLanguage distribution:")
print(
    df.groupby(["split", "language_type"])
    .size()
    .unstack(fill_value=0)
)

print("\nAll raw text datasets saved successfully!")