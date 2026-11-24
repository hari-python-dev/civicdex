
import pandas as pd
import os

# Load original dataset
file_path = "data/raw/civicdex_main.csv"
df = pd.read_csv(file_path)

print("\n===== ORIGINAL DATASET =====")
print("Original Shape:", df.shape)

# Remove exact duplicate rows
before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\n===== EXACT DUPLICATE REMOVAL =====")
print("Duplicates Removed:", before - after)
print("Current Shape:", df.shape)

# Check duplicate raw complaints
print("\n===== RAW TEXT DUPLICATES =====")
print("Duplicate Raw Text:", df["raw_text"].duplicated().sum())

# Check duplicate normalized complaints
print("\n===== NORMALIZED TEXT DUPLICATES =====")
print("Duplicate Normalized Text:",
      df["normalized_text"].duplicated().sum())


print("\n===== DUPLICATE COMPLAINT DETAILS =====")

duplicate_data = df[
    df["normalized_text"].duplicated(keep=False)
]

duplicate_data = duplicate_data.sort_values("normalized_text")

print(
    duplicate_data[
        ["raw_text", "normalized_text", "category",
         "urgency", "split"]
    ].to_string(index=False)
)

print("\n===== DUPLICATE SPLIT CHECK =====")

split_check = (
    duplicate_data.groupby("normalized_text")["split"]
    .nunique()
)

print("Duplicates Across Different Splits:",
      (split_check > 1).sum())

print("\n===== DUPLICATE CATEGORY CHECK =====")

category_check = (
    duplicate_data.groupby("normalized_text")["category"]
    .nunique()
)

print("Duplicates With Different Categories:",
      (category_check > 1).sum())


print("\n===== URGENCY LABEL CONFLICT CHECK =====")

label_check = df.groupby("normalized_text").agg(
    total_records=("normalized_text", "size"),
    category_count=("category", "nunique"),
    urgency_count=("urgency", "nunique"),
    department_count=("department", "nunique"),
    split_count=("split", "nunique")
)

conflicts = label_check[
    (label_check["category_count"] > 1) |
    (label_check["urgency_count"] > 1) |
    (label_check["department_count"] > 1)
]

print("Total Conflicting Complaints:", len(conflicts))

if len(conflicts) > 0:
    print("\n===== CONFLICT DETAILS =====")

    conflict_texts = conflicts.index

    print(
        df[df["normalized_text"].isin(conflict_texts)]
        [["raw_text", "normalized_text", "category",
          "urgency", "department", "split"]]
        .sort_values("normalized_text")
        .to_string(index=False)
    )
else:
    print("No label conflicts found.")


# ===== MANUAL URGENCY CONFLICT RESOLUTION =====

print("\n===== RESOLVING URGENCY CONFLICTS =====")

# Roads complaint -> Medium
df.loc[
    df["normalized_text"] == "சாலை பழுது இன்னும் சரி செய்யப்படவில்லை",
    "urgency"
] = "medium"

# Water Supply complaint -> High
df.loc[
    df["normalized_text"] == "தண்ணீர் விநியோகம் சரியாக இல்லை",
    "urgency"
] = "high"

print("Roads urgency updated to: Medium")
print("Water Supply urgency updated to: High")

# Verify the changes
print("\n===== UPDATED URGENCY LABELS =====")

print(
    df[
        df["normalized_text"].isin([
            "சாலை பழுது இன்னும் சரி செய்யப்படவில்லை",
            "தண்ணீர் விநியோகம் சரியாக இல்லை"
        ])
    ][["raw_text", "category", "urgency", "split"]]
    .to_string(index=False)
)

# Check remaining label conflicts
label_check = df.groupby("normalized_text").agg(
    category_count=("category", "nunique"),
    urgency_count=("urgency", "nunique"),
    department_count=("department", "nunique")
)

conflicts = label_check[
    (label_check["category_count"] > 1) |
    (label_check["urgency_count"] > 1) |
    (label_check["department_count"] > 1)
]

print("\nRemaining Label Conflicts:", len(conflicts))


from sklearn.model_selection import train_test_split

print("\n===== FINAL DATA CLEANING =====")

# Stop if label conflicts still exist
if len(conflicts) > 0:
    raise ValueError("Resolve label conflicts before splitting!")

# Remove duplicate normalized complaints
df_clean = df.drop_duplicates(
    subset=["normalized_text"],
    keep="first"
).copy()

print("Original Records:", len(df))
print("Records After Duplicate Removal:", len(df_clean))
print("Duplicates Removed:", len(df) - len(df_clean))

# Remove old split information
df_clean = df_clean.drop(columns=["split"])

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.30,
    random_state=42,
    stratify=df_clean["category"]
)

# Second split: 15% validation, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["category"]
)

# Assign new split labels
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

# Combine datasets
final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# Create processed directory
os.makedirs("data/processed", exist_ok=True)

# Save datasets
final_df.to_csv(
    "data/processed/civicdex_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

train_df.to_csv(
    "data/processed/train.csv",
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    "data/processed/validation.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== FINAL DATA SPLIT =====")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
print("Total:", len(final_df))

print("\n===== CATEGORY DISTRIBUTION =====")
print(pd.crosstab(final_df["category"], final_df["split"]))

print("\n===== FINAL DUPLICATE CHECK =====")
print(
    "Duplicate Normalized Text:",
    final_df["normalized_text"].duplicated().sum()
)

print("\n===== FILES SAVED SUCCESSFULLY =====")
print("data/processed/civicdex_cleaned.csv")
print("data/processed/train.csv")
print("data/processed/validation.csv")
print("data/processed/test.csv")
