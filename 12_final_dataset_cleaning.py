import pandas as pd

# Load original cleaned dataset
df = pd.read_csv("data/processed/civicdex_cleaned.csv")

# Load audit decisions
audit_df = pd.read_csv(
    "data/processed/generic_complaint_decisions.csv"
)

# Identify records that should not enter category classifier
exclude_texts = audit_df[
    audit_df["audit_decision"] != "KEEP"
]["normalized_text"].tolist()

# Remove only audited non-classifier records
final_df = df[
    ~df["normalized_text"].isin(exclude_texts)
].copy()

# Save final classifier dataset
final_df.to_csv(
    "data/processed/civicdex_classifier_ready.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== FINAL DATASET CLEANING =====")

print("Original records:", len(df))
print("Excluded records:", len(df) - len(final_df))
print("Final records:", len(final_df))

print("\n===== CATEGORY DISTRIBUTION =====")
print(final_df["category"].value_counts())

print("\nFinal classifier dataset saved successfully!")