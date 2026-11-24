import pandas as pd

df = pd.read_csv("data/processed/civicdex_cleaned.csv")

print("\n===== LANGUAGE DISTRIBUTION =====")
print(df["language_type"].value_counts())

print("\n===== RAW TEXT AND NORMALIZED TEXT =====")

for language in df["language_type"].unique():

    print(f"\n===== {language} =====")

    sample = df[df["language_type"] == language].head(5)

    for _, row in sample.iterrows():
        print("\nRAW:", row["raw_text"])
        print("NORMALIZED:", row["normalized_text"])

print("\n===== NORMALIZED TEXT LANGUAGE CHECK =====")

for language in df["language_type"].unique():

    print(f"\n===== {language} =====")

    sample = df[df["language_type"] == language].sample(
        min(5, len(df[df["language_type"] == language])),
        random_state=42
    )

    for _, row in sample.iterrows():
        print("RAW:", row["raw_text"])
        print("NORMALIZED:", row["normalized_text"])
        print("CATEGORY:", row["category"])
        print("-" * 50)

print("\n===== EMPTY TEXT CHECK =====")

print("Raw Text Empty:", df["raw_text"].isna().sum())
print("Normalized Text Empty:", df["normalized_text"].isna().sum())

print("\n===== EMPTY STRING CHECK =====")

print(
    "Raw Empty Strings:",
    (df["raw_text"].str.strip() == "").sum()
)

print(
    "Normalized Empty Strings:",
    (df["normalized_text"].str.strip() == "").sum()
)

print("\n===== TEXT LENGTH CHECK =====")

print("Raw Text Average Length:",
      df["raw_text"].str.len().mean())

print("Normalized Text Average Length:",
      df["normalized_text"].str.len().mean())