import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/civicdex_cleaned.csv")

print("\n===== RAW vs NORMALIZED TEXT =====")

for category in ["roads", "welfare", "sanitation"]:

    print(f"\n===== {category.upper()} =====")

    sample = df[df["category"] == category].head(10)

    for _, row in sample.iterrows():

        print("\nOriginal:", row["raw_text"])
        print("Normalized:", row["normalized_text"])