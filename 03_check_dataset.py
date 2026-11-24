
import pandas as pd

df = pd.read_csv("data/raw/civicdex_main.csv")

print("\n===== CATEGORY BY DATA SPLIT =====")
print(pd.crosstab(df["category"], df["split"]))

print("\n===== URGENCY BY DATA SPLIT =====")
print(pd.crosstab(df["urgency"], df["split"]))

print("\n===== DEPARTMENT AND CATEGORY MAPPING =====")
print(df.groupby("category")["department"].nunique())

print("\n===== SAMPLE COMPLAINTS =====")
print(
    df[["raw_text", "normalized_text", "english_gloss",
        "category", "department", "urgency", "split"]]
    .head(10)
    .to_string(index=False)
)

print("\n===== DUPLICATES ACROSS SPLITS =====")
duplicate_rows = df[df.duplicated("raw_text", keep=False)]

if len(duplicate_rows) > 0:
    print(duplicate_rows[["raw_text", "split"]].sort_values("raw_text").to_string(index=False))
else:
    print("No duplicate complaints found.")
