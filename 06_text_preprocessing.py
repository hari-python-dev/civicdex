import pandas as pd
import re
import os

# Load cleaned dataset
df = pd.read_csv("data/processed/civicdex_cleaned.csv")

print("\n===== TEXT PREPROCESSING =====")

# Function to clean text
def clean_text(text):

    text = str(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


# Apply cleaning
df["model_text"] = df["normalized_text"].apply(clean_text)

# Check results
print("\n===== BEFORE AND AFTER =====")

for i in range(5):

    print("\nOriginal:", df["normalized_text"].iloc[i])
    print("Cleaned:", df["model_text"].iloc[i])


# Check empty values
print("\n===== FINAL CHECK =====")
print("Empty Model Text:", (df["model_text"].str.strip() == "").sum())

# Save processed dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/civicdex_model_ready.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nModel Ready Dataset Saved Successfully!")