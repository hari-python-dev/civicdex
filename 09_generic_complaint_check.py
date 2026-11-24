import pandas as pd

df = pd.read_csv("data/processed/civicdex_cleaned.csv")

keywords = [
    "இந்த பிரச்சனை",
    "உடனடி நடவடிக்கை",
    "உடனடி தீர்வு",
    "விரைவில் தீர்வு",
    "பதில் இல்லை",
    "மேலதிகாரி"
]

print("\n===== GENERIC COMPLAINTS =====")

for _, row in df.iterrows():

    text = str(row["normalized_text"])

    if any(word in text for word in keywords):

        print("\nCategory:", row["category"])
        print("Complaint:", text)