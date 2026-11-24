import pandas as pd

df = pd.read_csv("data/processed/civicdex_cleaned.csv")

keywords = [
    "இந்த பிரச்சனை",
    "உடனடி நடவடிக்கை",
    "உடனடி தீர்வு",
    "விரைவில் தீர்வு",
    "பதில் இல்லை",
    "மேலதிகாரி",
    "யாரை தொடர்பு",
    "எங்கு தொடர்பு"
]

mask = df["normalized_text"].astype(str).apply(
    lambda text: any(word in text for word in keywords)
)

generic_df = df[mask].copy()

print("Total records:", len(df))
print("Flagged records:", len(generic_df))

print("\n===== AUDIT RECORDS =====")

print(
    generic_df[
        ["raw_text", "normalized_text", "category", "urgency"]
    ].to_string(index=False)
)

generic_df.to_csv(
    "data/processed/generic_complaint_audit.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAudit file saved successfully!")