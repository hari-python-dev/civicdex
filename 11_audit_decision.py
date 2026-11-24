import pandas as pd

df = pd.read_csv(
    "data/processed/generic_complaint_audit.csv"
)

generic_phrases = [
    "இந்த பிரச்சனைக்கு உடனடி நடவடிக்கை எடுக்க வேண்டும்",
    "இந்த பிரச்சனைக்கு விரைவில் தீர்வு தரவும்",
    "இந்த பிரச்சனைக்கு உடனடி தீர்வு வழங்கவும்",
    "இந்த பிரச்சனைக்கு உடனடி தீர்வு வழங்க வேண்டும்",
    "இந்த பிரச்சனைக்கு உடனடி நடவடிக்கை தேவை",
    "இந்த பிரச்சனைக்கு உடனடி தீர்வு தேவை"
]

information_phrases = [
    "இந்த பிரச்சனைக்கு பொறுப்பு அதிகாரி யார்?",
    "இந்த பிரச்சனைக்கு யாரை தொடர்பு கொள்ள வேண்டும்?",
    "இந்த பிரச்சனைக்கு எங்கு தொடர்பு கொள்ள வேண்டும்?",
    "இந்த பிரச்சனைக்கு யாரை தொடர்பு கொள்ளலாம்?"
]

followup_phrases = [
    "already escalate செய்தும் பதில் இல்லை",
    "என் முந்தைய புகாருக்கு இன்னும் பதில் இல்லை",
    "இந்த பிரச்சனை இன்னும் சரி செய்யவில்லை என்றால் மேலதிகாரிக்கு தெரிவிப்பேன்",
    "மேலதிகாரிக்கு சென்றும் பதில் இல்லை"
]

def decide(text):

    text = str(text).strip()

    if text in generic_phrases:
        return "REVIEW_GENERIC"

    elif text in information_phrases:
        return "INFORMATION_REQUEST"

    elif text in followup_phrases:
        return "SEPARATE_FOLLOWUP"

    else:
        return "KEEP"


df["audit_decision"] = df["normalized_text"].apply(decide)

print("\n===== FINAL AUDIT RESULTS =====")

print(df[[
    "normalized_text",
    "category",
    "audit_decision"
]].to_string(index=False))

df.to_csv(
    "data/processed/generic_complaint_decisions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAudit decisions saved successfully!")

print(df["audit_decision"].value_counts())