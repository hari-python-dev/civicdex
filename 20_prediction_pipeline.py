
import joblib

# Load trained model
model = joblib.load("models/civicdex_raw_model.pkl")
tfidf = joblib.load("models/civicdex_raw_vectorizer.pkl")


# Department mapping
department_mapping = {
    "certificates": "Revenue Department",
    "drainage": "Drainage Department",
    "electricity": "Electricity Department",
    "public_health": "Public Health Department",
    "roads": "Roads Department",
    "sanitation": "Sanitation Department",
    "street_lighting": "Electricity Department",
    "transport": "Transport Department",
    "water_supply": "Water Supply Department",
    "welfare": "Welfare Department"
}


# Initial confidence threshold
CONFIDENCE_THRESHOLD = 50


def predict_complaint(complaint_text, selected_category=None):

    # Welfare bypass
    if selected_category == "welfare":
        return {
            "category": "welfare",
            "department": "Welfare Department",
            "confidence": None,
            "routing": "Direct Welfare Routing",
            "status": "Direct Routing"
        }

    # Convert raw complaint into TF-IDF features
    text_vector = tfidf.transform([complaint_text])

    # Predict category
    prediction = model.predict(text_vector)[0]

    # Get confidence score
    probabilities = model.predict_proba(text_vector)[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    # Confidence-based review
    if confidence < CONFIDENCE_THRESHOLD:
        status = "Manual Review Required"
    else:
        status = "Prediction Available"

    return {
        "category": prediction,
        "department": department_mapping[prediction],
        "confidence": confidence,
        "routing": "ML Classification",
        "status": status
    }


# Test complaint
complaint = "road la romba pothole iruku"

result = predict_complaint(complaint)

print("\n===== CIVICDEX PREDICTION RESULT =====")
print("Category:", result["category"])
print("Department:", result["department"])
print("Confidence:", result["confidence"], "%")
print("Routing:", result["routing"])
print("Status:", result["status"])


# Test Welfare bypass
welfare_result = predict_complaint(
    "ஓய்வூதியம் கிடைக்கவில்லை",
    selected_category="welfare"
)

print("\n===== WELFARE ROUTING TEST =====")
print("Category:", welfare_result["category"])
print("Department:", welfare_result["department"])
print("Routing:", welfare_result["routing"])
print("Status:", welfare_result["status"])
