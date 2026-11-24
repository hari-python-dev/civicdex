
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/civicdex_main.csv")

print("\n===== CATEGORY DISTRIBUTION =====")
print(df["category"].value_counts())

print("\n===== DEPARTMENT DISTRIBUTION =====")
print(df["department"].value_counts())

print("\n===== URGENCY DISTRIBUTION =====")
print(df["urgency"].value_counts())

print("\n===== LANGUAGE DISTRIBUTION =====")
print(df["language_type"].value_counts())

print("\n===== DATA SOURCE DISTRIBUTION =====")
print(df["source_type"].value_counts())

print("\n===== TRAIN / TEST SPLIT =====")
print(df["split"].value_counts())

print("\n===== DUPLICATE COMPLAINTS =====")
print(df["raw_text"].duplicated().sum())

print("\n===== UNIQUE VALUES =====")
print("Categories:", df["category"].nunique())
print("Departments:", df["department"].nunique())
print("Urgency levels:", df["urgency"].nunique())

print("\n===== CATEGORY COUNTS =====")
print(df["category"].value_counts().to_string())

print("\n===== DEPARTMENT COUNTS =====")
print(df["department"].value_counts().to_string())

print("\n===== URGENCY COUNTS =====")
print(df["urgency"].value_counts().to_string())

print("\n===== LANGUAGE COUNTS =====")
print(df["language_type"].value_counts().to_string())

print("\n===== SOURCE TYPE COUNTS =====")
print(df["source_type"].value_counts().to_string())

print("\n===== TRAIN TEST COUNTS =====")
print(df["split"].value_counts().to_string())