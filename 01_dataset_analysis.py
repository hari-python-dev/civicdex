import pandas as pd

# Load dataset
file_path = "data/raw/civicdex_main.csv"
df = pd.read_csv(file_path)

print("\n===== DATASET LOADED SUCCESSFULLY =====")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
df.info()