import pandas as pd

DATA_PATH = "data/raw/WELFake_Dataset.csv"

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

print("\nLabel distribution:")
print(data["label"].value_counts())