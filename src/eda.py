import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/cleaned_news.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading cleaned dataset...")

data = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
data.info()

print("\nLabel distribution:")
print(data["label"].value_counts())

label_counts = data["label"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["Real News (0)", "Fake News (1)"], label_counts.values)
plt.title("Distribution of Real and Fake News")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")
plt.tight_layout()

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "label_distribution.png")
plt.savefig(OUTPUT_PATH)
plt.close()

print("\nEDA completed!")
print("Graph saved to:", OUTPUT_PATH)