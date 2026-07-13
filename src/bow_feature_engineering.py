import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/cleaned_news.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading cleaned dataset...")

data = pd.read_csv(DATA_PATH)

data = data.dropna(subset=["cleaned_text"])

X = data["cleaned_text"]
y = data["label"]

print("Creating Bag-of-Words features...")

vectorizer = CountVectorizer(
    max_features=5000
)

X_bow = vectorizer.fit_transform(X)

print("Bag-of-Words shape:", X_bow.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_bow,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "bow_vectorizer.joblib")
)

joblib.dump(
    (X_train, X_test, y_train, y_test),
    os.path.join(MODEL_DIR, "bow_train_test_data.joblib")
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("Bag-of-Words feature engineering completed!")
print("Vectorizer and train-test data saved.")