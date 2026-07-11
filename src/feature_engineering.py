import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/cleaned_news.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading cleaned dataset...")

data = pd.read_csv(DATA_PATH)

X = data["cleaned_text"]
y = data["label"]

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

print("TF-IDF shape:", X_vectorized.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

joblib.dump(
    (X_train, X_test, y_train, y_test),
    os.path.join(MODEL_DIR, "train_test_data.joblib")
)

print("Feature engineering completed!")
print("TF-IDF vectorizer and train-test data saved.")