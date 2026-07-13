import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/cleaned_news.csv"
MODEL_DIR = "models"

EMBEDDING_DIMENSIONS = 100

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading cleaned dataset...")

data = pd.read_csv(DATA_PATH)
data = data.dropna(subset=["cleaned_text"])

X = data["cleaned_text"]
y = data["label"]

print("Creating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

print("Creating dense document embeddings using LSA...")

embedding_model = TruncatedSVD(
    n_components=EMBEDDING_DIMENSIONS,
    random_state=42
)

X_embeddings = embedding_model.fit_transform(X_tfidf)

print("Embedding shape:", X_embeddings.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "embedding_tfidf_vectorizer.joblib")
)

joblib.dump(
    embedding_model,
    os.path.join(MODEL_DIR, "lsa_embedding_model.joblib")
)

joblib.dump(
    (X_train, X_test, y_train, y_test),
    os.path.join(
        MODEL_DIR,
        "embedding_train_test_data.joblib"
    )
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("Embedding feature engineering completed!")
print("LSA embedding model and train-test data saved.")