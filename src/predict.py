import os
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

MODEL_DIR = "models"

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


print("Loading model...")

model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.joblib")
)

vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
)

news = input("\nEnter news article text:\n")

cleaned_news = clean_text(news)

news_vector = vectorizer.transform([cleaned_news])

prediction = model.predict(news_vector)[0]

if prediction == 0:
    print("\nPrediction: FAKE NEWS")
else:
    print("\nPrediction: REAL NEWS")