import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

DATA_PATH = "data/raw/WELFake_Dataset.csv"
OUTPUT_PATH = "data/processed/cleaned_news.csv"

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


print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

data = data.drop(columns=["Unnamed: 0"])

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data["content"] = data["title"] + " " + data["text"]

print("Cleaning text... This may take a few minutes.")

data["cleaned_text"] = data["content"].apply(clean_text)

data = data[["cleaned_text", "label"]]

data.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing completed!")
print("Cleaned dataset saved to:", OUTPUT_PATH)
print("Final dataset shape:", data.shape)