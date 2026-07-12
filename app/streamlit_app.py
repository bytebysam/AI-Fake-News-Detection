import os
import re
import joblib
import streamlit as st
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


@st.cache_resource
def load_models():
    model = joblib.load(
        os.path.join(MODEL_DIR, "random_forest.joblib")
    )

    vectorizer = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    )

    return model, vectorizer


st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("AI-Powered Fake News Detector")

st.write(
    "Enter a news article below to classify it as fake or real "
    "using Natural Language Processing and Machine Learning."
)

model, vectorizer = load_models()

news_text = st.text_area(
    "News Article",
    placeholder="Paste the complete news article here...",
    height=300
)

if st.button("Analyze News", type="primary"):

    if not news_text.strip():
        st.warning("Please enter a news article.")

    else:
        cleaned_news = clean_text(news_text)

        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)[0]

        probabilities = model.predict_proba(news_vector)[0]

        confidence = probabilities[prediction] * 100

        if prediction == 0:
            st.error("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "This system performs text classification based on patterns "
            "learned from the training dataset. It does not perform "
            "real-time fact-checking."
        )