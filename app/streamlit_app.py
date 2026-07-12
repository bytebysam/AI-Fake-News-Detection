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
    page_title="TruthLens AI",
    page_icon="🔍",
    layout="wide"
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 3rem;
    }

    .hero-title {
        font-size: 3.7rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }

    .model-card {
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .result-real {
        padding: 25px;
        border-radius: 14px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .result-fake {
        padding: 25px;
        border-radius: 14px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        font-size: 1.5rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<p class="hero-title">TruthLens AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="hero-subtitle">
    AI-powered fake news classification using Natural Language Processing
    and Machine Learning.
    </p>
    """,
    unsafe_allow_html=True
)


model, vectorizer = load_models()


left_column, right_column = st.columns([2, 1])


with left_column:

    st.subheader("Analyze a News Article")

    news_text = st.text_area(
        "News Article",
        placeholder="Paste the complete news article here...",
        height=350
    )

    analyze = st.button(
        "Analyze Article",
        type="primary",
        use_container_width=True
    )


with right_column:

    st.subheader("Model Information")

    st.markdown(
        """
        <div class="model-card">
        <b>Best Model</b><br>
        Random Forest
        <br><br>

        <b>Test Accuracy</b><br>
        95.58%
        <br><br>

        <b>Feature Extraction</b><br>
        TF-IDF — 5,000 Features
        <br><br>

        <b>Training Dataset</b><br>
        72,080 News Articles
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "TruthLens identifies linguistic patterns. "
        "It does not perform live internet fact-checking."
    )


if analyze:

    if not news_text.strip():

        st.warning(
            "Please enter a news article before starting the analysis."
        )

    else:

        with st.spinner("Analyzing linguistic patterns..."):

            cleaned_news = clean_text(news_text)

            news_vector = vectorizer.transform(
                [cleaned_news]
            )

            prediction = model.predict(
                news_vector
            )[0]

            probabilities = model.predict_proba(
                news_vector
            )[0]

            confidence = probabilities[prediction] * 100


        st.divider()

        st.subheader("Analysis Result")


        if prediction == 0:

            st.markdown(
                """
                <div class="result-fake">
                FAKE NEWS PATTERN DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-real">
                REAL NEWS PATTERN DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )


        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        st.caption(
            "The prediction represents the machine learning model's "
            "classification based on linguistic patterns learned from "
            "the WELFake dataset."
        )


st.divider()

st.caption(
    "TruthLens AI • NLP and Machine Learning Fake News Detection Project"
)