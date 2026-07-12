# AI-Powered Fake News Detection

A Natural Language Processing and Machine Learning project for classifying news articles as **Fake** or **Real** using text classification techniques.

## Overview

The rapid growth of digital media has increased the spread of misinformation and fake news. This project implements a complete machine learning pipeline to automatically classify news articles using Natural Language Processing.

Four machine learning models were trained and compared:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest
- Neural Network

Random Forest achieved the best accuracy of **95.58%**.

## Dataset

The project uses the WELFake dataset containing more than 72,000 news articles.

Dataset features:

- Title
- Article text
- Classification label

Labels:

- `0` - Fake News
- `1` - Real News

After preprocessing, **72,080 articles** were used for model development.

The dataset is not included in this repository because of its large file size.

## Machine Learning Pipeline

1. Dataset collection
2. Data inspection
3. Text preprocessing
4. Exploratory Data Analysis (EDA)
5. TF-IDF feature extraction
6. Train-test splitting
7. Model training
8. Model evaluation
9. Fake news prediction

## Text Preprocessing

The preprocessing pipeline includes:

- Lowercase conversion
- HTML tag removal
- URL removal
- Special character removal
- Stopword removal
- Word lemmatization
- Empty sample removal

## Feature Engineering

TF-IDF was used to transform news text into numerical features.

- Maximum TF-IDF features: `5000`
- Training samples: `57,664`
- Testing samples: `14,416`

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 95.58% | 94.57% | 96.98% | 95.76% |
| Neural Network | 94.76% | 94.55% | 95.30% | 94.93% |
| Logistic Regression | 94.58% | 94.32% | 95.18% | 94.75% |
| KNN | 71.37% | 65.29% | 94.56% | 77.25% |

## Best Model

Random Forest achieved the highest classification accuracy of **95.58%**.

The ensemble model effectively identified patterns in the high-dimensional TF-IDF feature space.

## Project Structure

```text
AI-Fake-News-Detection/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── data_inspection.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_knn.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_neural_network.py
│   ├── compare_models.py
│   └── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/bytebysam/AI-Fake-News-Detection.git
```

Enter the project directory:

```bash
cd AI-Fake-News-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run preprocessing:

```bash
python src/preprocessing.py
```

Generate TF-IDF features:

```bash
python src/feature_engineering.py
```

Train the models using the training scripts available in the `src` directory.

Run the fake news predictor:

```bash
python src/predict.py
```

## Limitations

The system performs text classification based on linguistic patterns learned from the training dataset.

It does not perform live internet fact-checking or verify claims using external news sources.

## Future Scope

- Transformer models such as BERT
- Real-time fact-checking
- Multilingual fake news detection
- Explainable AI
- External fact-checking API integration
- Web deployment

## Author

Developed by bytebysam.

## License

This project is developed for educational and research purposes.