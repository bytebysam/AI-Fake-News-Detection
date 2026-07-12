# AI-Powered Fake News Detection Using Text Classification

## Abstract

The rapid growth of digital media and online news platforms has increased the spread of misinformation and fake news. Manual verification of large volumes of news articles is time-consuming and difficult. This project presents a machine learning-based fake news detection system using Natural Language Processing and text classification techniques. The WELFake dataset was used for model development. Text preprocessing, TF-IDF feature extraction, and four machine learning algorithms—K-Nearest Neighbors, Logistic Regression, Random Forest, and a Neural Network—were implemented and evaluated. Experimental results showed that the Random Forest classifier achieved the highest accuracy of 95.58%. The developed system demonstrates the effectiveness of machine learning and NLP techniques for automatic fake news classification.

## 1. Introduction

Fake news refers to false or misleading information presented in the form of legitimate news. The widespread use of social media and digital news platforms has made the rapid distribution of misinformation a significant problem.

Manual fact-checking requires considerable time and human effort. Artificial Intelligence and Machine Learning can assist in automatically identifying patterns associated with fake and real news articles.

The objective of this project is to design and implement a machine learning pipeline capable of classifying news articles as fake or real using Natural Language Processing and text classification techniques.

## 2. Dataset Description

The WELFake dataset was used in this project. The original dataset contained 72,134 news articles.

The dataset contained the following attributes:

- Title of the news article
- Text or content of the article
- Classification label

The class labels are represented as:

- 0 — Fake News
- 1 — Real News

The original label distribution was:

- Fake News: 35,028 articles
- Real News: 37,106 articles

The dataset contained 558 missing title values and 39 missing text values.

After preprocessing and removal of empty cleaned articles, 72,080 samples remained for model development.

## 3. Methodology

The proposed fake news detection system consists of the following stages:

1. Data collection
2. Data inspection
3. Text preprocessing
4. Exploratory data analysis
5. Feature extraction
6. Train-test splitting
7. Model training
8. Model evaluation
9. News classification

### 3.1 Text Preprocessing

The title and article text were combined to create a single textual feature.

The preprocessing pipeline performed the following operations:

- Conversion of text to lowercase
- Removal of HTML tags
- Removal of URLs
- Removal of punctuation and special characters
- Stopword removal
- Word lemmatization
- Removal of empty processed samples

The NLTK library was used for English stopwords and WordNet-based lemmatization.

### 3.2 Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) was used to transform textual news data into numerical features.

The TF-IDF vectorizer was configured with a maximum of 5,000 features.

The final TF-IDF feature matrix contained:

72,080 samples × 5,000 features.

### 3.3 Train-Test Split

The processed dataset was divided into training and testing sets using an 80:20 ratio.

- Training samples: 57,664
- Testing samples: 14,416

Stratified sampling was used to maintain the class distribution in both sets.

### 3.4 Machine Learning Models

Four classification algorithms were implemented.

#### K-Nearest Neighbors

K-Nearest Neighbors is a non-parametric machine learning algorithm that classifies samples based on the classes of nearby data points.

The value of K was configured as 5.

#### Logistic Regression

Logistic Regression is a parametric classification algorithm used to estimate the probability of a binary class.

It was applied to the TF-IDF feature vectors for fake and real news classification.

#### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

A total of 100 decision trees were used. The final prediction was determined using the combined decisions of the trees.

#### Neural Network

A Multi-Layer Perceptron classifier was implemented as a simple neural network.

The neural network contained one hidden layer with 100 neurons. Early stopping was applied to reduce unnecessary training and limit overfitting.

## 4. Evaluation Metrics

The models were evaluated using the following performance metrics:

### Accuracy

Accuracy represents the proportion of correctly classified samples.

### Precision

Precision measures the proportion of positive predictions that are correct.

### Recall

Recall measures the ability of the classifier to correctly identify positive samples.

### F1-Score

The F1-score is the harmonic mean of precision and recall.

Confusion matrices were also generated to analyze classification performance.

## 5. Results

The experimental results are shown below.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 95.58% | 94.57% | 96.98% | 95.76% |
| Neural Network | 94.76% | 94.55% | 95.30% | 94.93% |
| Logistic Regression | 94.58% | 94.32% | 95.18% | 94.75% |
| K-Nearest Neighbors | 71.37% | 65.29% | 94.56% | 77.25% |

Random Forest achieved the highest classification accuracy of 95.58%.

## 6. Discussion

The experimental results demonstrate that ensemble and linear classification techniques perform effectively on TF-IDF text features.

Random Forest achieved the highest accuracy and F1-score among the evaluated models. Its use of multiple decision trees enabled the classifier to identify complex patterns within the textual features.

Logistic Regression also achieved strong performance despite its relatively simple architecture. This demonstrates that TF-IDF features are highly suitable for linear text classification.

The Neural Network achieved an accuracy of 94.76%. Early stopping terminated training when validation performance stopped improving.

K-Nearest Neighbors achieved the lowest accuracy of 71.37%. KNN showed high recall but comparatively low precision. The results indicate that distance-based classification is less effective for high-dimensional sparse TF-IDF feature spaces.

## 7. Conclusion

This project successfully developed an AI-powered fake news detection system using Natural Language Processing and Machine Learning.

A complete text classification pipeline was implemented, including preprocessing, TF-IDF feature extraction, model training, and performance evaluation.

Among the four evaluated models, Random Forest achieved the best performance with an accuracy of 95.58%.

The results demonstrate that NLP and machine learning techniques can effectively identify linguistic patterns associated with fake and real news articles.

## 8. Limitations

The system identifies fake news based on linguistic patterns learned from the training dataset. It does not perform real-time fact-checking or verify information using external news sources.

Predictions may also be less reliable when only short headlines are provided instead of complete article text.

## 9. Future Scope

Future improvements may include:

- Transformer-based models such as BERT
- Real-time news source verification
- Multilingual fake news detection
- Explainable AI techniques
- Web-based deployment
- Integration with external fact-checking services

## 10. Appendix

The complete project source code includes:

- Dataset inspection
- Text preprocessing pipeline
- Exploratory data analysis
- TF-IDF feature engineering
- KNN classifier
- Logistic Regression classifier
- Random Forest classifier
- Neural Network classifier
- Model comparison system
- Fake news prediction system

The source code and project documentation are maintained using Git and GitHub.