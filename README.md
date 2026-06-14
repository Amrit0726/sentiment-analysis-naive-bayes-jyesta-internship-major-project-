# Sentiment Analysis — IMDB Movie Reviews

A natural language processing project that classifies movie reviews as **Positive** or **Negative** using the Naive Bayes algorithm. Built as part of the **Jyesta Internship Training Program**.

---

##  Project Overview

The goal is to build a sentiment classifier that understands whether a given piece of text expresses a positive or negative opinion. The project covers the complete NLP pipeline — from raw text preprocessing to model evaluation.

---

##  Dataset

- **IMDB Movie Reviews** — 25,000 training + 25,000 test reviews
- Binary labels: `0 = Negative`, `1 = Positive`
- Loaded directly via `tensorflow.keras.datasets.imdb`

---

##  Models Implemented

| Model | Feature Extraction | Accuracy (approx.) |
|---|---|---|
| Multinomial Naive Bayes | Bag of Words | ~85–86% |
| Multinomial Naive Bayes | TF-IDF | ~86–87% |
| Bernoulli Naive Bayes | Bag of Words | ~84–85% |
| **Best (tuned α)** | **TF-IDF** | **~87%+** |

---

##  Tech Stack

- Python 3.10
- Scikit-learn
- NLTK
- TensorFlow / Keras (dataset loading only)
- NumPy, Pandas
- Matplotlib, Seaborn
- WordCloud

---

##  How to Run

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/sentiment-analysis-naive-bayes.git
cd sentiment-analysis-naive-bayes
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Open the notebook**
```bash
jupyter notebook Sentiment_Analysis_NaiveBayes.ipynb
```
Or open directly in [Google Colab](https://colab.research.google.com/) by uploading the `.ipynb` file.

---

##  Notebook Structure

```
1. Data Loading
2. Text Preprocessing (lowercase, punctuation removal, stopwords, stemming)
3. Feature Extraction (Bag of Words vs TF-IDF with bigrams)
4. Data Splitting (train / validation / test)
5. Model Training (Multinomial NB, Bernoulli NB)
6. Model Evaluation (accuracy, precision, recall, F1, confusion matrix)
7. Hyperparameter Tuning (Laplace smoothing α)
8. Top predictive words per class
9. Custom review prediction
```

---

##  Key Concepts Covered

- **Bag of Words** vs **TF-IDF** feature extraction
- **Bigrams** to capture phrases like *"not good"* or *"very bad"*
- **Laplace Smoothing** to handle unseen words
- **Multinomial NB** for frequency-based features
- **Bernoulli NB** for binary word presence features
- WordCloud visualisation of most common positive/negative words

---

##  Sample Prediction

```
Review   : This movie was absolutely fantastic! The acting was brilliant.
Sentiment: ✅ POSITIVE  (Confidence: 97.3%)

Review   : Terrible film. Boring plot and a complete waste of time.
Sentiment: ❌ NEGATIVE  (Confidence: 95.1%)
```

---

##  Author

**Amrit Noor Singh**  
B.Tech (Data Science) — NIT Jalandhar, Batch of 2028  
