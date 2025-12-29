# SMS Spam Detection Project 

A Machine Learning project that compares three variants of the Naive Bayes algorithm to classify SMS messages as Spam or Ham (legitimate).

##  Overview
This project builds a model to automatically detect spam messages. The workflow including:

1.  **Data Cleaning & Preprocessing:** Lowercasing, tokenization, removing special characters/stopwords, and stemming.
   
2.  **EDA (Exploratory Data Analysis):** Visualizing spam vs. ham frequency and word clouds.
   
3.  **Feature Extraction:** Converting text to numerical vectors using **TF-IDF**.
   
4.  **Model Training:** Training and comparing three different Naive Bayes classifiers:
    * **Gaussian Naive Bayes (GNB)**
    * **Multinomial Naive Bayes (MNB)**
    * **Bernoulli Naive Bayes (BNB)**

## Tech Stack

* **Python 3.x**
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `scikit-learn`, `wordcloud`.

## Project Structure
* `spam_detection.ipynb`: Jupyter Notebook containing data analysis, preprocessing, and training logic.
* `spam.csv`: The original SMS dataset.
* `vectorizer.pkl`: The trained TF-IDF vectorizer (essential for transforming new input).

## Installation & Usage

##Thanks for reading

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud,steamlit
