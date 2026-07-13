# 🎬 IMDb Movie Review Sentiment Analyzer

An AI-powered web application that analyzes IMDb movie reviews and predicts whether the sentiment is **Positive 😊** or **Negative 😞** using Natural Language Processing (NLP) and Machine Learning. The application is built with Python and deployed using Streamlit.

---

## 📌 Project Overview

The IMDb Movie Review Sentiment Analyzer uses a trained Machine Learning model to classify movie reviews based on their sentiment. Users can enter any movie review, and the application predicts whether the review expresses a positive or negative opinion.

This project demonstrates the practical implementation of NLP techniques such as text preprocessing, tokenization, TF-IDF vectorization, and sentiment classification.

---

## 🚀 Features

- Predicts Positive or Negative sentiment
- User-friendly Streamlit interface
- Real-time sentiment analysis
- Text preprocessing for improved accuracy
- Machine Learning-based classification
- Easy to deploy on Streamlit Cloud

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Joblib
- TF-IDF Vectorizer

---

## 📂 Project Structure

```
IMDb-Movie-Review-Sentiment-Analyzer/
│── app.py
│── train_model.py
│── imdb_dataset.csv
│── sentiment_model.pkl
│── tfidf_vectorizer.pkl
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 📊 Dataset

Dataset: IMDb Movie Reviews Dataset

The dataset contains thousands of movie reviews labeled as:

- Positive
- Negative

The model is trained using TF-IDF vectorization and a Machine Learning classifier.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/imdb-movie-review-sentiment-analyzer.git
```

### Move into the project folder

```bash
cd imdb-movie-review-sentiment-analyzer
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧠 Machine Learning Workflow

1. Load IMDb dataset
2. Clean and preprocess text
3. Remove stopwords
4. Tokenize text
5. Convert text using TF-IDF Vectorizer
6. Train Machine Learning model
7. Save trained model using Joblib
8. Predict sentiment for user reviews

---

## 📷 Application Preview

- Enter a movie review
- Click **Analyze Sentiment**
- View the predicted sentiment (Positive or Negative)

---

## 🎯 Future Improvements

- Multi-class emotion detection
- Confidence score for predictions
- Deep Learning (LSTM/BERT) implementation
- Movie recommendation integration
- Review visualization dashboard
- Support for multiple languages

---

## 📈 Learning Outcomes

Through this project, you will learn:

- Natural Language Processing (NLP)
- Text preprocessing
- TF-IDF Vectorization
- Sentiment Analysis
- Machine Learning Classification
- Streamlit Deployment
- Model Serialization with Joblib

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Mansi Suryawanshi**

B.Tech Electronics & Telecommunication Engineering

Government College of Engineering, Nagpur

---
