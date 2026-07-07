import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (only once)
# These should ideally be present in the deployment environment
# st.cache_resource ensures these run only once across rerun
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except nltk.downloader.DownloadError:
        nltk.download('averaged_perceptron_tagger')

download_nltk_data()

# Load the trained model and TF-IDF vectorizer
@st.cache_resource
def load_resources():
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, tfidf_vectorizer = load_resources()

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing functions (must be identical to training)
def clean_text(text):
    text = re.sub(r'<br />', '', text) # Remove HTML break tags
    text = text.lower() # Convert to lowercase
    return text

def remove_punctuation(text):
    text = re.sub(r'[\W_]+', ' ', text) # Remove punctuation and special characters, replace with space
    return text.strip()

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_words(words):
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]

def preprocess_review(text):
    text = clean_text(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    tokens = word_tokenize(text)
    lemmas = lemmatize_words(tokens)
    return ' '.join(lemmas)

# Streamlit app layout
st.title('IMDB Movie Review Sentiment Analyzer')
st.write('Enter a movie review below to predict its sentiment (Positive or Negative).')

# Text input
user_input = st.text_area('Movie Review:', height=150, placeholder='Type your movie review here...')

if st.button('Analyze Sentiment'):
    if user_input:
        # Preprocess the input text
        preprocessed_input = preprocess_review(user_input)
        
        # Transform text using the loaded TF-IDF vectorizer
        # Use `transform` not `fit_transform`!
        input_vector = tfidf_vectorizer.transform([preprocessed_input])
        
        # Make prediction
        prediction = model.predict(input_vector)
        prediction_proba = model.predict_proba(input_vector)
        
        # Display result
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        st.subheader(f'Predicted Sentiment: {sentiment}')
        
        # Display probabilities
        positive_proba = prediction_proba[0][1] * 100
        negative_proba = prediction_proba[0][0] * 100
        st.write(f'Confidence (Positive): {positive_proba:.2f}%')
        st.write(f'Confidence (Negative): {negative_proba:.2f}%')
    else:
        st.warning('Please enter a review to analyze.')
