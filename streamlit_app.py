import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -------------------------------
# Download NLTK Data
# -------------------------------
@st.cache_resource
def download_nltk():
    packages = [
        "stopwords",
        "punkt",
        "wordnet",
        "averaged_perceptron_tagger",
        "omw-1.4"
    ]

    for package in packages:
        try:
            nltk.data.find(package)
        except:
            nltk.download(package)

download_nltk()

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


try:
    model, tfidf = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------
# NLP Objects
# -------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Preprocessing Functions
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text


def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def preprocess(text):
    text = clean_text(text)
    text = remove_stopwords(text)

    tokens = word_tokenize(text)

    lemmas = [
        lemmatizer.lemmatize(token, get_wordnet_pos(token))
        for token in tokens
    ]

    return " ".join(lemmas)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🎬 About")

st.sidebar.info(
"""
This application predicts whether an IMDB movie review is:

✅ Positive

❌ Negative

using:

• TF-IDF Vectorizer

• Logistic Regression

• Natural Language Processing
"""
)

st.sidebar.success("Model Accuracy: 89%")

# -------------------------------
# Main UI
# -------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analyzer")

st.write(
"Enter a movie review below and click **Analyze**."
)

# Example Reviews

examples = {
    "Positive Example":
    "This movie was absolutely amazing. I loved every scene.",

    "Negative Example":
    "Worst movie ever. Waste of time and money."
}

choice = st.selectbox(
    "Choose an example (optional)",
    ["None"] + list(examples.keys())
)

default_text = ""

if choice != "None":
    default_text = examples[choice]

review = st.text_area(
    "Movie Review",
    value=default_text,
    height=180
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        cleaned = preprocess(review)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        positive = probability[1]
        negative = probability[0]

        st.divider()

        if prediction == 1:
            st.success("😊 Positive Review")
            st.balloons()
        else:
            st.error("😞 Negative Review")

        st.subheader("Prediction Confidence")

        st.write(f"Positive : **{positive*100:.2f}%**")
        st.progress(float(positive))

        st.write(f"Negative : **{negative*100:.2f}%**")
        st.progress(float(negative))

        st.subheader("Processed Review")

        st.code(cleaned)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit, NLP, TF-IDF and Logistic Regression"
)
