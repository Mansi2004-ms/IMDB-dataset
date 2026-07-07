import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------------------------
# Download NLTK Resources
# --------------------------------------------------
@st.cache_resource
def download_nltk_data():
    resources = [
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng"
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

download_nltk_data()

# --------------------------------------------------
# Load Model & Vectorizer
# --------------------------------------------------
@st.cache_resource
def load_resources():
    with open("logistic_regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


try:
    model, tfidf_vectorizer = load_resources()
except Exception as e:
    st.error(f"Error loading model files:\n{e}")
    st.stop()

# --------------------------------------------------
# NLP Setup
# --------------------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# Preprocessing Functions
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text


def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def get_wordnet_pos(word):
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    except:
        return wordnet.NOUN

    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def preprocess(text):
    text = clean_text(text)
    text = remove_stopwords(text)

    # No word_tokenize() → avoids punkt errors
    tokens = text.split()

    lemmas = [
        lemmatizer.lemmatize(token, get_wordnet_pos(token))
        for token in tokens
    ]

    return " ".join(lemmas)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("ℹ️ About")

st.sidebar.write("""
### IMDB Movie Review Sentiment Analyzer

This application predicts whether a movie review is:

- 😊 Positive
- 😞 Negative

Model Used:

- Logistic Regression
- TF-IDF Vectorizer
- NLP Text Preprocessing
""")

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analyzer")

st.write("Enter a movie review below and click **Analyze Sentiment**.")

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Example: This movie was fantastic. I really enjoyed it!"
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        cleaned_review = preprocess(review)

        vector = tfidf_vectorizer.transform([cleaned_review])

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

        st.subheader("Confidence")

        st.write(f"Positive: **{positive*100:.2f}%**")
        st.progress(float(positive))

        st.write(f"Negative: **{negative*100:.2f}%**")
        st.progress(float(negative))

        with st.expander("Processed Text"):
            st.write(cleaned_review)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built using Streamlit • NLP • TF-IDF • Logistic Regression")
