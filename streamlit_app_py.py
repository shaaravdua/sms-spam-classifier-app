import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
nltk.download('punkt_tab')

# Ensure nltk_data folder exists
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Append folder to nltk path
nltk.data.path.append(nltk_data_dir)

# Download resources to that folder
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def load_model_and_vectorizer():
    import pickle
    with open("model (1).pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer (1).pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()


tfidf = pickle.load(open('vectorizer (1).pkl','rb'))
model = pickle.load(open('model (1).pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")