import pickle 

from sklearn.feature_extraction.text import TfidfTransformer

import nltk 
from nltk.stem import  WordNetLemmatizer

import tensorflow as tf 
from tensorflow.keras.models import load_model  
from scipy.sparse import hstack 

import streamlit as st

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

snowball = SnowballStemmer("english")
lemmetiz = WordNetLemmatizer()

def preprocessing(para):
    # take array of string and applies stem and lemmeization to it 
    words = para.split()

    final = []
    for word in words: 
        
        # word = snowball.stem(word)
        word = lemmetiz.lemmatize(word, 'v')
        final.append(word)

    final_string = " ".join(final)
    return final_string

with open("tfidf_subject.pkl", "rb") as f:
    subject_vectorizer = pickle.load(f)

with open("tfidf_body.pkl", "rb") as f:
    body_vectorizer = pickle.load(f)

model = load_model("spam_classifer_NN.keras")

def predict_email(sub_input, body_input):

    c_sub = preprocessing(sub_input)

    c_body = preprocessing(body_input)

    s = subject_vectorizer.transform([c_sub])

    b = body_vectorizer.transform([c_body])

    f = hstack([s, b])

    f = f.toarray()

    prediction = model.predict(f)

    prob = prediction[0][0]

    return prob


# =========================
# STREAMLIT PART STARTS
# =========================


# Creates big title on webpage
st.title("Email Spam Classifier")


# Single line input box for email subject
subject = st.text_input("Enter Email Subject")


# Multi-line input box for email body
body = st.text_area("Enter Email Body")


# Creates button
# Code inside runs only when button is clicked
if st.button("Predict"):


    # Calls your prediction function
    prob = predict_email(subject, body)


    # If probability > 0.5 then spam
    if prob > 0.5:

        # Red colored output box
        st.error("Spam Email Detected")

    else:

        # Green colored output box
        st.success("Not Spam (Ham)")


    # Shows probability value
    st.write(f"Spam Probability : {prob:.4f}")


    # Progress bar from 0 to 1
    st.progress(float(prob))