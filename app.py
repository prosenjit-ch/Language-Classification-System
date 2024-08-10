import streamlit as st
import os
from PIL import Image
import google.generativeai as genai



# initialize our streamlit app

st.set_page_config(page_title="Language Classifiaction")

st.header("Language Classifiaction")

input_content = st.text_input("Input Content: ", key = "input")





import pickle
import string
# from sklearn.feature_extraction.text import TfidfVectorizer


# Load the model from the file
with open('knn_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf1 = pickle.load(file)


 




submit=st.button("Submit")




# if submit button is clicked

if submit:

    st.write("Entered Content:", input_content)

    st.write("The response is:")

    clean_content = input_content.translate(str.maketrans('', '', string.punctuation))
    transform_text = tfidf1.transform([clean_content])
    pred = loaded_model.predict(transform_text)

    if(pred[0] == 0):
        st.write('Assamese Language')
    elif (pred[0] == 1):
        st.write('Bangla Language')
    elif(pred[0] == 2):
        st.write('Chakma Language')
    elif(pred[0] == 3):
        st.write('Chittagonian Language')
    else:
        st.write('Kokborok Language')


   










