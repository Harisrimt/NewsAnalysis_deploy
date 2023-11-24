#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:04:27 2023

@author: hari
"""

/home/adminuser/venv/bin/python -m pip install --upgrade pip

import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the pickled Random Forest model
# with open('/Users/hari/Desktop/Projects/P306/Model Building and deployment/random_forest_model.pkl', 'rb') as model_file:
with open('RFC.sav', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

# Load the pickled vectorizer
# with open('/Users/hari/Desktop/Projects/P306/Model Building and deployment/vectorizer.pkl', 'rb') as vectorizer_file:
with open('vect.sav', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
def main():
    st.title("Real and Fake News Prediction")

    # User input for news text
    news_text = st.text_area("Enter the news text:", "")

    # Make prediction when the user clicks the "Predict" button
    if st.button("Predict"):
        if news_text:
            # Vectorize the input text
            text_vectorized = vectorizer.transform([news_text])

            # Make prediction using the model
            prediction = random_forest_model.predict(text_vectorized)

            # Display the prediction result
            st.write("Prediction:", "Fake News" if prediction[0] == 0 else "Real News")
        else:
            st.warning("Please enter some news text for prediction.")

if __name__ == "__main__":
    main()
    
