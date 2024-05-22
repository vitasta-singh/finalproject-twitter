import streamlit as st
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to preprocess and vectorize text
def preprocess_and_vectorize(text):
    # Preprocess the input text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Remove leading/trailing spaces
    
    # Vectorize the text using the loaded vectorizer
    vectorized_text = vectorizer.transform([text])
    return vectorized_text

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess and vectorize the input text
    vectorized_text = preprocess_and_vectorize(text)
    # Make predictions using the loaded model
    prediction = model.predict(vectorized_text)
    return prediction

# Streamlit app
def main():
    # Set title and description
    st.title('Sentiment Analysis App')
    st.markdown("## This app predicts the sentiment of text. The dataset contain millions of tweets. If the Prediction value is 1 then it is positive comment else it is a negative comment. ")

    # Add a text input for user to enter text
    text = st.text_area('Enter text here:')

    # Add a button to make predictions
    if st.button('Predict'):
        if text:  # Ensure that the text is not empty
            # Make prediction when the button is clicked
            prediction = predict_sentiment(text)
            sentiment = 'Positive' if prediction == 1 else 'Negative'  # Adjust based on your model
            st.write('Prediction:', sentiment)
        else:
            st.write('Please enter text to analyze.')

if __name__ == '__main__':
    main()