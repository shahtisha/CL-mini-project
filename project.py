import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load datasets
@st.cache_data
def load_data():
    df_hindi = pd.read_csv(r"C:\Users\shaht\Downloads\hindi_data.csv")
    df_punjabi = pd.read_csv(r"C:\Users\shaht\Downloads\punjabi_lexicon.csv")
    df_gujrati = pd.read_csv(r"C:\Users\shaht\Downloads\gujarati_data.csv")
    df_marathi = pd.read_csv(r"C:\Users\shaht\Downloads\final_marathi_data.csv")
    return df_hindi, df_punjabi, df_gujrati, df_marathi

# Preprocess and train models
@st.cache_data
def preprocess_and_train(df_hindi, df_marathi, df_gujrati, df_punjabi):
    tfidf_vectorizer = TfidfVectorizer()

    X_hindi = tfidf_vectorizer.fit_transform(df_hindi['Sentence'])
    X_punjabi = tfidf_vectorizer.transform(df_punjabi['Word'])
    X_gujrati = tfidf_vectorizer.transform(df_gujrati['Sentence'])
    X_marathi = tfidf_vectorizer.transform(df_marathi['Sentence'])

    y_hindi = df_hindi['Score']
    y_punjabi = (df_punjabi['Positive Score'] > df_punjabi['Negative Score']).astype(int)
    y_gujrati = df_gujrati['Score']
    y_marathi = df_marathi['Score']

    X_hindi_train, X_hindi_test, y_hindi_train, y_hindi_test = train_test_split(X_hindi, y_hindi, test_size=0.2, random_state=42)
    X_marathi_train, X_marathi_test, y_marathi_train, y_marathi_test = train_test_split(X_marathi, y_marathi, test_size=0.2, random_state=42)
    X_gujrati_train, X_gujrati_test, y_gujrati_train, y_gujrati_test = train_test_split(X_gujrati, y_gujrati, test_size=0.2, random_state=42)
    X_punjabi_train, X_punjabi_test, y_punjabi_train, y_punjabi_test = train_test_split(X_punjabi, y_punjabi, test_size=0.2, random_state=42)

    model_hindi = RandomForestClassifier()
    model_hindi.fit(X_hindi_train, y_hindi_train)

    model_marathi = RandomForestClassifier()
    model_marathi.fit(X_marathi_train, y_marathi_train)

    model_gujrati = RandomForestClassifier()
    model_gujrati.fit(X_gujrati_train, y_gujrati_train)

    model_punjabi = RandomForestClassifier()
    model_punjabi.fit(X_punjabi_train, y_punjabi_train)

    # Save the TF-IDF vectorizer to a pickle file
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    return model_hindi, model_marathi, model_gujrati, model_punjabi

def main():
    st.title("Sentiment Analysis for Indian Languages")

    df_hindi, df_punjabi, df_gujrati, df_marathi = load_data()
    model_hindi, model_marathi, model_gujrati, model_punjabi = preprocess_and_train(df_hindi, df_marathi, df_gujrati, df_punjabi)

    # Load the pickled TF-IDF vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    option = st.sidebar.selectbox("Select Language", ["Hindi", "Marathi", "Gujarati", "Punjabi"])

    if option == "Hindi":
        input_text = st.text_area("Enter Hindi text for sentiment analysis")

        if input_text:
            X_input = tfidf_vectorizer.transform([input_text])
            y_pred = model_hindi.predict(X_input)[0]

            if y_pred == 1:
                sentiment_label = "Positive"
            elif y_pred == -1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            st.write(f"Sentiment Label: {sentiment_label}")

    elif option == "Marathi":
        input_text = st.text_area("Enter Marathi text for sentiment analysis")

        if input_text:
            X_input = tfidf_vectorizer.transform([input_text])
            y_pred = model_marathi.predict(X_input)[0]

            if y_pred == 1:
                sentiment_label = "Positive"
            elif y_pred == -1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            st.write(f"Sentiment Label: {sentiment_label}")

    elif option == "Gujarati":
        input_text = st.text_area("Enter Gujarati text for sentiment analysis")

        if input_text:
            X_input = tfidf_vectorizer.transform([input_text])
            y_pred = model_gujrati.predict(X_input)[0]

            if y_pred == 1:
                sentiment_label = "Positive"
            elif y_pred == 0:
                sentiment_label = "Neutral"
            else:
                sentiment_label = "Negative"

            st.write(f"Sentiment Label: {sentiment_label}")

    elif option == "Punjabi":
        input_text = st.text_area("Enter Punjabi text for sentiment analysis")

        if input_text:
            X_input = tfidf_vectorizer.transform([input_text])
            y_pred = model_punjabi.predict(X_input)[0]

            if y_pred == 1:
                sentiment_label = "Positive"
            else:
                sentiment_label = "Negative"

            st.write(f"Sentiment Label: {sentiment_label}")

if __name__ == "__main__":
    main()