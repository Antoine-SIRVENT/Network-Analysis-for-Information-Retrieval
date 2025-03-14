import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()




def search_engine(query, vectorizer, X, data, top_n=5):
    """Given a user query, vectorizer, doc matrix X, returns top_n results."""
    query_vector = vectorizer.transform([query])
    cos_sim = cosine_similarity(query_vector, X).flatten()
    top_indices = cos_sim.argsort()[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = cos_sim[idx]
        title = data.loc[idx, 'title']
        abstract_text = data.loc[idx, 'abstract'] if pd.notnull(data.loc[idx, 'abstract']) else ""
        results.append((rank, score, title, abstract_text))
    return results

def preprocess_lemm(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

