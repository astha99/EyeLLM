import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

def calculate_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2) or text1.strip() == "" or text2.strip() == "":
        return None  # Handle empty or NaN values

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return round(similarity_score, 4)

def get_openai_embedding(text):
    if pd.isna(text) or text.strip() == "":
        return None  # Handle empty or NaN values
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return np.array(response["data"][0]["embedding"])

def calculate_semantic_similarity(text1, text2):
    if isinstance(text1, tuple):
        text1 = text1[0]  # Extract the actual text if it's inside a tuple
    if isinstance(text2, tuple):
        text2 = text2[0]

    emb1 = get_openai_embedding(text1)
    emb2 = get_openai_embedding(text2)

    if emb1 is None or emb2 is None:
        return None

    similarity_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return round(similarity_score, 4)

def calculate_jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts."""
    if pd.isna(text1) or pd.isna(text2) or text1.strip() == "" or text2.strip() == "":
        return None  # Handle empty or NaN values

    set1, set2 = set(text1.split()), set(text2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return round(intersection / union, 4) if union != 0 else 0.0

def calculate_f1_score(text1, text2):
    """Calculate F1 score based on word overlap between two texts."""
    if pd.isna(text1) or pd.isna(text2) or text1.strip() == "" or text2.strip() == "":
        return None  # Handle empty or NaN values

    set1, set2 = set(text1.split()), set(text2.split())
    intersection = len(set1 & set2)

    if intersection == 0:
        return 0.0  # No common words, F1 score is 0

    precision = intersection / len(set2)  # How many words from text2 appear in text1
    recall = intersection / len(set1)     # How many words from text1 appear in text2

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)