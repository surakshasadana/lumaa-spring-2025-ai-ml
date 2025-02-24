"""
Content-Based Movie Recommendation System (Overview Only)

This script loads a CSV dataset ("movie_dataset.csv") containing movie data,
and uses the movie's overview (plot summary) to build a content-based recommendation system.
It converts the overviews into TF-IDF vectors and computes cosine similarity between
the user's input description and each movie's overview. The top 5 recommendations are printed
with their similarity scores.

Usage:
    python recommend.py

Requirements:
    - The CSV dataset (movie_dataset.csv) must be in the same directory.
    - The dataset must include the columns "original_title" and "overview".
"""

import re
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_text(text):
    """
    Preprocess text by:
      - Converting to lowercase.
      - Removing punctuation (by replacing them with spaces).
      - Removing extra whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()

def build_tfidf_matrix(text_series):
    """
    Build a TF-IDF matrix for the given text data.
    Uses unigrams and bigrams.
    
    Args:
        text_series (pd.Series): Series containing text data.
        
    Returns:
        tfidf_matrix: The TF-IDF feature matrix.
        vectorizer: The fitted TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(text_series)
    return tfidf_matrix, vectorizer

def get_recommendations(query, df, tfidf_matrix, vectorizer, top_n=5):
    """
    Compute cosine similarity between the user's query and the TF-IDF vectors
    of the movie overviews, and return the top N recommendations.
    
    Args:
        query (str): User's text query.
        df (pd.DataFrame): DataFrame containing movies.
        tfidf_matrix: TF-IDF matrix of the movie overviews.
        vectorizer: Fitted TfidfVectorizer.
        top_n (int): Number of top recommendations to return.
        
    Returns:
        pd.DataFrame: DataFrame with recommended movies including title, overview, and similarity score.
    """
    query_processed = preprocess_text(query)
    query_vec = vectorizer.transform([query_processed])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity'] = cosine_sim[top_indices]
    return recommendations[['original_title', 'overview', 'similarity']]

def main():
    dataset_path = "movie_dataset.csv"
    
    # Load dataset
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Check for required columns
    if 'original_title' not in df.columns or 'overview' not in df.columns:
        print("Error: The dataset must contain 'original_title' and 'overview' columns.")
        sys.exit(1)
    
    # Fill missing overviews
    df['overview'] = df['overview'].fillna('')
    
    # Preprocess the overviews
    df['overview'] = df['overview'].apply(preprocess_text)
    
    # Build TF-IDF matrix on the "overview" column
    tfidf_matrix, vectorizer = build_tfidf_matrix(df['overview'])
    
    # Prompt user for input
    print("Enter a description of your movie preferences:")
    user_query = input(">> ")
    
    # Get top 5 recommendations
    recommendations = get_recommendations(user_query, df, tfidf_matrix, vectorizer, top_n=5)
    
    # Print the recommendations
    print("\nRecommended Movies:")
    if recommendations.empty:
        print("No recommendations found.")
        sys.exit(0)
    
    for i, (index, row) in enumerate(recommendations.iterrows(), start=1):
        title = row['original_title']
        similarity = row['similarity']
        overview = row['overview']
        print(f"\n{i}. {title}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Overview: {overview}")

if __name__ == "__main__":
    main()
