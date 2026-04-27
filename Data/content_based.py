import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# LOAD DATA
# =====================
movies = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/movies.csv")
ratings = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/ratings.csv")

# =====================
# USE POPULAR MOVIES ONLY
# =====================
# Keep only movies with 1000+ ratings
popular_movies = ratings.groupby('movieId').filter(
    lambda x: len(x) >= 1000)['movieId'].unique()

# Filter movies dataset
movies = movies[movies['movieId'].isin(popular_movies)].reset_index(drop=True)

print("=" * 50)
print("RecoAI — Content Based Filtering")
print("=" * 50)
print(f"Working with {len(movies)} popular movies!")

# =====================
# PREPARE DATA
# =====================
movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)

# =====================
# CREATE TF-IDF MATRIX
# =====================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# =====================
# CALCULATE SIMILARITY
# =====================
print("\nCalculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("✅ Similarity calculated!")

# =====================
# RECOMMENDATION FUNCTION
# =====================
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, n=10):
    if title not in movie_indices:
        print(f"❌ Movie '{title}' not found!")
        return None
    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_idx = [i[0] for i in sim_scores]
    similarity = [round(i[1], 3) for i in sim_scores]
    recommendations = movies['title'].iloc[movie_idx].reset_index(drop=True)
    result = pd.DataFrame({
        'Movie': recommendations,
        'Similarity Score': similarity
    })
    return result

# =====================
# TEST RECOMMENDATIONS
# =====================
test_movies = [
    "Toy Story (1995)",
    "Matrix, The (1999)",
    "Forrest Gump (1994)"
]

for movie in test_movies:
    print("\n" + "=" * 50)
    print(f"🎬 Movies similar to: {movie}")
    print("=" * 50)
    recommendations = get_recommendations(movie)
    if recommendations is not None:
        print(recommendations.to_string(index=False))

print("\n✅ Content Based Filtering Complete!")