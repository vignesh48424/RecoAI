import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# =====================
# LOAD DATA
# =====================
movies = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/movies.csv")
ratings = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/ratings.csv")

print("=" * 50)
print("RecoAI — Collaborative Filtering")
print("=" * 50)
print(f"Movies: {movies.shape[0]}")
print(f"Ratings: {ratings.shape[0]}")
print(f"Users: {ratings['userId'].nunique()}")

# =====================
# FILTER POPULAR MOVIES
# =====================
# Keep movies with 50+ ratings
movie_stats = ratings.groupby('movieId').agg(
    rating_count=('rating', 'count'),
    avg_rating=('rating', 'mean')
).reset_index()

popular_movies = movie_stats[movie_stats['rating_count'] >= 50]['movieId']
ratings_filtered = ratings[ratings['movieId'].isin(popular_movies)]

# Keep active users with 50+ ratings
user_stats = ratings_filtered.groupby('userId').agg(
    rating_count=('rating', 'count')
).reset_index()

active_users = user_stats[user_stats['rating_count'] >= 50]['userId']
ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(active_users)]

print(f"\nAfter filtering:")
print(f"Movies: {ratings_filtered['movieId'].nunique()}")
print(f"Users: {ratings_filtered['userId'].nunique()}")
print(f"Ratings: {ratings_filtered.shape[0]}")

# =====================
# CREATE USER-MOVIE MATRIX
# =====================
print("\nCreating User-Movie Matrix...")
user_movie_matrix = ratings_filtered.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)

print(f"Matrix Shape: {user_movie_matrix.shape}")
print(f"Users × Movies = {user_movie_matrix.shape[0]} × {user_movie_matrix.shape[1]}")

# Convert to sparse matrix for efficiency
sparse_matrix = csr_matrix(user_movie_matrix.values)

# =====================
# TRAIN KNN MODEL
# =====================
print("\nTraining KNN model...")
model = NearestNeighbors(
    metric='cosine',
    algorithm='brute',
    n_neighbors=10
)
model.fit(sparse_matrix.T)  # Transpose for movie based filtering
print("✅ KNN Model trained!")

# =====================
# RECOMMENDATION FUNCTION
# =====================
# Create movie index mapping
movie_mapper = dict(zip(movies['movieId'], movies['title']))
movie_inv_mapper = dict(zip(movies['title'], movies['movieId']))

def get_movie_recommendations(movie_title, n=10):
    # Check if movie exists
    if movie_title not in movie_inv_mapper:
        print(f"❌ Movie '{movie_title}' not found!")
        return None

    movie_id = movie_inv_mapper[movie_title]

    # Check if movie is in matrix
    if movie_id not in user_movie_matrix.columns:
        print(f"❌ Movie '{movie_title}' not in filtered dataset!")
        return None

    # Get movie index in matrix
    movie_idx = list(user_movie_matrix.columns).index(movie_id)

    # Find similar movies
    distances, indices = model.kneighbors(
        sparse_matrix.T[movie_idx],
        n_neighbors=n+1
    )

    # Get recommendations
    similar_movies = []
    similarity_scores = []

    for i, idx in enumerate(indices.flatten()):
        if idx != movie_idx:
            movie_id_rec = user_movie_matrix.columns[idx]
            similar_movies.append(movie_mapper.get(movie_id_rec, "Unknown"))
            similarity_scores.append(round(1 - distances.flatten()[i], 3))

    result = pd.DataFrame({
        'Movie': similar_movies[:n],
        'Similarity Score': similarity_scores[:n]
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
    recommendations = get_movie_recommendations(movie)
    if recommendations is not None:
        print(recommendations.to_string(index=False))

print("\n✅ Collaborative Filtering Complete!")