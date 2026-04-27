import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

print("=" * 60)
print("       RecoAI — Hybrid Recommendation System")
print("       Combining Content Based + Collaborative")
print("=" * 60)

# =====================
# LOAD DATA
# =====================
movies = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/movies.csv")
ratings = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/ratings.csv")

# =====================
# FILTER POPULAR MOVIES
# =====================
movie_stats = ratings.groupby('movieId').agg(
    rating_count=('rating', 'count'),
    avg_rating=('rating', 'mean')
).reset_index()

popular_movies = movie_stats[movie_stats['rating_count'] >= 50]['movieId']
ratings_filtered = ratings[ratings['movieId'].isin(popular_movies)]

user_stats = ratings_filtered.groupby('userId').agg(
    rating_count=('rating', 'count')
).reset_index()

active_users = user_stats[user_stats['rating_count'] >= 50]['userId']
ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(active_users)]

movies_filtered = movies[movies['movieId'].isin(
    ratings_filtered['movieId'].unique())].reset_index(drop=True)

print(f"\n✅ Working with {len(movies_filtered)} popular movies!")
print(f"✅ Working with {ratings_filtered['userId'].nunique()} active users!")

# =====================
# MODULE 1 — CONTENT BASED
# =====================
print("\n📊 Building Content Based Model...")
movies_filtered['genres_clean'] = movies_filtered['genres'].str.replace(
    '|', ' ', regex=False)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_filtered['genres_clean'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_indices = pd.Series(
    movies_filtered.index,
    index=movies_filtered['title']
).drop_duplicates()

print("✅ Content Based Model Ready!")

# =====================
# MODULE 2 — COLLABORATIVE
# =====================
print("\n🤝 Building Collaborative Model...")
user_movie_matrix = ratings_filtered.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)

sparse_matrix = csr_matrix(user_movie_matrix.values)
knn_model = NearestNeighbors(
    metric='cosine',
    algorithm='brute',
    n_neighbors=10
)
knn_model.fit(sparse_matrix.T)

movie_mapper = dict(zip(movies_filtered['movieId'], movies_filtered['title']))
movie_inv_mapper = dict(zip(movies_filtered['title'], movies_filtered['movieId']))

print("✅ Collaborative Model Ready!")

# =====================
# HYBRID RECOMMENDATION FUNCTION
# =====================
def get_hybrid_recommendations(movie_title, n=10,
                                content_weight=0.4,
                                collab_weight=0.6):
    print(f"\n{'=' * 60}")
    print(f"🎬 RecoAI Recommendations for: {movie_title}")
    print(f"{'=' * 60}")

    scores = {}

    # --- Content Based Scores ---
    if movie_title in movie_indices:
        idx = movie_indices[movie_title]
        content_scores = list(enumerate(content_sim[idx]))
        content_scores = sorted(content_scores,
                                key=lambda x: x[1], reverse=True)
        content_scores = content_scores[1:n*3+1]

        for i, score in content_scores:
            title = movies_filtered['title'].iloc[i]
            if title != movie_title:
                scores[title] = scores.get(title, 0) + (score * content_weight)

    # --- Collaborative Scores ---
    if movie_title in movie_inv_mapper:
        movie_id = movie_inv_mapper[movie_title]
        if movie_id in user_movie_matrix.columns:
            movie_idx = list(user_movie_matrix.columns).index(movie_id)
            distances, indices = knn_model.kneighbors(
                sparse_matrix.T[movie_idx],
                n_neighbors=n*2+1
            )
            for i, idx in enumerate(indices.flatten()):
                movie_id_rec = user_movie_matrix.columns[idx]
                title = movie_mapper.get(movie_id_rec, "Unknown")
                if title != movie_title:
                    sim_score = 1 - distances.flatten()[i]
                    scores[title] = scores.get(title, 0) + (sim_score * collab_weight)

    # --- Combine & Sort ---
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = sorted_scores[:n]

    result = pd.DataFrame(top_recommendations,
                          columns=['Movie', 'Hybrid Score'])
    result['Hybrid Score'] = result['Hybrid Score'].round(3)
    print(result.to_string(index=False))
    return result

# =====================
# TEST HYBRID SYSTEM
# =====================
test_movies = [
    "Toy Story (1995)",
    "Matrix, The (1999)",
    "Forrest Gump (1994)"
]

all_results = {}
for movie in test_movies:
    result = get_hybrid_recommendations(movie)
    all_results[movie] = result

# =====================
# VISUALIZATION
# =====================
print("\n📊 Generating Visualization...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("RecoAI — Hybrid Recommendations", fontsize=16, fontweight='bold')

colors = ['blue', 'red', 'green']
for ax, (movie, result), color in zip(axes, all_results.items(), colors):
    ax.barh(result['Movie'], result['Hybrid Score'], color=color)
    ax.set_title(f"Similar to:\n{movie}", fontsize=10)
    ax.set_xlabel("Hybrid Score")
    ax.invert_yaxis()

plt.tight_layout()
plt.show()

print("\n✅ Hybrid System Complete!")
print("🚀 RecoAI is now working like Netflix + Amazon!")