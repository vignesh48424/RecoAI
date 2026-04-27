from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

app = Flask(__name__)

# =====================
# LOAD DATA
# =====================
movies = pd.read_csv("Data/movies.csv")
ratings = pd.read_csv("Data/ratings.csv")

# =====================
# FILTER POPULAR MOVIES
# =====================
movie_stats = ratings.groupby('movieId').agg(
    rating_count=('rating', 'count')
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

# =====================
# CONTENT BASED MODEL
# =====================
movies_filtered['genres_clean'] = movies_filtered['genres'].str.replace(
    '|', ' ', regex=False)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_filtered['genres_clean'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(
    movies_filtered.index,
    index=movies_filtered['title']
).drop_duplicates()

# =====================
# COLLABORATIVE MODEL
# =====================
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

print("✅ RecoAI Models Loaded Successfully!")

# =====================
# HYBRID FUNCTION
# =====================
def get_recommendations(movie_title, n=10):
    scores = {}

    # Content Based
    if movie_title in movie_indices:
        idx = movie_indices[movie_title]
        content_scores = list(enumerate(content_sim[idx]))
        content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)
        content_scores = content_scores[1:n*3+1]
        for i, score in content_scores:
            title = movies_filtered['title'].iloc[i]
            if title != movie_title:
                scores[title] = scores.get(title, 0) + (score * 0.4)

    # Collaborative
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
                    scores[title] = scores.get(title, 0) + (sim_score * 0.6)

    # Sort & return
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:n]

# =====================
# GET ALL MOVIE TITLES
# =====================
all_movies = sorted(movies_filtered['title'].tolist())

# =====================
# FLASK ROUTES
# =====================
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_movie = ""
    error = ""

    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        if selected_movie:
            results = get_recommendations(selected_movie)
            if results:
                recommendations = [
                    {'title': title, 'score': round(score, 3)}
                    for title, score in results
                ]
            else:
                error = "No recommendations found!"

    return render_template('index.html',
                           movies=all_movies,
                           recommendations=recommendations,
                           selected_movie=selected_movie,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)