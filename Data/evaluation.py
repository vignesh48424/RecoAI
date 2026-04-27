import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

print("=" * 60)
print("       RecoAI — Model Evaluation")
print("=" * 60)

# =====================
# LOAD DATA
# =====================
movies = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/movies.csv")
ratings = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/ratings.csv")

# =====================
# FILTER DATA
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
ratings_filtered = ratings_filtered[
    ratings_filtered['userId'].isin(active_users)]

print(f"✅ Working with {ratings_filtered.shape[0]} ratings!")
print(f"✅ Users: {ratings_filtered['userId'].nunique()}")
print(f"✅ Movies: {ratings_filtered['movieId'].nunique()}")

# =====================
# SPLIT DATA
# =====================
print("\n📊 Splitting data into Train & Test...")
train_data, test_data = train_test_split(
    ratings_filtered,
    test_size=0.2,
    random_state=42
)
print(f"✅ Training set: {train_data.shape[0]} ratings")
print(f"✅ Testing set: {test_data.shape[0]} ratings")

# =====================
# BUILD MODEL
# =====================
print("\n🤝 Building Recommendation Model...")
user_movie_matrix = train_data.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)

sparse_matrix = csr_matrix(user_movie_matrix.values)
model = NearestNeighbors(
    metric='cosine',
    algorithm='brute',
    n_neighbors=10
)
model.fit(sparse_matrix)
print("✅ Model trained!")

# =====================
# EVALUATE — RMSE & MAE
# =====================
print("\n📊 Evaluating Model...")

actual_ratings = []
predicted_ratings = []

# Test on sample of test data
test_sample = test_data.sample(min(500, len(test_data)), random_state=42)

for _, row in test_sample.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    actual_rating = row['rating']

    # Check if user and movie exist in training data
    if (user_id in user_movie_matrix.index and
            movie_id in user_movie_matrix.columns):

        # Get user index
        user_idx = list(user_movie_matrix.index).index(user_id)

        # Find similar users
        distances, indices = model.kneighbors(
            sparse_matrix[user_idx],
            n_neighbors=5
        )

        # Predict rating based on similar users
        similar_users = indices.flatten()[1:]
        similar_distances = distances.flatten()[1:]

        # Get ratings from similar users
        movie_col_idx = list(user_movie_matrix.columns).index(movie_id)
        similar_ratings = []

        for sim_user_idx in similar_users:
            rating = user_movie_matrix.iloc[sim_user_idx, movie_col_idx]
            if rating > 0:
                similar_ratings.append(rating)

        if similar_ratings:
            predicted_rating = np.mean(similar_ratings)
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
mae = mean_absolute_error(actual_ratings, predicted_ratings)

print(f"\n{'=' * 40}")
print(f"📊 RecoAI Performance Metrics:")
print(f"{'=' * 40}")
print(f"✅ RMSE: {round(rmse, 4)}")
print(f"✅ MAE:  {round(mae, 4)}")
print(f"✅ Predictions made: {len(actual_ratings)}")

# =====================
# EVALUATE — COVERAGE
# =====================
total_movies = ratings_filtered['movieId'].nunique()
covered_movies = user_movie_matrix.shape[1]
coverage = round((covered_movies / total_movies) * 100, 2)
print(f"✅ Coverage: {coverage}%")

# =====================
# RATING DISTRIBUTION
# =====================
avg_actual = round(np.mean(actual_ratings), 2)
avg_predicted = round(np.mean(predicted_ratings), 2)
print(f"\n📊 Rating Analysis:")
print(f"Average Actual Rating:    {avg_actual}")
print(f"Average Predicted Rating: {avg_predicted}")

# =====================
# VISUALIZATION
# =====================
print("\n📊 Generating Evaluation Charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RecoAI — Model Evaluation Dashboard",
             fontsize=16, fontweight='bold')

# Chart 1 — Actual vs Predicted
axes[0, 0].scatter(actual_ratings, predicted_ratings,
                   alpha=0.5, color='blue')
axes[0, 0].plot([0, 5], [0, 5], 'r--', label='Perfect Prediction')
axes[0, 0].set_title("Actual vs Predicted Ratings")
axes[0, 0].set_xlabel("Actual Rating")
axes[0, 0].set_ylabel("Predicted Rating")
axes[0, 0].legend()

# Chart 2 — Error Distribution
errors = [a - p for a, p in zip(actual_ratings, predicted_ratings)]
axes[0, 1].hist(errors, bins=20, color='orange', edgecolor='black')
axes[0, 1].set_title("Prediction Error Distribution")
axes[0, 1].set_xlabel("Error (Actual - Predicted)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].axvline(x=0, color='red', linestyle='--')

# Chart 3 — Metrics Comparison
metrics = ['RMSE', 'MAE']
values = [rmse, mae]
axes[1, 0].bar(metrics, values, color=['blue', 'green'])
axes[1, 0].set_title("Error Metrics")
axes[1, 0].set_ylabel("Error Value")
for i, v in enumerate(values):
    axes[1, 0].text(i, v + 0.01, str(round(v, 4)),
                    ha='center', fontweight='bold')

# Chart 4 — Rating Distribution Comparison
axes[1, 1].hist(actual_ratings, bins=10, alpha=0.5,
                color='blue', label='Actual')
axes[1, 1].hist(predicted_ratings, bins=10, alpha=0.5,
                color='red', label='Predicted')
axes[1, 1].set_title("Rating Distribution")
axes[1, 1].set_xlabel("Rating")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n✅ Model Evaluation Complete!")
print(f"\n🎯 RecoAI Summary:")
print(f"   RMSE: {round(rmse, 4)} → Average prediction error")
print(f"   MAE:  {round(mae, 4)} → Average absolute error")
print(f"   Coverage: {coverage}% → Movies RecoAI can recommend")