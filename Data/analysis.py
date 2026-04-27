import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# LOAD DATA
# =====================
# ✅ Correct
movies = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/movies.csv")
ratings = pd.read_csv("C:/Users/vigne/OneDrive/Desktop/RecoAI/data/ratings.csv")

# =====================
# EXPLORE MOVIES
# =====================
print("=" * 40)
print("MOVIES DATASET")
print("=" * 40)
print(movies.head())
print("\nShape:", movies.shape)
print("\nColumns:", movies.columns.tolist())
print("\nMissing values:\n", movies.isnull().sum())

# =====================
# EXPLORE RATINGS
# =====================
print("\n" + "=" * 40)
print("RATINGS DATASET")
print("=" * 40)
print(ratings.head())
print("\nShape:", ratings.shape)
print("\nColumns:", ratings.columns.tolist())
print("\nMissing values:\n", ratings.isnull().sum())

# =====================
# BASIC STATS
# =====================
print("\n" + "=" * 40)
print("BASIC STATS")
print("=" * 40)
print("Total Movies:", movies.shape[0])
print("Total Ratings:", ratings.shape[0])
print("Total Users:", ratings['userId'].nunique())
print("Average Rating:", round(ratings['rating'].mean(), 2))
print("Min Rating:", ratings['rating'].min())
print("Max Rating:", ratings['rating'].max())

# =====================
# TOP 10 RATED MOVIES
# =====================
print("\n" + "=" * 40)
print("TOP 10 MOST RATED MOVIES")
print("=" * 40)
movie_ratings = ratings.groupby('movieId').agg(
    total_ratings=('rating', 'count'),
    avg_rating=('rating', 'mean')
).reset_index()

movie_ratings = movie_ratings.merge(movies, on='movieId')
top_movies = movie_ratings.sort_values('total_ratings', ascending=False).head(10)
print(top_movies[['title', 'total_ratings', 'avg_rating']])

# =====================
# VISUALIZATIONS
# =====================

# Chart 1 - Rating Distribution
plt.figure(figsize=(8, 5))
ratings['rating'].value_counts().sort_index().plot(kind='bar', color='blue')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Chart 2 - Top 10 Most Rated Movies
plt.figure(figsize=(12, 6))
plt.barh(top_movies['title'], top_movies['total_ratings'], color='green')
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of Ratings")
plt.tight_layout()
plt.show()

# Chart 3 - Genre Distribution
genres = movies['genres'].str.split('|').explode()
top_genres = genres.value_counts().head(10)
plt.figure(figsize=(12, 6))
top_genres.plot(kind='bar', color='orange')
plt.title("Top 10 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("\n✅ Data Analysis Complete!")