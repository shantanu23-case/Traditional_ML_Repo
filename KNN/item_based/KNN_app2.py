from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# =====================
#  PHASE 1: Calculate Item-to-Item Similarity (based on other users' ratings)
# =====================

# Load ratings
movies_rating = pd.read_csv("data/ml-10m/ml-10M100K/ratings.dat", sep="::", 
                            engine="python", 
                            names=["userId", "movieId", "rating", "timestamp"])

# Load movie titles
movies = pd.read_csv("data/ml-10m/ml-10M100K/movies.dat", sep="::", 
                     engine="python", 
                     names=["movieId", "title", "genres"])

# Create user-movie rating matrix (rows = users, columns = movies)
user_movie_matrix = movies_rating.pivot(index='userId', columns='movieId', values='rating')

# Transpose to get movie-user matrix (rows = movies, columns = users)
user_movie_matrix = user_movie_matrix.T

# Fill missing values with 0 (to apply cosine similarity)
user_movie_matrix.fillna(0, inplace=True)

# Compute cosine similarity between movies (items)
cosine_sim = cosine_similarity(user_movie_matrix)

# Convert to DataFrame for easy lookup
similarity_df = pd.DataFrame(cosine_sim, 
                             index=user_movie_matrix.index, 
                             columns=user_movie_matrix.index)

# Movie ID to title mapping
movie_titles = movies[['movieId', 'title']].drop_duplicates().set_index('movieId')

# =====================
# PHASE 2: Personalized Recommendation using user's liked movie
# =====================

def recommend_similar_movies(movie_id, top_n=5):
    if movie_id not in similarity_df.index:
        return f"Movie ID {movie_id} not found in similarity matrix."

    # Get similarity scores for the input movie
    similar_scores = similarity_df[movie_id].sort_values(ascending=False)

    # Exclude the movie itself and select top N similar
    top_similar_ids = similar_scores.iloc[1:top_n+1].index

    # Fetch titles of recommended movies
    recommended_movies = movie_titles.loc[top_similar_ids].reset_index()
    return recommended_movies

# 🧪 Example usage:
recommend_similar_movies(1, top_n=5)  # Try with movieId = 1 (e.g., Toy Story)
