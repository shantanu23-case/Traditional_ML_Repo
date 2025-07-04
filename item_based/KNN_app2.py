from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Final Key point Before you dive into this >
# So KNN has two logics with one it is predicting and with another one it 
# is calcualting the similarty and then aligning your decison.
movies_rating = pd.read_csv("data/ml-10m/ml-10M100K/ratings.dat",sep="::", 
                            engine="python", 
                            names=["userId", "movieId", "rating", "timestamp"])
movies=pd.read_csv("data/ml-10m/ml-10M100K/movies.dat",
                   sep="::", 
                   engine="python", names=["movieId", "title", "genres"])

user_movie_matrix = movies_rating.pivot(index='userId', columns='movieId', values='rating')
user_movie_matrix = user_movie_matrix.T
user_movie_matrix.fillna(0, inplace=True)


#Compute cosine similarity between movies
cosine_sim = cosine_similarity(user_movie_matrix)


model = KNeighborsClassifier(metric="cosine",algorithm="brute")
model.fit(user_movie_matrix)