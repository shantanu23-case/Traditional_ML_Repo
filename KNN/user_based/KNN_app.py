from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
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
#Each row becomes is a userID ,index parameter means the rows.
#Each Coloumn is movieID .
#values='rating'. Values means the values of userid row
user_movie_matrix.fillna(0, inplace=True)
#Some user might not have rated any moveies so used fillna function to change NA to 0.

model = KNeighborsClassifier(metric="cosine",algorithm="brute")
#Metric - cosine similarity to measure how similar two users are based on their movie ratings.
# Imagine each user is a warrior with different styles (ratings).
# We are not judging them by how many battles they fought (magnitude).But how similar their fighting style is — 
# angle of attack, movement, etc.
# Cosine similarity = comparing fighting style direction
# Brute = match each warrior against every other to find the most similar ones.
model.fit(user_movie_matrix)
distances, indices = model.kneighbors(user_movie_matrix.iloc[0:1], n_neighbors=3)
#Finds the 3 most similar movies to the first movie in user_movie_matrix.
#iloc[0:1] selects the first movie row (as a DataFrame).
#A DataFrame is a 2-dimensional table-like structure provided by the pandas library in Python.
#Think of it like an Excel sheet or a SQL table
# kneighbors() returns:
# distances: how far (or dissimilar) each neighbor is
# indices: the row positions (not IDs) of the similar movies
#>>>>>>>>>>>>> Why We Don't Use Cross-Validation in User-Based KNN >>>>>>>>>>>

#User-based KNN (a memory-based) is not a supervised learning model like regression or classification. So:
#There's no "training" in the typical ML sense.
#You're not predicting labels based on features — you're calculating similarity between users or items.
#Your "model" (like NearestNeighbors) just stores the matrix and helps you find nearest neighbors.

#Cross-validation is used only when you're "training" a predictive model — that is, in supervised learning, like:

